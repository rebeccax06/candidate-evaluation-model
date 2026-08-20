"""
Background worker for processing batch evaluation jobs.

This worker runs on Railway (or any server) and polls Supabase for pending jobs.
It processes each candidate evaluation and saves results back to Supabase.

Usage:
    python -m candidate_evaluator.worker

Environment variables required:
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase service role key (for worker)
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from uuid import uuid4

from candidate_evaluator.database import Database, get_supabase_client
from candidate_evaluator.storage import Storage, cleanup_temp_files
from candidate_evaluator.core.evaluator import CandidateEvaluator
from candidate_evaluator.core.processing import (
    EvaluationMode,
    eval_concurrency,
    evaluate_chunk_concurrently,
)
from candidate_evaluator.utils.config import Config, APIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

WORKER_ID = f"worker-{uuid4().hex[:8]}"
POLL_INTERVAL = 5  # seconds
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global SHUTDOWN_REQUESTED
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class Worker:
    """Background worker for processing evaluation jobs."""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.db = Database(self.client)
        self.storage = Storage(self.client)
        logger.info(f"Worker {WORKER_ID} initialized")
    
    def get_user_api_key(self, user_id: str) -> str | None:
        """Get the Anthropic API key for a user."""
        return self.db.get_user_api_key(user_id)
    
    def create_evaluator(self, api_key: str) -> CandidateEvaluator:
        """Create an evaluator instance with the user's API key."""
        config = Config(api=APIConfig(anthropic_api_key=api_key))
        return CandidateEvaluator(config)
    
    def process_job(self, job: dict) -> None:
        """Process a single batch job."""
        job_id = job["id"]
        user_id = job["user_id"]
        file_paths = job.get("file_paths", [])
        mode = EvaluationMode.coerce(job.get("evaluation_mode"))
        config = Database.get_job_config(job)
        role = config.get("role")
        screen_description = (config.get("screen_description") or "").strip()

        logger.info(f"Processing job {job_id} with {len(file_paths)} candidates")

        if mode is EvaluationMode.SCREEN and not screen_description:
            logger.error(f"Screening job {job_id} has no target profile description")
            self.db.complete_job(job_id, error="Screening job is missing a target profile description")
            return

        api_key = self.get_user_api_key(user_id)
        if not api_key:
            logger.error(f"No API key found for user {user_id}")
            self.db.complete_job(job_id, error="No Anthropic API key configured")
            return
        
        try:
            evaluator = self.create_evaluator(api_key)
        except Exception as e:
            logger.error(f"Failed to create evaluator: {e}")
            self.db.complete_job(job_id, error=f"Failed to initialize evaluator: {e}")
            return
        
        # Candidates already evaluated for this job (so re-queued jobs that had files
        # appended only process the newly added candidates instead of duplicating).
        already_done = {
            e.get("candidate_id")
            for e in self.db.get_job_evaluations(job_id)
            if e.get("candidate_id")
        }
        completed = len(already_done)
        failed = 0
        
        pending = []
        for storage_path in file_paths:
            candidate_id = self._extract_candidate_id(storage_path)
            if candidate_id in already_done:
                logger.info(f"Skipping already-evaluated candidate: {candidate_id}")
                continue
            pending.append((candidate_id, storage_path))

        # Candidates are processed in chunks: files download and results save
        # on the main thread (the Supabase client isn't guaranteed
        # thread-safe); only the Claude calls run concurrently.
        concurrency = eval_concurrency()
        for chunk_start in range(0, len(pending), concurrency):
            if SHUTDOWN_REQUESTED:
                logger.info("Shutdown requested, stopping job processing")
                break

            chunk = pending[chunk_start:chunk_start + concurrency]
            logger.info(
                f"Processing candidates {chunk_start + 1}-{chunk_start + len(chunk)}"
                f"/{len(pending)}: {', '.join(cid for cid, _ in chunk)}"
            )
            self.db.update_job_progress(
                job_id, completed, failed, ", ".join(cid for cid, _ in chunk)
            )

            temp_paths = {}
            tasks = []
            for candidate_id, storage_path in chunk:
                try:
                    temp_path = self.storage.download_to_temp(storage_path)
                    temp_paths[candidate_id] = temp_path
                    tasks.append((candidate_id, [temp_path]))
                except Exception as e:
                    logger.error(f"Error downloading {candidate_id}: {e}")
                    failed += 1

            try:
                outcomes = evaluate_chunk_concurrently(
                    evaluator,
                    mode,
                    tasks,
                    role=role,
                    description=screen_description,
                    max_workers=concurrency,
                )
                for candidate_id, _result, result_dict, error in outcomes:
                    if error is not None:
                        logger.error(f"Error evaluating {candidate_id}:\n{error}")
                        failed += 1
                        continue
                    try:
                        self.db.save_evaluation(
                            user_id=user_id,
                            candidate_id=candidate_id,
                            evaluation_type=mode.value,
                            result=result_dict,
                            job_id=job_id
                        )
                        completed += 1
                        already_done.add(candidate_id)
                        logger.info(f"Completed evaluation for {candidate_id}")
                    except Exception as e:
                        logger.error(f"Error saving evaluation for {candidate_id}: {e}")
                        failed += 1
            finally:
                cleanup_temp_files(list(temp_paths.values()))

            self.db.update_job_progress(job_id, completed, failed)
        
        if SHUTDOWN_REQUESTED:
            logger.info(f"Job {job_id} interrupted by shutdown")
            self.db.update_job_progress(job_id, completed, failed)
        else:
            if failed > 0 and completed == 0:
                self.db.complete_job(job_id, error=f"All {failed} candidates failed")
            else:
                self.db.complete_job(job_id)
                logger.info(f"Job {job_id} completed: {completed} success, {failed} failed")
    
    def _extract_candidate_id(self, storage_path: str) -> str:
        """Extract candidate ID from storage path."""
        filename = storage_path.split("/")[-1]
        if "_" in filename:
            filename = "_".join(filename.split("_")[1:])
        
        name = filename.rsplit(".", 1)[0] if "." in filename else filename
        return name
    
    def run(self) -> None:
        """Main worker loop."""
        logger.info(f"Worker {WORKER_ID} starting main loop")
        
        while not SHUTDOWN_REQUESTED:
            try:
                pending_jobs = self.db.get_pending_jobs(limit=1)
                
                if pending_jobs:
                    job = pending_jobs[0]
                    job_id = job["id"]
                    
                    if self.db.claim_job(job_id, WORKER_ID):
                        logger.info(f"Claimed job {job_id}")
                        self.process_job(job)
                    else:
                        logger.debug(f"Job {job_id} was claimed by another worker")
                else:
                    logger.debug("No pending jobs")
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
            
            if not SHUTDOWN_REQUESTED:
                time.sleep(POLL_INTERVAL)
        
        logger.info(f"Worker {WORKER_ID} shutting down")


def main():
    """Entry point for the worker."""
    logger.info("Starting candidate evaluation worker")
    
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)
    
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    main()
