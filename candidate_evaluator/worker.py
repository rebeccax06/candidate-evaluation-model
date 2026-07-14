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
from candidate_evaluator.utils.config import Config, APIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

WORKER_ID = f"worker-{uuid4().hex[:8]}"
POLL_INTERVAL = 5  # seconds
MAX_RETRIES = 3
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
        evaluation_mode = job.get("evaluation_mode", "criteria")
        is_holistic = evaluation_mode == "holistic"
        role = job.get("role")
        
        logger.info(f"Processing job {job_id} with {len(file_paths)} candidates")
        
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
        
        for i, storage_path in enumerate(file_paths):
            if SHUTDOWN_REQUESTED:
                logger.info("Shutdown requested, stopping job processing")
                break
            
            candidate_id = self._extract_candidate_id(storage_path)

            if candidate_id in already_done:
                logger.info(f"Skipping already-evaluated candidate: {candidate_id}")
                continue

            logger.info(f"Processing candidate {i+1}/{len(file_paths)}: {candidate_id}")
            
            self.db.update_job_progress(job_id, completed, failed, candidate_id)
            
            temp_paths = []
            try:
                temp_path = self.storage.download_to_temp(storage_path)
                temp_paths.append(temp_path)
                
                if is_holistic:
                    result = evaluator.evaluate_candidate_holistic(
                        candidate_id=candidate_id,
                        material_paths=temp_paths,
                        role=role
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(
                        result_dict["candidate"]["evaluation_date"]
                    )
                else:
                    result = evaluator.evaluate_candidate(
                        candidate_id=candidate_id,
                        material_paths=temp_paths,
                        role=role
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(
                        result_dict["candidate"]["evaluation_date"]
                    )
                    for score in result_dict.get("scores", []):
                        if "criterion" in score:
                            crit = score["criterion"]
                            score["criterion"] = crit.value if hasattr(crit, 'value') else str(crit)
                
                self.db.save_evaluation(
                    user_id=user_id,
                    candidate_id=candidate_id,
                    evaluation_type=evaluation_mode,
                    result=result_dict,
                    job_id=job_id
                )
                
                completed += 1
                already_done.add(candidate_id)
                logger.info(f"Completed evaluation for {candidate_id}")
                
            except Exception as e:
                logger.error(f"Error evaluating {candidate_id}: {e}")
                failed += 1
                
            finally:
                cleanup_temp_files(temp_paths)
            
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
