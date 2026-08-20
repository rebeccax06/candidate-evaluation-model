"""Background worker for persistent candidate evaluations.

This module runs evaluations in a separate process that continues
even when the web UI is closed or navigated away from.
"""

import json
import os
import sys
import time
import signal
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

# Add parent to path for imports when run as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from candidate_evaluator.core.evaluator import CandidateEvaluator
from candidate_evaluator.core.processing import (
    EvaluationMode,
    eval_concurrency,
    evaluate_chunk_concurrently,
    result_filename,
    summary_entry,
)
from candidate_evaluator.utils.config import Config, APIConfig
from candidate_evaluator.exporters.json_exporter import JSONExporter


class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackgroundWorker:
    """Processes evaluation jobs in the background."""

    def __init__(self, jobs_dir: Path, setup_signals: bool = False):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.should_stop = False

        # Only setup signal handlers when running as main script
        if setup_signals:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.should_stop = True

    def create_job(
        self,
        job_id: str,
        candidate_files: dict,  # candidate_id -> file_path
        output_dir: str,
        config_dict: Optional[dict] = None
    ) -> Path:
        """Create a new evaluation job."""
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "candidate_files": candidate_files,
            "output_dir": output_dir,
            "config": config_dict or {},
            "progress": {
                "total": len(candidate_files),
                "completed": 0,
                "failed": 0,
                "current_candidate": None
            },
            "results": [],
            "errors": []
        }

        job_path = self.jobs_dir / f"{job_id}.json"
        with open(job_path, 'w') as f:
            json.dump(job_data, f, indent=2)

        return job_path

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get current status of a job."""
        job_path = self.jobs_dir / f"{job_id}.json"
        if not job_path.exists():
            return None

        with open(job_path, 'r') as f:
            return json.load(f)

    def update_job(self, job_id: str, updates: dict):
        """Update job data."""
        job_path = self.jobs_dir / f"{job_id}.json"
        if not job_path.exists():
            return

        with open(job_path, 'r') as f:
            job_data = json.load(f)

        job_data.update(updates)
        job_data["updated_at"] = datetime.now().isoformat()

        with open(job_path, 'w') as f:
            json.dump(job_data, f, indent=2)

    def process_job(self, job_id: str):
        """Process a single job."""
        job_data = self.get_job_status(job_id)
        if not job_data:
            return

        # Update status to running
        self.update_job(job_id, {"status": JobStatus.RUNNING, "started_at": datetime.now().isoformat()})

        try:
            # Initialize evaluator with config from job
            job_config = job_data.get("config", {})
            api_key = job_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            max_tokens = job_config.get("max_tokens", 8192)
            mode = EvaluationMode.coerce(job_config.get("evaluation_mode"))
            role = job_config.get("role")
            description = job_config.get("description", "")

            if not api_key:
                raise ValueError("No API key provided in job config or environment")

            config = Config(
                api=APIConfig(
                    anthropic_api_key=api_key,
                    max_tokens=max_tokens
                )
            )

            evaluator = CandidateEvaluator(config)
            output_dir = Path(job_data["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            candidate_files = job_data["candidate_files"]
            total = len(candidate_files)
            completed = 0
            failed = 0
            results = []
            errors = []

            pending = []
            for candidate_id, file_path in candidate_files.items():
                if (output_dir / result_filename(mode, candidate_id)).exists():
                    completed += 1
                    results.append({
                        "candidate_id": candidate_id,
                        "status": "skipped",
                        "message": "Already evaluated"
                    })
                    continue
                pending.append((candidate_id, file_path))

            # Candidates are processed in chunks: result files are written on
            # the main thread; only the Claude calls run concurrently.
            concurrency = eval_concurrency()
            for chunk_start in range(0, len(pending), concurrency):
                if self.should_stop:
                    self.update_job(job_id, {"status": JobStatus.CANCELLED})
                    return

                chunk = pending[chunk_start:chunk_start + concurrency]
                self.update_job(job_id, {
                    "progress": {
                        "total": total,
                        "completed": completed,
                        "failed": failed,
                        "current_candidate": ", ".join(cid for cid, _ in chunk)
                    }
                })

                outcomes = evaluate_chunk_concurrently(
                    evaluator,
                    mode,
                    [(cid, [fp]) for cid, fp in chunk],
                    role=role,
                    description=description,
                    max_workers=concurrency,
                )
                for candidate_id, result, result_dict, error in outcomes:
                    if error is not None:
                        failed += 1
                        errors.append({
                            "candidate_id": candidate_id,
                            "error": error.strip().splitlines()[-1],
                            "traceback": error
                        })
                        continue

                    result_path = output_dir / result_filename(mode, candidate_id)
                    if mode is EvaluationMode.CRITERIA:
                        # Criteria results keep the exporter's file format so
                        # existing on-disk results stay loadable.
                        JSONExporter.export_evaluation(result, result_path)
                    else:
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(result_dict, f, indent=2, default=str)

                    completed += 1
                    results.append(summary_entry(mode, candidate_id, result))

                # Update progress after each chunk
                self.update_job(job_id, {
                    "progress": {
                        "total": total,
                        "completed": completed,
                        "failed": failed,
                        "current_candidate": None
                    },
                    "results": results,
                    "errors": errors
                })

            # Mark job as completed
            self.update_job(job_id, {
                "status": JobStatus.COMPLETED,
                "completed_at": datetime.now().isoformat(),
                "progress": {
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "current_candidate": None
                },
                "results": results,
                "errors": errors
            })

        except Exception as e:
            self.update_job(job_id, {
                "status": JobStatus.FAILED,
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    def list_jobs(self) -> list:
        """List all jobs."""
        jobs = []
        for job_file in self.jobs_dir.glob("*.json"):
            with open(job_file, 'r') as f:
                jobs.append(json.load(f))
        return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)

    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        self.update_job(job_id, {"status": JobStatus.CANCELLED})

    def delete_job(self, job_id: str):
        """Delete a job file."""
        job_path = self.jobs_dir / f"{job_id}.json"
        if job_path.exists():
            job_path.unlink()


def run_worker(jobs_dir: str, job_id: str):
    """Run a worker to process a specific job."""
    worker = BackgroundWorker(Path(jobs_dir), setup_signals=True)
    worker.process_job(job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background evaluation worker")
    parser.add_argument("--jobs-dir", required=True, help="Directory for job files")
    parser.add_argument("--job-id", required=True, help="Job ID to process")

    args = parser.parse_args()
    run_worker(args.jobs_dir, args.job_id)
