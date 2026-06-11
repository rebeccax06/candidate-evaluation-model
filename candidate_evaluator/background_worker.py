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
            evaluation_mode = job_config.get("evaluation_mode", "criteria")
            is_holistic = evaluation_mode == "holistic"
            role = job_config.get("role")

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

            for candidate_id, file_path in candidate_files.items():
                if self.should_stop:
                    self.update_job(job_id, {"status": JobStatus.CANCELLED})
                    return

                # Update current candidate
                self.update_job(job_id, {
                    "progress": {
                        "total": total,
                        "completed": completed,
                        "failed": failed,
                        "current_candidate": candidate_id
                    }
                })

                try:
                    # Check if already evaluated (use appropriate filename pattern)
                    if is_holistic:
                        result_path = output_dir / f"{candidate_id}_holistic_evaluation.json"
                    else:
                        result_path = output_dir / f"{candidate_id}_evaluation.json"

                    if result_path.exists():
                        completed += 1
                        results.append({
                            "candidate_id": candidate_id,
                            "status": "skipped",
                            "message": "Already evaluated"
                        })
                        continue

                    # Run evaluation based on mode
                    if is_holistic:
                        result = evaluator.evaluate_candidate_holistic(
                            candidate_id=candidate_id,
                            material_paths=[file_path],
                            candidate_name=None,
                            role=role
                        )
                        # Save holistic result
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(result.model_dump(), f, indent=2, default=str)

                        completed += 1
                        results.append({
                            "candidate_id": candidate_id,
                            "status": "success",
                            "overall_score": result.overall_score,
                            "recommendation": result.recommendation,
                            "interview_decision": result.interview_decision,
                            "evaluation_mode": "holistic"
                        })
                    else:
                        result = evaluator.evaluate_candidate(
                            candidate_id=candidate_id,
                            material_paths=[file_path],
                            candidate_name=None,
                            role=role
                        )
                        # Save criteria-based result
                        JSONExporter.export_evaluation(result, result_path)

                        completed += 1
                        results.append({
                            "candidate_id": candidate_id,
                            "status": "success",
                            "overall_score": result.overall_score,
                            "recommendation": result.recommendation,
                            "evaluation_mode": "criteria"
                        })

                except Exception as e:
                    failed += 1
                    errors.append({
                        "candidate_id": candidate_id,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })

                # Update progress after each candidate
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
