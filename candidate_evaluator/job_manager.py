"""Job manager for launching and tracking background evaluation jobs."""

import os
import sys
import json
import subprocess
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from candidate_evaluator.background_worker import BackgroundWorker, JobStatus


class JobManager:
    """Manages background evaluation jobs."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path.home() / ".candidate_evaluator"

        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / "jobs"
        self.logs_dir = self.base_dir / "logs"
        self.pids_dir = self.base_dir / "pids"

        # Create directories
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.pids_dir.mkdir(parents=True, exist_ok=True)

        self.worker = BackgroundWorker(self.jobs_dir)

    def submit_job(
        self,
        candidate_files: Dict[str, str],
        output_dir: str,
        config: Optional[dict] = None,
        job_name: Optional[str] = None
    ) -> str:
        """
        Submit a new evaluation job to run in the background.

        Args:
            candidate_files: Dict of candidate_id -> file_path
            output_dir: Directory to save evaluation results
            config: Optional configuration overrides
            job_name: Optional friendly name for the job

        Returns:
            job_id: Unique identifier for the job
        """
        # Generate job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        job_id = f"eval_{timestamp}_{short_uuid}"

        # Create job file
        job_path = self.worker.create_job(
            job_id=job_id,
            candidate_files=candidate_files,
            output_dir=output_dir,
            config_dict=config
        )

        # Add job name if provided
        if job_name:
            self.worker.update_job(job_id, {"job_name": job_name})

        # Launch background process
        self._launch_worker(job_id)

        return job_id

    def _launch_worker(self, job_id: str):
        """Launch a background worker process for a job."""
        # Get path to worker script
        worker_script = Path(__file__).parent / "background_worker.py"

        # Log file for this job
        log_file = self.logs_dir / f"{job_id}.log"

        # Launch as detached subprocess
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(worker_script),
                    "--jobs-dir", str(self.jobs_dir),
                    "--job-id", job_id
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
                cwd=str(Path(__file__).parent.parent)
            )

        # Save PID
        pid_file = self.pids_dir / f"{job_id}.pid"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job details."""
        return self.worker.get_job_status(job_id)

    def list_jobs(self, limit: int = 50) -> List[dict]:
        """List recent jobs."""
        jobs = self.worker.list_jobs()
        return jobs[:limit]

    def get_active_jobs(self) -> List[dict]:
        """Get currently running jobs."""
        jobs = self.worker.list_jobs()
        return [j for j in jobs if j.get("status") in [JobStatus.PENDING, JobStatus.RUNNING]]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.get_job(job_id)
        if not job:
            return False

        # Mark as cancelled
        self.worker.cancel_job(job_id)

        # Try to kill the process
        pid_file = self.pids_dir / f"{job_id}.pid"
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, 15)  # SIGTERM
            except (ProcessLookupError, ValueError):
                pass

        return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files."""
        # Cancel first if running
        self.cancel_job(job_id)

        # Delete job file
        self.worker.delete_job(job_id)

        # Delete log file
        log_file = self.logs_dir / f"{job_id}.log"
        if log_file.exists():
            log_file.unlink()

        # Delete pid file
        pid_file = self.pids_dir / f"{job_id}.pid"
        if pid_file.exists():
            pid_file.unlink()

        return True

    def get_job_logs(self, job_id: str, tail: int = 100) -> str:
        """Get logs for a job."""
        log_file = self.logs_dir / f"{job_id}.log"
        if not log_file.exists():
            return ""

        with open(log_file, 'r') as f:
            lines = f.readlines()

        return "".join(lines[-tail:])

    def cleanup_old_jobs(self, days: int = 7):
        """Remove jobs older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        for job in self.worker.list_jobs():
            created = job.get("created_at")
            if created:
                created_dt = datetime.fromisoformat(created)
                if created_dt < cutoff and job.get("status") in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    self.delete_job(job["job_id"])
