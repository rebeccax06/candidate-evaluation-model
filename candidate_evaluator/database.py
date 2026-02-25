"""Supabase database operations for candidate evaluator."""

import os
import json
from datetime import datetime
from typing import Optional
from uuid import UUID

from supabase import create_client, Client


def get_supabase_client() -> Client:
    """Get Supabase client from environment variables or Streamlit secrets."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        try:
            import streamlit as st
            # Try direct access (not .get()) as Streamlit secrets may not support .get()
            if hasattr(st, 'secrets'):
                url = url or st.secrets["SUPABASE_URL"]
                key = key or st.secrets["SUPABASE_KEY"]
        except KeyError as e:
            raise ValueError(f"Missing secret: {e}. Please add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")
        except Exception as e:
            raise ValueError(f"Error accessing Streamlit secrets: {e}")
    
    if not url or not key:
        raise ValueError(
            "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY "
            "environment variables or add them to .streamlit/secrets.toml"
        )
    
    return create_client(url, key)


class Database:
    """Database operations for candidate evaluator."""
    
    def __init__(self, client: Optional[Client] = None):
        self.client = client or get_supabase_client()
    
    # -------------------------------------------------------------------------
    # User Profile Operations
    # -------------------------------------------------------------------------
    
    def get_user_profile(self, user_id: str) -> Optional[dict]:
        """Get user profile by ID."""
        result = self.client.table("profiles").select("*").eq("id", user_id).execute()
        return result.data[0] if result.data else None
    
    def update_user_api_key(self, user_id: str, api_key: str) -> dict:
        """Update user's Anthropic API key."""
        result = self.client.table("profiles").update({
            "anthropic_api_key": api_key
        }).eq("id", user_id).execute()
        return result.data[0] if result.data else None
    
    def get_user_api_key(self, user_id: str) -> Optional[str]:
        """Get user's stored Anthropic API key."""
        profile = self.get_user_profile(user_id)
        return profile.get("anthropic_api_key") if profile else None
    
    # -------------------------------------------------------------------------
    # Job Queue Operations
    # -------------------------------------------------------------------------
    
    def create_job(
        self,
        user_id: str,
        job_name: str,
        job_type: str,
        total_candidates: int,
        file_paths: list[str],
        evaluation_mode: str = "criteria"
    ) -> dict:
        """Create a new batch job."""
        result = self.client.table("jobs").insert({
            "user_id": user_id,
            "job_name": job_name,
            "job_type": job_type,
            "status": "pending",
            "total_candidates": total_candidates,
            "completed_candidates": 0,
            "failed_candidates": 0,
            "file_paths": file_paths,
            "evaluation_mode": evaluation_mode
        }).execute()
        return result.data[0] if result.data else None
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID."""
        result = self.client.table("jobs").select("*").eq("id", job_id).execute()
        return result.data[0] if result.data else None
    
    def get_user_jobs(self, user_id: str, limit: int = 50) -> list[dict]:
        """Get all jobs for a user."""
        result = self.client.table("jobs").select("*").eq(
            "user_id", user_id
        ).order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    
    def get_pending_jobs(self, limit: int = 10) -> list[dict]:
        """Get pending jobs for worker processing."""
        result = self.client.table("jobs").select("*").eq(
            "status", "pending"
        ).order("created_at").limit(limit).execute()
        return result.data or []
    
    def get_processing_jobs(self) -> list[dict]:
        """Get jobs currently being processed."""
        result = self.client.table("jobs").select("*").eq(
            "status", "processing"
        ).execute()
        return result.data or []
    
    def claim_job(self, job_id: str, worker_id: str) -> bool:
        """Claim a job for processing (atomic operation)."""
        result = self.client.table("jobs").update({
            "status": "processing",
            "worker_id": worker_id,
            "started_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).eq("status", "pending").execute()
        return len(result.data) > 0 if result.data else False
    
    def update_job_progress(
        self,
        job_id: str,
        completed: int,
        failed: int = 0,
        current_candidate: Optional[str] = None
    ) -> dict:
        """Update job progress."""
        update_data = {
            "completed_candidates": completed,
            "failed_candidates": failed
        }
        if current_candidate:
            update_data["current_candidate"] = current_candidate
        
        result = self.client.table("jobs").update(update_data).eq("id", job_id).execute()
        return result.data[0] if result.data else None
    
    def complete_job(self, job_id: str, error: Optional[str] = None) -> dict:
        """Mark job as completed or failed."""
        status = "failed" if error else "completed"
        update_data = {
            "status": status,
            "completed_at": datetime.utcnow().isoformat()
        }
        if error:
            update_data["error"] = error
        
        result = self.client.table("jobs").update(update_data).eq("id", job_id).execute()
        return result.data[0] if result.data else None
    
    def cancel_job(self, job_id: str) -> dict:
        """Cancel a pending or processing job."""
        result = self.client.table("jobs").update({
            "status": "cancelled",
            "completed_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).in_("status", ["pending", "processing"]).execute()
        return result.data[0] if result.data else None
    
    # -------------------------------------------------------------------------
    # Evaluation Operations
    # -------------------------------------------------------------------------
    
    def save_evaluation(
        self,
        user_id: str,
        candidate_id: str,
        evaluation_type: str,
        result: dict,
        job_id: Optional[str] = None,
        candidate_name: Optional[str] = None
    ) -> dict:
        """Save an evaluation result."""
        data = {
            "user_id": user_id,
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "evaluation_type": evaluation_type,
            "result": result,
            "overall_score": result.get("overall_score"),
            "recommendation": result.get("recommendation")
        }
        if job_id:
            data["job_id"] = job_id
        
        db_result = self.client.table("evaluations").insert(data).execute()
        return db_result.data[0] if db_result.data else None
    
    def get_evaluation(self, evaluation_id: str) -> Optional[dict]:
        """Get evaluation by ID."""
        result = self.client.table("evaluations").select("*").eq("id", evaluation_id).execute()
        return result.data[0] if result.data else None
    
    def get_user_evaluations(
        self,
        user_id: str,
        evaluation_type: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Get all evaluations for a user."""
        query = self.client.table("evaluations").select("*").eq("user_id", user_id)
        if evaluation_type:
            query = query.eq("evaluation_type", evaluation_type)
        result = query.order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    
    def get_job_evaluations(self, job_id: str) -> list[dict]:
        """Get all evaluations for a specific job."""
        result = self.client.table("evaluations").select("*").eq(
            "job_id", job_id
        ).order("created_at").execute()
        return result.data or []
    
    def delete_evaluation(self, evaluation_id: str, user_id: str) -> bool:
        """Delete an evaluation (user must own it)."""
        result = self.client.table("evaluations").delete().eq(
            "id", evaluation_id
        ).eq("user_id", user_id).execute()
        return len(result.data) > 0 if result.data else False
    
    # -------------------------------------------------------------------------
    # File Metadata Operations
    # -------------------------------------------------------------------------
    
    def save_file_metadata(
        self,
        user_id: str,
        file_name: str,
        storage_path: str,
        file_type: str,
        evaluation_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> dict:
        """Save uploaded file metadata."""
        data = {
            "user_id": user_id,
            "file_name": file_name,
            "storage_path": storage_path,
            "file_type": file_type
        }
        if evaluation_id:
            data["evaluation_id"] = evaluation_id
        if job_id:
            data["job_id"] = job_id
        
        result = self.client.table("uploaded_files").insert(data).execute()
        return result.data[0] if result.data else None
    
    def get_user_files(self, user_id: str, limit: int = 100) -> list[dict]:
        """Get all uploaded files for a user."""
        result = self.client.table("uploaded_files").select("*").eq(
            "user_id", user_id
        ).order("uploaded_at", desc=True).limit(limit).execute()
        return result.data or []
    
    def get_job_files(self, job_id: str) -> list[dict]:
        """Get all files associated with a job."""
        result = self.client.table("uploaded_files").select("*").eq(
            "job_id", job_id
        ).execute()
        return result.data or []
