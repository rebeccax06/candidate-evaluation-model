"""
Cloud-ready Streamlit web interface with authentication and Supabase integration.

This version adds:
- User authentication (Supabase Auth)
- Cloud file storage (Supabase Storage)
- Background job processing (via Railway worker)
- Per-user evaluation history

Usage:
    streamlit run candidate_evaluator/web_app_cloud.py
"""

import streamlit as st
import tempfile
import os
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from candidate_evaluator.core.evaluator import CandidateEvaluator
from candidate_evaluator.core.models import (
    EvaluationResult,
    CandidateProfile,
    CriterionScore,
    EvaluationCriterion,
    Evidence,
    HolisticEvaluationResult,
    InnovationPotential,
    ProgramFit,
    NotableQuality
)
from candidate_evaluator.utils.config import get_default_config
from candidate_evaluator.auth import (
    init_auth_state,
    render_auth_ui,
    render_user_menu,
    get_current_user,
    is_logged_in,
    require_api_key,
    get_database
)
from candidate_evaluator.database import Database
from candidate_evaluator.storage import Storage, cleanup_temp_files


st.set_page_config(
    page_title="Candidate Evaluator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    init_auth_state()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'storage' not in st.session_state:
        st.session_state.storage = None


def get_storage() -> Storage:
    """Get or create Storage instance using the authenticated client."""
    if st.session_state.storage is None:
        from candidate_evaluator.auth import get_auth_client
        # Use the same authenticated client that was used for sign-in
        st.session_state.storage = Storage(get_auth_client())
    return st.session_state.storage


def get_evaluator(api_key: str) -> CandidateEvaluator:
    """Get or create evaluator with user's API key."""
    if st.session_state.evaluator is None or st.session_state.get('current_api_key') != api_key:
        # Create config directly with the user's API key
        from candidate_evaluator.utils.config import Config, APIConfig
        config = Config(
            api=APIConfig(anthropic_api_key=api_key)
        )
        st.session_state.evaluator = CandidateEvaluator(config)
        st.session_state.current_api_key = api_key
    return st.session_state.evaluator


def result_dict_to_evaluation_result(data: dict) -> EvaluationResult:
    """Convert database result dict to EvaluationResult object."""
    candidate = CandidateProfile(
        candidate_id=data['candidate']['candidate_id'],
        name=data['candidate'].get('name'),
        materials=data['candidate'].get('materials', []),
        evaluation_date=datetime.fromisoformat(str(data['candidate']['evaluation_date']).replace('Z', '+00:00'))
    )
    
    scores = []
    for score_data in data.get('scores', []):
        evidence_list = [
            Evidence(
                quote=ev['quote'],
                source=ev['source'],
                context=ev['context']
            )
            for ev in score_data.get('evidence', [])
        ]
        
        criterion_value = score_data['criterion']
        if isinstance(criterion_value, str):
            criterion_value = criterion_value.replace('EvaluationCriterion.', '')
        
        scores.append(CriterionScore(
            criterion=EvaluationCriterion(criterion_value),
            score=score_data['score'],
            reasoning=score_data['reasoning'],
            evidence=evidence_list,
            confidence=score_data.get('confidence', 'medium'),
            notes=score_data.get('notes')
        ))
    
    return EvaluationResult(
        candidate=candidate,
        scores=scores,
        overall_score=data['overall_score'],
        overall_assessment=data['overall_assessment'],
        strengths=data.get('strengths', []),
        areas_for_development=data.get('areas_for_development', []),
        recommendation=data['recommendation'],
        metadata=data.get('metadata', {})
    )


def result_dict_to_holistic_result(data: dict) -> HolisticEvaluationResult:
    """Convert database result dict to HolisticEvaluationResult object."""
    candidate = CandidateProfile(
        candidate_id=data['candidate']['candidate_id'],
        name=data['candidate'].get('name'),
        materials=data['candidate'].get('materials', []),
        evaluation_date=datetime.fromisoformat(str(data['candidate']['evaluation_date']).replace('Z', '+00:00'))
    )
    
    innovation_data = data.get('innovation_potential', {})
    innovation_potential = InnovationPotential(
        level=innovation_data.get('level', 'medium'),
        reasoning=innovation_data.get('reasoning', ''),
        key_evidence=innovation_data.get('key_evidence', [])
    )
    
    fit_data = data.get('program_fit', {})
    program_fit = ProgramFit(
        level=fit_data.get('level', 'moderate'),
        strengths_for_program=fit_data.get('strengths_for_program', []),
        concerns=fit_data.get('concerns', [])
    )
    
    notable_qualities = []
    for q in data.get('notable_qualities', []):
        if isinstance(q, dict):
            notable_qualities.append(NotableQuality(
                quality=q.get('quality', ''),
                evidence=q.get('evidence', ''),
                significance=q.get('significance', '')
            ))
    
    return HolisticEvaluationResult(
        candidate=candidate,
        overall_assessment=data.get('overall_assessment', ''),
        innovation_potential=innovation_potential,
        program_fit=program_fit,
        notable_qualities=notable_qualities,
        red_flags=data.get('red_flags', []),
        questions_for_interview=data.get('questions_for_interview', []),
        overall_score=float(data.get('overall_score', 5.0)),
        recommendation=data.get('recommendation', ''),
        interview_decision=data.get('interview_decision', False),
        metadata=data.get('metadata', {})
    )


def main():
    """Main application."""
    init_session_state()
    
    if not render_auth_ui():
        return
    
    user = get_current_user()
    api_key = require_api_key()
    
    if not api_key:
        st.stop()
    
    with st.sidebar:
        st.markdown("## Candidate Evaluator")
        render_user_menu()
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "New Evaluation", "Batch Jobs", "Results", "Analysis", "Research", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        db = get_database()
        active_jobs = [j for j in db.get_user_jobs(user["id"], limit=10) 
                       if j.get("status") == "processing"]
        
        if active_jobs:
            st.markdown(f"**Active Jobs:** {len(active_jobs)}")
            st.caption("Go to Batch Jobs to view")
        
        st.markdown("---")
        st.caption("AI-powered candidate evaluation")
    
    if page == "Dashboard":
        dashboard_page(user, api_key)
    elif page == "New Evaluation":
        new_evaluation_page(user, api_key)
    elif page == "Batch Jobs":
        batch_jobs_page(user)
    elif page == "Results":
        results_page(user)
    elif page == "Analysis":
        analysis_page(user)
    elif page == "Research":
        research_page(user)
    elif page == "Settings":
        settings_page(user)


def dashboard_page(user: dict, api_key: str):
    """Dashboard overview page."""
    st.title("Dashboard")
    
    db = get_database()
    evaluations = db.get_user_evaluations(user["id"], limit=100)
    jobs = db.get_user_jobs(user["id"], limit=10)
    
    active_jobs = [j for j in jobs if j.get("status") in ["pending", "processing"]]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Evaluations", len(evaluations))
    
    with col2:
        st.metric("Active Jobs", len(active_jobs))
    
    with col3:
        if evaluations:
            avg_score = sum(e.get("overall_score", 0) or 0 for e in evaluations) / len(evaluations)
            st.metric("Avg Score", f"{avg_score:.1f}/10")
        else:
            st.metric("Avg Score", "N/A")
    
    with col4:
        completed_jobs = len([j for j in jobs if j.get("status") == "completed"])
        st.metric("Completed Jobs", completed_jobs)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recent Evaluations")
        
        if evaluations:
            recent = evaluations[:5]
            
            for eval_data in recent:
                with st.container():
                    rcol1, rcol2, rcol3 = st.columns([3, 1, 1])
                    with rcol1:
                        st.markdown(f"**{eval_data['candidate_id']}**")
                        created = eval_data.get('created_at', '')
                        if created:
                            st.caption(str(created)[:19])
                    with rcol2:
                        score = eval_data.get('overall_score')
                        st.markdown(f"**{score:.1f}**/10" if score else "N/A")
                    with rcol3:
                        st.caption(eval_data.get('recommendation', '')[:20] or '-')
                    st.markdown("---")
        else:
            st.info("No evaluations yet. Select 'New Evaluation' to get started.")
    
    with col2:
        st.subheader("Active Jobs")
        
        if active_jobs:
            for job in active_jobs[:3]:
                completed = job.get("completed_candidates", 0)
                total = job.get("total_candidates", 1)
                pct = (completed / total) if total > 0 else 0
                
                st.markdown(f"**{job.get('job_name', 'Batch Job')[:25]}**")
                st.progress(pct)
                st.caption(f"{completed}/{total} candidates")
                st.markdown("---")
        else:
            st.info("No active jobs")


def new_evaluation_page(user: dict, api_key: str):
    """New evaluation page."""
    st.title("New Evaluation")
    
    tab1, tab2 = st.tabs(["Single Candidate", "Batch Upload"])
    
    with tab1:
        single_evaluation_form(user, api_key)
    
    with tab2:
        batch_evaluation_form(user)


def single_evaluation_form(user: dict, api_key: str):
    """Single candidate evaluation form."""
    st.markdown("### Evaluate a Single Candidate")
    
    eval_mode = st.radio(
        "Evaluation Mode",
        ["Criteria-Based (11 criteria)", "Holistic (program fit)"],
        horizontal=True
    )
    is_holistic = eval_mode == "Holistic (program fit)"
    
    if is_holistic:
        st.info("Holistic mode evaluates overall program fit and innovation potential.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        candidate_id = st.text_input(
            "Candidate ID",
            placeholder="e.g., CAND001"
        )
        
        candidate_name = st.text_input(
            "Name (optional)",
            placeholder="e.g., John Doe"
        )
        
        uploaded_files = st.file_uploader(
            "Upload Application Materials",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True
        )
    
    with col2:
        st.markdown("#### Supported Formats")
        st.markdown("- PDF, Word, Text, Markdown")
        st.markdown("#### Tips")
        st.markdown("- Include all relevant materials")
        st.markdown("- More context = better evaluation")
    
    if st.button("Evaluate Candidate", disabled=not (candidate_id and uploaded_files)):
        with st.spinner("Evaluating candidate... This may take a minute."):
            try:
                temp_dir = tempfile.mkdtemp()
                material_paths = []
                
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    material_paths.append(temp_path)
                
                evaluator = get_evaluator(api_key)
                db = get_database()
                
                if is_holistic:
                    result = evaluator.evaluate_candidate_holistic(
                        candidate_id=candidate_id,
                        material_paths=material_paths,
                        candidate_name=candidate_name or None
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(result_dict["candidate"]["evaluation_date"])
                    
                    db.save_evaluation(
                        user_id=user["id"],
                        candidate_id=candidate_id,
                        evaluation_type="holistic",
                        result=result_dict,
                        candidate_name=candidate_name
                    )
                    
                    st.success("Holistic evaluation completed!")
                    display_holistic_evaluation_result(result)
                else:
                    result = evaluator.evaluate_candidate(
                        candidate_id=candidate_id,
                        material_paths=material_paths,
                        candidate_name=candidate_name or None
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(result_dict["candidate"]["evaluation_date"])
                    for score in result_dict.get("scores", []):
                        if "criterion" in score:
                            score["criterion"] = str(score["criterion"])
                    
                    db.save_evaluation(
                        user_id=user["id"],
                        candidate_id=candidate_id,
                        evaluation_type="criteria",
                        result=result_dict,
                        candidate_name=candidate_name
                    )
                    
                    st.success("Evaluation completed!")
                    display_evaluation_result(result)
                
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
            finally:
                for path in material_paths:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass


def batch_evaluation_form(user: dict):
    """Batch evaluation form with background processing."""
    st.markdown("### Batch Evaluation")
    st.markdown("Upload multiple PDF files to evaluate in the background. You can close this page and the evaluation will continue.")
    
    eval_mode = st.radio(
        "Evaluation Mode",
        ["Criteria-Based (11 criteria)", "Holistic (program fit)"],
        horizontal=True,
        key="batch_eval_mode"
    )
    is_holistic = eval_mode == "Holistic (program fit)"
    
    uploaded_files = st.file_uploader(
        "Upload Candidate PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="One PDF per candidate. Filename becomes the candidate ID.",
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")
        
        with st.expander("View files"):
            for f in uploaded_files:
                st.text(f"- {f.name}")
        
        job_name = st.text_input(
            "Job Name (optional)",
            placeholder="e.g., Spring 2026 Applicants"
        )
        
        if st.button("Start Batch Evaluation", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                storage = get_storage()
                db = get_database()
                
                file_paths = []
                total_files = len(uploaded_files)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {i+1}/{total_files}: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / total_files)
                    
                    file_data = uploaded_file.getbuffer().tobytes()
                    storage_path = storage.upload_file(
                        user_id=user["id"],
                        file_data=file_data,
                        filename=uploaded_file.name,
                        content_type="application/pdf"
                    )
                    file_paths.append(storage_path)
                    
                    # Small delay to avoid rate limiting
                    if i < total_files - 1:
                        time.sleep(0.2)
                
                status_text.text("Creating job...")
                job = db.create_job(
                    user_id=user["id"],
                    job_name=job_name or f"Batch {datetime.now().strftime('%Y%m%d_%H%M')}",
                    job_type="batch",
                    total_candidates=len(file_paths),
                    file_paths=file_paths,
                    evaluation_mode="holistic" if is_holistic else "criteria"
                )
                
                progress_bar.progress(1.0)
                status_text.empty()
                st.success(f"Job created! ID: `{job['id']}`")
                st.info("The background worker will process your candidates. You can close this page.")
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error creating job: {e}")


def batch_jobs_page(user: dict):
    """View and manage batch jobs."""
    st.title("Batch Jobs")
    
    db = get_database()
    
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    
    jobs = db.get_user_jobs(user["id"], limit=50)
    
    tab1, tab2, tab3 = st.tabs(["Active", "Completed", "All Jobs"])
    
    with tab1:
        active_jobs = [j for j in jobs if j.get("status") in ["pending", "processing"]]
        
        if active_jobs:
            for job in active_jobs:
                render_job_card(job, db, user)
            
            if auto_refresh:
                time.sleep(3)
                st.rerun()
        else:
            st.info("No active jobs. Start a batch evaluation to see jobs here.")
    
    with tab2:
        completed_jobs = [j for j in jobs if j.get("status") == "completed"]
        
        if completed_jobs:
            for job in completed_jobs[:20]:
                render_job_card(job, db, user, show_actions=False)
        else:
            st.info("No completed jobs yet.")
    
    with tab3:
        if jobs:
            job_data = []
            for job in jobs:
                job_data.append({
                    "Job ID": str(job["id"])[:20] + "...",
                    "Name": (job.get("job_name") or "-")[:30],
                    "Status": job.get("status", "unknown"),
                    "Progress": f"{job.get('completed_candidates', 0)}/{job.get('total_candidates', 0)}",
                    "Created": str(job.get("created_at", ""))[:19]
                })
            
            st.dataframe(pd.DataFrame(job_data), hide_index=True, use_container_width=True)
        else:
            st.info("No jobs yet.")


def render_job_card(job: dict, db: Database, user: dict, show_actions: bool = True):
    """Render a job card with progress and actions."""
    job_id = job["id"]
    status = job.get("status", "unknown")
    completed = job.get("completed_candidates", 0)
    failed = job.get("failed_candidates", 0)
    total = job.get("total_candidates", 1)
    current = job.get("current_candidate")
    
    pct = (completed / total) if total > 0 else 0
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{job.get('job_name', str(job_id)[:25])}**")
            st.caption(f"ID: {str(job_id)[:30]}...")
        
        with col2:
            status_colors = {
                "pending": "🟡",
                "processing": "🔵",
                "completed": "🟢",
                "failed": "🔴",
                "cancelled": "⚪"
            }
            st.markdown(f"{status_colors.get(status, '')} **{status.upper()}**")
        
        with col3:
            st.markdown(f"**{completed}/{total}**")
            if failed > 0:
                st.caption(f"{failed} failed")
        
        if status in ["processing", "pending"]:
            st.progress(pct)
            if current:
                st.caption(f"Currently evaluating: {current}")
        
        if show_actions and status in ["processing", "pending"]:
            if st.button("Cancel", key=f"cancel_{job_id}"):
                db.cancel_job(str(job_id))
                st.rerun()
        
        if status == "completed":
            with st.expander("View Results Summary"):
                evaluations = db.get_job_evaluations(str(job_id))
                
                if evaluations:
                    result_data = []
                    for e in evaluations:
                        result_data.append({
                            "Candidate": e.get("candidate_id", ""),
                            "Score": f"{e.get('overall_score', 0):.1f}/10" if e.get('overall_score') else "N/A",
                            "Recommendation": (e.get("recommendation") or "")[:30]
                        })
                    st.dataframe(pd.DataFrame(result_data), hide_index=True)
                else:
                    st.write("No results available.")
        
        if status == "failed" or failed > 0:
            error = job.get("error")
            if error:
                st.error(f"Error: {error}")
        
        st.markdown("---")


def results_page(user: dict):
    """View all evaluation results."""
    st.title("Evaluation Results")
    
    db = get_database()
    evaluations = db.get_user_evaluations(user["id"], limit=200)
    
    if not evaluations:
        st.info("No evaluation results yet. Run some evaluations first!")
        return
    
    criteria_evals = [e for e in evaluations if e.get("evaluation_type") == "criteria"]
    holistic_evals = [e for e in evaluations if e.get("evaluation_type") == "holistic"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Criteria-Based", len(criteria_evals))
    with col2:
        st.metric("Holistic", len(holistic_evals))
    with col3:
        st.metric("Total", len(evaluations))
    with col4:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs([
        f"Criteria-Based ({len(criteria_evals)})",
        f"Holistic ({len(holistic_evals)})"
    ])
    
    with tab1:
        if not criteria_evals:
            st.info("No criteria-based evaluations yet.")
        else:
            summary_data = []
            sorted_evals = sorted(criteria_evals, key=lambda e: e.get("overall_score", 0) or 0, reverse=True)
            
            for rank, e in enumerate(sorted_evals, 1):
                summary_data.append({
                    'Rank': rank,
                    'Candidate ID': e.get('candidate_id', ''),
                    'Name': e.get('candidate_name') or '-',
                    'Score': f"{e.get('overall_score', 0):.1f}/10" if e.get('overall_score') else 'N/A',
                    'Date': str(e.get('created_at', ''))[:10],
                    'Recommendation': (e.get('recommendation') or '')[:30]
                })
            
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
            
            st.subheader("Candidate Details")
            options = {f"{e['candidate_id']} ({e.get('overall_score', 0):.1f})": e for e in sorted_evals}
            selected = st.selectbox("Select candidate", list(options.keys()), key="criteria_select")
            
            if selected:
                eval_data = options[selected]
                result = result_dict_to_evaluation_result(eval_data["result"])
                display_evaluation_result(result)
    
    with tab2:
        if not holistic_evals:
            st.info("No holistic evaluations yet.")
        else:
            summary_data = []
            sorted_evals = sorted(holistic_evals, key=lambda e: e.get("overall_score", 0) or 0, reverse=True)
            
            for rank, e in enumerate(sorted_evals, 1):
                result_data = e.get("result", {})
                summary_data.append({
                    'Rank': rank,
                    'Candidate ID': e.get('candidate_id', ''),
                    'Name': e.get('candidate_name') or '-',
                    'Score': f"{e.get('overall_score', 0):.1f}/10" if e.get('overall_score') else 'N/A',
                    'Interview': 'Yes' if result_data.get('interview_decision') else 'No',
                    'Date': str(e.get('created_at', ''))[:10]
                })
            
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
            
            st.subheader("Candidate Details")
            options = {f"{e['candidate_id']} ({e.get('overall_score', 0):.1f})": e for e in sorted_evals}
            selected = st.selectbox("Select candidate", list(options.keys()), key="holistic_select")
            
            if selected:
                eval_data = options[selected]
                result = result_dict_to_holistic_result(eval_data["result"])
                display_holistic_evaluation_result(result)


def settings_page(user: dict):
    """Settings page."""
    st.title("Settings")
    
    st.subheader("Account")
    st.markdown(f"**Email:** {user['email']}")
    st.markdown(f"**User ID:** {user['id']}")
    
    st.markdown("---")
    
    st.subheader("API Key")
    
    db = get_database()
    current_key = db.get_user_api_key(user["id"])
    
    if current_key:
        st.success("API key is configured")
        masked = current_key[:10] + "..." + current_key[-4:]
        st.code(masked)
    
    with st.form("update_api_key"):
        new_key = st.text_input(
            "Update Anthropic API Key",
            type="password",
            placeholder="sk-ant-..."
        )
        
        if st.form_submit_button("Update API Key"):
            if new_key and new_key.startswith("sk-ant-"):
                db.update_user_api_key(user["id"], new_key)
                st.success("API key updated!")
                st.rerun()
            else:
                st.error("Please enter a valid Anthropic API key")
    
    st.markdown("---")
    
    st.subheader("Data")
    
    evaluations = db.get_user_evaluations(user["id"])
    jobs = db.get_user_jobs(user["id"])
    
    st.markdown(f"**Total evaluations:** {len(evaluations)}")
    st.markdown(f"**Total jobs:** {len(jobs)}")


def display_evaluation_result(result: EvaluationResult):
    """Display a criteria-based evaluation result."""
    st.markdown(f"### {result.candidate.candidate_id}")
    if result.candidate.name:
        st.caption(f"Name: {result.candidate.name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{result.overall_score:.1f}/10")
    with col2:
        st.metric("Recommendation", result.recommendation)
    with col3:
        st.metric("Criteria Evaluated", len(result.scores))
    
    st.markdown("#### Overall Assessment")
    st.write(result.overall_assessment)
    
    if result.strengths:
        st.markdown("#### Strengths")
        for s in result.strengths:
            st.markdown(f"- {s}")
    
    if result.areas_for_development:
        st.markdown("#### Areas for Development")
        for a in result.areas_for_development:
            st.markdown(f"- {a}")
    
    st.markdown("#### Criterion Scores")
    
    for score in sorted(result.scores, key=lambda s: s.score, reverse=True):
        with st.expander(f"{score.criterion.value}: {score.score}/10"):
            st.markdown(f"**Reasoning:** {score.reasoning}")
            st.markdown(f"**Confidence:** {score.confidence}")
            
            if score.evidence:
                st.markdown("**Evidence:**")
                for ev in score.evidence:
                    st.markdown(f"> \"{ev.quote}\"")
                    st.caption(f"Source: {ev.source}")


def display_holistic_evaluation_result(result: HolisticEvaluationResult):
    """Display a holistic evaluation result."""
    st.markdown(f"### {result.candidate.candidate_id}")
    if result.candidate.name:
        st.caption(f"Name: {result.candidate.name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{result.overall_score:.1f}/10")
    with col2:
        st.metric("Interview?", "Yes" if result.interview_decision else "No")
    with col3:
        st.metric("Innovation", result.innovation_potential.level.capitalize())
    
    st.markdown("#### Overall Assessment")
    st.write(result.overall_assessment)
    
    st.markdown("#### Innovation Potential")
    st.markdown(f"**Level:** {result.innovation_potential.level.capitalize()}")
    st.write(result.innovation_potential.reasoning)
    
    st.markdown("#### Program Fit")
    st.markdown(f"**Level:** {result.program_fit.level.capitalize()}")
    
    if result.program_fit.strengths_for_program:
        st.markdown("**Strengths:**")
        for s in result.program_fit.strengths_for_program:
            st.markdown(f"- {s}")
    
    if result.program_fit.concerns:
        st.markdown("**Concerns:**")
        for c in result.program_fit.concerns:
            st.markdown(f"- {c}")
    
    if result.notable_qualities:
        st.markdown("#### Notable Qualities")
        for q in result.notable_qualities:
            with st.expander(q.quality):
                st.write(q.evidence)
                st.caption(f"Significance: {q.significance}")
    
    if result.red_flags:
        st.markdown("#### Red Flags")
        for flag in result.red_flags:
            st.warning(flag)
    
    if result.questions_for_interview:
        st.markdown("#### Suggested Interview Questions")
        for q in result.questions_for_interview:
            st.markdown(f"- {q}")


def get_user_results_as_objects(user: dict):
    """Load user's evaluations and convert to result objects."""
    db = get_database()
    evaluations = db.get_user_evaluations(user["id"], limit=500)
    
    criteria_results = []
    holistic_results = []
    
    for e in evaluations:
        try:
            if e.get("evaluation_type") == "criteria":
                result = result_dict_to_evaluation_result(e["result"])
                criteria_results.append(result)
            elif e.get("evaluation_type") == "holistic":
                result = result_dict_to_holistic_result(e["result"])
                holistic_results.append(result)
        except Exception:
            pass
    
    return criteria_results, holistic_results


def analysis_page(user: dict):
    """Analysis dashboard."""
    st.title("Analysis Dashboard")
    
    criteria_results, holistic_results = get_user_results_as_objects(user)
    total_results = len(criteria_results) + len(holistic_results)
    
    if total_results == 0:
        st.warning("No evaluation results found. Run some evaluations first!")
        return
    
    st.markdown(f"**Analyzing {len(criteria_results)} criteria-based + {len(holistic_results)} holistic evaluations**")
    
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Recommendations", "Score Comparison"])
    
    with tab1:
        if criteria_results:
            distribution_analysis(criteria_results)
        else:
            st.info("No criteria-based evaluations to analyze.")
    
    with tab2:
        if criteria_results:
            recommendation_analysis(criteria_results)
        else:
            st.info("Recommendations analysis requires criteria-based evaluations.")
    
    with tab3:
        if criteria_results and holistic_results:
            score_comparison_analysis(criteria_results, holistic_results)
        else:
            st.info("Need both criteria-based and holistic evaluations for comparison.")


def distribution_analysis(all_results):
    """Score distribution analysis."""
    st.subheader("Score Distribution Analysis")
    
    overall_scores = [r.overall_score for r in all_results]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Score", f"{sum(overall_scores)/len(overall_scores):.2f}")
    with col2:
        st.metric("Median Score", f"{sorted(overall_scores)[len(overall_scores)//2]:.2f}")
    with col3:
        st.metric("Min Score", f"{min(overall_scores):.2f}")
    with col4:
        st.metric("Max Score", f"{max(overall_scores):.2f}")
    
    st.markdown("---")
    
    # Score distribution chart
    st.subheader("Score Distribution")
    score_df = pd.DataFrame({'Score': overall_scores})
    st.bar_chart(score_df['Score'].value_counts().sort_index())
    
    # Criterion analysis
    st.subheader("Criterion Score Averages")
    
    criterion_scores = {}
    for result in all_results:
        for score in result.scores:
            crit_name = score.criterion.value.replace('_', ' ').title()
            if crit_name not in criterion_scores:
                criterion_scores[crit_name] = []
            criterion_scores[crit_name].append(score.score)
    
    crit_data = []
    for crit, scores in criterion_scores.items():
        crit_data.append({
            'Criterion': crit,
            'Average': sum(scores) / len(scores),
            'Min': min(scores),
            'Max': max(scores)
        })
    
    crit_df = pd.DataFrame(crit_data).sort_values('Average', ascending=False)
    st.dataframe(crit_df, hide_index=True, use_container_width=True)


def recommendation_analysis(all_results):
    """Recommendation breakdown analysis."""
    st.subheader("Recommendation Analysis")
    
    recommendations = {}
    for result in all_results:
        rec = result.recommendation if result.recommendation else "Unknown"
        if rec not in recommendations:
            recommendations[rec] = []
        recommendations[rec].append(result)
    
    st.markdown(f"**{len(recommendations)} unique recommendation types**")
    
    for rec_type, candidates in sorted(recommendations.items(), key=lambda x: -len(x[1])):
        with st.expander(f"{rec_type} ({len(candidates)} candidates)"):
            if candidates:
                avg = sum(c.overall_score for c in candidates) / len(candidates)
                st.metric("Average Score", f"{avg:.2f}")
                
                for c in sorted(candidates, key=lambda x: -x.overall_score)[:5]:
                    st.text(f"- {c.candidate.candidate_id}: {c.overall_score:.1f}/10")


def score_comparison_analysis(criteria_results, holistic_results):
    """Compare criteria-based vs holistic scores."""
    st.subheader("Criteria vs Holistic Score Comparison")
    
    criteria_by_id = {r.candidate.candidate_id: r for r in criteria_results}
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}
    both_ids = set(criteria_by_id.keys()) & set(holistic_by_id.keys())
    
    if not both_ids:
        st.info("No candidates have been evaluated with both methods.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Criteria-Based Stats")
            scores = [r.overall_score for r in criteria_results]
            st.metric("Mean Score", f"{sum(scores)/len(scores):.2f}")
            st.metric("Count", len(criteria_results))
        
        with col2:
            st.markdown("### Holistic Stats")
            scores = [r.overall_score for r in holistic_results]
            interview_yes = sum(1 for r in holistic_results if r.interview_decision)
            st.metric("Mean Score", f"{sum(scores)/len(scores):.2f}")
            st.metric("Interview Yes", f"{interview_yes}/{len(holistic_results)}")
        return
    
    st.markdown(f"**{len(both_ids)} candidates evaluated with both methods**")
    
    comparison_data = []
    for cid in both_ids:
        cr = criteria_by_id[cid]
        hr = holistic_by_id[cid]
        comparison_data.append({
            'Candidate': cid,
            'Criteria Score': cr.overall_score,
            'Holistic Score': hr.overall_score,
            'Difference': cr.overall_score - hr.overall_score,
            'Interview': 'Yes' if hr.interview_decision else 'No'
        })
    
    df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    with col1:
        if len(df) > 1:
            correlation = df['Criteria Score'].corr(df['Holistic Score'])
            st.metric("Score Correlation", f"{correlation:.3f}")
        mad = df['Difference'].abs().mean()
        st.metric("Mean Absolute Difference", f"{mad:.2f} points")
    
    with col2:
        higher = sum(1 for d in df['Difference'] if d > 0.5)
        lower = sum(1 for d in df['Difference'] if d < -0.5)
        similar = len(df) - higher - lower
        st.markdown("**Score Comparison:**")
        st.markdown(f"- Criteria higher: {higher}")
        st.markdown(f"- Holistic higher: {lower}")
        st.markdown(f"- Similar: {similar}")
    
    st.dataframe(df.sort_values('Difference', key=abs, ascending=False), hide_index=True, use_container_width=True)


def research_page(user: dict):
    """Research dashboard for comparing evaluation methods."""
    st.title("Research Dashboard")
    st.markdown("Compare evaluation methods and analyze AI performance for research purposes.")
    
    criteria_results, holistic_results = get_user_results_as_objects(user)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Criteria-Based Evaluations", len(criteria_results))
    with col2:
        st.metric("Holistic Evaluations", len(holistic_results))
    with col3:
        criteria_ids = {r.candidate.candidate_id for r in criteria_results}
        holistic_ids = {r.candidate.candidate_id for r in holistic_results}
        both_ids = criteria_ids & holistic_ids
        st.metric("Candidates with Both", len(both_ids))
    
    if len(criteria_results) == 0 and len(holistic_results) == 0:
        st.warning("No evaluation results found. Run some evaluations first!")
        return
    
    tab1, tab2, tab3 = st.tabs(["Method Comparison", "Side-by-Side View", "Export Report"])
    
    with tab1:
        research_method_comparison(criteria_results, holistic_results)
    
    with tab2:
        research_side_by_side(criteria_results, holistic_results)
    
    with tab3:
        research_export_report(criteria_results, holistic_results)


def research_method_comparison(criteria_results, holistic_results):
    """Compare criteria-based vs holistic evaluation methods."""
    st.subheader("Evaluation Method Comparison")
    
    criteria_by_id = {r.candidate.candidate_id: r for r in criteria_results}
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}
    both_ids = set(criteria_by_id.keys()) & set(holistic_by_id.keys())
    
    if len(both_ids) == 0:
        st.info("No candidates have been evaluated with both methods.")
        
        if criteria_results:
            st.markdown("### Criteria-Based Results")
            scores = [r.overall_score for r in criteria_results]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Score", f"{sum(scores)/len(scores):.2f}")
            with col2:
                st.metric("Min Score", f"{min(scores):.2f}")
            with col3:
                st.metric("Max Score", f"{max(scores):.2f}")
        
        if holistic_results:
            st.markdown("### Holistic Results")
            scores = [r.overall_score for r in holistic_results]
            interview_yes = sum(1 for r in holistic_results if r.interview_decision)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Score", f"{sum(scores)/len(scores):.2f}")
            with col2:
                st.metric("Min Score", f"{min(scores):.2f}")
            with col3:
                st.metric("Max Score", f"{max(scores):.2f}")
            with col4:
                st.metric("Interview Yes", f"{interview_yes}/{len(holistic_results)}")
        return
    
    st.markdown(f"**{len(both_ids)} candidates evaluated with both methods**")
    
    comparison_data = []
    for cid in both_ids:
        cr = criteria_by_id[cid]
        hr = holistic_by_id[cid]
        comparison_data.append({
            'candidate_id': cid,
            'criteria_score': cr.overall_score,
            'holistic_score': hr.overall_score,
            'criteria_rec': cr.recommendation,
            'holistic_rec': hr.recommendation,
            'holistic_interview': hr.interview_decision,
            'score_diff': cr.overall_score - hr.overall_score
        })
    
    df = pd.DataFrame(comparison_data)
    
    st.markdown("### Agreement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(df) > 1:
            correlation = df['criteria_score'].corr(df['holistic_score'])
            st.metric("Score Correlation", f"{correlation:.3f}")
        else:
            st.metric("Score Correlation", "N/A")
        
        mad = df['score_diff'].abs().mean()
        st.metric("Mean Absolute Difference", f"{mad:.2f} points")
    
    with col2:
        higher = sum(1 for d in df['score_diff'] if d > 0.5)
        lower = sum(1 for d in df['score_diff'] if d < -0.5)
        similar = len(df) - higher - lower
        
        st.markdown("**Score Comparison:**")
        st.markdown(f"- Criteria higher: {higher} ({higher/len(df)*100:.1f}%)")
        st.markdown(f"- Holistic higher: {lower} ({lower/len(df)*100:.1f}%)")
        st.markdown(f"- Similar (within 0.5): {similar} ({similar/len(df)*100:.1f}%)")
    
    st.markdown("### Detailed Comparison")
    display_df = df[['candidate_id', 'criteria_score', 'holistic_score', 'score_diff', 'holistic_interview']].copy()
    display_df.columns = ['Candidate', 'Criteria Score', 'Holistic Score', 'Difference', 'Interview Rec']
    display_df = display_df.sort_values('Difference', key=abs, ascending=False)
    st.dataframe(display_df, hide_index=True, use_container_width=True)


def research_side_by_side(criteria_results, holistic_results):
    """Side-by-side view of individual candidate evaluations."""
    st.subheader("Side-by-Side Candidate View")
    
    criteria_by_id = {r.candidate.candidate_id: r for r in criteria_results}
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}
    all_ids = sorted(set(criteria_by_id.keys()) | set(holistic_by_id.keys()))
    
    if not all_ids:
        st.info("No candidates to display.")
        return
    
    selected_id = st.selectbox("Select Candidate", all_ids)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Criteria-Based Evaluation")
        if selected_id in criteria_by_id:
            cr = criteria_by_id[selected_id]
            st.metric("Overall Score", f"{cr.overall_score:.1f}/10")
            st.markdown(f"**Recommendation:** {cr.recommendation}")
            
            if cr.strengths:
                st.markdown("**Strengths:**")
                for s in cr.strengths[:3]:
                    st.markdown(f"- {s}")
            
            if cr.areas_for_development:
                st.markdown("**Areas for Development:**")
                for a in cr.areas_for_development[:3]:
                    st.markdown(f"- {a}")
            
            with st.expander("Score Breakdown"):
                for score in sorted(cr.scores, key=lambda x: -x.score):
                    st.markdown(f"- {score.criterion.value.replace('_', ' ').title()}: {score.score}/10")
        else:
            st.info("No criteria-based evaluation for this candidate.")
    
    with col2:
        st.markdown("### Holistic Evaluation")
        if selected_id in holistic_by_id:
            hr = holistic_by_id[selected_id]
            st.metric("Overall Score", f"{hr.overall_score:.1f}/10")
            interview_text = "Yes" if hr.interview_decision else "No"
            st.metric("Interview Decision", interview_text)
            st.markdown(f"**Recommendation:** {hr.recommendation}")
            st.markdown(f"**Innovation Potential:** {hr.innovation_potential.level.capitalize()}")
            st.markdown(f"**Program Fit:** {hr.program_fit.level.capitalize()}")
            
            if hr.notable_qualities:
                st.markdown("**Notable Qualities:**")
                for q in hr.notable_qualities[:3]:
                    st.markdown(f"- {q.quality}")
            
            if hr.red_flags:
                st.markdown("**Red Flags:**")
                for flag in hr.red_flags[:3]:
                    flag_text = flag.flag if hasattr(flag, 'flag') else str(flag)
                    st.warning(flag_text)
            
            with st.expander("Full Assessment"):
                st.markdown(hr.overall_assessment)
        else:
            st.info("No holistic evaluation for this candidate.")


def research_export_report(criteria_results, holistic_results):
    """Export research comparison report."""
    st.subheader("Export Research Report")
    
    criteria_by_id = {r.candidate.candidate_id: r for r in criteria_results}
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}
    both_ids = set(criteria_by_id.keys()) & set(holistic_by_id.keys())
    
    st.markdown("Generate a markdown report summarizing evaluation comparisons.")
    
    col1, col2 = st.columns(2)
    with col1:
        include_criteria = st.checkbox("Include criteria-based results", value=True)
    with col2:
        include_holistic = st.checkbox("Include holistic results", value=True)
    
    if st.button("Generate Report"):
        report_lines = [
            "# Candidate Evaluation Research Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary Statistics",
            f"\n- Total criteria-based evaluations: {len(criteria_results)}",
            f"- Total holistic evaluations: {len(holistic_results)}",
            f"- Candidates with both evaluations: {len(both_ids)}",
        ]
        
        if criteria_results and include_criteria:
            scores = [r.overall_score for r in criteria_results]
            report_lines.extend([
                "\n## Criteria-Based Evaluation Summary",
                f"\n- Mean score: {sum(scores)/len(scores):.2f}",
                f"- Score range: {min(scores):.2f} - {max(scores):.2f}",
            ])
        
        if holistic_results and include_holistic:
            scores = [r.overall_score for r in holistic_results]
            interview_yes = sum(1 for r in holistic_results if r.interview_decision)
            report_lines.extend([
                "\n## Holistic Evaluation Summary",
                f"\n- Mean score: {sum(scores)/len(scores):.2f}",
                f"- Score range: {min(scores):.2f} - {max(scores):.2f}",
                f"- Interview recommendations: {interview_yes}/{len(holistic_results)} ({interview_yes/len(holistic_results)*100:.1f}%)",
            ])
        
        if both_ids:
            report_lines.extend([
                "\n## Method Comparison (candidates with both evaluations)",
            ])
            for cid in sorted(both_ids):
                cr = criteria_by_id[cid]
                hr = holistic_by_id[cid]
                diff = cr.overall_score - hr.overall_score
                report_lines.append(f"\n### {cid}")
                report_lines.append(f"- Criteria score: {cr.overall_score:.1f}")
                report_lines.append(f"- Holistic score: {hr.overall_score:.1f}")
                report_lines.append(f"- Difference: {diff:+.1f}")
                report_lines.append(f"- Interview decision (holistic): {'Yes' if hr.interview_decision else 'No'}")
        
        report_content = "\n".join(report_lines)
        
        st.success("Report generated!")
        
        st.markdown("### Report Preview")
        st.markdown(report_content)
        
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()
