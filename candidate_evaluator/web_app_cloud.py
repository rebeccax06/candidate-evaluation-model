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
import streamlit.components.v1 as components
import tempfile
import os
import json
import time
import hashlib
import re
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from candidate_evaluator.core.evaluator import CandidateEvaluator
from candidate_evaluator.core.distribution_analyzer import DistributionAnalyzer
from candidate_evaluator.core.expert_comparison import ExpertComparisonAnalyzer
from candidate_evaluator.core.models import (
    EvaluationResult,
    CandidateProfile,
    CriterionScore,
    EvaluationCriterion,
    Evidence,
    HolisticEvaluationResult,
    InnovationPotential,
    ProgramFit,
    NotableQuality,
    AdmitPatternAnalysisResult,
    InterviewSelectionResult,
    parse_role_specific_assessment,
)
from candidate_evaluator.utils.role_results import (
    ROLE_ORDER,
    ROLE_LABELS,
    eval_row_role,
    group_eval_rows_by_role,
    sort_eval_rows_by_role_score,
    build_role_ranking_row_from_eval_row,
    build_role_dimension_heatmap_html,
    ROLE_HEATMAP_HEIGHT,
    COMBINED_DIMENSION_KEYS,
)
from candidate_evaluator.core.pattern_analyzer import AdmitPatternAnalyzer
from candidate_evaluator.exporters import CSVExporter, JSONExporter
from candidate_evaluator.prompt_manager import PromptManager
from candidate_evaluator.utils.config import get_default_config, Config, APIConfig, CriteriaWeights
from candidate_evaluator.utils.worker_health import is_background_worker_likely_available
from candidate_evaluator.auth import (
    init_auth_state,
    render_auth_ui,
    render_user_menu,
    get_current_user,
    is_logged_in,
    require_api_key,
    get_database,
    get_auth_client
)
from candidate_evaluator.database import Database
from candidate_evaluator.storage import Storage, cleanup_temp_files
from candidate_evaluator.help_assistant import render_guide_page, render_help_chat
from candidate_evaluator.branding import apply_branding, LOGO_PATH


st.set_page_config(
    page_title="Catalyst Candidate Assessment",
    page_icon=LOGO_PATH,
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
    """Get or create Storage instance, ensuring the user JWT is set on the storage client."""
    client = get_auth_client()
    
    # supabase-py auth→storage propagation is unreliable in Streamlit's rerun model,
    # so explicitly re-apply the access token before every storage operation.
    access_token = st.session_state.get('access_token')
    if access_token:
        try:
            client.storage.set_auth(access_token)
        except Exception:
            pass
    
    if st.session_state.storage is None:
        st.session_state.storage = Storage(client)
    return st.session_state.storage


def get_evaluator(api_key: str) -> CandidateEvaluator:
    """Get or create evaluator with user's API key."""
    if st.session_state.evaluator is None or st.session_state.get('current_api_key') != api_key:
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
            # Strip prefix if present and normalize to lowercase
            criterion_value = criterion_value.replace('EvaluationCriterion.', '').lower()
        
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
        role=data.get('role'),
        role_specific_assessment=parse_role_specific_assessment(
            data.get('role_specific_assessment')
        ),
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
                confidence=q.get('confidence', 'medium'),
                evidence=q.get('evidence', ''),
                significance=q.get('significance', ''),
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
        role=data.get('role'),
        role_specific_assessment=parse_role_specific_assessment(
            data.get('role_specific_assessment')
        ),
        metadata=data.get('metadata', {})
    )


def main():
    """Main application."""
    init_session_state()

    apply_branding()

    if not render_auth_ui():
        return
    
    user = get_current_user()
    api_key = require_api_key()
    
    if not api_key:
        st.stop()
    
    with st.sidebar:
        st.markdown("## Catalyst Candidate Assessment")
        render_user_menu()
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "New Evaluation", "Batch Jobs", "Results", "Analysis", "Guide", "Help", "Settings"],
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
    elif page == "Guide":
        render_guide_page()
    elif page == "Help":
        help_page(user, api_key)
    elif page == "Settings":
        settings_page(user)


def help_page(user: dict, api_key: str):
    """Render the Help chatbot using the user's configured Claude client."""
    evaluator = get_evaluator(api_key)
    client = getattr(evaluator, "client", None)
    model = evaluator.config.api.model if evaluator else "claude-sonnet-4-5-20250929"
    session_key = f"help_chat_messages_{user['id']}"
    render_help_chat(client, model, session_key=session_key)


_SELECTION_EVAL_TYPE = "interview_selection"
_SELECTION_CANDIDATE_ID = "__interview_selection__"


def _save_selection_result_cloud(user_id: str, result: InterviewSelectionResult) -> None:
    """Persist an InterviewSelectionResult to Supabase (upsert via delete+insert)."""
    db = get_database()
    # Remove any previous selection result for this user before inserting the new one
    try:
        existing = db.get_user_evaluations(
            user_id, evaluation_type=_SELECTION_EVAL_TYPE, limit=10
        )
        for row in existing:
            db.delete_evaluation(row["id"], user_id)
    except Exception:
        pass
    db.save_evaluation(
        user_id=user_id,
        candidate_id=_SELECTION_CANDIDATE_ID,
        evaluation_type=_SELECTION_EVAL_TYPE,
        result=json.loads(json.dumps(result.model_dump(), default=str)),
    )


def _load_selection_result_cloud(user_id: str) -> InterviewSelectionResult | None:
    """Load the most recent InterviewSelectionResult from Supabase, or None."""
    db = get_database()
    rows = db.get_user_evaluations(user_id, evaluation_type=_SELECTION_EVAL_TYPE, limit=1)
    if not rows:
        return None
    try:
        return InterviewSelectionResult(**rows[0]["result"])
    except Exception:
        return None


def interview_selection_page(user: dict, api_key: str):
    """Two-phase, context-window-efficient interview-selection page."""
    st.title("Interview Selection")
    st.markdown(
        "Select the top *N* candidates to interview from a pool of holistic evaluations. "
        "The system performs a compact first-pass ranking across all candidates, then a deeper "
        "final comparison restricted to the top pool."
    )

    # Load holistic evaluations from Supabase
    _, holistic_results = get_user_results_as_objects(user)

    if not holistic_results:
        st.warning(
            "No holistic evaluations found. "
            "Run evaluations in **Holistic** mode first (New Evaluation or Batch Jobs)."
        )
        return

    st.info(f"Found **{len(holistic_results)}** holistic evaluation(s) available.")

    # Restore last saved result into session state on first load
    session_key = f"interview_selection_result_{user['id']}"
    if session_key not in st.session_state:
        st.session_state[session_key] = _load_selection_result_cloud(user["id"])

    # ------------------------------------------------------------------ #
    # Configuration                                                         #
    # ------------------------------------------------------------------ #
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)

    with col1:
        n_interviews = st.number_input(
            "Number of candidates to interview",
            min_value=1,
            max_value=len(holistic_results),
            value=min(3, len(holistic_results)),
            step=1,
            help="How many candidates should be selected for interviews.",
        )

    with col2:
        pool_multiplier = st.slider(
            "Pool multiplier (Phase-2 pool = N × multiplier)",
            min_value=1.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help=(
                "Controls how many candidates enter the deeper Phase-2 comparison. "
                "Higher values are more thorough but use more context window."
            ),
        )

    pool_size = min(
        len(holistic_results),
        max(int(n_interviews) + 2, int(int(n_interviews) * pool_multiplier)),
    )
    st.caption(
        f"Phase-2 pool will contain **{pool_size}** candidate(s) "
        f"(out of {len(holistic_results)} total)."
    )

    # ------------------------------------------------------------------ #
    # Candidate overview                                                    #
    # ------------------------------------------------------------------ #
    with st.expander("Available candidates", expanded=False):
        overview = []
        for r in sorted(holistic_results, key=lambda x: x.overall_score, reverse=True):
            overview.append(
                {
                    "Candidate ID": r.candidate.candidate_id,
                    "Name": r.candidate.name or "-",
                    "Score": f"{r.overall_score:.1f}/10",
                    "Interview (individual)": "Yes" if r.interview_decision else "No",
                    "Innovation": r.innovation_potential.level.capitalize(),
                    "Program Fit": r.program_fit.level.capitalize(),
                }
            )
        st.dataframe(pd.DataFrame(overview), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Run selection                                                         #
    # ------------------------------------------------------------------ #
    if st.button("Run Interview Selection", use_container_width=True, type="primary"):
        evaluator = get_evaluator(api_key)
        with st.spinner("Running two-phase selection …"):
            try:
                result: InterviewSelectionResult = evaluator.select_interviews_holistic(
                    evaluations=holistic_results,
                    n_interviews=int(n_interviews),
                    pool_multiplier=float(pool_multiplier),
                )
                st.session_state[session_key] = result
                _save_selection_result_cloud(user["id"], result)
            except Exception as exc:
                st.error(f"Selection failed: {exc}")
                return

    # ------------------------------------------------------------------ #
    # Display results (persists across reloads via Supabase + session)     #
    # ------------------------------------------------------------------ #
    result: InterviewSelectionResult = st.session_state.get(session_key)
    if result is None:
        return

    col_hdr, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("Clear result", use_container_width=True):
            st.session_state[session_key] = None
            try:
                db = get_database()
                existing = db.get_user_evaluations(
                    user["id"], evaluation_type=_SELECTION_EVAL_TYPE, limit=10
                )
                for row in existing:
                    db.delete_evaluation(row["id"], user["id"])
            except Exception:
                pass
            st.rerun()

    st.success(
        f"Selected **{len(result.selected_candidate_ids)}** candidate(s) for interview "
        f"from a pool of {len(result.pool_candidate_ids)} finalists "
        f"(total evaluated: {result.metadata.get('n_total_candidates', '?')})."
    )

    # Final selections table
    st.markdown("### Selected for Interview")
    eval_by_id = {r.candidate.candidate_id: r for r in holistic_results}
    selected_rows = []
    for rank, cid in enumerate(result.selected_candidate_ids, 1):
        ev = eval_by_id.get(cid)
        note = result.candidate_notes.get(cid, "")
        selected_rows.append(
            {
                "Rank": rank,
                "Candidate ID": cid,
                "Name": ev.candidate.name if ev else "-",
                "Score": f"{ev.overall_score:.1f}/10" if ev else "-",
                "Program Fit": ev.program_fit.level.capitalize() if ev else "-",
                "Innovation": ev.innovation_potential.level.capitalize() if ev else "-",
                "Red Flags": len(ev.red_flags) if ev and isinstance(ev.red_flags, list) else 0,
                "Note": note,
            }
        )
    st.dataframe(pd.DataFrame(selected_rows), hide_index=True, use_container_width=True)

    if result.selection_rationale:
        st.markdown("**Selection rationale:**")
        st.info(result.selection_rationale)

    # Phase-1 full ranking
    if result.all_candidate_ids_ranked:
        with st.expander("Phase-1 ranking (all candidates)", expanded=False):
            phase1_rows = []
            phase1_rationale = result.metadata.get("phase1_rationale", "")
            for rank, cid in enumerate(result.all_candidate_ids_ranked, 1):
                ev = eval_by_id.get(cid)
                in_pool = cid in result.pool_candidate_ids
                phase1_rows.append(
                    {
                        "Rank": rank,
                        "Candidate ID": cid,
                        "Name": ev.candidate.name if ev else "-",
                        "Score": f"{ev.overall_score:.1f}/10" if ev else "-",
                        "In Phase-2 Pool": "Yes" if in_pool else "No",
                        "Selected": "✓" if cid in result.selected_candidate_ids else "",
                    }
                )
            st.dataframe(
                pd.DataFrame(phase1_rows), hide_index=True, use_container_width=True
            )
            if phase1_rationale:
                st.caption(f"Ranking rationale: {phase1_rationale}")

    elapsed = result.metadata.get("processing_time_seconds")
    if elapsed:
        st.caption(
            f"Completed in {elapsed:.1f}s using {result.metadata.get('model', 'unknown model')}."
        )


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
                        rec = (eval_data.get('recommendation') or '')[:20] or '-'
                        st.caption(rec)
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


ROLE_OPTIONS = {
    "Combined (All Dimensions)": "combined",
}


def new_evaluation_page(user: dict, api_key: str):
    """New evaluation page."""
    st.title("New Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["Single Candidate", "Batch Upload", "Role Dimensions Evaluation"])
    
    with tab1:
        single_evaluation_form(user, api_key, key_prefix="new_single_")
    
    with tab2:
        batch_evaluation_form(user, key_prefix="new_batch_")

    with tab3:
        role_specific_evaluation_form(user, api_key)


def _batch_target_selector(user: dict, key_prefix: str = "") -> dict | None:
    """Optional 'add to existing batch' picker.

    Returns the selected batch job dict, or None to create a standalone evaluation.
    """
    try:
        jobs = get_database().get_user_batch_jobs(user["id"])
    except Exception:
        jobs = []
    # Holistic-only: never attach new candidates to a criteria-based batch.
    jobs = [j for j in jobs if j.get("evaluation_mode", "holistic") == "holistic"]
    if not jobs:
        return None

    options: dict[str, dict | None] = {"— Don't add to a batch —": None}
    for j in jobs:
        created = str(j.get("created_at", ""))[:10]
        mode = j.get("evaluation_mode", "holistic")
        done = j.get("completed_candidates", 0)
        total = j.get("total_candidates", 0)
        label = f"{j.get('job_name') or '(unnamed)'} · {created} · {mode} · {done}/{total}"
        options[label] = j

    choice = st.selectbox(
        "Add to existing batch (optional)",
        list(options.keys()),
        key=f"{key_prefix}add_to_batch_selector",
        help=(
            "Append these candidate(s) to an existing batch so they appear together "
            "in Batch Jobs and Rankings. New candidates inherit the batch's evaluation "
            "mode and role."
        ),
    )
    return options[choice]


def _sanitize_candidate_id(name: str) -> str:
    """Turn a filename into a safe, short candidate ID."""
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (cleaned or "candidate")[:40]


def _generate_candidate_id_cloud(user: dict, uploaded_files) -> str:
    """Auto-generate a unique candidate ID from the uploaded file name.

    Uses the first file's name; if that ID already exists for the user, a
    timestamp suffix is appended to keep it unique.
    """
    base = _sanitize_candidate_id(uploaded_files[0].name) if uploaded_files else "candidate"
    try:
        existing = {
            e.get("candidate_id")
            for e in get_database().get_user_evaluations(user["id"])
        }
    except Exception:
        existing = set()
    if base not in existing:
        return base
    return f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def single_evaluation_form(user: dict, api_key: str, role: str | None = None, key_prefix: str = ""):
    """Single candidate evaluation form."""
    title = "### Evaluate a Single Candidate"
    if role:
        role_label = next((k for k, v in ROLE_OPTIONS.items() if v == role), role)
        title = f"### Evaluate a Single {role_label} Candidate"
    st.markdown(title)

    target_job = _batch_target_selector(user, key_prefix)

    # Holistic-only: evaluation mode is fixed to holistic across the app.
    is_holistic = True
    if target_job:
        eff_role = target_job.get("role")
        st.info(
            f"Adding to batch **{target_job.get('job_name') or '(unnamed)'}** — "
            f"**Holistic (program fit)**"
            + (f", role **{eff_role}**" if eff_role else "")
            + "."
        )
    else:
        eff_role = role
        if role:
            st.caption("Role guidance is appended to the system prompt for this holistic evaluation.")
        st.info("Holistic mode evaluates overall program fit and innovation potential.")

    col1, col2 = st.columns([2, 1])

    with col1:
        candidate_name = st.text_input(
            "Name (optional)",
            placeholder="e.g., John Doe",
            key=f"{key_prefix}single_candidate_name"
        )

        uploaded_files = st.file_uploader(
            "Upload Application Materials",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            key=f"{key_prefix}single_uploader"
        )
        st.caption("A candidate ID is generated automatically from the uploaded file name.")

    candidate_id = _generate_candidate_id_cloud(user, uploaded_files) if uploaded_files else ""

    with col2:
        st.markdown("#### Supported Formats")
        st.markdown("- PDF, Word, Text, Markdown")
        st.markdown("#### Tips")
        st.markdown("- Include all relevant materials")
        st.markdown("- More context = better evaluation")

    if st.button("Evaluate Candidate", disabled=not uploaded_files, key=f"{key_prefix}single_eval_btn"):
        with st.spinner("Evaluating candidate... This may take a minute."):
            material_paths = []
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

                target_job_id = target_job["id"] if target_job else None
                if is_holistic:
                    result = evaluator.evaluate_candidate_holistic(
                        candidate_id=candidate_id,
                        material_paths=material_paths,
                        candidate_name=candidate_name or None,
                        role=eff_role
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(result_dict["candidate"]["evaluation_date"])

                    db.save_evaluation(
                        user_id=user["id"],
                        candidate_id=candidate_id,
                        evaluation_type="holistic",
                        result=result_dict,
                        candidate_name=candidate_name,
                        job_id=target_job_id,
                    )

                    if target_job_id:
                        db.bump_job_counts(target_job_id, add_total=1, add_completed=1)
                        st.success(f"Holistic evaluation completed and added to batch '{target_job.get('job_name') or ''}'.")
                    else:
                        st.success("Holistic evaluation completed!")
                    display_holistic_evaluation_result(result)
                else:
                    result = evaluator.evaluate_candidate(
                        candidate_id=candidate_id,
                        material_paths=material_paths,
                        candidate_name=candidate_name or None,
                        role=eff_role
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(result_dict["candidate"]["evaluation_date"])
                    for score in result_dict.get("scores", []):
                        if "criterion" in score:
                            crit = score["criterion"]
                            score["criterion"] = crit.value if hasattr(crit, 'value') else str(crit)

                    db.save_evaluation(
                        user_id=user["id"],
                        candidate_id=candidate_id,
                        evaluation_type="criteria",
                        result=result_dict,
                        candidate_name=candidate_name,
                        job_id=target_job_id,
                    )

                    if target_job_id:
                        db.bump_job_counts(target_job_id, add_total=1, add_completed=1)
                        st.success(f"Evaluation completed and added to batch '{target_job.get('job_name') or ''}'.")
                    else:
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


def _run_role_batch_sequential_cloud(
    user: dict,
    api_key: str,
    uploaded_files,
    is_holistic: bool,
    role: str,
    target_job_id: str | None = None,
) -> None:
    """Run role-specific batch evaluations inline when the background worker is unavailable.

    When target_job_id is set, results are attached to that batch (skipping candidates
    already in it) and the batch's counters are updated.
    """
    evaluator = get_evaluator(api_key)
    db = get_database()
    evaluation_mode = "holistic" if is_holistic else "criteria"

    already_done: set[str] = set()
    if target_job_id:
        try:
            already_done = {
                e.get("candidate_id")
                for e in db.get_job_evaluations(target_job_id)
                if e.get("candidate_id")
            }
        except Exception:
            already_done = set()

    total = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed = 0
    skipped = 0
    added_total = 0
    added_completed = 0
    added_failed = 0
    failed: list[tuple[str, str]] = []

    st.info(
        "Running evaluations sequentially in your browser. "
        "Keep this tab open until finished."
    )

    temp_dir = tempfile.mkdtemp()
    try:
        for i, uploaded_file in enumerate(uploaded_files):
            candidate_id = Path(uploaded_file.name).stem
            progress_bar.progress(i / total if total else 0)

            if target_job_id and candidate_id in already_done:
                skipped += 1
                status_text.text(f"Skipping {candidate_id} (already in batch)")
                continue

            status_text.text(f"Evaluating {i + 1}/{total}: {candidate_id}")

            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                if is_holistic:
                    result = evaluator.evaluate_candidate_holistic(
                        candidate_id=candidate_id,
                        material_paths=[temp_path],
                        role=role,
                    )
                else:
                    result = evaluator.evaluate_candidate(
                        candidate_id=candidate_id,
                        material_paths=[temp_path],
                        role=role,
                    )

                result_dict = result.model_dump()
                result_dict["candidate"]["evaluation_date"] = str(
                    result_dict["candidate"]["evaluation_date"]
                )
                if not is_holistic:
                    for score in result_dict.get("scores", []):
                        if "criterion" in score:
                            crit = score["criterion"]
                            score["criterion"] = crit.value if hasattr(crit, "value") else str(crit)

                db.save_evaluation(
                    user_id=user["id"],
                    candidate_id=candidate_id,
                    evaluation_type=evaluation_mode,
                    result=result_dict,
                    job_id=target_job_id,
                )
                completed += 1
                if target_job_id:
                    added_total += 1
                    added_completed += 1
                    already_done.add(candidate_id)
            except Exception as e:
                failed.append((candidate_id, str(e)))
                if target_job_id:
                    added_total += 1
                    added_failed += 1
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        if target_job_id and added_total:
            try:
                db.bump_job_counts(
                    target_job_id,
                    add_total=added_total,
                    add_completed=added_completed,
                    add_failed=added_failed,
                )
            except Exception:
                pass
    finally:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    progress_bar.progress(1.0)
    status_text.empty()

    if completed:
        msg = f"Completed {completed}/{total} evaluations."
        if target_job_id:
            msg += " Added to the selected batch."
        st.success(msg)
    if skipped:
        st.info(f"Skipped {skipped} candidate(s) already in the batch.")
    if failed:
        st.error(f"Failed {len(failed)} candidate(s):")
        for candidate_id, err in failed:
            st.text(f"- {candidate_id}: {err}")


def batch_evaluation_form(
    user: dict,
    role: str | None = None,
    key_prefix: str = "",
    api_key: str | None = None,
):
    """Batch evaluation form with background processing."""
    title = "### Batch Evaluation"
    if role:
        role_label = next((k for k, v in ROLE_OPTIONS.items() if v == role), role)
        title = f"### Batch {role_label} Evaluation"
    st.markdown(title)
    if role:
        st.markdown(
            "Upload multiple PDF files to evaluate in the background when the worker is available. "
            "If the background worker is down, evaluations run sequentially in this browser session."
        )
    else:
        st.markdown(
            "Upload multiple PDF files to evaluate in the background. "
            "You can close this page and the evaluation will continue."
        )

    target_job = _batch_target_selector(user, key_prefix)

    # Holistic-only: evaluation mode is fixed to holistic across the app.
    is_holistic = True
    if target_job:
        eff_role = target_job.get("role")
        st.info(
            f"Adding to batch **{target_job.get('job_name') or '(unnamed)'}** — "
            f"new candidates are evaluated as **Holistic (program fit)**"
            + (f", role **{eff_role}**" if eff_role else "")
            + ". Files are appended and the background worker evaluates only the new candidates."
        )
    else:
        eff_role = role
        if role:
            st.caption("Role guidance is appended to the system prompt for these holistic evaluations.")

    uploaded_files = st.file_uploader(
        "Upload Candidate PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="One PDF per candidate. Filename becomes the candidate ID.",
        key=f"{key_prefix}batch_uploader"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")

        with st.expander("View files"):
            for f in uploaded_files:
                st.text(f"- {f.name}")

        if target_job:
            job_name = None
            add_label = "Add to Batch (Background)"
        else:
            job_name = st.text_input(
                "Job Name (optional)",
                placeholder="e.g., Spring 2026 Applicants",
                key=f"{key_prefix}batch_job_name"
            )
            add_label = "Start Batch Evaluation (Background)"

        run_sequential = False
        if api_key:
            col_bg, col_seq = st.columns(2)
            with col_bg:
                start_clicked = st.button(
                    add_label,
                    use_container_width=True,
                    key=f"{key_prefix}batch_eval_btn",
                )
            with col_seq:
                run_sequential = st.button(
                    "Run Sequentially Now",
                    use_container_width=True,
                    key=f"{key_prefix}batch_seq_btn",
                    help="Process candidates one-by-one in this browser tab. Use when the background worker is down.",
                )
        else:
            start_clicked = st.button(
                add_label,
                use_container_width=True,
                key=f"{key_prefix}batch_eval_btn",
            )

        if run_sequential:
            _run_role_batch_sequential_cloud(
                user, api_key, uploaded_files, is_holistic, eff_role,
                target_job_id=target_job["id"] if target_job else None,
            )
            return

        if start_clicked:
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

                    if i < total_files - 1:
                        time.sleep(0.2)

                if target_job:
                    status_text.text("Adding to batch...")
                    db.add_files_to_job(target_job["id"], file_paths)
                    progress_bar.progress(1.0)
                    status_text.empty()
                    st.success(
                        f"Added {len(file_paths)} candidate(s) to batch "
                        f"'{target_job.get('job_name') or ''}'."
                    )
                    st.info("The background worker will evaluate the newly added candidates.")
                else:
                    status_text.text("Creating job...")
                    default_job_name = f"Batch {datetime.now().strftime('%Y%m%d_%H%M')}"
                    if eff_role:
                        role_label = next((k for k, v in ROLE_OPTIONS.items() if v == eff_role), eff_role)
                        default_job_name = f"{role_label} {default_job_name}"
                    job = db.create_job(
                        user_id=user["id"],
                        job_name=job_name or default_job_name,
                        job_type="batch",
                        total_candidates=len(file_paths),
                        file_paths=file_paths,
                        evaluation_mode="holistic" if is_holistic else "criteria",
                        role=eff_role
                    )

                    progress_bar.progress(1.0)
                    status_text.empty()
                    st.success(f"Job created! ID: `{job['id']}`")
                    st.info("The background worker will process your candidates. You can close this page.")

                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"Error creating job: {e}")


def role_specific_evaluation_form(user: dict, api_key: str):
    """Role dimensions evaluation form using a single combined all-dimensions prompt."""
    st.markdown("### Role Dimensions Evaluation")
    st.caption(
        "Evaluate candidates with role-dimension guidance appended to the system prompt. "
        "Runs as a holistic evaluation."
    )

    role = "combined"

    st.info(
        "Every candidate is scored across **all** role dimensions "
        "(clinical, engineering, and research) using one combined prompt."
    )

    sub_tab1, sub_tab2 = st.tabs(["Single Candidate", "Batch Upload"])

    with sub_tab1:
        single_evaluation_form(user, api_key, role=role, key_prefix="role_single_")

    with sub_tab2:
        batch_evaluation_form(user, role=role, key_prefix="role_batch_", api_key=api_key)


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


def _group_evaluations_by_batch(evaluations: list, jobs: list) -> list:
    """Group evaluations by batch (job_id). Single/ad-hoc evals are sub-grouped by date. Returns list of (batch_key, batch_label, evals)."""
    job_map = {str(j["id"]): j for j in jobs}
    groups = {}  # batch_key -> list of evals
    for e in evaluations:
        jid = e.get("job_id")
        if jid:
            key = str(jid)
        else:
            # Sub-group single/ad-hoc by date (YYYY-MM-DD) so they don't all appear as one block
            created = e.get("created_at")
            date_part = str(created)[:10] if created else "no_date"
            key = f"_single_{date_part}"
        if key not in groups:
            groups[key] = []
        groups[key].append(e)

    # Batch jobs: (key, label, evals, sort_key)
    batch_list = []
    single_groups = []
    for key, evals in groups.items():
        if key.startswith("_single_"):
            date_part = key.replace("_single_", "")
            label = f"Single / ad-hoc — {date_part}" if date_part != "no_date" else "Single / ad-hoc (no date)"
            single_groups.append((key, label, evals, date_part))
        else:
            job = job_map.get(key, {})
            created = job.get("created_at") or ""
            batch_list.append((key, label_for_job(job, key), evals, created))
    batch_list.sort(key=lambda x: x[3] or "", reverse=True)
    single_groups.sort(key=lambda x: x[3], reverse=True)

    out = [(k, lbl, ev) for k, lbl, ev, _ in batch_list]
    for k, lbl, ev, _ in single_groups:
        out.append((k, lbl, ev))
    return out


def label_for_job(job: dict, job_id: str) -> str:
    name = (job.get("job_name") or "").strip() or str(job_id)[:24]
    created = job.get("created_at") or ""
    if created:
        try:
            date_part = str(created)[:10]
        except Exception:
            date_part = ""
    else:
        date_part = ""
    return f"{name}" + (f" ({date_part})" if date_part else "")


def _eval_row_id(e: dict) -> str:
    eid = e.get("id")
    return str(eid) if eid is not None else ""


def _sections_split_by_batch_jobs(jobs: list, db, all_evaluations: list) -> list:
    """
    Build one section per batch job using get_job_evaluations (same data as Batch Jobs tab),
    then sections for evaluations not tied to any job (by date).
    """
    placed: set = set()
    sections = []
    sorted_jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)
    for job in sorted_jobs:
        jid = str(job.get("id") or "")
        if not jid:
            continue
        evs = db.get_job_evaluations(jid)
        if not evs:
            continue
        short_id = f"{jid[:8]}…{jid[-4:]}" if len(jid) > 14 else jid
        label = f"{label_for_job(job, jid)} · Batch ID {short_id}"
        sections.append((jid, label, evs))
        for e in evs:
            rid = _eval_row_id(e)
            if rid:
                placed.add(rid)
    orphans = [e for e in all_evaluations if _eval_row_id(e) and _eval_row_id(e) not in placed]
    if not orphans:
        return sections
    by_date: dict = {}
    for e in orphans:
        d = str(e.get("created_at") or "")[:10] or "no_date"
        k = f"_orphan_{d}"
        by_date.setdefault(k, []).append(e)
    for k in sorted(by_date.keys(), reverse=True):
        d = k.replace("_orphan_", "")
        lbl = (
            f"Not from a batch job — {d}"
            if d != "no_date"
            else "Not from a batch job"
        )
        sections.append((k, lbl, by_date[k]))
    return sections


def _split_eval_rows_by_type(batch_evals: list) -> tuple:
    """Partition evaluation DB rows into criteria vs holistic lists (same as Results / Analysis)."""
    criteria_evals = [e for e in batch_evals if e.get("evaluation_type") == "criteria"]
    holistic_evals = [e for e in batch_evals if e.get("evaluation_type") == "holistic"]
    return criteria_evals, holistic_evals


def _batch_evals_to_parsed_results(batch_evals: list) -> tuple:
    """Build (criteria_results, holistic_results) pydantic objects from evaluation DB rows."""
    criteria_evals, holistic_evals = _split_eval_rows_by_type(batch_evals)
    criteria_results = []
    for e in criteria_evals:
        try:
            criteria_results.append(result_dict_to_evaluation_result(e["result"]))
        except Exception:
            pass
    holistic_results = []
    for e in holistic_evals:
        try:
            holistic_results.append(result_dict_to_holistic_result(e["result"]))
        except Exception:
            pass
    return criteria_results, holistic_results


def _render_batch_results(
    batch_key: str,
    batch_label: str,
    batch_evals: list,
) -> None:
    """Render Holistic and Role Rankings tabs for one batch of evaluations."""
    _criteria_evals, holistic_evals = _split_eval_rows_by_type(batch_evals)
    _criteria_results, holistic_results = _batch_evals_to_parsed_results(batch_evals)
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}

    role_eval_count = sum(1 for e in batch_evals if eval_row_role(e))
    tab2, tab4 = st.tabs([
        f"Holistic ({len(holistic_evals)})",
        f"Role Rankings ({role_eval_count})",
    ])
    k = batch_key.replace("-", "_")[:30]

    with tab2:
        if not holistic_evals:
            st.info("No holistic evaluations in this batch.")
        else:
            summary_data = []
            sorted_evals = sorted(holistic_evals, key=lambda e: e.get("overall_score", 0) or 0, reverse=True)
            n_h = len(sorted_evals)
            for rank, e in enumerate(sorted_evals, 1):
                result_data = e.get("result", {})
                pct = 100 if n_h == 1 else round(100.0 * (n_h - rank) / (n_h - 1))
                summary_data.append({
                    'Rank': rank,
                    'Percentile': f"{pct}th",
                    'Candidate ID': e.get('candidate_id', ''),
                    'Name': e.get('candidate_name') or '-',
                    'Score': f"{e.get('overall_score', 0):.1f}/10" if e.get('overall_score') else 'N/A',
                    'Interview': 'Yes' if result_data.get('interview_decision') else 'No',
                    'Innovation': result_data.get('innovation_potential', {}).get('level', '-').capitalize() if isinstance(result_data.get('innovation_potential'), dict) else '-',
                    'Program Fit': result_data.get('program_fit', {}).get('level', '-').capitalize() if isinstance(result_data.get('program_fit'), dict) else '-',
                    'Date': str(e.get('created_at', ''))[:10]
                })
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
            st.subheader("Candidate Details")
            options = {f"{e['candidate_id']} ({e.get('overall_score', 0):.1f})": e for e in sorted_evals}
            selected = st.selectbox("Select candidate", list(options.keys()), key=f"hol_{k}")
            if selected:
                result = result_dict_to_holistic_result(options[selected]["result"])
                display_holistic_evaluation_result(result)

    with tab4:
        _render_role_specific_rankings(batch_evals, key_prefix=f"batch_{k}_")


def _render_role_specific_rankings(batch_evals: list, key_prefix: str = "") -> None:
    """Render role-specific leaderboards grouped by specialty."""
    role_evals = [e for e in batch_evals if eval_row_role(e)]
    if not role_evals:
        st.info(
            "No role dimensions evaluations in this set. Use **New Evaluation → "
            "Role Dimensions Evaluation** to evaluate across role dimensions."
        )
        return

    st.caption(
        "Candidates ranked by **role-specific score** across all role dimensions "
        "(ties broken by overall Catalyst score)."
    )
    grouped = group_eval_rows_by_role(role_evals)

    role_tabs = st.tabs([
        f"{ROLE_LABELS[role]} ({len(grouped.get(role, []))})"
        for role in ROLE_ORDER
    ])

    for role, role_tab in zip(ROLE_ORDER, role_tabs):
        with role_tab:
            rows = grouped.get(role, [])
            if not rows:
                st.info(f"No {ROLE_LABELS[role]} evaluations yet.")
                continue
            sorted_rows = sort_eval_rows_by_role_score(rows)
            n = len(sorted_rows)
            summary = [
                build_role_ranking_row_from_eval_row(rank, row, n)
                for rank, row in enumerate(sorted_rows, 1)
            ]
            st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

            options = {}
            for row in sorted_rows:
                cid = row.get("candidate_id", "")
                rs = build_role_ranking_row_from_eval_row(1, row, n)["Role Score"]
                options[f"{cid} ({rs})"] = row
            selected = st.selectbox(
                f"View {ROLE_LABELS[role]} candidate",
                list(options.keys()),
                key=f"{key_prefix}role_sel_{role}",
            )
            if selected:
                row = options[selected]
                result_data = row["result"]
                if row.get("evaluation_type") == "holistic":
                    display_holistic_evaluation_result(result_dict_to_holistic_result(result_data))
                else:
                    display_evaluation_result(result_dict_to_evaluation_result(result_data))


def _analysis_widget_key(batch_key: str, widget_family: str) -> str:
    """Stable short key for Streamlit widgets when the same analysis UI is repeated per batch."""
    digest = hashlib.md5(f"{batch_key}|{widget_family}".encode()).hexdigest()[:16]
    return f"an_{digest}"


def _infer_job_ids_for_evaluations(evaluations: list, jobs: list, db) -> None:
    """When job_id is missing on evaluations, infer it from per-job queries (same as Batch Jobs tab). Mutates evaluations in place."""
    eval_ids_missing_job = {e.get("id") for e in evaluations if not e.get("job_id")}
    if not eval_ids_missing_job:
        return
    eval_id_to_job_id = {}
    for job in jobs:
        jid = str(job.get("id") or "")
        if not jid:
            continue
        for ev in db.get_job_evaluations(jid):
            eid = ev.get("id")
            if eid and eid in eval_ids_missing_job:
                eval_id_to_job_id[eid] = jid
    for e in evaluations:
        if not e.get("job_id") and e.get("id") in eval_id_to_job_id:
            e["job_id"] = eval_id_to_job_id[e["id"]]


_NO_BATCH_SCOPE_MSG = (
    "No batch-linked evaluations found. Run a batch job from **Batch Jobs**, "
    "or switch to **Everything together**."
)
_NO_BATCH_SCOPE_MSG_EXPERT = (
    "No batch-linked evaluations found. Switch to **Everything together** "
    "or run evaluations from **Batch Jobs**."
)


def _load_user_evaluations_context(user: dict, limit_evals: int = 500, limit_jobs: int = 300):
    """
    Shared load path for Results and Analysis: DB rows, job inference, typed row splits.
    Returns None if the user has no evaluations.
    """
    db = get_database()
    evaluations = db.get_user_evaluations(user["id"], limit=limit_evals)
    jobs = db.get_user_jobs(user["id"], limit=limit_jobs)
    if not evaluations:
        return None
    _infer_job_ids_for_evaluations(evaluations, jobs, db)
    criteria_evals, holistic_evals = _split_eval_rows_by_type(evaluations)
    return {
        "db": db,
        "evaluations": evaluations,
        "jobs": jobs,
        "criteria_evals": criteria_evals,
        "holistic_evals": holistic_evals,
        "n_criteria": len(criteria_evals),
        "n_holistic": len(holistic_evals),
    }


def _parsed_sections_for_scope(split_by_batch: bool, batches_split, evaluations: list) -> list:
    """
    Build (batch_key, batch_label, batch_eval_rows, criteria_results, holistic_results) per section.
    Used so Analysis parses each batch once and matches Results batch grouping.
    """
    if split_by_batch:
        if not batches_split:
            return []
        return [
            (k, lbl, ev, *_batch_evals_to_parsed_results(ev))
            for k, lbl, ev in batches_split
        ]
    c_res, h_res = _batch_evals_to_parsed_results(evaluations)
    return [("_all", "All evaluations", evaluations, c_res, h_res)]


def results_page(user: dict):
    """View all evaluation results, separated by batch."""
    st.title("Evaluation Results")

    ctx = _load_user_evaluations_context(user)
    if ctx is None:
        st.info("No evaluation results yet. Run some evaluations first!")
        return

    db = ctx["db"]
    evaluations = ctx["evaluations"]
    jobs = ctx["jobs"]
    criteria_evals = ctx["criteria_evals"]
    holistic_evals = ctx["holistic_evals"]
    
    st.markdown("---")
    display_mode = st.selectbox(
        "**Display mode**",
        [
            "Split by batch — one section per job",
            "Everything together — single combined list",
        ],
        index=0,
        key="results_display_mode_main",
        help="Use “Split by batch” to see each run in its own section.",
    )
    split_by_batch = display_mode.startswith("Split")
    batches_split = None
    if split_by_batch:
        batches_split = _sections_split_by_batch_jobs(jobs, db, evaluations)
    job_section_count = (
        sum(1 for k, _, _ in batches_split if not str(k).startswith("_orphan_"))
        if batches_split
        else "—"
    )
    st.markdown("---")
    role_eval_count = sum(1 for e in evaluations if eval_row_role(e))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Batches", job_section_count)
    with col2:
        st.metric("Holistic", len(holistic_evals))
    with col3:
        st.metric("Role Dimensions", role_eval_count)
    with col4:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    if split_by_batch:
        if not batches_split:
            st.info(_NO_BATCH_SCOPE_MSG)
        for batch_key, batch_label, batch_evals in (batches_split or []):
            n = len(batch_evals)
            with st.expander(
                f"{batch_label} — {n} evaluation(s)",
                expanded=(len(batches_split or []) <= 3),
            ):
                _render_batch_results(batch_key, batch_label, batch_evals)
    else:
        _criteria_results, holistic_results = _batch_evals_to_parsed_results(evaluations)
        holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}
        
        tab2, tab4 = st.tabs([
            f"Holistic ({len(holistic_evals)})",
            f"Role Rankings ({role_eval_count})",
        ])
        
        with tab2:
            if not holistic_evals:
                st.info("No holistic evaluations yet.")
            else:
                summary_data = []
                sorted_evals = sorted(holistic_evals, key=lambda e: e.get("overall_score", 0) or 0, reverse=True)
                n_h = len(sorted_evals)
                for rank, e in enumerate(sorted_evals, 1):
                    result_data = e.get("result", {})
                    pct = 100 if n_h == 1 else round(100.0 * (n_h - rank) / (n_h - 1))
                    summary_data.append({
                        'Rank': rank,
                        'Percentile': f"{pct}th",
                        'Candidate ID': e.get('candidate_id', ''),
                        'Name': e.get('candidate_name') or '-',
                        'Score': f"{e.get('overall_score', 0):.1f}/10" if e.get('overall_score') else 'N/A',
                        'Interview': 'Yes' if result_data.get('interview_decision') else 'No',
                        'Innovation': result_data.get('innovation_potential', {}).get('level', '-').capitalize() if isinstance(result_data.get('innovation_potential'), dict) else '-',
                        'Program Fit': result_data.get('program_fit', {}).get('level', '-').capitalize() if isinstance(result_data.get('program_fit'), dict) else '-',
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

        with tab4:
            _render_role_specific_rankings(evaluations, key_prefix="all_")


def display_disparity_analysis(criteria_by_id: dict, holistic_by_id: dict, both_ids: list):
    """Display statistical analysis of method disparity vs candidate spread."""
    if len(both_ids) < 3:
        st.info("Need at least 3 candidates with both evaluations for disparity analysis.")
        return

    criteria_scores = [criteria_by_id[cid].overall_score for cid in both_ids]
    holistic_scores = [holistic_by_id[cid].overall_score for cid in both_ids]
    score_diffs = [criteria_by_id[cid].overall_score - holistic_by_id[cid].overall_score
                   for cid in both_ids]

    criteria_std = np.std(criteria_scores)
    holistic_std = np.std(holistic_scores)
    candidate_std = (criteria_std + holistic_std) / 2
    method_disparity_std = np.std(score_diffs)
    correlation = np.corrcoef(criteria_scores, holistic_scores)[0, 1]
    disparity_ratio = method_disparity_std / candidate_std if candidate_std > 0 else 0

    st.markdown("### Method Disparity Analysis")
    st.caption("Comparing the spread of scores across candidates vs. the disagreement between evaluation methods.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Candidate Spread (σ)", f"{candidate_std:.2f} pts")
    with col2:
        st.metric("Method Disparity (σ)", f"{method_disparity_std:.2f} pts")
    with col3:
        st.metric("Disparity Ratio", f"{disparity_ratio:.2f}")
    with col4:
        st.metric("Correlation", f"{correlation:.2f}")

    if disparity_ratio < 0.3:
        st.success(f"**Methods strongly agree.** Method differences ({method_disparity_std:.2f}) are much smaller than candidate differences ({candidate_std:.2f}).")
    elif disparity_ratio < 0.5:
        st.info(f"**Methods mostly agree.** Method differences are moderate compared to candidate spread.")
    elif disparity_ratio < 0.7:
        st.warning(f"**Moderate disagreement.** Method choice affects scores nearly as much as candidate quality.")
    else:
        st.error(f"**Significant disagreement.** Method differences ({method_disparity_std:.2f}) are large relative to candidate spread ({candidate_std:.2f}).")

    st.markdown("---")


def display_comparison_summary(criteria_result, holistic_result):
    """Display a comparison summary between criteria-based and holistic evaluations."""
    score_diff = criteria_result.overall_score - holistic_result.overall_score

    st.markdown("### Comparison Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Criteria Score", f"{criteria_result.overall_score:.1f}")
    with col2:
        st.metric("Holistic Score", f"{holistic_result.overall_score:.1f}")
    with col3:
        if score_diff > 0:
            delta_label = "Criteria higher"
        elif score_diff < 0:
            delta_label = "Holistic higher"
        else:
            delta_label = "Equal"
        st.metric("Score Difference", f"{abs(score_diff):.1f}", delta=delta_label)
    with col4:
        interview = "Yes" if holistic_result.interview_decision else "No"
        st.metric("Interview (Holistic)", interview)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Criteria Recommendation:** {criteria_result.recommendation}")
    with col2:
        st.markdown(f"**Holistic Recommendation:** {holistic_result.recommendation}")

    recs_match = criteria_result.recommendation.lower() == holistic_result.recommendation.lower()
    if recs_match:
        st.success("Recommendations align")
    elif abs(score_diff) > 1.5:
        st.warning(f"Significant score difference: {abs(score_diff):.1f} points")

    st.markdown("---")


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

    st.markdown("---")

    st.subheader("Prompt Editor")
    st.caption("View and customize the prompts used for candidate evaluation.")

    if 'prompt_manager' not in st.session_state:
        st.session_state.prompt_manager = PromptManager()

    prompt_manager = st.session_state.prompt_manager

    prompt_options = {
        "System Prompt": "system",
        "Holistic Template": "holistic",
        "Combined Role Prompt (All Dimensions)": "combined",
    }

    selected_prompt_name = st.selectbox(
        "Select Prompt to View/Edit",
        list(prompt_options.keys()),
        key="prompt_selector"
    )
    selected_prompt_type = prompt_options[selected_prompt_name]

    metadata = prompt_manager.get_prompt_metadata(selected_prompt_type)

    if metadata["is_custom"]:
        st.info(f"Status: **Custom** (modified {metadata['updated_at'][:10] if metadata['updated_at'] else 'unknown'})")
    else:
        st.success("Status: **Default** (using built-in prompt)")

    if selected_prompt_type in PromptManager.ROLE_PROMPT_TYPES:
        st.caption(
            "This role prompt is **appended** to the System Prompt during role dimensions evaluations."
        )

    if selected_prompt_type == "system":
        current_content = prompt_manager.get_system_prompt()
    elif selected_prompt_type == "criteria":
        current_content = prompt_manager.get_criteria_template()
    elif selected_prompt_type == "holistic":
        current_content = prompt_manager.get_holistic_template()
    elif selected_prompt_type == "ranking":
        current_content = prompt_manager.get_ranking_template()
    elif selected_prompt_type == "selection":
        current_content = prompt_manager.get_selection_template()
    else:
        current_content = prompt_manager.get_combined_prompt()

    edited_content = st.text_area(
        f"Edit {selected_prompt_name}",
        value=current_content,
        height=400,
        key=f"prompt_editor_{selected_prompt_type}"
    )

    st.caption(f"Character count: {len(edited_content):,}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Changes", type="primary", use_container_width=True):
            if edited_content != current_content:
                if selected_prompt_type == "system":
                    prompt_manager.save_system_prompt(edited_content)
                elif selected_prompt_type == "criteria":
                    prompt_manager.save_criteria_template(edited_content)
                elif selected_prompt_type == "holistic":
                    prompt_manager.save_holistic_template(edited_content)
                elif selected_prompt_type == "ranking":
                    prompt_manager.save_ranking_template(edited_content)
                elif selected_prompt_type == "selection":
                    prompt_manager.save_selection_template(edited_content)
                elif selected_prompt_type in PromptManager.ROLE_PROMPT_TYPES:
                    prompt_manager.save_role_prompt(selected_prompt_type, edited_content)
                st.success("Prompt saved successfully!")
                st.rerun()
            else:
                st.warning("No changes to save.")
    with col2:
        if st.button("Reset to Default", use_container_width=True):
            if metadata["is_custom"]:
                prompt_manager.reset_to_default(selected_prompt_type)
                st.success("Reset to default prompt!")
                st.rerun()
            else:
                st.info("Already using default prompt.")
    with col3:
        if st.button("View Default", use_container_width=True):
            default_content = prompt_manager.get_default_prompt(selected_prompt_type)
            st.text_area(
                "Default Prompt (read-only)",
                value=default_content,
                height=300,
                disabled=True,
                key="default_prompt_view"
            )

    if selected_prompt_type in ["criteria", "holistic", "ranking", "selection"]:
        with st.expander("Template Variables"):
            if selected_prompt_type == "criteria":
                st.markdown("""
**Available placeholders:**
- `{materials}` — Candidate application materials (required)
- `{criteria_details}` — Formatted criteria rubrics (required)

**Note:** Do not remove these placeholders or the evaluation will fail.
""")
            elif selected_prompt_type == "holistic":
                st.markdown("""
**Available placeholders:**
- `{materials}` — Candidate application materials (required)

**Note:** Do not remove this placeholder or the evaluation will fail.
""")
            elif selected_prompt_type == "ranking":
                st.markdown("""
**Available placeholders:**
- `{n_candidates}` — Total number of candidates being ranked (required)
- `{candidates_data}` — Formatted candidate profiles (required)

**Note:** Candidate profiles are generated automatically from each candidate's holistic evaluation.
""")
            elif selected_prompt_type == "selection":
                st.markdown("""
**Available placeholders:**
- `{n_pool}` — Number of candidates in the final pool (required)
- `{n_interviews}` — Number of candidates to select (required)
- `{candidates_data}` — Formatted candidate profiles, including notable qualities (required)

**Note:** Candidate profiles are generated automatically from each candidate's holistic evaluation.
""")


def _render_role_dimension_heatmap(assessment) -> None:
    """Render a 3x3 red (worst) -> green (best) heatmap across the 9 role dimensions."""
    try:
        heatmap_html = build_role_dimension_heatmap_html(assessment)
    except Exception:
        return
    st.markdown("**Dimension Heatmap** (red = weaker, green = stronger")
    components.html(heatmap_html, height=ROLE_HEATMAP_HEIGHT, scrolling=False)


def display_role_specific_assessment(assessment) -> None:
    """Display structured role-specific assessment when present."""
    if not assessment:
        return
    if hasattr(assessment, "model_dump"):
        assessment = assessment.model_dump()

    role_label = assessment.get("role", "unknown").replace("_", " ").title()
    marker = assessment.get("marker", "Role-specific assessment")
    score = assessment.get("score")
    confidence = assessment.get("confidence", "medium")

    st.markdown(f"#### Role Dimensions Assessment ({role_label})")
    cols = st.columns(3)
    with cols[0]:
        if score is not None:
            st.metric(marker, f"{score}/10")
    with cols[1]:
        st.metric("Confidence", str(confidence).capitalize())
    with cols[2]:
        rated = sum(1 for k in COMBINED_DIMENSION_KEYS if assessment.get(k))
        st.metric("Dimensions Rated", f"{rated}/{len(COMBINED_DIMENSION_KEYS)}")

    if assessment.get("reasoning"):
        st.markdown(assessment["reasoning"])

    _render_role_dimension_heatmap(assessment)

    # Show any non-standard extra fields that are not part of the 9 dimensions.
    _standard = {"role", "marker", "score", "confidence", "reasoning", "evidence", "evidence_gaps"}
    for key, value in assessment.items():
        if key in _standard or key in COMBINED_DIMENSION_KEYS:
            continue
        if value:
            st.caption(f"**{key.replace('_', ' ').title()}:** {str(value).replace('_', ' ')}")

    evidence = assessment.get("evidence") or []
    if evidence:
        with st.expander("Role Dimensions Evidence"):
            for ev in evidence:
                if isinstance(ev, dict):
                    _holistic_evidence_dict_block(ev)
                else:
                    st.markdown(str(ev))
                st.markdown("---")

    gaps = assessment.get("evidence_gaps") or []
    if gaps:
        st.markdown("**Evidence gaps for interview:**")
        for gap in gaps:
            st.markdown(f"- {gap}")


def display_evaluation_result(result: EvaluationResult):
    """Display a criteria-based evaluation result."""
    if result.candidate.name:
        st.caption(f"Name: {result.candidate.name}")

    if result.role:
        st.caption(f"Role context: **{result.role.replace('_', ' ').title()}**")

    if len(result.scores) < 11:
        st.warning(f"Incomplete evaluation: only {len(result.scores)}/11 criteria scored. "
                   f"This may be due to output truncation. Re-run this candidate for a complete evaluation.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{result.overall_score:.1f}/10")
    with col2:
        st.metric("Criteria", f"{len(result.scores)}/11")
    with col3:
        high_conf = sum(1 for s in result.scores if s.confidence == 'high')
        st.metric("High Confidence", f"{high_conf}/{len(result.scores)}")
    with col4:
        st.metric("Recommendation", result.recommendation)

    col1, col2 = st.columns(2)
    with col1:
        if result.strengths:
            st.markdown("#### Strengths")
            for s in result.strengths:
                st.markdown(f"- {s}")
    with col2:
        if result.areas_for_development:
            st.markdown("#### Areas for Development")
            for a in result.areas_for_development:
                st.markdown(f"- {a}")

    st.markdown("#### Scores by Criterion")
    score_data = pd.DataFrame([
        {
            'Criterion': score.criterion.value.replace('_', ' ').title(),
            'Score': score.score,
            'Confidence': score.confidence.capitalize()
        }
        for score in sorted(result.scores, key=lambda s: s.score, reverse=True)
    ])
    st.dataframe(score_data, hide_index=True, use_container_width=True)

    chart_data = pd.DataFrame([
        {'Criterion': s.criterion.value.replace('_', ' ').title(), 'Score': s.score}
        for s in result.scores
    ])
    st.bar_chart(chart_data.set_index('Criterion'))

    if result.overall_assessment:
        with st.expander("Overall Assessment"):
            st.write(result.overall_assessment)

    display_role_specific_assessment(result.role_specific_assessment)

    with st.expander("View Detailed Evidence"):
        for score in result.scores:
            st.markdown(f"**{score.criterion.value.replace('_', ' ').title()}** - {score.score}/10 ({score.confidence})")
            st.markdown(score.reasoning)
            if score.evidence:
                for ev in score.evidence:
                    st.markdown(f"> *{ev.source}:* {ev.quote}")
            st.markdown("---")


def _holistic_evidence_dict_block(ev: dict) -> None:
    """Render quote/source/context evidence dict as readable text (not Python repr)."""
    quote = (ev.get("quote") or "").strip()
    source = (ev.get("source") or "").strip()
    context = (ev.get("context") or "").strip()
    if source:
        st.markdown(f"**Source:** {source}")
    if context:
        st.markdown(f"**Context:** {context}")
    if quote:
        st.markdown(quote)


def _render_notable_quality_evidence(evidence) -> None:
    """Notable quality evidence: string, dict, list of dicts, or HolisticEvidence-like objects."""
    if evidence is None or evidence == "":
        return
    if isinstance(evidence, dict):
        _holistic_evidence_dict_block(evidence)
        return
    if isinstance(evidence, str):
        st.markdown(evidence)
        return
    if isinstance(evidence, list):
        if evidence:
            st.markdown("**Evidence:**")
        for ev in evidence:
            if isinstance(ev, dict):
                _holistic_evidence_dict_block(ev)
            elif hasattr(ev, "quote"):
                source_text = f" *({ev.source})*" if getattr(ev, "source", None) else ""
                st.markdown(f"> \"{ev.quote}\"{source_text}")
                if getattr(ev, "context", None):
                    st.caption(ev.context)
            else:
                st.markdown(f"> {ev}")
        return
    st.markdown(str(evidence))


def _red_flag_parts(flag) -> tuple:
    """Return (flag_text, evidence_text, severity) for RedFlag models or dicts."""
    if isinstance(flag, dict):
        return (
            (flag.get("flag") or "").strip(),
            (flag.get("evidence") or "").strip(),
            str(flag.get("severity") or "medium").lower(),
        )
    if hasattr(flag, "flag"):
        return (
            str(flag.flag).strip(),
            str(getattr(flag, "evidence", "") or "").strip(),
            str(getattr(flag, "severity", "medium") or "medium").lower(),
        )
    return str(flag).strip(), "", "medium"


def _interview_question_parts(q) -> tuple:
    """Return (category, purpose, question) for InterviewQuestion models or dicts."""
    if isinstance(q, dict):
        return (
            (q.get("category") or "General").strip(),
            (q.get("purpose") or "").strip(),
            (q.get("question") or "").strip(),
        )
    if hasattr(q, "question"):
        return (
            str(getattr(q, "category", None) or "General").strip(),
            str(getattr(q, "purpose", None) or "").strip(),
            str(q.question).strip(),
        )
    return "General", "", str(q).strip()


def display_holistic_evaluation_result(result: HolisticEvaluationResult):
    """Display a holistic evaluation result (supports both old and enhanced formats)."""
    if result.candidate.name:
        st.caption(f"Name: {result.candidate.name}")

    if result.role:
        st.caption(f"Role context: **{result.role.replace('_', ' ').title()}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{result.overall_score:.1f}/10")
    with col2:
        interview_text = "Yes" if result.interview_decision else "No"
        st.metric("Interview Recommendation", interview_text)
    with col3:
        innovation_conf = getattr(result.innovation_potential, 'confidence', 'medium')
        st.metric("Innovation Potential", f"{result.innovation_potential.level.capitalize()} ({innovation_conf})")
    with col4:
        fit_conf = getattr(result.program_fit, 'confidence', 'medium')
        st.metric("Program Fit", f"{result.program_fit.level.capitalize()} ({fit_conf})")

    st.markdown(f"**Recommendation:** {result.recommendation}")

    if hasattr(result, 'score_justification') and result.score_justification:
        with st.expander("Score Justification"):
            st.markdown(result.score_justification)

    if hasattr(result, 'interview_decision_reasoning') and result.interview_decision_reasoning:
        with st.expander("Interview Decision Reasoning"):
            st.markdown(result.interview_decision_reasoning)

    st.markdown("#### Overall Assessment")
    st.markdown(result.overall_assessment)

    display_role_specific_assessment(result.role_specific_assessment)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Innovation Potential")
        st.markdown(f"**Level:** {result.innovation_potential.level.capitalize()} (Confidence: {getattr(result.innovation_potential, 'confidence', 'medium')})")
        st.markdown(result.innovation_potential.reasoning)

        if hasattr(result.innovation_potential, 'evidence') and result.innovation_potential.evidence:
            st.markdown("**Evidence:**")
            for ev in result.innovation_potential.evidence:
                if hasattr(ev, 'quote'):
                    source_text = f" *({ev.source})*" if ev.source else ""
                    st.markdown(f"> \"{ev.quote}\"{source_text}")
                    if ev.context:
                        st.caption(f"Context: {ev.context}")
        elif result.innovation_potential.key_evidence:
            st.markdown("**Key Evidence:**")
            for ev in result.innovation_potential.key_evidence:
                st.markdown(f"> {ev}")

    with col2:
        st.markdown("#### Program Fit")
        st.markdown(f"**Level:** {result.program_fit.level.capitalize()} (Confidence: {getattr(result.program_fit, 'confidence', 'medium')})")

        if hasattr(result.program_fit, 'detailed_analysis') and result.program_fit.detailed_analysis:
            st.markdown(result.program_fit.detailed_analysis)

        if result.program_fit.strengths_for_program:
            st.markdown("**Strengths:**")
            for s in result.program_fit.strengths_for_program:
                st.markdown(f"- {s}")
        if result.program_fit.concerns:
            st.markdown("**Concerns:**")
            for c in result.program_fit.concerns:
                st.markdown(f"- {c}")

        if hasattr(result.program_fit, 'evidence') and result.program_fit.evidence:
            with st.expander("Supporting Evidence"):
                for ev in result.program_fit.evidence:
                    if hasattr(ev, 'quote'):
                        source_text = f" *({ev.source})*" if ev.source else ""
                        st.markdown(f"> \"{ev.quote}\"{source_text}")
                        if ev.context:
                            st.caption(f"Context: {ev.context}")

    if result.notable_qualities:
        st.markdown("#### Notable Qualities")
        for q in result.notable_qualities:
            confidence = getattr(q, 'confidence', 'medium')
            with st.expander(f"{q.quality} (Confidence: {confidence})"):
                _render_notable_quality_evidence(q.evidence)
                st.markdown(f"**Significance:** {q.significance}")

    if result.red_flags:
        st.markdown("#### Red Flags")
        for flag in result.red_flags:
            flag_text, evidence_text, severity = _red_flag_parts(flag)
            if not flag_text:
                st.warning(str(flag))
                continue
            st.markdown(f"#### {flag_text}")
            st.caption(f"Severity: {severity.capitalize()}")
            if evidence_text:
                st.markdown(evidence_text)

    if result.questions_for_interview:
        st.markdown("#### Suggested Interview Questions")
        for q in result.questions_for_interview:
            category, purpose, question = _interview_question_parts(q)
            st.markdown(f"#### {category}")
            if purpose:
                st.markdown(f"**{purpose}**")
            if question:
                st.markdown(question)
            st.markdown("")


def get_user_results_as_objects(user: dict):
    """Load user's evaluations and convert to result objects (same parsing as Results / Analysis)."""
    ctx = _load_user_evaluations_context(user)
    if ctx is None:
        return [], []
    return _batch_evals_to_parsed_results(ctx["evaluations"])


def analysis_page(user: dict):
    """Analysis dashboard."""
    st.title("Analysis Dashboard")

    ctx = _load_user_evaluations_context(user)
    if ctx is None:
        st.warning("No evaluation results found. Run some evaluations first!")
        return

    db = ctx["db"]
    evaluations = ctx["evaluations"]
    jobs = ctx["jobs"]
    n_criteria = ctx["n_criteria"]
    n_holistic = ctx["n_holistic"]

    display_mode = st.selectbox(
        "**Analysis scope**",
        [
            "Everything together — all evaluations",
            "By batch — one section per batch job",
        ],
        index=0,
        key="analysis_display_mode",
        help="Match a single batch job (batch ID) or combine every evaluation in your account.",
    )
    split_by_batch = display_mode.startswith("By batch")
    batches_split = (
        _sections_split_by_batch_jobs(jobs, db, evaluations) if split_by_batch else None
    )
    sections = _parsed_sections_for_scope(split_by_batch, batches_split, evaluations)
    expanded_batches = len(sections) <= 3 if split_by_batch else False

    st.markdown(f"**{n_holistic} holistic evaluations**")

    st.subheader("Distribution Analysis")
    if split_by_batch and not sections:
        st.info(_NO_BATCH_SCOPE_MSG)
    elif not n_holistic:
        st.info("No holistic evaluations to analyze. Run holistic evaluations first.")
    elif split_by_batch:
        for batch_key, batch_label, _ev, _c, h_res in sections:
            nh = len(h_res)
            with st.expander(
                f"{batch_label} — {nh} holistic",
                expanded=expanded_batches,
            ):
                if h_res:
                    holistic_distribution_analysis(h_res)
                else:
                    st.info("No holistic evaluations in this batch.")
    else:
        _, _, _, _c, h_res = sections[0]
        holistic_distribution_analysis(h_res)


def distribution_analysis(all_results, key_prefix="dist_default"):
    """Score distribution analysis."""
    st.subheader("Score Distribution Analysis")

    analyzer = DistributionAnalyzer(all_results)

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

    percentile = st.slider(
        "Percentile Split", 25, 75, 50, key=f"{key_prefix}_percentile_split"
    )

    if st.button("Analyze", key=f"{key_prefix}_analyze_distribution"):
        top_group, bottom_group = analyzer.segment_by_percentile(percentile)
        comparison = analyzer.compare_groups(top_group, bottom_group)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Top {100-percentile}% ({len(top_group)} candidates)")
            if top_group:
                top_analysis = analyzer.analyze_group(top_group)
                st.metric("Mean Score", f"{top_analysis.mean_overall_score:.2f}")
        with col2:
            st.markdown(f"### Bottom {percentile}% ({len(bottom_group)} candidates)")
            if bottom_group:
                bottom_analysis = analyzer.analyze_group(bottom_group)
                st.metric("Mean Score", f"{bottom_analysis.mean_overall_score:.2f}")
        
        st.markdown("### Most Discriminating Criteria")
        for i, crit in enumerate(comparison.get('most_discriminating_criteria', []), 1):
            name = crit['criterion'].replace('_', ' ').title()
            st.markdown(f"**{i}. {name}**: {crit['difference']:.1f} point gap")
        
        st.markdown("### Insights")
        for insight in comparison.get('insights', []):
            st.info(insight)
    
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


def recommendation_analysis(all_results, key_prefix="rec_default"):
    """Recommendation breakdown analysis."""
    st.subheader("Recommendation Analysis")

    recommendations = {}
    for result in all_results:
        rec = result.recommendation if result.recommendation else "Unknown"
        if rec not in recommendations:
            recommendations[rec] = []
        recommendations[rec].append(result)

    st.markdown(f"**{len(recommendations)} unique recommendation types**")

    for i, (rec_type, candidates) in enumerate(
        sorted(recommendations.items(), key=lambda x: -len(x[1]))
    ):
        with st.expander(
            f"{rec_type} ({len(candidates)} candidates)",
            key=f"{key_prefix}_rec_exp_{i}",
        ):
            if candidates:
                avg = sum(c.overall_score for c in candidates) / len(candidates)
                st.metric("Average Score", f"{avg:.2f}")
                
                for c in sorted(candidates, key=lambda x: -x.overall_score)[:5]:
                    st.text(f"- {c.candidate.candidate_id}: {c.overall_score:.1f}/10")


def holistic_distribution_analysis(holistic_results):
    """Score distribution analysis for holistic evaluations."""
    st.subheader("Holistic Score Distribution Analysis")

    overall_scores = [r.overall_score for r in holistic_results]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Score", f"{sum(overall_scores)/len(overall_scores):.2f}")
    with col2:
        st.metric("Median Score", f"{sorted(overall_scores)[len(overall_scores)//2]:.2f}")
    with col3:
        st.metric("Min Score", f"{min(overall_scores):.2f}")
    with col4:
        st.metric("Max Score", f"{max(overall_scores):.2f}")

    interview_yes = sum(1 for r in holistic_results if r.interview_decision)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Interview Recommended", f"{interview_yes}/{len(holistic_results)}")
    with col2:
        st.metric("Interview Rate", f"{interview_yes/len(holistic_results)*100:.1f}%")
    with col3:
        avg_score_interview = (
            sum(r.overall_score for r in holistic_results if r.interview_decision) / interview_yes
            if interview_yes else 0
        )
        st.metric("Avg Score (Interview Yes)", f"{avg_score_interview:.2f}" if interview_yes else "N/A")

    st.markdown("---")

    # Score distribution chart
    st.subheader("Score Distribution")
    score_df = pd.DataFrame({'Score': overall_scores})
    st.bar_chart(score_df['Score'].value_counts().sort_index())

    # Innovation potential breakdown
    st.subheader("Innovation Potential Distribution")
    innovation_counts = {}
    for r in holistic_results:
        level = r.innovation_potential.level.capitalize()
        innovation_counts[level] = innovation_counts.get(level, 0) + 1

    innov_data = []
    for level in ['High', 'Medium', 'Low']:
        count = innovation_counts.get(level, 0)
        pct = count / len(holistic_results) * 100
        avg = (
            sum(r.overall_score for r in holistic_results if r.innovation_potential.level.capitalize() == level) / count
            if count else 0
        )
        innov_data.append({'Level': level, 'Count': count, 'Percentage': f"{pct:.1f}%", 'Avg Score': f"{avg:.2f}" if count else 'N/A'})
    st.dataframe(pd.DataFrame(innov_data), hide_index=True, use_container_width=True)

    # Program fit breakdown
    st.subheader("Program Fit Distribution")
    fit_counts = {}
    for r in holistic_results:
        level = r.program_fit.level.capitalize()
        fit_counts[level] = fit_counts.get(level, 0) + 1

    fit_data = []
    for level in ['Strong', 'Moderate', 'Weak', 'High', 'Medium', 'Low']:
        count = fit_counts.get(level, 0)
        if count == 0:
            continue
        pct = count / len(holistic_results) * 100
        avg = (
            sum(r.overall_score for r in holistic_results if r.program_fit.level.capitalize() == level) / count
            if count else 0
        )
        fit_data.append({'Level': level, 'Count': count, 'Percentage': f"{pct:.1f}%", 'Avg Score': f"{avg:.2f}"})
    if fit_data:
        st.dataframe(pd.DataFrame(fit_data), hide_index=True, use_container_width=True)

    # Interview decision by score band
    st.subheader("Interview Decision by Score Band")
    bands = [(0, 4, "0–4"), (4, 6, "4–6"), (6, 8, "6–8"), (8, 10.1, "8–10")]
    band_data = []
    for low, high, label in bands:
        group = [r for r in holistic_results if low <= r.overall_score < high]
        if group:
            yes = sum(1 for r in group if r.interview_decision)
            band_data.append({
                'Score Band': label,
                'Candidates': len(group),
                'Interview Yes': yes,
                'Interview Rate': f"{yes/len(group)*100:.1f}%"
            })
    if band_data:
        st.dataframe(pd.DataFrame(band_data), hide_index=True, use_container_width=True)


def holistic_recommendation_analysis(holistic_results, key_prefix="hol_rec_default"):
    """Recommendation and decision breakdown for holistic evaluations."""
    st.subheader("Holistic Recommendation Analysis")

    recommendations = {}
    for result in holistic_results:
        rec = result.recommendation if result.recommendation else "Unknown"
        if rec not in recommendations:
            recommendations[rec] = []
        recommendations[rec].append(result)

    st.markdown(f"**{len(recommendations)} unique recommendation types**")

    for i, (rec_type, candidates) in enumerate(
        sorted(recommendations.items(), key=lambda x: -len(x[1]))
    ):
        with st.expander(
            f"{rec_type} ({len(candidates)} candidates)",
            key=f"{key_prefix}_hol_rec_exp_{i}",
        ):
            if candidates:
                avg = sum(c.overall_score for c in candidates) / len(candidates)
                interview_yes = sum(1 for c in candidates if c.interview_decision)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{avg:.2f}")
                with col2:
                    st.metric("Interview Recommended", f"{interview_yes}/{len(candidates)}")
                with col3:
                    st.metric("Interview Rate", f"{interview_yes/len(candidates)*100:.1f}%")

                for c in sorted(candidates, key=lambda x: -x.overall_score)[:5]:
                    interview_icon = "✓" if c.interview_decision else "✗"
                    st.text(f"- {c.candidate.candidate_id}: {c.overall_score:.1f}/10  {interview_icon} Interview")

    st.markdown("---")
    st.subheader("Innovation Potential by Recommendation")

    innov_rec_data = []
    for result in holistic_results:
        innov_rec_data.append({
            'Candidate': result.candidate.candidate_id,
            'Score': result.overall_score,
            'Recommendation': result.recommendation or 'Unknown',
            'Innovation': result.innovation_potential.level.capitalize(),
            'Program Fit': result.program_fit.level.capitalize(),
            'Interview': 'Yes' if result.interview_decision else 'No'
        })

    df = pd.DataFrame(innov_rec_data).sort_values('Score', ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_expert_comparison_inner(all_results, expert_ratings, key_prefix: str):
    """Run AI vs expert comparison UI for a fixed expert_ratings list and criteria AI results."""
    analyzer = ExpertComparisonAnalyzer()
    analyzer.expert_ratings = expert_ratings
    analyzer.set_ai_results(all_results)

    threshold = st.slider(
        "Interview Threshold",
        1.0,
        10.0,
        6.0,
        key=f"{key_prefix}_interview_threshold",
    )

    if st.button("Run Comparison", key=f"{key_prefix}_run_expert_comparison"):
        metrics = analyzer.compare(interview_threshold=threshold)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matched", f"{metrics.matched_candidates}/{metrics.total_candidates}")
        with col2:
            st.metric("MAE", f"{metrics.overall_mae:.2f}")
        with col3:
            st.metric("Correlation", f"{metrics.overall_correlation:.2f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sensitivity", f"{metrics.sensitivity*100:.1f}%" if metrics.sensitivity else "N/A")
        with col2:
            st.metric("Specificity", f"{metrics.specificity*100:.1f}%" if metrics.specificity else "N/A")
        with col3:
            st.metric("Cohen's Kappa", f"{metrics.cohens_kappa:.2f}" if metrics.cohens_kappa else "N/A")

        if metrics.ai_bias:
            st.warning(f"Detected bias: {metrics.ai_bias}")


def expert_comparison_analysis(all_results):
    """AI vs Expert comparison (single scope; analysis dashboard uses batch-aware flow)."""
    st.subheader("AI vs Expert Comparison")
    st.info("Upload expert ratings to compare with AI evaluations.")

    expert_file = st.file_uploader(
        "Upload Expert Ratings",
        type=["xlsx", "xls", "csv"],
        help="Excel or CSV with expert ratings",
        key="standalone_expert_comparison_upload",
    )

    if expert_file:
        try:
            _tmp = tempfile.mkdtemp()
            _tmp_path = os.path.join(_tmp, expert_file.name)
            with open(_tmp_path, "wb") as f:
                f.write(expert_file.getbuffer())

            with st.spinner("Loading expert ratings..."):
                _loader = ExpertComparisonAnalyzer()
                expert_ratings = _loader.load_expert_ratings_from_excel(_tmp_path)

            st.success(f"Loaded {len(expert_ratings)} expert ratings")
            _render_expert_comparison_inner(all_results, expert_ratings, "standalone_expert_cmp")

        except Exception as e:
            st.error(f"Error: {e}")


def decision_comparison_analysis(criteria_results, holistic_results):
    """Compare AI predictions against actual admission decisions."""
    st.subheader("Decision Comparison")
    st.markdown("Compare AI evaluation predictions against actual admission decisions.")

    with st.expander("How to use", expanded=False):
        st.markdown("""
        **Upload a CSV file with columns:**
        - `candidate_id`: Must match the candidate IDs from evaluations
        - `admitted`: Yes/No — whether the candidate was actually admitted

        **Example:**
        ```
        candidate_id,admitted
        Tomberg_Spencer_app,yes
        Prado Larrea_Michaela_app,yes
        Nagesh_Nitish_app2,no
        ```

        **The tool will calculate:**
        - Sensitivity (true positive rate)
        - Specificity (true negative rate)
        - Cohen's Kappa (agreement statistic)
        - Confusion matrix
        - Per-dimension bias breakdown (holistic evaluations only)
        """)

    uploaded_file = st.file_uploader(
        "Upload decisions file",
        type=['csv', 'xlsx'],
        help="CSV with candidate_id and admitted columns"
    )

    if uploaded_file is None:
        st.info("Upload a decisions file to compare AI predictions against actual admit outcomes.")
        return

    try:
        if uploaded_file.name.endswith('.xlsx'):
            decisions_df = pd.read_excel(uploaded_file)
        else:
            decisions_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    decisions_df.columns = decisions_df.columns.str.lower().str.strip()

    if 'candidate_id' not in decisions_df.columns:
        st.error("File must have a `candidate_id` column")
        return
    if 'admitted' not in decisions_df.columns:
        st.error("File must have an `admitted` column (yes/no)")
        return

    def normalize_bool(val):
        if pd.isna(val):
            return None
        if isinstance(val, bool):
            return val
        return str(val).lower().strip() in ['yes', 'true', '1', 'y']

    decisions_df['admitted'] = decisions_df['admitted'].apply(normalize_bool)

    st.success(f"Loaded {len(decisions_df)} admission records")

    st.markdown("### Select Evaluation Data")
    data_source = st.radio(
        "Compare against:",
        ["Criteria-Based Evaluations", "Holistic Evaluations", "Both"],
        horizontal=True
    )

    threshold = st.slider(
        "Score threshold for positive AI prediction",
        min_value=1.0, max_value=10.0, value=6.0, step=0.5,
        help="Candidates scoring at or above this threshold are predicted as 'interview'"
    )

    if st.button("Run Comparison"):
        results_to_compare = []

        if data_source in ["Criteria-Based Evaluations", "Both"]:
            for result in criteria_results:
                results_to_compare.append({
                    'candidate_id': result.candidate.candidate_id,
                    'overall_score': result.overall_score,
                    'ai_prediction': result.overall_score >= threshold,
                    'source': 'criteria'
                })

        if data_source in ["Holistic Evaluations", "Both"]:
            for result in holistic_results:
                results_to_compare.append({
                    'candidate_id': result.candidate.candidate_id,
                    'overall_score': result.overall_score,
                    'ai_prediction': result.interview_decision,
                    'source': 'holistic'
                })

        if not results_to_compare:
            st.warning("No evaluation results found for selected data source.")
            return

        ai_df = pd.DataFrame(results_to_compare)
        merged = ai_df.merge(decisions_df, on='candidate_id', how='inner')

        if len(merged) == 0:
            st.error("No matching candidates found. Check that candidate IDs match.")
            return

        st.markdown(f"**Matched {len(merged)} candidates**")
        st.markdown("### Admit Decision Comparison")

        actual_admitted = merged['admitted'].values
        ai_predicted = merged['ai_prediction'].values

        valid_mask = [a is not None for a in actual_admitted]
        actual_admitted = [actual_admitted[i] for i in range(len(valid_mask)) if valid_mask[i]]
        ai_predicted = [ai_predicted[i] for i in range(len(valid_mask)) if valid_mask[i]]

        if len(actual_admitted) == 0:
            st.warning("No valid admit decisions to compare.")
            return

        tp = sum(1 for a, p in zip(actual_admitted, ai_predicted) if a and p)
        tn = sum(1 for a, p in zip(actual_admitted, ai_predicted) if not a and not p)
        fp = sum(1 for a, p in zip(actual_admitted, ai_predicted) if not a and p)
        fn = sum(1 for a, p in zip(actual_admitted, ai_predicted) if a and not p)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Confusion Matrix")
            cm_df = pd.DataFrame(
                [[tp, fn], [fp, tn]],
                columns=['AI: Yes', 'AI: No'],
                index=['Actual Admit: Yes', 'Actual Admit: No']
            )
            st.dataframe(cm_df, use_container_width=True)

        with col2:
            st.markdown("#### Key Metrics")
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            st.metric("Sensitivity (True Positive Rate)", f"{sensitivity:.1%}",
                      help=f"AI identified {sensitivity:.1%} of candidates who were actually admitted")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            st.metric("Specificity (True Negative Rate)", f"{specificity:.1%}",
                      help=f"AI correctly rejected {specificity:.1%} of candidates not admitted")
            accuracy = (tp + tn) / len(actual_admitted) if len(actual_admitted) > 0 else 0
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
            p_o = (tp + tn) / len(actual_admitted) if len(actual_admitted) > 0 else 0
            p_yes = ((tp + fn) / len(actual_admitted)) * ((tp + fp) / len(actual_admitted))
            p_no = ((fp + tn) / len(actual_admitted)) * ((fn + tn) / len(actual_admitted))
            p_e = p_yes + p_no
            kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0
            kappa_interpretation = "Poor" if kappa < 0.2 else "Fair" if kappa < 0.4 else "Moderate" if kappa < 0.6 else "Good" if kappa < 0.8 else "Excellent"
            st.metric("Cohen's Kappa", f"{kappa:.3f} ({kappa_interpretation})")

        st.markdown("### Disagreement Analysis")
        disagreements = merged[merged['ai_prediction'] != merged['admitted']]
        if len(disagreements) > 0:
            st.markdown(f"**{len(disagreements)} candidates where AI and actual admit decision disagree:**")
            false_positives = disagreements[disagreements['ai_prediction'] & ~disagreements['admitted']]
            false_negatives = disagreements[~disagreements['ai_prediction'] & disagreements['admitted']]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**False Positives ({len(false_positives)})** — AI recommended, not admitted:")
                for _, row in false_positives.head(10).iterrows():
                    st.text(f"- {row['candidate_id']}: AI score {row['overall_score']:.1f}")
            with col2:
                st.markdown(f"**False Negatives ({len(false_negatives)})** — Not recommended, was admitted:")
                for _, row in false_negatives.head(10).iterrows():
                    st.text(f"- {row['candidate_id']}: AI score {row['overall_score']:.1f}")
        else:
            st.success("Perfect agreement between AI predictions and actual admit decisions!")

        st.markdown("### Bias Analysis")
        ai_positive_rate = sum(ai_predicted) / len(ai_predicted) if len(ai_predicted) > 0 else 0
        human_positive_rate = sum(actual_admitted) / len(actual_admitted) if len(actual_admitted) > 0 else 0
        if ai_positive_rate > human_positive_rate + 0.1:
            st.warning(f"AI may be too lenient: AI recommends {ai_positive_rate:.1%} vs actual admit rate {human_positive_rate:.1%}")
        elif ai_positive_rate < human_positive_rate - 0.1:
            st.warning(f"AI may be too strict: AI recommends {ai_positive_rate:.1%} vs actual admit rate {human_positive_rate:.1%}")
        else:
            st.success(f"AI and actual admit rates are similar: AI {ai_positive_rate:.1%}, Actual {human_positive_rate:.1%}")

        # Per-dimension systematic bias (holistic only)
        holistic_in_merged = merged[merged['source'] == 'holistic'] if 'source' in merged.columns else pd.DataFrame()

        if not holistic_in_merged.empty:
            _, holistic_results = get_user_results_as_objects(user)
            holistic_obj_map = {r.candidate.candidate_id: r for r in holistic_results}

            rows_dim = []
            for _, row in holistic_in_merged.iterrows():
                obj = holistic_obj_map.get(row['candidate_id'])
                if obj is None:
                    continue
                outcome = "Correct" if row['ai_prediction'] == row['admitted'] else (
                    "False Positive" if row['ai_prediction'] else "False Negative"
                )
                rows_dim.append({
                    'candidate_id': row['candidate_id'],
                    'overall_score': obj.overall_score,
                    'program_fit': obj.program_fit.level.lower(),
                    'program_fit_confidence': obj.program_fit.confidence.lower(),
                    'innovation': obj.innovation_potential.level.lower(),
                    'innovation_confidence': obj.innovation_potential.confidence.lower(),
                    'red_flags': len(obj.red_flags) if isinstance(obj.red_flags, list) else 0,
                    'outcome': outcome,
                    'ai_said': row['ai_prediction'],
                    'actual': row['admitted'],
                })

            if rows_dim:
                dim_df = pd.DataFrame(rows_dim)
                fp_df = dim_df[dim_df['outcome'] == 'False Positive']
                fn_df = dim_df[dim_df['outcome'] == 'False Negative']
                correct_df = dim_df[dim_df['outcome'] == 'Correct']

                st.markdown("#### Per-Dimension Breakdown")
                st.caption(
                    "Compares average dimension signals between correctly predicted, "
                    "false-positive, and false-negative candidates."
                )

                def _level_to_num(val):
                    return {'high': 3, 'strong': 3, 'medium': 2, 'moderate': 2,
                            'low': 1, 'weak': 1}.get(str(val), 2)

                for label, subset in [("False Positives (AI over-predicted)", fp_df),
                                       ("False Negatives (AI under-predicted)", fn_df),
                                       ("Correct predictions", correct_df)]:
                    if subset.empty:
                        continue
                    with st.expander(f"{label} — {len(subset)} candidate(s)"):
                        avg_score = subset['overall_score'].mean()
                        avg_fit = subset['program_fit'].map(_level_to_num).mean()
                        avg_inn = subset['innovation'].map(_level_to_num).mean()
                        avg_rf = subset['red_flags'].mean()

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Avg score", f"{avg_score:.1f}/10")
                        c2.metric("Avg program fit", f"{avg_fit:.2f}/3")
                        c3.metric("Avg innovation", f"{avg_inn:.2f}/3")
                        c4.metric("Avg red flags", f"{avg_rf:.1f}")

                        fit_counts = subset['program_fit'].value_counts().to_dict()
                        inn_counts = subset['innovation'].value_counts().to_dict()
                        st.markdown(
                            f"Program fit: {fit_counts}  |  Innovation: {inn_counts}"
                        )

                st.markdown("#### Score Distribution by Outcome")
                score_summary = (
                    dim_df.groupby('outcome')['overall_score']
                    .agg(['mean', 'min', 'max', 'count'])
                    .rename(columns={'mean': 'Mean', 'min': 'Min', 'max': 'Max', 'count': 'N'})
                    .round(2)
                )
                st.dataframe(score_summary, use_container_width=True)


def admit_pattern_analysis_page(user: dict, api_key: str):
    """Admit Pattern Analysis - discover what distinguishes admitted from rejected candidates."""
    st.title("Admit Pattern Analysis")

    page_tab1, page_tab2 = st.tabs(["Pattern Analysis", "Decision Comparison"])

    with page_tab1:
        _admit_pattern_tab(user, api_key)

    with page_tab2:
        criteria_results, holistic_results = get_user_results_as_objects(user)
        st.markdown("Compare AI evaluation predictions against actual interview and admission decisions.")
        decision_comparison_analysis(criteria_results, holistic_results)


def _admit_pattern_tab(user: dict, api_key: str):
    """Inner content for the Pattern Analysis tab."""
    st.markdown("Upload candidate applications with admit/reject labels to discover distinguishing patterns.")

    if 'admit_analysis_progress' in st.session_state:
        progress = st.session_state['admit_analysis_progress']
        col1, col2 = st.columns([3, 1])
        with col1:
            if progress.get('completed'):
                st.success(f"Previous analysis completed: {len(progress.get('candidate_summaries', []))} candidates evaluated")
            else:
                st.info(f"Analysis in progress: {progress.get('current_index', 0)} / {progress.get('total_candidates', '?')} candidates evaluated")
        with col2:
            if st.button("Reset / Start New", use_container_width=True):
                for key in ['admit_analysis_progress', 'admit_analysis_file_paths', 'last_pattern_analysis', 'admit_mapping', 'admit_analysis_holistic']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        st.markdown("---")

    with st.expander("How to use", expanded=True):
        st.markdown("""
        **Step 1: Upload candidate application files**
        - Upload PDF files containing candidate applications
        - Each file should be one candidate's complete application

        **Step 2: Upload admit mapping CSV**
        - Create a CSV file with two columns: `filename` and `admit_status`
        - Example:
        ```
        filename,admit_status
        candidate_001_application.pdf,yes
        candidate_002_application.pdf,no
        ```

        **Step 3: Run analysis**
        - The system will evaluate each candidate, then analyze patterns.

        **Note**: This process may take significant time (~1-2 minutes per candidate).
        """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Candidate Applications")
        uploaded_files = st.file_uploader(
            "Upload candidate PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload all candidate application PDFs",
            key="admit_pattern_files"
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded")
            with st.expander("View files"):
                for f in uploaded_files:
                    st.text(f"- {f.name}")

    with col2:
        st.subheader("2. Upload Admit Mapping CSV")
        mapping_file = st.file_uploader(
            "Upload admit mapping CSV",
            type=['csv'],
            help="CSV with filename and admit_status columns",
            key="admit_mapping_file"
        )
        if mapping_file:
            try:
                mapping_df = pd.read_csv(mapping_file)
                mapping_df.columns = mapping_df.columns.str.lower().str.strip()

                if 'filename' not in mapping_df.columns:
                    st.error("CSV must have a 'filename' column")
                elif 'admit_status' not in mapping_df.columns:
                    st.error("CSV must have an 'admit_status' column")
                else:
                    def normalize_admit(val):
                        if pd.isna(val):
                            return None
                        return str(val).lower().strip() in ['yes', 'true', '1', 'y', 'admitted', 'admit']

                    mapping_df['admit_status'] = mapping_df['admit_status'].apply(normalize_admit)
                    admitted_count = int(mapping_df['admit_status'].sum())
                    rejected_count = len(mapping_df) - admitted_count
                    st.success(f"Loaded {len(mapping_df)} mappings")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Admitted", admitted_count)
                    with col_b:
                        st.metric("Rejected", rejected_count)
                    st.session_state['admit_mapping'] = mapping_df
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")
    st.subheader("3. Analysis Options")

    eval_mode = st.radio(
        "Evaluation mode for candidates",
        ["Holistic (faster, recommended)", "Criteria-Based (detailed scores)"],
        horizontal=True,
        help="Holistic mode is faster and provides overall fit assessment."
    )
    use_holistic = eval_mode == "Holistic (faster, recommended)"

    can_proceed = (uploaded_files and mapping_file and 'admit_mapping' in st.session_state)

    analysis_in_progress = (
        'admit_analysis_progress' in st.session_state and
        not st.session_state['admit_analysis_progress'].get('completed', False)
    )

    if analysis_in_progress:
        st.info("Continuing analysis...")
        if 'admit_analysis_holistic' not in st.session_state:
            st.session_state['admit_analysis_holistic'] = use_holistic
        run_admit_pattern_analysis(uploaded_files, st.session_state['admit_analysis_holistic'], user, api_key)
    elif st.button("Run Admit Pattern Analysis", disabled=not can_proceed, use_container_width=True):
        st.session_state['admit_analysis_holistic'] = use_holistic
        run_admit_pattern_analysis(uploaded_files, use_holistic, user, api_key)

    st.markdown("---")
    st.subheader("Previous Analysis Results")
    display_admit_pattern_results_cloud()


def run_admit_pattern_analysis(uploaded_files, use_holistic: bool, user: dict, api_key: str):
    """Run the full admit pattern analysis pipeline (cloud version)."""
    mapping_df = st.session_state.get('admit_mapping')
    if mapping_df is None:
        st.error("No admit mapping found")
        return

    admit_map = dict(zip(mapping_df['filename'], mapping_df['admit_status']))

    # Save files to temp dir if not already done
    if 'admit_analysis_file_paths' not in st.session_state:
        file_paths = {}
        if uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                save_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(save_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_paths[uploaded_file.name] = save_path
        st.session_state['admit_analysis_file_paths'] = file_paths
    else:
        file_paths = st.session_state['admit_analysis_file_paths']

    matched_candidates = []
    unmatched_files = []
    for filename, filepath in file_paths.items():
        if filename in admit_map:
            matched_candidates.append({
                'filename': filename,
                'filepath': filepath,
                'admit_status': admit_map[filename]
            })
        else:
            unmatched_files.append(filename)

    if unmatched_files:
        st.warning(f"{len(unmatched_files)} files not found in mapping: {', '.join(unmatched_files[:5])}")

    if not matched_candidates:
        st.error("No files matched the admit mapping. Check that filenames in CSV match uploaded files exactly.")
        return

    if 'admit_analysis_progress' not in st.session_state:
        st.session_state['admit_analysis_progress'] = {
            'current_index': 0,
            'total_candidates': len(matched_candidates),
            'candidate_summaries': [],
            'errors': [],
            'completed': False,
            'use_holistic': use_holistic,
        }

    progress = st.session_state['admit_analysis_progress']

    if progress['completed']:
        st.success("Analysis already completed! See results below.")
        if 'last_pattern_analysis' in st.session_state:
            _display_pattern_analysis_from_dict(st.session_state['last_pattern_analysis'])
        return

    st.info(f"Processing {len(matched_candidates)} matched candidates...")

    evaluator = get_evaluator(api_key)
    db = get_database()

    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(progress['current_index'] / len(matched_candidates))
        status_text = st.empty()
        error_container = st.empty()
        skipped_container = st.empty()
        skipped_count = 0

        start_index = progress['current_index']

        for i in range(start_index, len(matched_candidates)):
            candidate = matched_candidates[i]
            candidate_id = Path(candidate['filename']).stem

            status_text.text(f"Evaluating {i+1}/{len(matched_candidates)}: {candidate_id}")

            try:
                if use_holistic:
                    result = evaluator.evaluate_candidate_holistic(
                        candidate_id=candidate_id,
                        material_paths=[candidate['filepath']]
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(result_dict["candidate"]["evaluation_date"])

                    db.save_evaluation(
                        user_id=user["id"],
                        candidate_id=candidate_id,
                        evaluation_type="holistic",
                        result=result_dict
                    )

                    summary = {
                        'candidate_id': candidate_id,
                        'admit_status': candidate['admit_status'],
                        'overall_score': result.overall_score,
                        'recommendation': result.recommendation,
                        'innovation_potential': result.innovation_potential.level,
                        'program_fit': result.program_fit.level,
                        'interview_decision': result.interview_decision,
                        'strengths': result.program_fit.strengths_for_program,
                        'weaknesses': result.program_fit.concerns,
                        'red_flags': [rf.flag if hasattr(rf, 'flag') else str(rf) for rf in result.red_flags[:3]],
                        'notable_qualities': [q.quality for q in result.notable_qualities[:5]]
                    }
                else:
                    result = evaluator.evaluate_candidate(
                        candidate_id=candidate_id,
                        material_paths=[candidate['filepath']]
                    )
                    result_dict = result.model_dump()
                    result_dict["candidate"]["evaluation_date"] = str(result_dict["candidate"]["evaluation_date"])
                    for score in result_dict.get("scores", []):
                        if "criterion" in score:
                            crit = score["criterion"]
                            score["criterion"] = crit.value if hasattr(crit, 'value') else str(crit)

                    db.save_evaluation(
                        user_id=user["id"],
                        candidate_id=candidate_id,
                        evaluation_type="criteria",
                        result=result_dict
                    )

                    summary = {
                        'candidate_id': candidate_id,
                        'admit_status': candidate['admit_status'],
                        'overall_score': result.overall_score,
                        'recommendation': result.recommendation,
                        'scores': {score.criterion.value: score.score for score in result.scores},
                        'strengths': result.strengths,
                        'weaknesses': result.areas_for_development
                    }

                progress['candidate_summaries'].append(summary)

            except Exception as e:
                error_msg = f"Failed to evaluate {candidate_id}: {e}"
                progress['errors'].append(error_msg)
                error_container.warning(error_msg)

            progress['current_index'] = i + 1
            progress_bar.progress((i + 1) / len(matched_candidates))
            st.session_state['admit_analysis_progress'] = progress

            if i < len(matched_candidates) - 1:
                time.sleep(0.5)
                st.rerun()

    candidate_summaries = progress['candidate_summaries']
    status_text.text(f"Evaluated {len(candidate_summaries)}/{len(matched_candidates)} candidates")

    if len(candidate_summaries) < 2:
        st.error("Need at least 2 successfully evaluated candidates for pattern analysis")
        return

    st.info("Running pattern analysis...")

    try:
        config = Config(api=APIConfig(anthropic_api_key=api_key))
        analyzer = AdmitPatternAnalyzer(
            api_key=api_key,
            model=config.api.model,
            max_tokens=config.api.max_tokens
        )

        basic_stats = analyzer.calculate_basic_statistics(candidate_summaries)

        st.subheader("Basic Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", basic_stats['total_candidates'])
        with col2:
            st.metric("Admitted", basic_stats['admitted_count'])
        with col3:
            st.metric("Rejected", basic_stats['rejected_count'])
        with col4:
            rate = basic_stats.get('admission_rate', 0) * 100
            st.metric("Admission Rate", f"{rate:.1f}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Admitted Mean Score", f"{basic_stats.get('admitted_mean_score', 0):.2f}")
        with col2:
            st.metric("Rejected Mean Score", f"{basic_stats.get('rejected_mean_score', 0):.2f}")
        with col3:
            st.metric("Score Difference", f"{basic_stats.get('score_difference', 0):.2f}")

        if basic_stats.get('criterion_differences'):
            st.markdown("### Most Discriminating Criteria")
            for crit in basic_stats['criterion_differences'][:5]:
                name = crit['criterion'].replace('_', ' ').title()
                st.markdown(f"- **{name}**: {crit['difference']:.1f} point gap (Admitted: {crit['admitted_mean']:.1f}, Rejected: {crit['rejected_mean']:.1f})")

        st.info("Running detailed pattern analysis with Claude...")
        pattern_result = analyzer.analyze_patterns(candidate_summaries)

        progress['completed'] = True
        st.session_state['admit_analysis_progress'] = progress
        st.session_state['last_pattern_analysis'] = pattern_result.model_dump()

        st.success("Analysis complete!")
        display_pattern_analysis_result(pattern_result)

    except Exception as e:
        st.error(f"Pattern analysis failed: {e}")
        st.code(traceback.format_exc())


def display_admit_pattern_results_cloud():
    """Display previous admit pattern analysis results from session state."""
    if 'last_pattern_analysis' not in st.session_state:
        st.info("No previous pattern analyses found in this session. Run an analysis to see results here.")
        return

    _display_pattern_analysis_from_dict(st.session_state['last_pattern_analysis'])


def display_pattern_analysis_result(result: AdmitPatternAnalysisResult):
    """Display a pattern analysis result."""
    _display_pattern_analysis_from_dict(result.model_dump())


def _display_pattern_analysis_from_dict(data):
    """Display pattern analysis from a dict or AdmitPatternAnalysisResult object."""
    if hasattr(data, 'model_dump'):
        data = data.model_dump()

    st.markdown("---")
    st.header("Pattern Analysis Results")

    st.subheader("Executive Summary")
    st.markdown(data.get('executive_summary', 'No summary available'))

    st.subheader("Score Comparison")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyzed", data.get('total_candidates', 0))
    with col2:
        st.metric("Admitted", data.get('admitted_count', 0))
    with col3:
        st.metric("Rejected", data.get('rejected_count', 0))
    with col4:
        st.metric("Score Gap", f"{data.get('score_difference', 0):.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Admitted Mean Score", f"{data.get('admitted_mean_score', 0):.2f}/10")
    with col2:
        st.metric("Rejected Mean Score", f"{data.get('rejected_mean_score', 0):.2f}/10")

    key_patterns = data.get('key_patterns', [])
    if key_patterns:
        st.subheader("Key Distinguishing Patterns")
        for category in key_patterns:
            if hasattr(category, 'importance'):
                importance = category.importance
                category_name = category.category_name
                description = category.description
                patterns = category.patterns or []
            else:
                importance = category.get('importance', 'medium')
                category_name = category.get('category_name', 'Unknown')
                description = category.get('description', '')
                patterns = category.get('patterns', [])

            importance_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(importance, '⚪')
            with st.expander(f"{importance_color} {category_name} ({importance.upper()} importance)"):
                st.markdown(description)
                for pattern in patterns:
                    if hasattr(pattern, 'pattern'):
                        p_text = pattern.pattern
                        admitted_ex = pattern.admitted_examples or []
                        rejected_ex = pattern.rejected_examples or []
                    else:
                        p_text = pattern.get('pattern', '')
                        admitted_ex = pattern.get('admitted_examples', [])
                        rejected_ex = pattern.get('rejected_examples', [])
                    st.markdown(f"**Pattern:** {p_text}")
                    if admitted_ex:
                        st.markdown("*Admitted examples:*")
                        for ex in admitted_ex[:3]:
                            st.markdown(f"  - {ex}")
                    if rejected_ex:
                        st.markdown("*Rejected counter-examples:*")
                        for ex in rejected_ex[:3]:
                            st.markdown(f"  - {ex}")
                    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Common Strengths (Admitted)")
        for strength in data.get('admitted_strengths', []):
            st.markdown(f"✓ {strength}")
        if not data.get('admitted_strengths'):
            st.info("No common strengths identified")
    with col2:
        st.subheader("Common Weaknesses (Rejected)")
        for weakness in data.get('rejected_weaknesses', []):
            st.markdown(f"✗ {weakness}")
        if not data.get('rejected_weaknesses'):
            st.info("No common weaknesses identified")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Surprising Admits")
        surprising_admits = data.get('surprising_admits', [])
        for case in surprising_admits:
            c = case if isinstance(case, dict) else (case.model_dump() if hasattr(case, 'model_dump') else {})
            st.warning(f"**{c.get('candidate_id', 'Unknown')}** (Score: {c.get('score', 'N/A')})")
            st.caption(c.get('reason', ''))
            if c.get('possible_explanation'):
                st.caption(f"Possible explanation: {c.get('possible_explanation')}")
        if not surprising_admits:
            st.info("No surprising admits identified")
    with col2:
        st.subheader("Surprising Rejects")
        surprising_rejects = data.get('surprising_rejects', [])
        for case in surprising_rejects:
            c = case if isinstance(case, dict) else (case.model_dump() if hasattr(case, 'model_dump') else {})
            st.warning(f"**{c.get('candidate_id', 'Unknown')}** (Score: {c.get('score', 'N/A')})")
            st.caption(c.get('reason', ''))
            if c.get('possible_explanation'):
                st.caption(f"Possible explanation: {c.get('possible_explanation')}")
        if not surprising_rejects:
            st.info("No surprising rejects identified")

    metadata = data.get('metadata', {})
    if isinstance(metadata, dict) and metadata.get('predictive_factors'):
        st.subheader("Predictive Factors")
        for factor in metadata['predictive_factors']:
            f = factor if isinstance(factor, dict) else {}
            strength_icon = {'strong': '💪', 'moderate': '👍', 'weak': '👌'}.get(f.get('strength', ''), '•')
            st.markdown(f"{strength_icon} **{f.get('factor', '')}**: {f.get('direction', '')}")
            st.caption(f.get('evidence', ''))

    if data.get('methodology_notes'):
        with st.expander("Methodology Notes"):
            st.markdown(data['methodology_notes'])

    with st.expander("View All Candidate Data"):
        candidate_summaries = data.get('candidate_summaries', [])
        if candidate_summaries:
            display_data = []
            for c in candidate_summaries:
                row = {
                    'Candidate ID': c.get('candidate_id', ''),
                    'Admit Status': 'Admitted' if c.get('admit_status') else 'Rejected',
                    'Score': c.get('overall_score', 0),
                    'Recommendation': c.get('recommendation', '')
                }
                display_data.append(row)
            df = pd.DataFrame(display_data).sort_values('Score', ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.download_button(
        label="Download Analysis (JSON)",
        data=json.dumps(data, indent=2, default=str),
        file_name=f"admit_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


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
    
    st.markdown("### Score Comparison Chart")
    chart_df = df[['criteria_score', 'holistic_score']].copy()
    chart_df.columns = ['Criteria-Based', 'Holistic']
    st.scatter_chart(chart_df)

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
                    if hasattr(flag, 'flag'):
                        severity = getattr(flag, 'severity', 'medium').lower()
                        severity_icons = {'high': '🔴', 'medium': '🟡', 'low': '🔵'}
                        icon = severity_icons.get(severity, '🟡')
                        st.markdown(f"{icon} {flag.flag}  \n<small>Severity: {severity.capitalize()}</small>", unsafe_allow_html=True)
                    else:
                        st.warning(str(flag))
            
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
