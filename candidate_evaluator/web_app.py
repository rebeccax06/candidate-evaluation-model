"""Modern Streamlit web interface for candidate evaluator with background processing."""

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
    AdmitPatternAnalysisResult,
    InterviewSelectionResult,
    parse_role_specific_assessment,
)
from candidate_evaluator.core.pattern_analyzer import AdmitPatternAnalyzer
from candidate_evaluator.utils.role_results import (
    ROLE_ORDER,
    ROLE_LABELS,
    group_results_by_role,
    sort_results_by_role_score,
    build_role_ranking_row_from_result,
    count_role_evaluations_from_results,
)
from candidate_evaluator.utils.config import load_config, get_default_config
from candidate_evaluator.exporters import (
    JSONExporter,
    MarkdownExporter,
    HTMLExporter,
    CSVExporter
)
from candidate_evaluator.job_manager import JobManager
from candidate_evaluator.background_worker import JobStatus
from candidate_evaluator.prompt_manager import PromptManager

# No custom CSS - using Streamlit defaults for reliability
CUSTOM_CSS = ""


def load_result_from_json(json_path: Path) -> EvaluationResult:
    """Load an EvaluationResult from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    candidate = CandidateProfile(
        candidate_id=data['candidate']['candidate_id'],
        name=data['candidate'].get('name'),
        materials=data['candidate'].get('materials', []),
        evaluation_date=datetime.fromisoformat(data['candidate']['evaluation_date'])
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

        scores.append(CriterionScore(
            criterion=EvaluationCriterion(score_data['criterion']),
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


def load_holistic_result_from_json(json_path: Path) -> HolisticEvaluationResult:
    """Load a HolisticEvaluationResult from a JSON file."""
    from candidate_evaluator.core.models import (
        InnovationPotential, ProgramFit, NotableQuality
    )

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    candidate = CandidateProfile(
        candidate_id=data['candidate']['candidate_id'],
        name=data['candidate'].get('name'),
        materials=data['candidate'].get('materials', []),
        evaluation_date=datetime.fromisoformat(data['candidate']['evaluation_date'])
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
        role=data.get('role'),
        role_specific_assessment=parse_role_specific_assessment(
            data.get('role_specific_assessment')
        ),
        metadata=data.get('metadata', {})
    )


def load_all_results_from_disk(results_dir: Path) -> list:
    """Load all evaluation results from JSON files in the results directory."""
    results = []
    if not results_dir.exists():
        return results

    for json_file in results_dir.glob("*_evaluation.json"):
        # Skip holistic evaluations
        if "_holistic_evaluation.json" in str(json_file):
            continue
        try:
            result = load_result_from_json(json_file)
            results.append(result)
        except Exception:
            pass

    return results


def load_all_holistic_results_from_disk(results_dir: Path) -> list:
    """Load all holistic evaluation results from JSON files in the results directory."""
    results = []
    if not results_dir.exists():
        return results

    for json_file in results_dir.glob("*_holistic_evaluation.json"):
        try:
            result = load_holistic_result_from_json(json_file)
            results.append(result)
        except Exception:
            pass

    return results


# Page config
st.set_page_config(
    page_title="Candidate Evaluator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
# Using Streamlit default theme


def init_session_state():
    """Initialize session state variables"""
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'job_manager' not in st.session_state:
        st.session_state.job_manager = JobManager()


def load_configuration():
    """Load configuration"""
    if st.session_state.config is None:
        try:
            config = load_config()
        except Exception:
            try:
                config = get_default_config()
            except Exception as e:
                st.error(f"Error loading configuration: {e}")
                st.info("Please set ANTHROPIC_API_KEY environment variable or create config.yaml")
                return None

        st.session_state.config = config
        st.session_state.evaluator = CandidateEvaluator(config)

    return st.session_state.config


_SELECTION_RESULT_FILE = Path("./results/interview_selection_result.json")


def _save_selection_result(result: InterviewSelectionResult) -> None:
    """Persist an InterviewSelectionResult to disk as JSON."""
    _SELECTION_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_SELECTION_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)


def _load_selection_result() -> InterviewSelectionResult | None:
    """Load the last saved InterviewSelectionResult from disk, or None if absent."""
    if not _SELECTION_RESULT_FILE.exists():
        return None
    try:
        with open(_SELECTION_RESULT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return InterviewSelectionResult(**data)
    except Exception:
        return None


def interview_selection_page():
    """Two-phase, context-window-efficient interview-selection page."""
    st.title("Interview Selection")
    st.markdown(
        "Select the top *N* candidates to interview from a pool of holistic evaluations. "
        "The system performs a compact first-pass ranking across all candidates, then a deeper "
        "final comparison restricted to the top pool."
    )

    output_dir = Path("./results")
    holistic_results = load_all_holistic_results_from_disk(output_dir)

    # Restore last result from disk into session state on first load
    if "interview_selection_result" not in st.session_state:
        st.session_state["interview_selection_result"] = _load_selection_result()

    if not holistic_results:
        st.warning(
            "No holistic evaluations found in `./results/`. "
            "Run holistic evaluations first (New Evaluation → Holistic mode, or Batch Jobs → Holistic)."
        )
        return

    st.info(f"Found **{len(holistic_results)}** holistic evaluation(s) available.")

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
        evaluator = st.session_state.get("evaluator")
        if evaluator is None:
            st.error("Evaluator not initialised — check your API key in Settings.")
            return

        with st.spinner("Running two-phase selection …"):
            try:
                result: InterviewSelectionResult = evaluator.select_interviews_holistic(
                    evaluations=holistic_results,
                    n_interviews=int(n_interviews),
                    pool_multiplier=float(pool_multiplier),
                )
                st.session_state["interview_selection_result"] = result
                _save_selection_result(result)
            except Exception as exc:
                st.error(f"Selection failed: {exc}")
                return

    # ------------------------------------------------------------------ #
    # Display results (persists across reloads via disk + session_state)    #
    # ------------------------------------------------------------------ #
    result: InterviewSelectionResult = st.session_state.get("interview_selection_result")
    if result is None:
        return

    col_hdr, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("Clear result", use_container_width=True):
            st.session_state["interview_selection_result"] = None
            if _SELECTION_RESULT_FILE.exists():
                _SELECTION_RESULT_FILE.unlink()
            st.rerun()

    st.success(
        f"Selected **{len(result.selected_candidate_ids)}** candidate(s) for interview "
        f"from a pool of {len(result.pool_candidate_ids)} finalists "
        f"(total evaluated: {result.metadata.get('n_total_candidates', '?')})."
    )

    # Final selections
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
                "Red Flags": ev.metadata.get("red_flags_count", len(ev.red_flags) if ev else 0),
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

    # Timing metadata
    elapsed = result.metadata.get("processing_time_seconds")
    if elapsed:
        st.caption(f"Completed in {elapsed:.1f}s using {result.metadata.get('model', 'unknown model')}.")


def main():
    """Main application"""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("## Candidate Evaluator")
        st.markdown("---")

        page_options = ["Dashboard", "New Evaluation", "Batch Jobs", "Results", "Interview Selection", "Analysis", "Admit Patterns", "Research", "Settings"]

        page = st.radio(
            "Navigation",
            page_options,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats
        job_manager = st.session_state.job_manager
        active_jobs = len(job_manager.get_active_jobs())

        if active_jobs > 0:
            st.markdown(f"**Active Jobs:** {active_jobs}")
            st.caption("Go to Batch Jobs to view")

        st.markdown("---")
        st.caption("AI-powered candidate evaluation. Evaluate candidates against 11 research-backed criteria.")

    # Load config
    config = load_configuration()
    if config is None:
        st.stop()

    # Route pages
    if page == "Dashboard":
        dashboard_page()
    elif page == "New Evaluation":
        new_evaluation_page()
    elif page == "Batch Jobs":
        batch_jobs_page()
    elif page == "Results":
        results_page()
    elif page == "Interview Selection":
        interview_selection_page()
    elif page == "Analysis":
        analysis_page()
    elif page == "Admit Patterns":
        admit_pattern_analysis_page()
    elif page == "Research":
        research_page()
    elif page == "Settings":
        settings_page()


def dashboard_page():
    """Dashboard overview page."""
    st.title("Dashboard")

    output_dir = Path("./results")
    all_results = load_all_results_from_disk(output_dir)
    job_manager = st.session_state.job_manager

    # Stats row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Evaluations", len(all_results))

    with col2:
        active_jobs = len(job_manager.get_active_jobs())
        st.metric("Active Jobs", active_jobs)

    with col3:
        if all_results:
            avg_score = sum(r.overall_score for r in all_results) / len(all_results)
            st.metric("Avg Score", f"{avg_score:.1f}/10")
        else:
            st.metric("Avg Score", "N/A")

    with col4:
        recent_jobs = job_manager.list_jobs(limit=10)
        completed = len([j for j in recent_jobs if j.get("status") == JobStatus.COMPLETED])
        st.metric("Recent Completed", completed)

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recent Evaluations")

        if all_results:
            recent = sorted(all_results, key=lambda r: r.candidate.evaluation_date, reverse=True)[:5]

            for result in recent:
                with st.container():
                    rcol1, rcol2, rcol3 = st.columns([3, 1, 1])
                    with rcol1:
                        st.markdown(f"**{result.candidate.candidate_id}**")
                        st.caption(result.candidate.evaluation_date.strftime("%Y-%m-%d %H:%M"))
                    with rcol2:
                        st.markdown(f"**{result.overall_score:.1f}**/10")
                    with rcol3:
                        st.caption(result.recommendation)
                    st.markdown("---")
        else:
            st.info("No evaluations yet. Select 'New Evaluation' from the sidebar to get started.")

    with col2:
        st.subheader("Active Jobs")

        active_jobs = job_manager.get_active_jobs()

        if active_jobs:
            for job in active_jobs[:3]:
                progress = job.get("progress", {})
                completed = progress.get("completed", 0)
                total = progress.get("total", 1)
                pct = (completed / total * 100) if total > 0 else 0

                st.markdown(f"**{job.get('job_name', job['job_id'][:20])}**")
                st.progress(pct / 100)
                st.caption(f"{completed}/{total} candidates")
                st.markdown("---")
        else:
            st.info("No active jobs")

        st.subheader("Quick Actions")
        st.caption("Use sidebar to navigate to New Evaluation")


ROLE_OPTIONS = {
    "Clinician": "clinician",
    "Engineer / Tech": "engineer",
    "PhD": "phd",
}


def new_evaluation_page():
    """New evaluation page - single or batch."""
    st.title("New Evaluation")

    tab1, tab2, tab3 = st.tabs(["Single Candidate", "Batch Upload", "Role-Specific Evaluation"])

    with tab1:
        single_evaluation_form()

    with tab2:
        batch_evaluation_form()

    with tab3:
        role_specific_evaluation_form()


def single_evaluation_form(role: str | None = None, key_prefix: str = ""):
    """Single candidate evaluation form."""
    title = "### Evaluate a Single Candidate"
    if role:
        role_label = next((k for k, v in ROLE_OPTIONS.items() if v == role), role)
        title = f"### Evaluate a Single {role_label} Candidate"
    st.markdown(title)

    st.markdown("#### Evaluation Mode")
    eval_mode = st.radio(
        "Select evaluation approach",
        ["Criteria-Based (11 criteria)", "Holistic (program fit)"],
        horizontal=True,
        help="Criteria-Based uses 11 predefined criteria. Holistic evaluates overall program fit without specific criteria.",
        key=f"{key_prefix}single_eval_mode"
    )
    is_holistic = eval_mode == "Holistic (program fit)"

    if role:
        st.caption("Role-specific guidance is appended to the system prompt and works with any evaluation mode.")
    if is_holistic:
        st.info("Holistic mode evaluates candidates based on overall program fit, innovation potential, and notable qualities without using predefined criteria.")

    col1, col2 = st.columns([2, 1])

    with col1:
        candidate_id = st.text_input(
            "Candidate ID",
            placeholder="e.g., CAND001",
            help="Unique identifier for the candidate",
            key=f"{key_prefix}single_candidate_id"
        )

        candidate_name = st.text_input(
            "Name (optional)",
            placeholder="e.g., John Doe",
            key=f"{key_prefix}single_candidate_name"
        )

        uploaded_files = st.file_uploader(
            "Upload Application Materials",
            type=['pdf', 'docx', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload resume, cover letter, writing samples, etc.",
            key=f"{key_prefix}single_uploader"
        )

    with col2:
        st.markdown("#### Supported Formats")
        st.markdown("""
        - PDF documents
        - Word documents (.docx)
        - Plain text (.txt)
        - Markdown (.md)
        """)

        st.markdown("#### Tips")
        st.markdown("""
        - Include all relevant materials
        - PDFs with interview responses work best
        - More context = better evaluation
        """)

    if st.button("Evaluate Candidate", disabled=not (candidate_id and uploaded_files), key=f"{key_prefix}single_eval_btn"):
        with st.spinner("Evaluating candidate... This may take a minute."):
            try:
                temp_dir = tempfile.mkdtemp()
                material_paths = []

                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    material_paths.append(temp_path)

                evaluator = st.session_state.evaluator
                output_dir = Path("./results")
                output_dir.mkdir(exist_ok=True)

                if is_holistic:
                    result = evaluator.evaluate_candidate_holistic(
                        candidate_id=candidate_id,
                        material_paths=material_paths,
                        candidate_name=candidate_name or None,
                        role=role
                    )
                    json_path = output_dir / f"{candidate_id}_holistic_evaluation.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result.model_dump(), f, indent=2, default=str)
                    st.success("Holistic evaluation completed!")
                    display_holistic_evaluation_result(result)
                else:
                    result = evaluator.evaluate_candidate(
                        candidate_id=candidate_id,
                        material_paths=material_paths,
                        candidate_name=candidate_name or None,
                        role=role
                    )
                    json_path = output_dir / f"{candidate_id}_evaluation.json"
                    JSONExporter.export_evaluation(result, json_path)
                    st.session_state.evaluation_results.append(result)
                    st.success("Evaluation completed!")
                    display_evaluation_result(result)

            except Exception as e:
                st.error(f"Error during evaluation: {e}")


def batch_evaluation_form(role: str | None = None, key_prefix: str = ""):
    """Batch evaluation form with background processing."""
    title = "### Batch Evaluation"
    if role:
        role_label = next((k for k, v in ROLE_OPTIONS.items() if v == role), role)
        title = f"### Batch {role_label} Evaluation"
    st.markdown(title)
    st.markdown("Upload multiple PDF files to evaluate in the background. You can navigate away and the evaluation will continue running.")

    st.markdown("#### Evaluation Mode")
    batch_eval_mode = st.radio(
        "Select evaluation approach",
        ["Criteria-Based (11 criteria)", "Holistic (program fit)"],
        horizontal=True,
        help="Criteria-Based uses 11 predefined criteria. Holistic evaluates overall program fit without specific criteria.",
        key=f"{key_prefix}batch_eval_mode"
    )
    batch_is_holistic = batch_eval_mode == "Holistic (program fit)"

    if role:
        st.caption("Role-specific guidance is appended to the system prompt and works with any evaluation mode.")
    if batch_is_holistic:
        st.info("Holistic mode will evaluate candidates based on overall program fit and provide binary interview recommendations.")

    uploaded_files = st.file_uploader(
        "Upload Candidate PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDFs, one per candidate. Filename becomes the candidate ID.",
        key=f"{key_prefix}batch_uploader"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")

        with st.expander("View files"):
            for f in uploaded_files:
                st.text(f"- {f.name}")

        col1, col2 = st.columns(2)

        with col1:
            job_name = st.text_input(
                "Job Name (optional)",
                placeholder="e.g., Spring 2026 Applicants",
                help="Give this batch a friendly name",
                key=f"{key_prefix}batch_job_name"
            )

        with col2:
            output_dir = st.text_input(
                "Output Directory",
                value="./results",
                help="Where to save evaluation results",
                key=f"{key_prefix}batch_output_dir"
            )

        if st.button("Start Batch Evaluation", use_container_width=True, key=f"{key_prefix}batch_eval_btn"):
            temp_dir = tempfile.mkdtemp()
            candidate_files = {}

            for uploaded_file in uploaded_files:
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                candidate_id = temp_path.stem
                candidate_files[candidate_id] = str(temp_path)

            config = st.session_state.config
            job_config = {
                "api_key": config.api.anthropic_api_key,
                "max_tokens": config.api.max_tokens,
                "evaluation_mode": "holistic" if batch_is_holistic else "criteria",
            }
            if role:
                job_config["role"] = role

            job_manager = st.session_state.job_manager
            mode_suffix = " (Holistic)" if batch_is_holistic else ""
            if role:
                role_label = next((k for k, v in ROLE_OPTIONS.items() if v == role), role)
                mode_suffix = f" ({role_label}{mode_suffix})"
            job_id = job_manager.submit_job(
                candidate_files=candidate_files,
                output_dir=output_dir,
                config=job_config,
                job_name=(job_name or f"Batch {datetime.now().strftime('%Y%m%d_%H%M')}") + mode_suffix
            )

            st.success(f"Job submitted! ID: `{job_id}`")
            st.info("You can navigate away - the evaluation will continue in the background.")

            time.sleep(1)
            st.rerun()


def role_specific_evaluation_form():
    """Role-specific evaluation form with role and mode selectors."""
    st.markdown("### Role-Specific Evaluation")
    st.caption(
        "Evaluate candidates with role-tailored guidance appended to the system prompt. "
        "Works with both Criteria-Based and Holistic modes."
    )

    selected_role_label = st.radio(
        "Candidate Role",
        list(ROLE_OPTIONS.keys()),
        horizontal=True,
        key="role_specific_role_selector"
    )
    role = ROLE_OPTIONS[selected_role_label]

    st.info(f"Evaluating as **{selected_role_label}** — role-specific calibration will be applied.")

    sub_tab1, sub_tab2 = st.tabs(["Single Candidate", "Batch Upload"])

    with sub_tab1:
        single_evaluation_form(role=role, key_prefix="role_single_")

    with sub_tab2:
        batch_evaluation_form(role=role, key_prefix="role_batch_")


def batch_jobs_page():
    """View and manage batch jobs."""
    st.title("Batch Jobs")

    job_manager = st.session_state.job_manager

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    if auto_refresh:
        time.sleep(0.1)  # Small delay to prevent too rapid refreshing

    # Tabs for job status
    tab1, tab2, tab3 = st.tabs(["Active", "Completed", "All Jobs"])

    with tab1:
        active_jobs = job_manager.get_active_jobs()

        if active_jobs:
            for job in active_jobs:
                render_job_card(job, job_manager)

            if auto_refresh:
                time.sleep(2)
                st.rerun()
        else:
            st.info("No active jobs. Start a new batch evaluation to see it here.")

    with tab2:
        all_jobs = job_manager.list_jobs(limit=50)
        completed_jobs = [j for j in all_jobs if j.get("status") == JobStatus.COMPLETED]

        if completed_jobs:
            for job in completed_jobs[:20]:
                render_job_card(job, job_manager, show_actions=False)
        else:
            st.info("No completed jobs yet.")

    with tab3:
        all_jobs = job_manager.list_jobs(limit=50)

        if all_jobs:
            # Summary table
            job_data = []
            for job in all_jobs:
                progress = job.get("progress", {})
                job_data.append({
                    "Job ID": job["job_id"][:20] + "...",
                    "Name": job.get("job_name", "-")[:30],
                    "Status": job.get("status", "unknown"),
                    "Progress": f"{progress.get('completed', 0)}/{progress.get('total', 0)}",
                    "Created": job.get("created_at", "")[:19]
                })

            st.dataframe(pd.DataFrame(job_data), hide_index=True, use_container_width=True)

            # Cleanup old jobs
            st.markdown("---")
            if st.button("Cleanup Old Jobs (> 7 days)"):
                job_manager.cleanup_old_jobs(days=7)
                st.success("Old jobs cleaned up!")
                st.rerun()
        else:
            st.info("No jobs yet.")


def render_job_card(job, job_manager, show_actions=True):
    """Render a job card with progress and actions."""
    job_id = job["job_id"]
    status = job.get("status", "unknown")
    progress = job.get("progress", {})
    completed = progress.get("completed", 0)
    failed = progress.get("failed", 0)
    total = progress.get("total", 1)
    current = progress.get("current_candidate")

    pct = (completed / total * 100) if total > 0 else 0

    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(f"**{job.get('job_name', job_id[:25])}**")
            st.caption(f"ID: {job_id[:30]}...")

        with col2:
            st.markdown(f"**{status.upper()}**")

        with col3:
            st.markdown(f"**{completed}/{total}**")
            if failed > 0:
                st.caption(f"{failed} failed")

        # Progress bar
        if status in [JobStatus.RUNNING, JobStatus.PENDING]:
            st.progress(pct / 100)
            if current:
                st.caption(f"Currently evaluating: {current}")

        # Actions
        if show_actions and status in [JobStatus.RUNNING, JobStatus.PENDING]:
            if st.button("Cancel", key=f"cancel_{job_id}"):
                job_manager.cancel_job(job_id)
                st.rerun()

        # Results for completed jobs
        if status == JobStatus.COMPLETED:
            with st.expander("View Results Summary"):
                # Try to get results from job data first
                results = job.get("results", [])
                result_data = []

                if results:
                    for r in results:
                        if r.get("status") == "success":
                            result_data.append({
                                "Candidate": r.get("candidate_id", ""),
                                "Score": f"{r.get('overall_score', 0):.1f}/10",
                                "Recommendation": r.get("recommendation", "") or ""
                            })

                # If no results in job data, try loading from disk
                if not result_data:
                    output_dir = Path(job.get("output_dir", "./results"))
                    if output_dir.exists():
                        disk_results = load_all_results_from_disk(output_dir)
                        # Filter to candidates in this job
                        job_candidates = set(job.get("candidate_files", {}).keys())
                        for r in disk_results:
                            if not job_candidates or r.candidate.candidate_id in job_candidates:
                                result_data.append({
                                    "Candidate": r.candidate.candidate_id,
                                    "Score": f"{r.overall_score:.1f}/10",
                                    "Recommendation": r.recommendation or ""
                                })

                if result_data:
                    st.dataframe(pd.DataFrame(result_data), hide_index=True)
                else:
                    st.write("No results available for this job.")

        # Errors for failed jobs
        if status == JobStatus.FAILED or failed > 0:
            errors = job.get("errors", [])
            if errors:
                with st.expander(f"View Errors ({len(errors)}) - Click to retry"):
                    for err in errors:
                        st.error(f"**{err.get('candidate_id')}**: {err.get('error')}")

                    st.markdown("---")

                    # Retry failed candidates
                    failed_ids = [err.get('candidate_id') for err in errors if err.get('candidate_id')]
                    if failed_ids and st.button(f"Retry {len(failed_ids)} Failed Candidates", key=f"retry_{job_id}"):
                        # Get original candidate files from job
                        candidate_files = job.get("candidate_files", {})
                        retry_files = {cid: path for cid, path in candidate_files.items() if cid in failed_ids}

                        if retry_files:
                            # Submit new job for failed candidates
                            # Preserve evaluation_mode from original job
                            original_config = job.get("config", {})
                            config = st.session_state.config
                            job_config = {
                                "api_key": config.api.anthropic_api_key,
                                "max_tokens": original_config.get("max_tokens", config.api.max_tokens),
                                "evaluation_mode": original_config.get("evaluation_mode", "criteria")
                            }

                            new_job_id = job_manager.submit_job(
                                candidate_files=retry_files,
                                output_dir=job.get("output_dir", "./results"),
                                config=job_config,
                                job_name=f"Retry: {job.get('job_name', 'Failed candidates')}"
                            )
                            st.success(f"Retry job submitted: {new_job_id}")
                            st.rerun()
                        else:
                            st.warning("Could not find original files for failed candidates")

        st.markdown("---")


def results_page():
    """View all evaluation results."""
    st.title("Evaluation Results")

    output_dir = Path("./results")

    # Load BOTH types of results
    criteria_results = load_all_results_from_disk(output_dir)
    holistic_results = load_all_holistic_results_from_disk(output_dir)

    total = len(criteria_results) + len(holistic_results)

    if total == 0:
        st.info("No evaluation results yet. Run some evaluations first!")
        return

    all_results = criteria_results + holistic_results
    role_eval_count = count_role_evaluations_from_results(all_results)

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Criteria-Based", len(criteria_results))
    with col2:
        st.metric("Holistic", len(holistic_results))
    with col3:
        st.metric("Role-Specific", role_eval_count)
    with col4:
        st.metric("Total", total)
    with col5:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Build lookup dicts for combined view
    criteria_by_id = {r.candidate.candidate_id: r for r in criteria_results}
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}

    # Find candidates with both evaluations
    both_ids = sorted(set(criteria_by_id.keys()) & set(holistic_by_id.keys()))

    # Tab view for different evaluation types
    tab1, tab2, tab3, tab4 = st.tabs([
        f"Criteria-Based ({len(criteria_results)})",
        f"Holistic ({len(holistic_results)})",
        f"Combined View ({len(both_ids)})",
        f"Role Rankings ({role_eval_count})",
    ])

    with tab1:
        if not criteria_results:
            st.info("No criteria-based evaluations yet.")
        else:
            # Export buttons for criteria-based
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                if st.button("Export Criteria (CSV)", use_container_width=True, key="export_csv"):
                    csv_path = output_dir / "criteria_results.csv"
                    CSVExporter.export_batch(criteria_results, csv_path)
                    with open(csv_path, 'rb') as f:
                        st.download_button(
                            "Download CSV",
                            data=f,
                            file_name=f"criteria_evaluations_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="dl_csv"
                        )
            with exp_col2:
                if st.button("Export Criteria (JSON)", use_container_width=True, key="export_json"):
                    json_path = output_dir / "criteria_results.json"
                    JSONExporter.export_batch(criteria_results, json_path)
                    with open(json_path, 'rb') as f:
                        st.download_button(
                            "Download JSON",
                            data=f,
                            file_name=f"criteria_evaluations_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            key="dl_json"
                        )

            # Results table
            st.subheader("Criteria-Based Results")
            summary_data = []
            for rank, result in enumerate(sorted(criteria_results, key=lambda r: r.overall_score, reverse=True), 1):
                summary_data.append({
                    'Rank': rank,
                    'Candidate ID': result.candidate.candidate_id,
                    'Name': result.candidate.name or '-',
                    'Score': f"{result.overall_score:.1f}/10",
                    'Date': result.candidate.evaluation_date.strftime("%Y-%m-%d"),
                    'Recommendation': result.recommendation
                })
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

            # Individual candidate view
            st.subheader("Candidate Details")
            candidate_options = {
                f"{r.candidate.candidate_id} ({r.overall_score:.1f})": r
                for r in sorted(criteria_results, key=lambda r: r.overall_score, reverse=True)
            }
            selected = st.selectbox("Select candidate", list(candidate_options.keys()), key="criteria_select")
            if selected:
                result = candidate_options[selected]
                display_evaluation_result(result)

    with tab2:
        if not holistic_results:
            st.info("No holistic evaluations yet. Run evaluations in Holistic mode to see results here.")
        else:
            # Results table for holistic
            st.subheader("Holistic Results")
            sorted_holistic = sorted(holistic_results, key=lambda r: r.overall_score, reverse=True)
            n_holistic = len(sorted_holistic)
            holistic_summary = []
            for rank, result in enumerate(sorted_holistic, 1):
                pct = 100.0 if n_holistic == 1 else round(100.0 * (n_holistic - rank) / (n_holistic - 1))
                holistic_summary.append({
                    'Rank': rank,
                    'Percentile': f"{pct}th",
                    'Candidate ID': result.candidate.candidate_id,
                    'Name': result.candidate.name or '-',
                    'Score': f"{result.overall_score:.1f}/10",
                    'Interview': 'Yes' if result.interview_decision else 'No',
                    'Innovation': result.innovation_potential.level.capitalize(),
                    'Program Fit': result.program_fit.level.capitalize(),
                    'Date': result.candidate.evaluation_date.strftime("%Y-%m-%d")
                })
            st.dataframe(pd.DataFrame(holistic_summary), hide_index=True, use_container_width=True)

            # Individual holistic candidate view
            st.subheader("Candidate Details")
            holistic_options = {
                f"{r.candidate.candidate_id} ({r.overall_score:.1f})": r
                for r in sorted(holistic_results, key=lambda r: r.overall_score, reverse=True)
            }
            selected_holistic = st.selectbox("Select candidate", list(holistic_options.keys()), key="holistic_select")
            if selected_holistic:
                result = holistic_options[selected_holistic]
                display_holistic_evaluation_result(result)

    with tab3:
        if not both_ids:
            st.info("No candidates have both evaluation types yet. Run both criteria-based and holistic evaluations on the same candidates to compare.")
        else:
            st.subheader(f"Candidates with Both Evaluations: {len(both_ids)}")

            # Disparity analysis (overall statistics)
            display_disparity_analysis(criteria_by_id, holistic_by_id, both_ids)

            # Individual candidate comparison
            st.subheader("Individual Candidate Comparison")

            # Candidate selector
            selected_id = st.selectbox(
                "Select Candidate",
                both_ids,
                key="combined_view_selector"
            )

            if selected_id:
                criteria_result = criteria_by_id[selected_id]
                holistic_result = holistic_by_id[selected_id]

                # Comparison summary
                display_comparison_summary(criteria_result, holistic_result)

                # Side-by-side display
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Criteria-Based Evaluation")
                    display_evaluation_result(criteria_result)
                with col2:
                    st.markdown("### Holistic Evaluation")
                    display_holistic_evaluation_result(holistic_result)

    with tab4:
        _render_role_specific_rankings_local(all_results)


def _render_role_specific_rankings_local(results: list) -> None:
    """Render role-specific leaderboards from loaded result objects."""
    role_results = [r for r in results if getattr(r, "role", None)]
    if not role_results:
        st.info(
            "No role-specific evaluations yet. Use **New Evaluation → "
            "Role-Specific Evaluation** to evaluate by specialty."
        )
        return

    st.caption(
        "Candidates ranked by **role-specific score** within each specialty "
        "(ties broken by overall Catalyst score)."
    )
    grouped = group_results_by_role(role_results)

    for role in ROLE_ORDER:
        items = grouped.get(role, [])
        if not items:
            continue
        sorted_items = sort_results_by_role_score(items)
        n = len(sorted_items)
        st.markdown(f"### {ROLE_LABELS[role]} ({n})")
        summary = [
            build_role_ranking_row_from_result(rank, result, n)
            for rank, result in enumerate(sorted_items, 1)
        ]
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

        options = {}
        for result in sorted_items:
            rs = build_role_ranking_row_from_result(1, result, n)["Role Score"]
            options[f"{result.candidate.candidate_id} ({rs})"] = result
        selected = st.selectbox(
            f"View {ROLE_LABELS[role]} candidate",
            list(options.keys()),
            key=f"local_role_sel_{role}",
        )
        if selected:
            result = options[selected]
            if isinstance(result, HolisticEvaluationResult):
                display_holistic_evaluation_result(result)
            else:
                display_evaluation_result(result)
        st.markdown("---")


def display_disparity_analysis(criteria_by_id: dict, holistic_by_id: dict, both_ids: list):
    """Display statistical analysis of method disparity vs candidate spread."""
    import numpy as np

    if len(both_ids) < 3:
        st.info("Need at least 3 candidates with both evaluations for disparity analysis.")
        return

    # Calculate scores and differences
    criteria_scores = [criteria_by_id[cid].overall_score for cid in both_ids]
    holistic_scores = [holistic_by_id[cid].overall_score for cid in both_ids]
    score_diffs = [criteria_by_id[cid].overall_score - holistic_by_id[cid].overall_score
                   for cid in both_ids]

    # Candidate spread (average std of both methods)
    criteria_std = np.std(criteria_scores)
    holistic_std = np.std(holistic_scores)
    candidate_std = (criteria_std + holistic_std) / 2

    # Method disparity
    method_disparity_std = np.std(score_diffs)
    method_disparity_mean = np.mean(np.abs(score_diffs))

    # Correlation
    correlation = np.corrcoef(criteria_scores, holistic_scores)[0, 1]

    # Disparity ratio
    disparity_ratio = method_disparity_std / candidate_std if candidate_std > 0 else 0

    st.markdown("### Method Disparity Analysis")
    st.caption("Comparing the spread of scores across candidates vs. the disagreement between evaluation methods.")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Candidate Spread (σ)", f"{candidate_std:.2f} pts")

    with col2:
        st.metric("Method Disparity (σ)", f"{method_disparity_std:.2f} pts")

    with col3:
        st.metric("Disparity Ratio", f"{disparity_ratio:.2f}")

    with col4:
        st.metric("Correlation", f"{correlation:.2f}")

    # Interpretation
    if disparity_ratio < 0.3:
        st.success(f"**Methods strongly agree.** Method differences ({method_disparity_std:.2f}) are much smaller than candidate differences ({candidate_std:.2f}).")
    elif disparity_ratio < 0.5:
        st.info(f"**Methods mostly agree.** Method differences are moderate compared to candidate spread.")
    elif disparity_ratio < 0.7:
        st.warning(f"**Moderate disagreement.** Method choice affects scores nearly as much as candidate quality.")
    else:
        st.error(f"**Significant disagreement.** Method differences ({method_disparity_std:.2f}) are large relative to candidate spread ({candidate_std:.2f}). Consider investigating why methods diverge.")

    # Scatter plot with confidence band
    with st.expander("View Scatter Plot", expanded=True):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot points
        ax.scatter(criteria_scores, holistic_scores, alpha=0.6, s=50, c='steelblue')

        # Perfect agreement line
        min_score = min(min(criteria_scores), min(holistic_scores)) - 0.5
        max_score = max(max(criteria_scores), max(holistic_scores)) + 0.5
        ax.plot([min_score, max_score], [min_score, max_score], 'k--', alpha=0.5, label='Perfect Agreement')

        # Confidence band (±1σ method disparity)
        x_line = np.linspace(min_score, max_score, 100)
        ax.fill_between(x_line, x_line - method_disparity_std, x_line + method_disparity_std,
                        alpha=0.2, color='orange', label=f'±1σ Band ({method_disparity_std:.2f} pts)')

        ax.set_xlabel('Criteria-Based Score')
        ax.set_ylabel('Holistic Score')
        ax.set_title('Criteria vs. Holistic Scores')
        ax.legend(loc='lower right')
        ax.set_xlim(min_score, max_score)
        ax.set_ylim(min_score, max_score)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    st.markdown("---")


def display_comparison_summary(criteria_result, holistic_result):
    """Display a comparison summary between criteria-based and holistic evaluations."""
    score_diff = criteria_result.overall_score - holistic_result.overall_score

    st.markdown("### Comparison Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Criteria Score",
            f"{criteria_result.overall_score:.1f}",
            delta=None
        )

    with col2:
        st.metric(
            "Holistic Score",
            f"{holistic_result.overall_score:.1f}",
            delta=None
        )

    with col3:
        if score_diff > 0:
            delta_label = "Criteria higher"
        elif score_diff < 0:
            delta_label = "Holistic higher"
        else:
            delta_label = "Equal"
        st.metric(
            "Score Difference",
            f"{abs(score_diff):.1f}",
            delta=delta_label
        )

    with col4:
        interview = "Yes" if holistic_result.interview_decision else "No"
        st.metric("Interview (Holistic)", interview)

    # Recommendations comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Criteria Recommendation:** {criteria_result.recommendation}")
    with col2:
        st.markdown(f"**Holistic Recommendation:** {holistic_result.recommendation}")

    # Agreement indicator
    recs_match = criteria_result.recommendation.lower() == holistic_result.recommendation.lower()
    if recs_match:
        st.success("Recommendations align")
    elif abs(score_diff) > 1.5:
        st.warning(f"Significant score difference: {abs(score_diff):.1f} points")

    st.markdown("---")


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

    st.markdown(f"#### Role-Specific Assessment ({role_label})")
    cols = st.columns(3)
    with cols[0]:
        if score is not None:
            st.metric(marker, f"{score}/10")
    with cols[1]:
        st.metric("Confidence", str(confidence).capitalize())
    with cols[2]:
        extra_fields = {
            k: v for k, v in assessment.items()
            if k not in {
                "role", "marker", "score", "confidence", "reasoning",
                "evidence", "evidence_gaps",
            } and v
        }
        if extra_fields:
            first_key, first_val = next(iter(extra_fields.items()))
            st.metric(first_key.replace("_", " ").title(), str(first_val).replace("_", " "))

    if assessment.get("reasoning"):
        st.markdown(assessment["reasoning"])

    for key, value in assessment.items():
        if key in {"role", "marker", "score", "confidence", "reasoning", "evidence", "evidence_gaps"}:
            continue
        if value:
            st.caption(f"**{key.replace('_', ' ').title()}:** {str(value).replace('_', ' ')}")

    evidence = assessment.get("evidence") or []
    if evidence:
        with st.expander("Role-Specific Evidence"):
            for ev in evidence:
                if isinstance(ev, dict):
                    quote = (ev.get("quote") or "").strip()
                    source = (ev.get("source") or "").strip()
                    context = (ev.get("context") or "").strip()
                    if source:
                        st.markdown(f"**Source:** {source}")
                    if context:
                        st.markdown(f"**Context:** {context}")
                    if quote:
                        st.markdown(quote)
                else:
                    st.markdown(str(ev))
                st.markdown("---")

    gaps = assessment.get("evidence_gaps") or []
    if gaps:
        st.markdown("**Evidence gaps for interview:**")
        for gap in gaps:
            st.markdown(f"- {gap}")


def display_evaluation_result(result):
    """Display a single evaluation result."""
    if getattr(result, "role", None):
        st.caption(f"Role context: **{result.role.replace('_', ' ').title()}**")

    # Warn if evaluation is incomplete
    if len(result.scores) < 11:
        st.warning(f"Incomplete evaluation: only {len(result.scores)}/11 criteria scored. "
                   f"This is likely due to output truncation (max_tokens too low). "
                   f"Re-run this candidate for a complete evaluation.")

    # Score summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Score", f"{result.overall_score:.1f}/10")
    with col2:
        st.metric("Criteria", f"{len(result.scores)}/11")
    with col3:
        high_conf = sum(1 for s in result.scores if s.confidence == 'high')
        st.metric("High Confidence", f"{high_conf}/{len(result.scores)}")
    with col4:
        st.metric("Date", result.candidate.evaluation_date.strftime("%Y-%m-%d"))

    st.markdown(f"**Recommendation:** {result.recommendation}")

    # Strengths and areas
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

    # Score breakdown
    st.markdown("#### Scores by Criterion")

    score_data = pd.DataFrame([
        {
            'Criterion': score.criterion.display_name,
            'Score': score.score,
            'Confidence': score.confidence.capitalize()
        }
        for score in sorted(result.scores, key=lambda s: s.score, reverse=True)
    ])

    st.dataframe(score_data, hide_index=True, use_container_width=True)

    # Bar chart
    chart_data = pd.DataFrame([
        {'Criterion': s.criterion.value.replace('_', ' ').title(), 'Score': s.score}
        for s in result.scores
    ])
    st.bar_chart(chart_data.set_index('Criterion'))

    display_role_specific_assessment(getattr(result, "role_specific_assessment", None))

    # Detailed evidence
    with st.expander("View Detailed Evidence"):
        for score in result.scores:
            st.markdown(f"**{score.criterion.display_name}** - {score.score}/10 ({score.confidence})")
            st.markdown(score.reasoning)

            if score.evidence:
                for ev in score.evidence:
                    st.markdown(f"> *{ev.source}:* {ev.quote}")

            st.markdown("---")


def display_holistic_evaluation_result(result: HolisticEvaluationResult):
    """Display a holistic evaluation result (supports both old and enhanced formats)."""
    if result.role:
        st.caption(f"Role context: **{result.role.replace('_', ' ').title()}**")

    # Score summary with interview decision and confidence
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

    # Score justification if present (enhanced format)
    if hasattr(result, 'score_justification') and result.score_justification:
        with st.expander("Score Justification"):
            st.markdown(result.score_justification)

    # Interview decision reasoning if present (enhanced format)
    if hasattr(result, 'interview_decision_reasoning') and result.interview_decision_reasoning:
        with st.expander("Interview Decision Reasoning"):
            st.markdown(result.interview_decision_reasoning)

    # Overall Assessment
    st.markdown("#### Overall Assessment")
    st.markdown(result.overall_assessment)

    display_role_specific_assessment(result.role_specific_assessment)

    # Innovation and Program Fit details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Innovation Potential")
        st.markdown(f"**Level:** {result.innovation_potential.level.capitalize()} (Confidence: {getattr(result.innovation_potential, 'confidence', 'medium')})")
        st.markdown(result.innovation_potential.reasoning)

        # Display structured evidence if present (enhanced format)
        if hasattr(result.innovation_potential, 'evidence') and result.innovation_potential.evidence:
            st.markdown("**Evidence:**")
            for ev in result.innovation_potential.evidence:
                if hasattr(ev, 'quote'):
                    source_text = f" *({ev.source})*" if ev.source else ""
                    st.markdown(f"> \"{ev.quote}\"{source_text}")
                    if ev.context:
                        st.caption(f"Context: {ev.context}")
        # Fallback to legacy key_evidence
        elif result.innovation_potential.key_evidence:
            st.markdown("**Key Evidence:**")
            for ev in result.innovation_potential.key_evidence:
                st.markdown(f"> {ev}")

    with col2:
        st.markdown("#### Program Fit")
        st.markdown(f"**Level:** {result.program_fit.level.capitalize()} (Confidence: {getattr(result.program_fit, 'confidence', 'medium')})")

        # Detailed analysis if present (enhanced format)
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

        # Display structured evidence if present (enhanced format)
        if hasattr(result.program_fit, 'evidence') and result.program_fit.evidence:
            with st.expander("Supporting Evidence"):
                for ev in result.program_fit.evidence:
                    if hasattr(ev, 'quote'):
                        source_text = f" *({ev.source})*" if ev.source else ""
                        st.markdown(f"> \"{ev.quote}\"{source_text}")
                        if ev.context:
                            st.caption(f"Context: {ev.context}")

    # Notable Qualities
    if result.notable_qualities:
        st.markdown("#### Notable Qualities")
        for q in result.notable_qualities:
            confidence = getattr(q, 'confidence', 'medium')
            with st.expander(f"{q.quality} (Confidence: {confidence})"):
                # Handle both string evidence and list of structured evidence
                if isinstance(q.evidence, list):
                    st.markdown("**Evidence:**")
                    for ev in q.evidence:
                        if hasattr(ev, 'quote'):
                            source_text = f" *({ev.source})*" if ev.source else ""
                            st.markdown(f"> \"{ev.quote}\"{source_text}")
                            if ev.context:
                                st.caption(ev.context)
                        else:
                            st.markdown(f"> {ev}")
                else:
                    st.markdown(f"**Evidence:** {q.evidence}")
                st.markdown(f"**Significance:** {q.significance}")

    # Red Flags (supports both string and structured formats)
    if result.red_flags:
        st.markdown("#### Red Flags")
        for flag in result.red_flags:
            if hasattr(flag, 'flag'):
                # Enhanced format with severity
                severity_colors = {'high': 'error', 'medium': 'warning', 'low': 'info'}
                severity = getattr(flag, 'severity', 'medium')
                if severity == 'high':
                    st.error(f"**{flag.flag}**")
                elif severity == 'low':
                    st.info(f"{flag.flag}")
                else:
                    st.warning(f"{flag.flag}")
                if flag.evidence:
                    st.caption(f"Evidence: {flag.evidence}")
            else:
                # Legacy string format
                st.warning(flag)

    # Interview Questions (supports both string and structured formats)
    if result.questions_for_interview:
        st.markdown("#### Suggested Interview Questions")
        for i, q in enumerate(result.questions_for_interview, 1):
            if hasattr(q, 'question'):
                # Enhanced format with category and purpose
                category = getattr(q, 'category', 'General')
                st.markdown(f"**{i}. [{category}]** {q.question}")
                if q.purpose:
                    st.caption(f"Purpose: {q.purpose}")
            else:
                # Legacy string format
                st.markdown(f"{i}. {q}")


def analysis_page():
    """Analysis dashboard."""
    st.title("Analysis Dashboard")

    output_dir = Path("./results")
    all_results = load_all_results_from_disk(output_dir)

    # Also load holistic results
    holistic_results = load_all_holistic_results_from_disk(output_dir)

    total_results = len(all_results) + len(holistic_results)

    if total_results == 0:
        st.warning("No evaluation results found. Run some evaluations first!")
        return

    st.markdown(f"**Analyzing {len(all_results)} criteria-based + {len(holistic_results)} holistic evaluations**")

    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "AI vs Expert", "Recommendations"])

    with tab1:
        if all_results:
            distribution_analysis(all_results)
        else:
            st.info("No criteria-based evaluations to analyze. Run criteria-based evaluations first.")

    with tab2:
        if all_results:
            expert_comparison_analysis(all_results, output_dir)
        else:
            st.info("Expert comparison requires criteria-based evaluations.")

    with tab3:
        if all_results:
            recommendation_analysis(all_results)
        else:
            st.info("Recommendations analysis requires criteria-based evaluations.")


def distribution_analysis(all_results):
    """Score distribution analysis."""
    st.subheader("Score Distribution Analysis")

    from candidate_evaluator.core.distribution_analyzer import DistributionAnalyzer

    analyzer = DistributionAnalyzer(all_results)

    # Overall stats
    col1, col2, col3, col4 = st.columns(4)

    overall_scores = [r.overall_score for r in all_results]

    with col1:
        st.metric("Mean Score", f"{sum(overall_scores)/len(overall_scores):.2f}")
    with col2:
        st.metric("Median Score", f"{sorted(overall_scores)[len(overall_scores)//2]:.2f}")
    with col3:
        st.metric("Min Score", f"{min(overall_scores):.2f}")
    with col4:
        st.metric("Max Score", f"{max(overall_scores):.2f}")

    st.markdown("---")

    # Percentile analysis
    percentile = st.slider("Percentile Split", 25, 75, 50)

    if st.button("Analyze"):
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


def expert_comparison_analysis(all_results, output_dir):
    """AI vs Expert comparison."""
    st.subheader("AI vs Expert Comparison")

    st.info("Upload expert ratings to compare with AI evaluations.")

    expert_file = st.file_uploader(
        "Upload Expert Ratings",
        type=['xlsx', 'xls', 'csv'],
        help="Excel or CSV with expert ratings"
    )

    if expert_file:
        try:
            from candidate_evaluator.core.expert_comparison import ExpertComparisonAnalyzer

            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / expert_file.name
            with open(temp_path, 'wb') as f:
                f.write(expert_file.getbuffer())

            analyzer = ExpertComparisonAnalyzer()

            with st.spinner("Loading expert ratings..."):
                expert_ratings = analyzer.load_expert_ratings_from_excel(temp_path)

            st.success(f"Loaded {len(expert_ratings)} expert ratings")

            analyzer.set_ai_results(all_results)

            threshold = st.slider("Interview Threshold", 1.0, 10.0, 6.0)

            if st.button("Run Comparison"):
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

        except Exception as e:
            st.error(f"Error: {e}")


def decision_comparison_analysis(criteria_results, holistic_results, output_dir):
    """Compare AI predictions against actual admission decisions."""
    st.subheader("Decision Comparison")
    st.markdown("Compare AI evaluation predictions against actual admission decisions.")

    # Instructions
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

    # File upload
    uploaded_file = st.file_uploader(
        "Upload decisions file",
        type=['csv', 'xlsx'],
        help="CSV with candidate_id and admitted columns",
        key="decision_comparison_uploader"
    )

    if uploaded_file is None:
        st.info("Upload a decisions file to compare AI predictions against actual admit outcomes.")
        return

    # Load decisions
    try:
        if uploaded_file.name.endswith('.xlsx'):
            decisions_df = pd.read_excel(uploaded_file)
        else:
            decisions_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Normalize column names
    decisions_df.columns = decisions_df.columns.str.lower().str.strip()

    if 'candidate_id' not in decisions_df.columns:
        st.error("File must have a `candidate_id` column")
        return

    if 'admitted' not in decisions_df.columns:
        st.error("File must have an `admitted` column (yes/no)")
        return

    # Normalize boolean column
    def normalize_bool(val):
        if pd.isna(val):
            return None
        if isinstance(val, bool):
            return val
        val_str = str(val).lower().strip()
        return val_str in ['yes', 'true', '1', 'y']

    decisions_df['admitted'] = decisions_df['admitted'].apply(normalize_bool)

    st.success(f"Loaded {len(decisions_df)} admission records")

    # Select data source for comparison
    st.markdown("### Select Evaluation Data")
    data_source = st.radio(
        "Compare against:",
        ["Criteria-Based Evaluations", "Holistic Evaluations", "Both"],
        horizontal=True
    )

    # Score threshold for criteria-based
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

        # Merge with decisions
        merged = ai_df.merge(decisions_df, on='candidate_id', how='inner')

        if len(merged) == 0:
            st.error("No matching candidates found between evaluations and decisions file. Check that candidate IDs match.")
            return

        st.markdown(f"**Matched {len(merged)} candidates**")

        # Calculate metrics
        st.markdown("### Admit Decision Comparison")

        actual_admitted = merged['admitted'].values
        ai_predicted = merged['ai_prediction'].values

        # Remove any None values
        valid_mask = [a is not None for a in actual_admitted]
        actual_admitted = [actual_admitted[i] for i in range(len(valid_mask)) if valid_mask[i]]
        ai_predicted = [ai_predicted[i] for i in range(len(valid_mask)) if valid_mask[i]]

        if len(actual_admitted) == 0:
            st.warning("No valid admit decisions to compare.")
            return

        # Calculate confusion matrix
        tp = sum(1 for a, p in zip(actual_admitted, ai_predicted) if a and p)
        tn = sum(1 for a, p in zip(actual_admitted, ai_predicted) if not a and not p)
        fp = sum(1 for a, p in zip(actual_admitted, ai_predicted) if not a and p)
        fn = sum(1 for a, p in zip(actual_admitted, ai_predicted) if a and not p)

        # Display confusion matrix
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
            st.metric(
                "Sensitivity (True Positive Rate)",
                f"{sensitivity:.1%}",
                help=f"AI identified {sensitivity:.1%} of candidates who were actually admitted"
            )

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            st.metric(
                "Specificity (True Negative Rate)",
                f"{specificity:.1%}",
                help=f"AI correctly rejected {specificity:.1%} of candidates not admitted"
            )

            accuracy = (tp + tn) / len(actual_admitted) if len(actual_admitted) > 0 else 0
            st.metric("Overall Accuracy", f"{accuracy:.1%}")

            p_o = (tp + tn) / len(actual_admitted) if len(actual_admitted) > 0 else 0
            p_yes = ((tp + fn) / len(actual_admitted)) * ((tp + fp) / len(actual_admitted))
            p_no = ((fp + tn) / len(actual_admitted)) * ((fn + tn) / len(actual_admitted))
            p_e = p_yes + p_no
            kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0

            kappa_interpretation = "Poor" if kappa < 0.2 else "Fair" if kappa < 0.4 else "Moderate" if kappa < 0.6 else "Good" if kappa < 0.8 else "Excellent"
            st.metric(
                "Cohen's Kappa",
                f"{kappa:.3f} ({kappa_interpretation})",
                help="Measures agreement beyond chance"
            )

        # Disagreement analysis
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

        # Bias detection
        st.markdown("### Bias Analysis")
        ai_positive_rate = sum(ai_predicted) / len(ai_predicted) if len(ai_predicted) > 0 else 0
        human_positive_rate = sum(actual_admitted) / len(actual_admitted) if len(actual_admitted) > 0 else 0

        if ai_positive_rate > human_positive_rate + 0.1:
            st.warning(f"AI may be too lenient: AI recommends {ai_positive_rate:.1%} vs actual admit rate {human_positive_rate:.1%}")
        elif ai_positive_rate < human_positive_rate - 0.1:
            st.warning(f"AI may be too strict: AI recommends {ai_positive_rate:.1%} vs actual admit rate {human_positive_rate:.1%}")
        else:
            st.success(f"AI and actual admit rates are similar: AI {ai_positive_rate:.1%}, Actual {human_positive_rate:.1%}")

        # Per-dimension systematic bias (holistic only — requires dimension fields)
        holistic_in_merged = merged[merged['source'] == 'holistic'] if 'source' in merged.columns else pd.DataFrame()

        if not holistic_in_merged.empty:
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
                    return {'high': 3, 'strong': 3, 'medium': 2, 'moderate': 2, 'low': 1, 'weak': 1}.get(str(val), 2)

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

                        # Distribution of program fit levels
                        fit_counts = subset['program_fit'].value_counts().to_dict()
                        inn_counts = subset['innovation'].value_counts().to_dict()
                        st.markdown(
                            f"Program fit: {fit_counts}  |  "
                            f"Innovation: {inn_counts}"
                        )

                # Score distribution comparison table
                st.markdown("#### Score Distribution by Outcome")
                score_summary = (
                    dim_df.groupby('outcome')['overall_score']
                    .agg(['mean', 'min', 'max', 'count'])
                    .rename(columns={'mean': 'Mean', 'min': 'Min', 'max': 'Max', 'count': 'N'})
                    .round(2)
                )
                st.dataframe(score_summary, use_container_width=True)


def recommendation_analysis(all_results):
    """Recommendation breakdown analysis."""
    st.subheader("Recommendation Analysis")

    # Group by recommendation
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

                # Show candidate list
                for c in sorted(candidates, key=lambda x: -x.overall_score)[:5]:
                    st.text(f"- {c.candidate.candidate_id}: {c.overall_score:.1f}/10")


def _get_progress_file_path() -> Path:
    """Get the path to the progress file."""
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    return output_dir / "admit_analysis_progress.json"


def _load_progress_from_disk() -> dict:
    """Load analysis progress from disk."""
    progress_file = _get_progress_file_path()
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_progress_to_disk(progress: dict):
    """Save analysis progress to disk."""
    progress_file = _get_progress_file_path()
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2, default=str)


def _clear_progress_from_disk():
    """Clear progress file from disk."""
    progress_file = _get_progress_file_path()
    if progress_file.exists():
        progress_file.unlink()


def admit_pattern_analysis_page():
    """Admit Pattern Analysis - discover patterns and compare AI accuracy against actual decisions."""
    st.title("Admit Pattern Analysis")

    page_tab1, page_tab2 = st.tabs(["Pattern Analysis", "Decision Comparison"])

    with page_tab1:
        _admit_pattern_tab()

    with page_tab2:
        output_dir = Path("./results")
        criteria_results = load_all_results_from_disk(output_dir)
        holistic_results = load_all_holistic_results_from_disk(output_dir)
        decision_comparison_analysis(criteria_results, holistic_results, output_dir)


def _admit_pattern_tab():
    """Core admit pattern analysis content (formerly the full page)."""
    st.markdown("Upload candidate applications with admit/reject labels to discover distinguishing patterns.")
    
    # Load progress from disk if not in session state
    if 'admit_analysis_progress' not in st.session_state:
        disk_progress = _load_progress_from_disk()
        if disk_progress:
            st.session_state['admit_analysis_progress'] = disk_progress
            # Also restore file paths if they exist
            if disk_progress.get('file_paths'):
                st.session_state['admit_analysis_file_paths'] = disk_progress['file_paths']
            if disk_progress.get('admit_map'):
                # Reconstruct mapping dataframe (pd is imported globally at top of file)
                mapping_df = pd.DataFrame(list(disk_progress['admit_map'].items()), 
                                         columns=['filename', 'admit_status'])
                st.session_state['admit_mapping'] = mapping_df
    
    # Reset button if analysis is in progress or completed
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
                keys_to_delete = [
                    'admit_analysis_progress',
                    'admit_analysis_temp_dir', 
                    'admit_analysis_file_paths',
                    'last_pattern_analysis',
                    'admit_mapping',
                    'admit_analysis_holistic'
                ]
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                _clear_progress_from_disk()
                st.rerun()
        st.markdown("---")
    
    # Instructions
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
        candidate_003_application.pdf,yes
        ```
        
        **Step 3: Run analysis**
        - The system will evaluate each candidate
        - Then analyze patterns distinguishing admitted from rejected
        
        **Note**: This process may take significant time for large batches (expect ~1-2 minutes per candidate).
        """)
    
    st.markdown("---")
    
    # File uploads
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
                    # Normalize admit_status
                    def normalize_admit(val):
                        if pd.isna(val):
                            return None
                        val_str = str(val).lower().strip()
                        return val_str in ['yes', 'true', '1', 'y', 'admitted', 'admit']
                    
                    mapping_df['admit_status'] = mapping_df['admit_status'].apply(normalize_admit)
                    
                    admitted_count = mapping_df['admit_status'].sum()
                    rejected_count = len(mapping_df) - admitted_count
                    
                    st.success(f"Loaded {len(mapping_df)} mappings")
                    st.metric("Admitted", admitted_count)
                    st.metric("Rejected", rejected_count)
                    
                    # Store in session state
                    st.session_state['admit_mapping'] = mapping_df
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    st.markdown("---")
    
    # Analysis options
    st.subheader("3. Analysis Options")
    
    eval_mode = st.radio(
        "Evaluation mode for candidates",
        ["Holistic (faster, recommended)", "Criteria-Based (detailed scores)"],
        horizontal=True,
        help="Holistic mode is faster and provides overall fit assessment. Criteria-based provides 11 detailed scores."
    )
    
    use_holistic = eval_mode == "Holistic (faster, recommended)"
    
    # Check if we can proceed
    can_proceed = (
        uploaded_files and 
        mapping_file and 
        'admit_mapping' in st.session_state
    )
    
    # Check if analysis is in progress (resume automatically)
    analysis_in_progress = (
        'admit_analysis_progress' in st.session_state and 
        not st.session_state['admit_analysis_progress'].get('completed', False)
    )
    
    if analysis_in_progress:
        st.info("Continuing analysis...")
        # Store evaluation mode in session state on first run
        if 'admit_analysis_holistic' not in st.session_state:
            st.session_state['admit_analysis_holistic'] = use_holistic
        run_admit_pattern_analysis(uploaded_files, st.session_state['admit_analysis_holistic'])
    elif st.button("Run Admit Pattern Analysis", disabled=not can_proceed, use_container_width=True):
        # Store evaluation mode
        st.session_state['admit_analysis_holistic'] = use_holistic
        run_admit_pattern_analysis(uploaded_files, use_holistic)
    
    st.markdown("---")
    
    # Display previous results if available
    st.subheader("Previous Analysis Results")
    display_admit_pattern_results()


def run_admit_pattern_analysis(uploaded_files, use_holistic: bool):
    """Run the full admit pattern analysis pipeline."""
    config = st.session_state.config
    mapping_df = st.session_state.get('admit_mapping')
    
    if mapping_df is None:
        st.error("No admit mapping found")
        return
    
    # Create filename -> admit_status mapping
    admit_map = dict(zip(mapping_df['filename'], mapping_df['admit_status']))
    
    # Save files to a persistent directory (not temp) so they survive reloads
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    upload_dir = output_dir / "admit_analysis_uploads"
    upload_dir.mkdir(exist_ok=True)
    
    # Save files if not already saved
    if 'admit_analysis_file_paths' not in st.session_state:
        file_paths = {}
        
        for uploaded_file in uploaded_files:
            save_path = upload_dir / uploaded_file.name
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths[uploaded_file.name] = str(save_path)
        
        st.session_state['admit_analysis_file_paths'] = file_paths
    else:
        file_paths = st.session_state['admit_analysis_file_paths']
    
    # Match files with admit status
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
        st.warning(f"{len(unmatched_files)} files not found in mapping: {', '.join(unmatched_files[:5])}...")
    
    if not matched_candidates:
        st.error("No files matched the admit mapping. Check that filenames in CSV match uploaded files exactly.")
        return
    
    # Initialize or get progress tracking from session state
    if 'admit_analysis_progress' not in st.session_state:
        st.session_state['admit_analysis_progress'] = {
            'current_index': 0,
            'total_candidates': len(matched_candidates),
            'candidate_summaries': [],
            'errors': [],
            'completed': False,
            'use_holistic': use_holistic,
            'file_paths': file_paths,
            'admit_map': {k: bool(v) for k, v in admit_map.items()}  # Ensure JSON serializable
        }
        # Save initial progress to disk
        _save_progress_to_disk(st.session_state['admit_analysis_progress'])
    
    progress = st.session_state['admit_analysis_progress']
    
    # Check if already completed
    if progress['completed']:
        st.success("Analysis already completed! See results below.")
        # Try to load the last pattern analysis from disk
        if 'last_pattern_analysis' not in st.session_state:
            # Find the most recent analysis file
            analysis_files = list(output_dir.glob("admit_pattern_analysis_*.json"))
            if analysis_files:
                latest = max(analysis_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest, 'r') as f:
                        data = json.load(f)
                    st.session_state['last_pattern_analysis'] = data
                except Exception:
                    pass
        
        if 'last_pattern_analysis' in st.session_state:
            # Display using dict directly since it may not be the model object
            _display_pattern_analysis_from_dict(st.session_state['last_pattern_analysis'])
        return
    
    st.info(f"Processing {len(matched_candidates)} matched candidates...")
    
    # Phase 1: Evaluate all candidates
    evaluator = st.session_state.evaluator
    
    # Create a placeholder for status updates
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(progress['current_index'] / len(matched_candidates))
        status_text = st.empty()
        error_container = st.empty()
        skipped_container = st.empty()
        
        # Track skipped candidates
        skipped_count = 0
        
        # Process candidates one at a time, saving progress
        start_index = progress['current_index']
        
        for i in range(start_index, len(matched_candidates)):
            candidate = matched_candidates[i]
            candidate_id = Path(candidate['filename']).stem
            
            # Check if this candidate already has results in ./results/
            output_dir = Path("./results")
            suffix = "_holistic_evaluation.json" if use_holistic else "_evaluation.json"
            existing_result_path = output_dir / f"{candidate_id}{suffix}"
            
            if existing_result_path.exists():
                # Load existing result instead of re-evaluating
                status_text.text(f"Loading existing {i+1}/{len(matched_candidates)}: {candidate_id}")
                
                try:
                    with open(existing_result_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Build summary from existing data
                    if use_holistic:
                        # Get nested data safely
                        innovation = existing_data.get('innovation_potential', {})
                        program_fit = existing_data.get('program_fit', {})
                        
                        summary = {
                            'candidate_id': candidate_id,
                            'admit_status': candidate['admit_status'],
                            'overall_score': existing_data.get('overall_score', 0),
                            'recommendation': existing_data.get('recommendation', ''),
                            'innovation_potential': innovation.get('level', 'medium') if isinstance(innovation, dict) else 'medium',
                            'program_fit': program_fit.get('level', 'moderate') if isinstance(program_fit, dict) else 'moderate',
                            'interview_decision': existing_data.get('interview_decision', False),
                            'strengths': program_fit.get('strengths_for_program', []) if isinstance(program_fit, dict) else [],
                            'weaknesses': program_fit.get('concerns', []) if isinstance(program_fit, dict) else [],
                            'red_flags': [],
                            'notable_qualities': []
                        }
                    else:
                        summary = {
                            'candidate_id': candidate_id,
                            'admit_status': candidate['admit_status'],
                            'overall_score': existing_data.get('overall_score', 0),
                            'recommendation': existing_data.get('recommendation', ''),
                            'scores': {},
                            'strengths': existing_data.get('strengths', []),
                            'weaknesses': existing_data.get('areas_for_development', [])
                        }
                        # Extract scores from existing data
                        for score_data in existing_data.get('scores', []):
                            criterion = score_data.get('criterion', '')
                            score = score_data.get('score', 0)
                            if criterion:
                                summary['scores'][criterion] = score
                    
                    progress['candidate_summaries'].append(summary)
                    skipped_count += 1
                    skipped_container.info(f"Loaded {skipped_count} existing evaluations (skipping re-evaluation)")
                    
                except Exception as e:
                    # If loading fails, we'll re-evaluate
                    status_text.text(f"Re-evaluating {i+1}/{len(matched_candidates)}: {candidate_id} (load failed)")
            else:
                # No existing result - evaluate the candidate
                status_text.text(f"Evaluating {i+1}/{len(matched_candidates)}: {candidate_id}")
                
                try:
                    if use_holistic:
                        result = evaluator.evaluate_candidate_holistic(
                            candidate_id=candidate_id,
                            material_paths=[candidate['filepath']]
                        )
                        
                        # Build summary from holistic result
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
                            'red_flags': [
                                rf.flag if hasattr(rf, 'flag') else str(rf) 
                                for rf in result.red_flags[:3]
                            ],
                            'notable_qualities': [q.quality for q in result.notable_qualities[:5]]
                        }
                    else:
                        result = evaluator.evaluate_candidate(
                            candidate_id=candidate_id,
                            material_paths=[candidate['filepath']]
                        )
                        
                        # Build summary from criteria-based result
                        summary = {
                            'candidate_id': candidate_id,
                            'admit_status': candidate['admit_status'],
                            'overall_score': result.overall_score,
                            'recommendation': result.recommendation,
                            'scores': {
                                score.criterion.value: score.score
                                for score in result.scores
                            },
                            'strengths': result.strengths,
                            'weaknesses': result.areas_for_development
                        }
                    
                    progress['candidate_summaries'].append(summary)
                    
                    # Save individual result
                    output_dir.mkdir(exist_ok=True)
                    json_path = output_dir / f"{candidate_id}{suffix}"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result.model_dump(), f, indent=2, default=str)
                        
                except Exception as e:
                    error_msg = f"Failed to evaluate {candidate_id}: {e}"
                    progress['errors'].append(error_msg)
                    error_container.warning(error_msg)
            
            # Update progress
            progress['current_index'] = i + 1
            progress_bar.progress((i + 1) / len(matched_candidates))
            
            # Save progress to session state AND disk
            st.session_state['admit_analysis_progress'] = progress
            _save_progress_to_disk(progress)
            
            # Rerun to update UI and continue processing (prevents timeouts)
            # Only rerun if we actually did an API call (not for loaded results)
            if i < len(matched_candidates) - 1:
                if not existing_result_path.exists():
                    time.sleep(0.5)  # Brief pause to let UI update
                st.rerun()
    
    candidate_summaries = progress['candidate_summaries']
    status_text.text(f"Evaluated {len(candidate_summaries)}/{len(matched_candidates)} candidates")
    
    if len(candidate_summaries) < 2:
        st.error("Need at least 2 successfully evaluated candidates for pattern analysis")
        return
    
    # Phase 2: Run pattern analysis
    st.info("Running pattern analysis...")
    
    try:
        analyzer = AdmitPatternAnalyzer(
            api_key=config.api.anthropic_api_key,
            model=config.api.model,
            max_tokens=config.api.max_tokens
        )
        
        # First show basic statistics (no API call needed)
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
        
        # Show criterion differences if available
        if basic_stats.get('criterion_differences'):
            st.markdown("### Most Discriminating Criteria")
            for crit in basic_stats['criterion_differences'][:5]:
                name = crit['criterion'].replace('_', ' ').title()
                st.markdown(f"- **{name}**: {crit['difference']:.1f} point gap (Admitted: {crit['admitted_mean']:.1f}, Rejected: {crit['rejected_mean']:.1f})")
        
        # Run full Claude-powered analysis
        st.info("Running detailed pattern analysis with Claude...")
        pattern_result = analyzer.analyze_patterns(candidate_summaries)
        
        # Save result
        output_dir = Path("./results")
        result_path = output_dir / f"admit_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(pattern_result.model_dump(), f, indent=2, default=str)
        
        st.success(f"Analysis complete! Saved to {result_path}")
        
        # Mark as completed
        progress['completed'] = True
        st.session_state['admit_analysis_progress'] = progress
        _save_progress_to_disk(progress)
        
        # Store in session state for display
        st.session_state['last_pattern_analysis'] = pattern_result
        
        # Display results
        display_pattern_analysis_result(pattern_result)
        
    except Exception as e:
        st.error(f"Pattern analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_admit_pattern_results():
    """Display previous admit pattern analysis results from disk."""
    output_dir = Path("./results")
    
    pattern_files = list(output_dir.glob("admit_pattern_analysis_*.json"))
    
    if not pattern_files:
        st.info("No previous pattern analyses found. Run an analysis to see results here.")
        return
    
    # Sort by date (newest first)
    pattern_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    selected_file = st.selectbox(
        "Select previous analysis",
        pattern_files,
        format_func=lambda x: f"{x.stem} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
    )
    
    if selected_file:
        try:
            with open(selected_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct result object
            result = AdmitPatternAnalysisResult(
                analysis_date=datetime.fromisoformat(data.get('analysis_date', datetime.now().isoformat())),
                total_candidates=data.get('total_candidates', 0),
                admitted_count=data.get('admitted_count', 0),
                rejected_count=data.get('rejected_count', 0),
                admitted_mean_score=data.get('admitted_mean_score', 0),
                rejected_mean_score=data.get('rejected_mean_score', 0),
                score_difference=data.get('score_difference', 0),
                key_patterns=[],  # Simplified for display
                admitted_strengths=data.get('admitted_strengths', []),
                rejected_weaknesses=data.get('rejected_weaknesses', []),
                surprising_admits=data.get('surprising_admits', []),
                surprising_rejects=data.get('surprising_rejects', []),
                executive_summary=data.get('executive_summary', ''),
                methodology_notes=data.get('methodology_notes', ''),
                candidate_summaries=data.get('candidate_summaries', []),
                metadata=data.get('metadata', {})
            )
            
            display_pattern_analysis_result(result)
            
        except Exception as e:
            st.error(f"Error loading analysis: {e}")


def display_pattern_analysis_result(result: AdmitPatternAnalysisResult):
    """Display a pattern analysis result."""
    st.markdown("---")
    st.header("Pattern Analysis Results")
    
    # Executive Summary
    st.subheader("Executive Summary")
    st.markdown(result.executive_summary)
    
    # Score comparison
    st.subheader("Score Comparison")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyzed", result.total_candidates)
    with col2:
        st.metric("Admitted", result.admitted_count)
    with col3:
        st.metric("Rejected", result.rejected_count)
    with col4:
        st.metric("Score Gap", f"{result.score_difference:.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Admitted Mean Score", f"{result.admitted_mean_score:.2f}/10")
    with col2:
        st.metric("Rejected Mean Score", f"{result.rejected_mean_score:.2f}/10")
    
    # Key Patterns
    if result.key_patterns:
        st.subheader("Key Distinguishing Patterns")
        for category in result.key_patterns:
            importance_color = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(category.importance, '⚪')
            
            with st.expander(f"{importance_color} {category.category_name} ({category.importance.upper()} importance)"):
                st.markdown(category.description)
                
                if category.patterns:
                    for pattern in category.patterns:
                        st.markdown(f"**Pattern:** {pattern.pattern}")
                        
                        if pattern.admitted_examples:
                            st.markdown("*Admitted examples:*")
                            for ex in pattern.admitted_examples[:3]:
                                st.markdown(f"  - {ex}")
                        
                        if pattern.rejected_examples:
                            st.markdown("*Rejected counter-examples:*")
                            for ex in pattern.rejected_examples[:3]:
                                st.markdown(f"  - {ex}")
                        
                        st.markdown("---")
    
    # Strengths and Weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Common Strengths (Admitted)")
        if result.admitted_strengths:
            for strength in result.admitted_strengths:
                st.markdown(f"✓ {strength}")
        else:
            st.info("No common strengths identified")
    
    with col2:
        st.subheader("Common Weaknesses (Rejected)")
        if result.rejected_weaknesses:
            for weakness in result.rejected_weaknesses:
                st.markdown(f"✗ {weakness}")
        else:
            st.info("No common weaknesses identified")
    
    # Surprising cases
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Surprising Admits")
        if result.surprising_admits:
            for case in result.surprising_admits:
                st.warning(f"**{case.get('candidate_id', 'Unknown')}** (Score: {case.get('score', 'N/A')})")
                st.caption(case.get('reason', ''))
                if case.get('possible_explanation'):
                    st.caption(f"Possible explanation: {case.get('possible_explanation')}")
        else:
            st.info("No surprising admits identified")
    
    with col2:
        st.subheader("Surprising Rejects")
        if result.surprising_rejects:
            for case in result.surprising_rejects:
                st.warning(f"**{case.get('candidate_id', 'Unknown')}** (Score: {case.get('score', 'N/A')})")
                st.caption(case.get('reason', ''))
                if case.get('possible_explanation'):
                    st.caption(f"Possible explanation: {case.get('possible_explanation')}")
        else:
            st.info("No surprising rejects identified")
    
    # Predictive factors from metadata
    if result.metadata.get('predictive_factors'):
        st.subheader("Predictive Factors")
        for factor in result.metadata['predictive_factors']:
            strength_icon = {'strong': '💪', 'moderate': '👍', 'weak': '👌'}.get(factor.get('strength', ''), '•')
            direction = factor.get('direction', '')
            st.markdown(f"{strength_icon} **{factor.get('factor', '')}**: {direction}")
            st.caption(factor.get('evidence', ''))
    
    # Methodology notes
    if result.methodology_notes:
        with st.expander("Methodology Notes"):
            st.markdown(result.methodology_notes)
    
    # Raw candidate data
    with st.expander("View All Candidate Data"):
        if result.candidate_summaries:
            # Convert to dataframe for display
            display_data = []
            for c in result.candidate_summaries:
                row = {
                    'Candidate ID': c.get('candidate_id', ''),
                    'Admit Status': 'Admitted' if c.get('admit_status') else 'Rejected',
                    'Score': c.get('overall_score', 0),
                    'Recommendation': c.get('recommendation', '')
                }
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            df = df.sort_values('Score', ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Download results
    st.download_button(
        label="Download Analysis (JSON)",
        data=json.dumps(result.model_dump(), indent=2, default=str),
        file_name=f"admit_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def _display_pattern_analysis_from_dict(data: dict):
    """Display pattern analysis from a dictionary (loaded from disk)."""
    st.markdown("---")
    st.header("Pattern Analysis Results")
    
    # Executive Summary
    st.subheader("Executive Summary")
    st.markdown(data.get('executive_summary', 'No summary available'))
    
    # Score comparison
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
    
    # Key Patterns
    key_patterns = data.get('key_patterns', [])
    if key_patterns:
        st.subheader("Key Distinguishing Patterns")
        for category in key_patterns:
            importance = category.get('importance', 'medium')
            importance_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(importance, '⚪')
            
            with st.expander(f"{importance_color} {category.get('category_name', 'Unknown')} ({importance.upper()} importance)"):
                st.markdown(category.get('description', ''))
                
                for pattern in category.get('patterns', []):
                    st.markdown(f"**Pattern:** {pattern.get('pattern', '')}")
                    
                    admitted_examples = pattern.get('admitted_examples', [])
                    if admitted_examples:
                        st.markdown("*Admitted examples:*")
                        for ex in admitted_examples[:3]:
                            st.markdown(f"  - {ex}")
                    
                    rejected_examples = pattern.get('rejected_examples', [])
                    if rejected_examples:
                        st.markdown("*Rejected counter-examples:*")
                        for ex in rejected_examples[:3]:
                            st.markdown(f"  - {ex}")
                    
                    st.markdown("---")
    
    # Strengths and Weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Common Strengths (Admitted)")
        admitted_strengths = data.get('admitted_strengths', [])
        if admitted_strengths:
            for strength in admitted_strengths:
                st.markdown(f"✓ {strength}")
        else:
            st.info("No common strengths identified")
    
    with col2:
        st.subheader("Common Weaknesses (Rejected)")
        rejected_weaknesses = data.get('rejected_weaknesses', [])
        if rejected_weaknesses:
            for weakness in rejected_weaknesses:
                st.markdown(f"✗ {weakness}")
        else:
            st.info("No common weaknesses identified")
    
    # Surprising cases
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Surprising Admits")
        surprising_admits = data.get('surprising_admits', [])
        if surprising_admits:
            for case in surprising_admits:
                st.warning(f"**{case.get('candidate_id', 'Unknown')}** (Score: {case.get('score', 'N/A')})")
                st.caption(case.get('reason', ''))
                if case.get('possible_explanation'):
                    st.caption(f"Possible explanation: {case.get('possible_explanation')}")
        else:
            st.info("No surprising admits identified")
    
    with col2:
        st.subheader("Surprising Rejects")
        surprising_rejects = data.get('surprising_rejects', [])
        if surprising_rejects:
            for case in surprising_rejects:
                st.warning(f"**{case.get('candidate_id', 'Unknown')}** (Score: {case.get('score', 'N/A')})")
                st.caption(case.get('reason', ''))
                if case.get('possible_explanation'):
                    st.caption(f"Possible explanation: {case.get('possible_explanation')}")
        else:
            st.info("No surprising rejects identified")
    
    # Methodology notes
    methodology_notes = data.get('methodology_notes', '')
    if methodology_notes:
        with st.expander("Methodology Notes"):
            st.markdown(methodology_notes)
    
    # Raw candidate data
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
            
            df = pd.DataFrame(display_data)
            df = df.sort_values('Score', ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    # Download results
    st.download_button(
        label="Download Analysis (JSON)",
        data=json.dumps(data, indent=2, default=str),
        file_name=f"admit_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def research_page():
    """Research dashboard for comparing evaluation methods."""
    st.title("Research Dashboard")
    st.markdown("Compare evaluation methods and analyze AI performance for research purposes.")

    output_dir = Path("./results")
    criteria_results = load_all_results_from_disk(output_dir)
    holistic_results = load_all_holistic_results_from_disk(output_dir)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Criteria-Based Evaluations", len(criteria_results))
    with col2:
        st.metric("Holistic Evaluations", len(holistic_results))
    with col3:
        # Find candidates with both types
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
        research_export_report(criteria_results, holistic_results, output_dir)


def research_method_comparison(criteria_results, holistic_results):
    """Compare criteria-based vs holistic evaluation methods."""
    st.subheader("Evaluation Method Comparison")

    # Find candidates evaluated with both methods
    criteria_by_id = {r.candidate.candidate_id: r for r in criteria_results}
    holistic_by_id = {r.candidate.candidate_id: r for r in holistic_results}
    both_ids = set(criteria_by_id.keys()) & set(holistic_by_id.keys())

    if len(both_ids) == 0:
        st.info("No candidates have been evaluated with both methods. Run evaluations with both Criteria-Based and Holistic modes on the same candidates to see comparisons.")

        # Show individual method stats if available
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

    # Build comparison data
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

    # Agreement metrics
    st.markdown("### Agreement Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Score correlation
        if len(df) > 1:
            correlation = df['criteria_score'].corr(df['holistic_score'])
            st.metric("Score Correlation", f"{correlation:.3f}")
        else:
            st.metric("Score Correlation", "N/A (need more data)")

        # Mean absolute difference
        mad = df['score_diff'].abs().mean()
        st.metric("Mean Absolute Difference", f"{mad:.2f} points")

    with col2:
        # Scores higher/lower/same
        higher = sum(1 for d in df['score_diff'] if d > 0.5)
        lower = sum(1 for d in df['score_diff'] if d < -0.5)
        similar = len(df) - higher - lower

        st.markdown("**Score Comparison:**")
        st.markdown(f"- Criteria higher: {higher} ({higher/len(df)*100:.1f}%)")
        st.markdown(f"- Holistic higher: {lower} ({lower/len(df)*100:.1f}%)")
        st.markdown(f"- Similar (within 0.5): {similar} ({similar/len(df)*100:.1f}%)")

    # Scatter plot comparison
    st.markdown("### Score Comparison Chart")
    chart_df = df[['criteria_score', 'holistic_score']].copy()
    chart_df.columns = ['Criteria-Based', 'Holistic']
    st.scatter_chart(chart_df)

    # Detailed comparison table
    st.markdown("### Detailed Comparison")
    display_df = df[['candidate_id', 'criteria_score', 'holistic_score', 'score_diff', 'holistic_interview']].copy()
    display_df.columns = ['Candidate', 'Criteria Score', 'Holistic Score', 'Difference', 'Interview Rec']
    display_df = display_df.sort_values('Difference', key=abs, ascending=False)
    st.dataframe(display_df, hide_index=True, use_container_width=True)


def research_side_by_side(criteria_results, holistic_results):
    """Side-by-side view of individual candidate evaluations."""
    st.subheader("Side-by-Side Candidate View")

    # Get all unique candidate IDs
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
                    st.warning(flag)

            with st.expander("Full Assessment"):
                st.markdown(hr.overall_assessment)
        else:
            st.info("No holistic evaluation for this candidate.")


def research_export_report(criteria_results, holistic_results, output_dir):
    """Export research comparison report."""
    st.subheader("Export Research Report")

    # Build report data
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

        # Save report
        report_path = output_dir / f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        st.success(f"Report saved to: {report_path}")

        # Show preview
        st.markdown("### Report Preview")
        st.markdown(report_content)

        # Download button
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


def settings_page():
    """Settings page."""
    st.title("Settings")

    config = st.session_state.config

    st.subheader("API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Model", value=config.api.model, disabled=True)
        st.number_input("Max Tokens", value=config.api.max_tokens, disabled=True)

    with col2:
        st.slider("Temperature", 0.0, 1.0, float(config.api.temperature), disabled=True)

    st.info("Edit config.yaml to modify settings, then restart the application.")

    st.subheader("Criteria Weights")

    weights = config.criteria.weights
    criteria = ['critical_thinking', 'coachability', 'curiosity', 'creativity',
                'collaboration', 'follow_through', 'problem_solving_motivation',
                'evidence_based', 'detail_orientation', 'communication', 'expertise_enabler']

    weight_data = []
    for c in criteria:
        weight_data.append({
            'Criterion': c.replace('_', ' ').title(),
            'Weight': getattr(weights, c, 10)
        })

    st.dataframe(pd.DataFrame(weight_data), hide_index=True, use_container_width=True)

    st.markdown("---")

    st.subheader("Job Management")

    job_manager = st.session_state.job_manager

    col1, col2 = st.columns(2)

    with col1:
        st.text(f"Jobs directory: {job_manager.jobs_dir}")
        st.text(f"Logs directory: {job_manager.logs_dir}")

    with col2:
        if st.button("Cleanup Old Jobs"):
            job_manager.cleanup_old_jobs(days=7)
            st.success("Cleaned up jobs older than 7 days")

    st.markdown("---")

    # Prompt Editor Section
    st.subheader("Prompt Editor")
    st.caption("View and customize the prompts used for candidate evaluation.")

    # Initialize prompt manager
    if 'prompt_manager' not in st.session_state:
        st.session_state.prompt_manager = PromptManager()

    prompt_manager = st.session_state.prompt_manager

    # Prompt type selector
    prompt_options = {
        "System Prompt": "system",
        "Criteria-Based Template": "criteria",
        "Holistic Template": "holistic",
        "Interview Ranking Template": "ranking",
        "Interview Selection Template": "selection",
        "Clinician Role Prompt": "clinician",
        "Engineer / Tech Role Prompt": "engineer",
        "PhD Role Prompt": "phd",
    }

    selected_prompt_name = st.selectbox(
        "Select Prompt to View/Edit",
        list(prompt_options.keys()),
        key="prompt_selector"
    )
    selected_prompt_type = prompt_options[selected_prompt_name]

    # Get current prompt and metadata
    metadata = prompt_manager.get_prompt_metadata(selected_prompt_type)

    # Status indicator
    if metadata["is_custom"]:
        st.info(f"Status: **Custom** (modified {metadata['updated_at'][:10] if metadata['updated_at'] else 'unknown'})")
    else:
        st.success("Status: **Default** (using built-in prompt)")

    if selected_prompt_type in PromptManager.ROLE_PROMPT_TYPES:
        st.caption(
            "This role prompt is **appended** to the System Prompt during role-specific evaluations. "
            "It works with both Criteria-Based and Holistic modes."
        )

    # Get current content
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
    elif selected_prompt_type == "clinician":
        current_content = prompt_manager.get_clinician_prompt()
    elif selected_prompt_type == "engineer":
        current_content = prompt_manager.get_engineer_prompt()
    else:
        current_content = prompt_manager.get_phd_prompt()

    # Text area for editing
    edited_content = st.text_area(
        f"Edit {selected_prompt_name}",
        value=current_content,
        height=400,
        key=f"prompt_editor_{selected_prompt_type}"
    )

    # Character count
    st.caption(f"Character count: {len(edited_content):,}")

    # Action buttons
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

    # Show placeholder info for templates
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

**Note:** Do not remove these placeholders. The candidate profiles are generated automatically
from each candidate's holistic evaluation and include score, program fit, innovation potential,
red flags, and key reasoning excerpts.
""")
            elif selected_prompt_type == "selection":
                st.markdown("""
**Available placeholders:**
- `{n_pool}` — Number of candidates in the final pool (required)
- `{n_interviews}` — Number of candidates to select (required)
- `{candidates_data}` — Formatted candidate profiles, including notable qualities (required)

**Note:** Do not remove these placeholders. Candidate profiles are generated automatically
from each candidate's holistic evaluation.
""")


if __name__ == "__main__":
    main()
