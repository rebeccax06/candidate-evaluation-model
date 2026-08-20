"""Shared Streamlit UI for the Screening feature.

Both web apps (local ``web_app.py`` and cloud ``web_app_cloud.py``) render the
same screening experience; only where jobs are submitted and results are
stored differs. The widgets live here so the two apps cannot drift.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st

from candidate_evaluator.core.models import (
    ScreeningResult,
    SCREENING_REVIEW_CONFIDENCE,
    SCREENING_OUTCOME_MATCH,
    SCREENING_OUTCOME_REVIEW,
    SCREENING_OUTCOME_NO_MATCH,
    SCREENING_OUTCOME_LABELS,
)

DEFAULT_SCREENING_DESCRIPTION = (
    "Primary capability is AI or Computer Science and does not have any "
    "experience with healthcare or medical domains."
)

SCREENING_PAGE_INTRO = (
    "Upload a batch of candidate PDFs and describe the target profile. Each "
    "candidate is screened in the background and sorted into **match**, "
    "**needs review** (the model isn't confident enough to decide), or "
    "**no match** — so you can leave this page while it runs. Watch progress "
    "under **Batch Jobs**, then come back to the **Results** tab here."
)

# Display order for outcome groups (best first)
_OUTCOME_ORDER = {
    SCREENING_OUTCOME_MATCH: 0,
    SCREENING_OUTCOME_REVIEW: 1,
    SCREENING_OUTCOME_NO_MATCH: 2,
}

_SHOW_FILTERS = {
    "Matches + needs review": {SCREENING_OUTCOME_MATCH, SCREENING_OUTCOME_REVIEW},
    "Matches only": {SCREENING_OUTCOME_MATCH},
    "Everything": set(_OUTCOME_ORDER),
}


def render_screening_submit_inputs(key_prefix: str = "screening_") -> Optional[dict]:
    """Render the screening submit inputs (description, PDFs, job name).

    Returns ``{"description", "uploaded_files", "job_name"}`` once the start
    button is clicked with valid inputs; otherwise None. The caller owns
    everything after that (job creation, storage, rerun).
    """
    st.markdown("### Screen Candidates")

    description = st.text_area(
        "Target profile description",
        value=DEFAULT_SCREENING_DESCRIPTION,
        height=120,
        help=(
            "Describe the candidate you're looking for. You can include both "
            "requirements (must have) and exclusions (must not have)."
        ),
        key=f"{key_prefix}description",
    )

    uploaded_files = st.file_uploader(
        "Upload Candidate PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one PDF per candidate. Filename becomes the candidate ID.",
        key=f"{key_prefix}uploader",
    )

    if not uploaded_files:
        return None

    st.markdown(f"**{len(uploaded_files)} files selected**")
    with st.expander("View files"):
        for f in uploaded_files:
            st.text(f"- {f.name}")

    job_name = st.text_input(
        "Job Name (optional)",
        placeholder="e.g., AI/CS screen",
        key=f"{key_prefix}job_name",
    )

    start_clicked = st.button(
        "Start Screening (Background)",
        use_container_width=True,
        key=f"{key_prefix}start_btn",
    )

    if not start_clicked:
        return None

    if not description.strip():
        st.error("Please provide a target profile description.")
        return None

    return {
        "description": description.strip(),
        "uploaded_files": uploaded_files,
        "job_name": job_name,
    }


def render_screening_results(results: List[ScreeningResult], key_prefix: str = "screening_") -> None:
    """Render the three-way screening results view for a list of results.

    The caller loads the results (from disk locally, from Supabase in the
    cloud) and applies any storage-specific filtering (e.g. by job) first.
    """
    if not results:
        st.info("No screening results yet. Run a screening job to see results here.")
        return

    descriptions = {r.description for r in results if r.description}
    if len(descriptions) == 1:
        st.caption(f"Target profile: {next(iter(descriptions))}")

    review_below = st.slider(
        "Review threshold",
        0.0, 1.0, SCREENING_REVIEW_CONFIDENCE, 0.05,
        help=(
            "Decisions with confidence below this are shown as 'Needs review' "
            "instead of a hard yes/no — the model's confidence is not "
            "calibrated, so borderline calls should get a human look."
        ),
        key=f"{key_prefix}review_threshold",
    )

    outcomes = {id(r): r.outcome(review_below=review_below) for r in results}
    counts = {k: 0 for k in _OUTCOME_ORDER}
    for outcome in outcomes.values():
        counts[outcome] += 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Screened", len(results))
    with col2:
        st.metric("Matches", counts[SCREENING_OUTCOME_MATCH])
    with col3:
        st.metric("Needs Review", counts[SCREENING_OUTCOME_REVIEW])
    with col4:
        st.metric("No Match", counts[SCREENING_OUTCOME_NO_MATCH])

    show_choice = st.radio(
        "Show",
        list(_SHOW_FILTERS.keys()),
        horizontal=True,
        key=f"{key_prefix}show_filter",
    )
    visible_outcomes = _SHOW_FILTERS[show_choice]

    filtered = [r for r in results if outcomes[id(r)] in visible_outcomes]
    filtered.sort(key=lambda r: (_OUTCOME_ORDER[outcomes[id(r)]], -r.confidence))

    if not filtered:
        st.warning("No candidates match the current filters.")
        return

    rows = []
    for r in filtered:
        rows.append({
            "Candidate": r.candidate.candidate_id,
            "Outcome": SCREENING_OUTCOME_LABELS[outcomes[id(r)]],
            "Confidence": f"{r.confidence:.2f}",
            "Reasoning": r.reasoning,
            "Disqualifiers": "; ".join(r.disqualifiers),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key=f"{key_prefix}csv_btn",
    )

    with st.expander("View evidence details"):
        for r in filtered:
            label = SCREENING_OUTCOME_LABELS[outcomes[id(r)]]
            st.markdown(f"**{r.candidate.candidate_id}** - {label} ({r.confidence:.2f})")
            if r.reasoning:
                st.caption(r.reasoning)
            if r.supporting_evidence:
                st.markdown("Supporting evidence:")
                for ev in r.supporting_evidence:
                    st.markdown(f"- {ev}")
            if r.disqualifiers:
                st.markdown("Disqualifiers:")
                for d in r.disqualifiers:
                    st.markdown(f"- {d}")
            st.markdown("---")
