"""Shared Guide page and Help chatbot for the candidate evaluator web apps.

This module centralizes:
- A human-readable "Guide" page that explains how the evaluation prompts work.
- A "Help" chatbot that answers user questions about the tool using the same
  Anthropic API key / model the evaluator is already configured with.

Both the local (`web_app.py`) and cloud (`web_app_cloud.py`) interfaces import
from here so the content and behavior stay in sync.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

import streamlit as st


# --------------------------------------------------------------------------- #
# Guide content
# --------------------------------------------------------------------------- #

GUIDE_OVERVIEW = """
This tool reads a candidate's application materials (resume, cover letter,
recommendation letters, etc.) and asks an AI model to act like an experienced,
skeptical reviewer. The **prompts** tell the AI *what to look for*, *how strict
to be*, and *what format to answer in*.

There are two layers of instruction working together on every evaluation:

- **System prompt** — the AI's standing "job description" (persona, values, and
  hard rules). Always sent first, before it ever sees a candidate.
- **Task prompt** — the specific assignment for a given review (the Holistic
  evaluation, optionally with a Role-dimensions lens layered on top).
"""

GUIDE_HOLISTIC = """
**Variable:** `HOLISTIC_EVALUATION_PROMPT` (in `evaluation_prompts.py`)

The holistic prompt asks the AI to return a complete, evidence-rich assessment:

- A **3–4 paragraph** overall assessment
- An **innovation potential** rating (high / medium / low) with evidence
- A **program fit** rating (strong / moderate / weak) with specific strengths
  and concerns
- A list of **notable qualities**
- **Red flags** — including calling out "fluff", such as generic claims like
  *"passionate about innovation"*
- Suggested **interview questions**
- A final **score (1–10)** and a **yes/no** interview recommendation

**It is deliberately skeptical.** The instructions tell the AI to assume most
applications are mostly filler, to demand a real quote from the documents for
every positive claim, and to never give the candidate the benefit of the doubt.
If something is *claimed* but not *shown*, the AI must say so.
"""

GUIDE_ROLE_INTRO = """
The role-dimensions prompt **does not run by itself.** It gets attached *on top
of* the holistic evaluation. Its job is to add a **specialized lens**: every
candidate is scored across **all** role dimensions (clinical, engineering, and
research) at once, using one combined prompt.
"""

GUIDE_ROLE_CLINICAL = """
**Variable:** `CLINICAL_PROMPT`

The AI focuses on whether the person spotted **real unmet clinical needs**,
understood the **surrounding system**, learned from **patients and staff**, and
**took action.**

**Clinical dimensions**

**1. `challenge_complexity`** — How hard/significant was the clinical problem the
candidate actually engaged with?
- **not_shown:** No specific clinical problem is described.
- **limited:** A routine or low-difficulty problem, or one only witnessed in passing.
- **moderate:** A genuine problem requiring real clinical or operational reasoning.
- **high:** A hard, significant problem affecting many patients/providers or the system.
- **exceptional:** An unusually complex, high-stakes problem the candidate meaningfully engaged with.

**2. `need_investigation_stage`** — How far did they go from noticing a problem to
doing something about it?
- **not_shown:** No clinical problem or engagement with one is described.
- **witnessed_only:** Saw or experienced the problem but took no further step.
- **need_identified:** Clearly framed an unmet need, distinct from its visible symptoms.
- **investigated:** Explored root causes and stakeholder perspectives.
- **intervention_proposed:** Proposed a concrete intervention or credible next step.
- **tested_or_implemented:** Tested or implemented an intervention.
- **measured:** Measured the effect of the intervention.
- **sustained_impact:** Achieved sustained adoption or broader impact.

**3. `systems_thinking`** — Did they understand the broader system around the
problem (workflow, staffing, cost, equity, incentives), not just one isolated
patient interaction?
- **not_shown:** No consideration of the surrounding system.
- **limited:** Minimal awareness of one or two contextual factors.
- **moderate:** Considered several system factors around the problem.
- **strong:** Rich understanding of how workflow, staffing, cost, equity, incentives, and adoption interact.
"""

GUIDE_ROLE_ENGINEERING = """
**Variable:** `ENGINEERING_PROMPT`

The AI focuses on **what the person actually built**, how far it got
(idea → prototype → tested → deployed → used by real people), **what they
personally owned**, and whether they built around a **real need** rather than
falling in love with a technology.

**Engineering / Tech dimensions**

**1. `build_stage`** — How far did the thing they built actually get? (A
progression from idea to lasting product.)
- **idea_only:** Described or proposed an idea.
- **designed:** Produced a design, specification, or architecture.
- **prototype_built:** Created a functioning initial artifact.
- **tested:** Evaluated the artifact against defined criteria.
- **iterated:** Changed the artifact based on test results or feedback.
- **deployed:** Used the artifact in a real environment.
- **used_by_real_users:** Documented use or adoption by intended users.
- **scaled_or_sustained:** Maintained, expanded, or operated the artifact over time.

**2. `ownership_clarity`** — How clearly was this candidate (vs. the team)
responsible for the work?
- **unclear:** Only vague involvement with no described individual role.
- **supporting_contributor:** Helped on components owned or directed by others.
- **substantial_contributor:** Personally designed or implemented significant parts.
- **primary_driver:** Personally drove the core technical work and key decisions.

**3. `user_grounding`** — Did they build around a real, understood user need
(interviews, real workflows, user testing) rather than just building technology
for its own sake?
- **not_shown:** No evidence of a real use case or user input.
- **limited:** Assumed a need with little validation.
- **moderate:** Some user or stakeholder input shaped the work.
- **strong:** Requirements grounded in real workflows, changes driven by stakeholder evidence, and validated value.
"""

GUIDE_ROLE_RESEARCH = """
**Variable:** `RESEARCH_PROMPT`

The AI focuses on the **depth and ownership** of their research: did they
originate the question, design the methods, and drive the work — vs. just being
listed on a paper?

**Research dimensions**

**1. `research_evidence_level`** — Overall strength of the evidence that they did
real, meaningful research.
- **none:** No research is documented.
- **limited:** Some involvement, but contribution or rigor is unclear.
- **moderate:** Clear substantive contribution to credible completed work.
- **strong:** Substantial, rigorous, clearly owned research.
- **exceptional:** Rigorous, field-relevant research with clearly demonstrated ownership.

**2. `publication_strength`** — Quality and significance of their research outputs
(judged by contribution and impact, not just count/prestige).
- **none_or_not_shown:** No research output is documented.
- **limited:** Output exists, but contribution or significance is unclear.
- **moderate:** Clear substantive contribution to credible completed work.
- **strong:** Multiple substantial contributions, or one highly significant, externally validated contribution.
- **exceptional:** Field-shaping or unusually influential work with clearly demonstrated ownership.

**3. `research_ownership`** — How much they personally drove the research (vs.
being a name on the lab's work).
- **unclear:** Ownership cannot be distinguished from the lab's or team's.
- **supporting_contributor:** Assisted work directed by others.
- **substantial_contributor:** Independently owned a major research component.
- **primary_driver:** Originated or reframed the question and drove the project's direction.
"""

GUIDE_SYSTEM_PROMPT = """
**Variable:** `SYSTEM_PROMPT`

If the holistic and role-based prompts are the *specific task instructions* for a
given review, the system prompt is the AI's **standing job description**. It
defines *who you are and how you must behave* and applies to **every**
evaluation, no matter which mode is used.

**Main themes**

- **Evidence only.** Every score must be backed by a direct quote from the
  candidate's materials. No assumptions, no inferring qualities that aren't shown.
- **Reject "fluff."** Stated qualities like *"I am creative," "passionate,"
  "strong work ethic,"* or generic supervisor praise are treated as
  near-worthless (score 1–2) unless backed by concrete actions, outcomes, or
  specific incidents.
- **Stated vs. demonstrated.** Saying you have a quality counts for little;
  *showing* it with a specific result counts.
- **Use the full 1–10 scale.** It explicitly tells the AI **not** to cluster
  everything in the safe middle (5–7): score low when evidence is weak and high
  when it's genuinely strong.
- **Be conservative when unsure.** If evidence is ambiguous, vague, or written in
  passive voice (*"improvements were made"* — by whom?), the AI must score
  **lower**, not give the benefit of the doubt.
- **No hallucination.** Only cite things that actually appear in the documents,
  quoted verbatim, with the source filename.
- Includes **worked calibration examples** (❌ bad vs. ✅ good evidence, and how a
  real expert would score borderline cases) so the AI's strictness matches how an
  experienced human reviewer would actually rate candidates.

**When it's used:** Always. It's automatically sent at the start of every
evaluation (holistic, and holistic with the role-dimensions lens). You don't
pick it — it's the constant backdrop that keeps every review skeptical,
evidence-driven, and consistently calibrated.
"""

GUIDE_INNOVATION = """
**Where:** in the holistic prompt.

The model is largely left to **define "innovation" itself** — the prompt only
tells it to provide a level, a confidence, reasoning, and evidence.

**The instruction:**

> **2. Innovation Potential Assessment** (detailed)
> - Level: high, medium, or low
> - Confidence: high, medium, or low (based on evidence quality)
> - Detailed reasoning (2+ paragraphs explaining your assessment)
> - Minimum 3 pieces of evidence with quotes, sources, and context

**The output-schema definition:**

```json
"innovation_potential": {
  "level": "high|medium|low",
  "confidence": "high|medium|low",
  "reasoning": "2+ paragraphs explaining the assessment with specific examples",
  "evidence": [
    {
      "quote": "Exact quote from materials",
      "source": "document_name.pdf",
      "context": "What this demonstrates and why it matters"
    }
  ]
}
```
"""

GUIDE_PROGRAM_FIT = """
**Where:** in the holistic prompt.

**Program fit** is defined entirely by reference to the **program's goals** — the
program-description bullets below. In other words, program fit literally means
*"how well does the evidence show this candidate matches those goals?"* Changing
the program description directly changes what "program fit" measures.

**The instruction:**

> **3. Program Fit Assessment** (detailed)
> - Level: strong, moderate, or weak
> - Confidence: high, medium, or low
> - Detailed analysis (2+ paragraphs)
> - Specific strengths aligned with program goals
> - Specific concerns or gaps
> - Minimum 3 pieces of evidence

**The output-schema definition:**

```json
"program_fit": {
  "level": "strong|moderate|weak",
  "confidence": "high|medium|low",
  "detailed_analysis": "2+ paragraphs analyzing fit with program goals",
  "strengths_for_program": ["Specific strength 1", "Specific strength 2"],
  "concerns": ["Specific concern 1", "Specific concern 2"],
  "evidence": [
    {
      "quote": "Exact quote from materials",
      "source": "document_name.pdf",
      "context": "What this demonstrates"
    }
  ]
}
```
"""

GUIDE_PROGRAM_DESCRIPTION = """
You are evaluating a candidate for an innovation and research program. Rather
than scoring on specific pre-defined criteria, you will provide an in-depth
holistic assessment based on the program's goals and the candidate's
demonstrated qualities.

This evaluation should be **THOROUGH** and **EVIDENCE-RICH** — comparable in
depth to a structured criteria-based evaluation.

**Program Description**

This is a competitive fellowship program seeking candidates who can:

- Drive innovation and creative problem-solving in their field
- Translate research into practical impact
- Work effectively across disciplines and with diverse teams
- Demonstrate sustained commitment to challenging problems
- Learn, grow, and adapt based on feedback and new evidence
- Communicate complex ideas effectively to varied audiences

The program values candidates who show genuine evidence of these qualities
through their actions and achievements, not just stated intentions.
"""


# --------------------------------------------------------------------------- #
# "How to use the app" content (cloud web app)
# --------------------------------------------------------------------------- #
# These sections describe the actual click-by-click flows in the hosted (cloud)
# web app so both the Guide page and the Help chatbot can walk users through
# real tasks. Page names match the sidebar exactly: Dashboard, New Evaluation,
# Batch Jobs, Results, Analysis, Guide, Help, Settings.

GUIDE_APP_GETTING_STARTED = """
The app is organized as a set of pages you switch between using the **sidebar
on the left**. The pages are: **Dashboard**, **New Evaluation**, **Batch Jobs**,
**Results**, **Analysis**, **Guide**, **Help**, and **Settings**.

**Before you can evaluate anyone, two things must be set up:**

1. **Sign in.** The app requires an account. On the sign-in screen use the
   **Sign In** tab (email + password), or the **Sign Up** tab to create an
   account (email, password, confirm password — password must be at least 6
   characters). Your evaluations, batch jobs, and API key are private to your
   account.
2. **Add your Anthropic API key.** The first time you sign in you'll be asked
   for an **Anthropic API Key** (it starts with `sk-ant-`). Paste it and click
   **Save API Key**. You can change it later on the **Settings** page under
   **Update Anthropic API Key**. Nothing can be evaluated until a valid key is
   saved, and the **Help** chat also needs it.

The **Dashboard** is the landing page. It shows summary metrics (Total
Evaluations, Active Jobs, Avg Score, Completed Jobs), your most recent
evaluations, and any batch jobs currently running. It's read-only — a place to
get your bearings, not to start work.
"""

GUIDE_APP_NEW_EVALUATION = """
This is where you actually run evaluations. Go to **New Evaluation** in the
sidebar. At the top you'll see three tabs — pick the one that matches your
situation:

- **Single Candidate** — evaluate one person, and upload several documents for
  them (resume + cover letter + letters, etc.).
- **Batch Upload** — evaluate many candidates at once, one PDF per candidate.
- **Role Dimensions Evaluation** — same as the above, but every candidate is also
  scored across the combined role dimensions (clinical + engineering + research).

Every evaluation runs as a **Holistic** evaluation (overall program fit,
innovation potential, notable qualities, red flags, suggested interview
questions, a 1–10 score, and a yes/no interview recommendation).

**To evaluate a single candidate (Single Candidate tab):**

1. *(Optional)* **Add to existing batch** — leave as **"— Don't add to a
   batch —"** unless you want this candidate grouped with an existing batch.
2. *(Optional)* enter a **Name**.
3. **Upload Application Materials** — you can drop in multiple files. Accepted
   types: **PDF, DOCX, TXT, MD**. A **candidate ID is generated automatically**
   from the uploaded file name (no need to type one).
4. Click **Evaluate Candidate**. It takes about a minute; the result appears
   right below and is saved to your account (view it later under **Results**).

**To evaluate many candidates at once (Batch Upload tab):**

1. *(Optional)* **Add to existing batch**, otherwise leave the default.
2. **Upload Candidate PDFs** — **PDF only**, one PDF per candidate. Each file's
   name (without `.pdf`) becomes that candidate's ID.
3. *(Optional)* Give the batch a **Job Name** (e.g. `Spring 2026 Applicants`).
4. Start it. You'll usually see two buttons:
   - **Start Batch Evaluation (Background)** — recommended. Runs on the server;
     you can close the page and check progress later under **Batch Jobs**.
   - **Run Sequentially Now** — runs in this browser tab and you must keep it
     open. Use this only if the background worker isn't processing.

**Role Dimensions Evaluation tab:** works exactly like the two flows above (it
has its own **Single Candidate** and **Batch Upload** sub-tabs), but it layers
the combined role prompt on top so you also get clinical/engineering/research
dimension scores. There is no separate picker for one role — every candidate is
scored on **all** dimensions at once.
"""

GUIDE_APP_CHOOSING = """
There are a few choices on **New Evaluation**, and the best one depends on what
you're trying to do. When in doubt, here's how to decide:

**Single Candidate vs. Batch Upload**
- Choose **Single Candidate** when you have **one** applicant, especially if
  their materials are split across several files (resume + cover letter + letters)
  or you want to set a specific name.
- Choose **Batch Upload** when you have **many** applicants and each one is a
  single PDF. It's the fast path for a whole applicant pool.

**When to use Role Dimensions Evaluation**
- Use it when a candidate's strength is tied to *what they built / researched /
  did clinically* and you want those dimensions scored. It runs on top of the
  holistic evaluation and powers the **Role Rankings** tab in **Results**.

**Background vs. Run Sequentially Now (batch only)**
- **Background** is almost always the right choice — it survives closing the
  tab and you monitor it in **Batch Jobs**.
- **Run Sequentially Now** only when the background worker isn't running and you
  can leave the tab open until it finishes.
"""

GUIDE_APP_BATCH_JOBS = """
**Batch Jobs** is where you **monitor** batch runs — it is *not* where you
upload files (that's **New Evaluation → Batch Upload**).

- Tabs: **Active**, **Completed**, and **All Jobs**.
- Each job card shows its name, status (PENDING / PROCESSING / COMPLETED /
  FAILED / CANCELLED), a progress bar (`completed / total`), and, while running,
  which candidate is being evaluated.
- **Auto-refresh** is on by default so progress updates itself.
- You can **Cancel** a job that's still running.
- On completed jobs, open **View Results Summary** for a quick table
  (Candidate, Score, Recommendation). Full details live on the **Results** page.
"""

GUIDE_APP_RESULTS = """
**Results** is the library of every evaluation you've run.

- Use the **Display mode** selector to either **split by batch** (one section
  per job) or show **everything together**.
- Results are organized into tabs: **Holistic** and **Role Rankings**
  (candidates run through Role Dimensions Evaluation).
- Each tab shows a ranked table; click into a candidate to see the full
  write-up (assessment, evidence quotes, strengths/concerns, red flags, suggested
  interview questions, and — for role-dimensions runs — a dimension heatmap).
"""

GUIDE_APP_ANALYSIS = """
**Analysis** gives you statistics across your holistic evaluations. Pick a scope
first (**everything together** or **by batch**), then view the
**Distribution Analysis**: the score distribution (mean/median/min/max), a score
histogram, interview-recommendation rate, and innovation/program-fit breakdowns.
"""

GUIDE_APP_SETTINGS = """
**Settings** is where you manage your account and configuration.

- **Account** — your email and user ID.
- **API Key** — see whether a key is configured and **Update Anthropic API Key**.
- **Data** — total counts of your evaluations and jobs.
- **Prompt Editor** — advanced: view and customize the underlying prompts
  (System Prompt, Holistic Template, Combined Role Prompt). You can **Save
  Changes**, **Reset to Default**, or **View Default**. Only change these if you
  understand how the prompts drive scoring.
"""

GUIDE_APP_QUICK_ANSWERS = """
**"How do I run a new evaluation?"** — Go to **New Evaluation**. For one person
use the **Single Candidate** tab; for many use **Batch Upload**. Upload the
materials (the candidate ID is generated automatically), then click **Evaluate
Candidate** (single) or **Start Batch Evaluation (Background)** (batch). Every
evaluation runs in Holistic mode.

**"Where do my results go?"** — The **Results** page. Batch progress shows on
**Batch Jobs** while running.

**"What file types can I upload?"** — Single Candidate: PDF, DOCX, TXT, MD.
Batch Upload: PDF only.

**"Where do I set my API key?"** — On first sign-in, or later under **Settings →
Update Anthropic API Key**.
"""


def render_guide_page() -> None:
    """Render the read-only Guide page explaining how the tool's prompts work."""
    st.title("Guide")
    st.caption(
        "How the candidate evaluator thinks"
    )

    st.header("Overview")
    st.markdown(GUIDE_OVERVIEW)

    st.markdown("---")
    st.header("How to use the app")
    st.caption("Step-by-step walkthroughs of each page")

    with st.expander("Getting started (sign in + API key)", expanded=True):
        st.markdown(GUIDE_APP_GETTING_STARTED)
    with st.expander("Run a new evaluation (New Evaluation)", expanded=False):
        st.markdown(GUIDE_APP_NEW_EVALUATION)
    with st.expander("Which option should I pick?", expanded=False):
        st.markdown(GUIDE_APP_CHOOSING)
    with st.expander("Track batch runs (Batch Jobs)", expanded=False):
        st.markdown(GUIDE_APP_BATCH_JOBS)
    with st.expander("Browse results (Results)", expanded=False):
        st.markdown(GUIDE_APP_RESULTS)
    with st.expander("Statistics (Analysis)", expanded=False):
        st.markdown(GUIDE_APP_ANALYSIS)
    with st.expander("Account & configuration (Settings)", expanded=False):
        st.markdown(GUIDE_APP_SETTINGS)
    with st.expander("Quick answers", expanded=False):
        st.markdown(GUIDE_APP_QUICK_ANSWERS)

    st.markdown("---")
    st.header("How the evaluator thinks")
    st.caption("The prompts and scoring philosophy behind every review")

    st.markdown("---")
    st.header("1. The Holistic Evaluation prompt")
    st.markdown(GUIDE_HOLISTIC)

    st.markdown("---")
    st.header("2. The Role Dimensions prompt")
    st.markdown(GUIDE_ROLE_INTRO)

    role_tab_clinical, role_tab_eng, role_tab_research = st.tabs(
        ["Clinical", "Engineer / Tech", "Research"]
    )
    with role_tab_clinical:
        st.markdown(GUIDE_ROLE_CLINICAL)
    with role_tab_eng:
        st.markdown(GUIDE_ROLE_ENGINEERING)
    with role_tab_research:
        st.markdown(GUIDE_ROLE_RESEARCH)

    st.markdown("---")
    st.header("3. The System prompt")
    st.markdown(GUIDE_SYSTEM_PROMPT)

    st.markdown("---")
    st.header("Innovation Potential — definition")
    st.markdown(GUIDE_INNOVATION)

    st.markdown("---")
    st.header("Program Fit — definition")
    st.markdown(GUIDE_PROGRAM_FIT)
    with st.expander("Program description (the six goals that define \"fit\")", expanded=False):
        st.markdown(GUIDE_PROGRAM_DESCRIPTION)


# --------------------------------------------------------------------------- #
# Help chatbot
# --------------------------------------------------------------------------- #

# The chatbot is grounded in the same guide content shown on the Guide page so
# its answers stay consistent with how the tool actually behaves.
HELP_SYSTEM_PROMPT = f"""You are the built-in Help assistant for the Candidate \
Evaluator tool. You help users understand how the tool works, how to use its \
features step by step, and how to interpret its outputs. Be concise, accurate, \
and friendly.

Ground your answers in the reference guide below. If a question is outside the \
scope of this tool (or the guide does not cover it), say so plainly and, where \
helpful, suggest what the user could check or who they might ask. Never invent \
features, prompt variables, dimensions, or rating levels that are not in the \
guide. When users ask "how do I..." questions, give short step-by-step guidance \
using the app's exact page names and button labels from the reference guide. \
The sidebar pages are: Dashboard, New Evaluation, Batch Jobs, Results, \
Analysis, Guide, Help, Settings.

Every evaluation runs in Holistic mode — there is no criteria-based mode, \
interview-selection, admit-patterns, or research page in this app, so never \
refer users to those. Candidate IDs are generated automatically from the \
uploaded file name (users do not type them).

IMPORTANT — help users choose between options. Some flows have more than one \
valid path (for example: Single Candidate vs. Batch Upload; running a batch in \
the Background vs. Run Sequentially Now). When a user's request could go more \
than one way, do NOT just assume one. Briefly lay out the relevant options with \
a one-line trade-off for each, recommend a default when there's a sensible one, \
and ask a short clarifying question so the user can pick the best fit. Once \
they've chosen (or if they've already made their intent clear), give the \
concrete step-by-step for that path. Use the "Which option should I pick?" \
guidance below to explain trade-offs.

This assistant describes the hosted (cloud) web app, which requires signing in \
and saving an Anthropic API key (starting with sk-ant-) before evaluating; the \
key can be updated on the Settings page.

=== REFERENCE GUIDE ===

OVERVIEW
{GUIDE_OVERVIEW}

HOW TO USE THE APP — GETTING STARTED
{GUIDE_APP_GETTING_STARTED}

HOW TO USE THE APP — RUN A NEW EVALUATION (New Evaluation page)
{GUIDE_APP_NEW_EVALUATION}

HOW TO USE THE APP — WHICH OPTION SHOULD I PICK? (help users choose)
{GUIDE_APP_CHOOSING}

HOW TO USE THE APP — BATCH JOBS
{GUIDE_APP_BATCH_JOBS}

HOW TO USE THE APP — RESULTS
{GUIDE_APP_RESULTS}

HOW TO USE THE APP — ANALYSIS
{GUIDE_APP_ANALYSIS}

HOW TO USE THE APP — SETTINGS
{GUIDE_APP_SETTINGS}

HOW TO USE THE APP — QUICK ANSWERS
{GUIDE_APP_QUICK_ANSWERS}

HOLISTIC EVALUATION PROMPT (HOLISTIC_EVALUATION_PROMPT)
{GUIDE_HOLISTIC}

ROLE-DIMENSIONS PROMPT (attached on top of the holistic review)
{GUIDE_ROLE_INTRO}

CLINICAL_PROMPT
{GUIDE_ROLE_CLINICAL}

ENGINEERING_PROMPT
{GUIDE_ROLE_ENGINEERING}

RESEARCH_PROMPT
{GUIDE_ROLE_RESEARCH}

SYSTEM_PROMPT
{GUIDE_SYSTEM_PROMPT}

INNOVATION POTENTIAL
{GUIDE_INNOVATION}

PROGRAM FIT
{GUIDE_PROGRAM_FIT}

PROGRAM DESCRIPTION (defines what "program fit" measures)
{GUIDE_PROGRAM_DESCRIPTION}

=== END REFERENCE GUIDE ===
"""

_HELP_SUGGESTIONS = [
    "How do I run a new evaluation?",
    "How do I evaluate a whole batch of candidates?",
    "What are the Role Dimensions?",
    "What file types can I upload?",
    "What does the Holistic evaluation return?",
    "Why does the tool reject \"fluff\"?",
    "What does \"program fit\" actually measure?",
]


def _stream_help_response(
    client: Any,
    model: str,
    messages: List[dict],
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Yield text chunks from a streaming Anthropic response for the help chat."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=0.3,
        system=HELP_SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def render_help_chat(
    client: Any,
    model: str,
    session_key: str = "help_chat_messages",
) -> None:
    """Render the Help chatbot.

    Args:
        client: An initialized ``anthropic.Anthropic`` client (reuses the same
            API key the evaluator is configured with).
        model: The Claude model id to use for the chat.
        session_key: Session-state key for this conversation's history. Use a
            per-user key in multi-user (cloud) deployments.
    """
    st.title("Help")
    st.caption(
        "Ask anything about how the evaluator works, what the ratings mean, or "
        "how to use a feature. Powered by the same Claude model as your "
        "evaluations."
    )

    if client is None:
        st.warning(
            "The help assistant needs a configured Anthropic API key. "
            "Set your API key in Settings to start chatting."
        )
        return

    if session_key not in st.session_state:
        st.session_state[session_key] = []

    messages: List[dict] = st.session_state[session_key]

    top_cols = st.columns([1, 4])
    with top_cols[0]:
        if st.button("Clear chat", use_container_width=True):
            st.session_state[session_key] = []
            st.rerun()

    if not messages:
        st.markdown("**Try asking:**")
        for i, suggestion in enumerate(_HELP_SUGGESTIONS):
            if st.button(suggestion, key=f"{session_key}_suggest_{i}"):
                st.session_state[f"{session_key}_pending"] = suggestion
                st.rerun()

    # Render existing conversation.
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # A queued suggestion (from a button) is treated like typed input.
    pending = st.session_state.pop(f"{session_key}_pending", None)
    user_input = st.chat_input("Ask a question about the evaluator...") or pending

    if not user_input:
        return

    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                _stream_help_response(client, model, messages)
            )
        except Exception as exc:  # noqa: BLE001 - surface any API error to the user
            response_text = (
                "Sorry, I ran into an error contacting the model: "
                f"`{exc}`\n\nPlease check that your API key is valid in Settings "
                "and try again."
            )
            st.error(response_text)

    messages.append({"role": "assistant", "content": response_text})
    st.session_state[session_key] = messages
