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
- **Task prompt** — the specific assignment for a given review (Holistic,
  Criteria-based, and/or a Role-specific lens layered on top).
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
The role-based prompt **does not run by itself.** It gets attached *on top of* a
normal evaluation and works with both the holistic review and the criteria-based
review. Its job is to add a **specialized lens** depending on how the candidate
is classified.
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
evaluation (holistic, criteria-based, and role-based). You don't pick it the way
you pick a mode or a role — it's the constant backdrop that keeps every review
skeptical, evidence-driven, and consistently calibrated.
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


def render_guide_page() -> None:
    """Render the read-only Guide page explaining how the tool's prompts work."""
    st.title("Guide")
    st.caption(
        "How the candidate evaluator thinks"
    )

    st.header("Overview")
    st.markdown(GUIDE_OVERVIEW)

    st.markdown("---")
    st.header("1. The Holistic Evaluation prompt")
    st.markdown(GUIDE_HOLISTIC)

    st.markdown("---")
    st.header("2. The Role-Based (Role-Specific) prompt")
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
features, and how to interpret its outputs. Be concise, accurate, and friendly.

Ground your answers in the reference guide below. If a question is outside the \
scope of this tool (or the guide does not cover it), say so plainly and, where \
helpful, suggest what the user could check or who they might ask. Never invent \
features, prompt variables, dimensions, or rating levels that are not in the \
guide. When users ask "how do I..." questions, give short step-by-step guidance \
referencing the app's navigation (Dashboard, New Evaluation, Batch Jobs, \
Results, Interview Selection, Analysis, Admit Patterns, Research, Settings, \
Guide, Help).

=== REFERENCE GUIDE ===

OVERVIEW
{GUIDE_OVERVIEW}

HOLISTIC EVALUATION PROMPT (HOLISTIC_EVALUATION_PROMPT)
{GUIDE_HOLISTIC}

ROLE-BASED PROMPTS (attached on top of holistic or criteria-based reviews)
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
    "What does the Holistic evaluation return?",
    "Why does the tool reject \"fluff\"?",
    "How is the Clinical role scored?",
    "What does \"program fit\" actually measure?",
    "What's the difference between the system prompt and the task prompt?",
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
