# Candidate Evaluation System — Reference Document

This document answers common questions about how the candidate evaluation model works:
where the program description lives, how the "skeptical" stance is enforced, how
role-based review layers on top of a normal evaluation, and the open design questions
around the clinician/engineer/researcher roles.

It is written for program reviewers (non-engineers) and points to the exact prompt
files so the team can see and edit the source of truth.

- System prompt (skepticism + calibration): `candidate_evaluator/prompts/evaluation_prompts.py` → `SYSTEM_PROMPT`
- Holistic prompt (program description): `candidate_evaluator/prompts/evaluation_prompts.py` → `HOLISTIC_EVALUATION_PROMPT`
- Role prompts (clinician / engineer / phd): `candidate_evaluator/prompts/role_prompts.py`
- Prompt assembly: `candidate_evaluator/prompt_manager.py` → `build_system_prompt()`

---

## 1. Where is the program description, and what does it say?

The program description is **embedded inside the holistic evaluation prompt** (and
repeated in the interview ranking and selection prompts). It is **not** a separate
uploaded document, which is why it does not appear as an attachment.

The exact text the model receives today:

> This is a competitive fellowship program seeking candidates who can:
> - Drive innovation and creative problem-solving in their field
> - Translate research into practical impact
> - Work effectively across disciplines and with diverse teams
> - Demonstrate sustained commitment to challenging problems
> - Learn, grow, and adapt based on feedback and new evidence
> - Communicate complex ideas effectively to varied audiences
>
> The program values candidates who show genuine evidence of these qualities through
> their actions and achievements, not just stated intentions.

**Notes / opportunities:**
- This is currently **generic fellowship language**, not a Catalyst-specific
  description. (The role prompts reference "MIT Catalyst," but this top-level
  description does not.)
- The code supports a custom `program_description` override, but the deployed web app
  does not currently expose it — so this default text is what is actually used.
- **Action item:** replace this with the program's real description to improve fit
  assessments.

---

## 2. How is the "deliberately skeptical" stance communicated?

Through the **system prompt** — a set of explicit instructions applied to every
evaluation (this is the long list of principles, likely shown at the end of the
reviewed document). Skepticism is not a toggle or setting; it is written into the
rules. Key mechanisms:

- **Evidence-only scoring** — every score must be tied to a direct quote; no assumptions.
- **Extreme skepticism of stated qualities** — "I am creative / passionate / hard-working"
  is treated as worthless (score 1–2) unless backed by a specific action or outcome.
- **Fluff detection** — generic phrases that could describe anyone are flagged and discounted.
- **Conservative scoring under uncertainty** — "when in doubt, score lower"; absence of
  evidence is treated as evidence of absence.
- **Passive-voice downgrade** — "improvements were made" (unclear who did it) is scored low.

---

## 3. Role-based review: how it layers on a normal evaluation

The role-based review **does not run on its own**. It is attached on top of a normal
evaluation and works with **both** the holistic review and the criteria-based review.

- The role addendum (clinician / engineer / phd) is **appended to the system prompt**
  via `build_system_prompt(role)`.
- A role-specific output block is added so the model returns a `role_specific_assessment`
  alongside its normal output.

**For the reviewed document:** it contains only holistic evaluations, so in that
instance the role guidance is combined with the **holistic** review.

Two scores result, produced in the same model call but kept separate:
- **Overall score** — holistic program fit across all common dimensions.
- **Role score** — the role-specific evidence only (e.g. research strength for a PhD).

By design, "a role-specific accomplishment must never substitute for overall program fit."

---

## 4. Clinician role: curiosity vs. complexity

**Question raised:** What we want most is the candidate's *curiosity about what might be
possible*. "Complexity" is hard to interpret, and without understanding the specific
problem, it's unclear how the AI could judge complexity either. Is the real goal of the
first part of the prompt the **articulation of the problem** rather than its complexity?

**Assessment:** Yes — that is the intended goal, and this question correctly identifies
the weak point.

- The clinician prompt's intent is **need-recognition and problem articulation**:
  observe → investigate → frame the unmet need → act. It explicitly says not to reward a
  situation just because it was "emotionally dramatic, clinically severe, or associated
  with a prestigious specialty."
- However, the scored dimension is named `challenge_complexity`, which is misleading.
  The AI has **no independent domain basis** to judge true clinical complexity; it can
  only infer it from *how the candidate describes the problem*. So in practice that
  dimension already measures **articulation and framing quality**, just under a label
  ("complexity") that invites the wrong interpretation.

**Recommendation:**
- Rename `challenge_complexity` to something like **"problem articulation"** (how clearly
  and insightfully the candidate frames the unmet need).
- Add an explicit **curiosity / sense of possibility** signal — evidence that the
  candidate wonders "what might be possible" and pursues it — since that is the quality
  the program values most.
- Keep clinical acuity as **context only (unscored)**.

---

## 5. Engineer vs. Scientist — should the split be academia vs. industry?

**Question raised:** For people in academia, "engineer" and "scientist" are potentially
much closer than the separate prompts suggest. Maybe the split should be **academic vs.
industry**, which might also help with the multiple-roles concern.

**Trade-offs:**
- The current split is by **activity / type of evidence**: did the candidate *build*
  something (engineer) vs. *do research* (scientist/PhD). For academics, these overlap.
- An **academia vs. industry** split cuts a **different axis** — environment and
  incentives rather than the kind of evidence produced. It may better match how reviewers
  think about candidates, but it doesn't by itself capture "what did this person actually
  do and own."
- **Neither axis alone resolves the multi-role problem** — a strong candidate may be both
  a builder and a researcher (see Section 6).

**Options to discuss:**
1. Keep activity-based roles but allow **more than one** to apply (Section 6).
2. Replace with **academia vs. industry**, accepting that evidence type becomes secondary.
3. Hybrid: a primary axis (academia/industry) plus optional evidence tags
   (built / researched / clinical).

This is a genuine design decision for the team — no change has been made yet.

---

## 6. Can we do dependent / conditional prompts?

**Question raised:** If someone claims to have *built* something, use the engineer
prompts; if they did *research*, use the research prompts.

**Current state:** Not supported. The role is a **single manual selection**, and the code
applies exactly one role addendum per evaluation.

**Feasible approaches (not yet built):**
1. **Routing pre-pass** — a quick first call where Claude classifies the candidate
   ("built something" → engineer; "did research" → researcher; "clinical practice" →
   clinician), then the matching addendum is attached automatically.
2. **Multiple roles** — let a reviewer (or the router) select more than one role and
   append multiple addenda; the output schema would be extended to hold more than one
   role-specific assessment.

Either approach directly addresses the multi-role concern from Section 5.

---

## 7. Problem-analysis prompt is appropriate for any applicant

**Agreed.** The need-recognition / problem-framing logic is role-agnostic. It is a good
candidate to **promote into the common prompt** so every applicant — regardless of role —
is assessed on how well they recognize, investigate, and articulate a problem. Role
prompts would then add only the role-specific evidence on top.

---

## 8. What the "worked calibration examples" include

The system prompt contains two calibration blocks that anchor the model's strictness to
how an experienced human reviewer would score:

**(a) Reject vs. credit pairs (❌ / ✅):** contrast vague claims against concrete evidence.
- ❌ "The candidate demonstrates interdisciplinary thinking" (too vague)
- ✅ "Combined machine learning with clinical data to develop a diagnostic tool (accuracy 92%)"
- ❌ "Strong endorsement from supervisor emphasizing work ethic" (generic)
- ✅ "Supervisor noted candidate stayed until 3am three nights in a row to fix a critical bug"

**(b) Expert calibration examples:** show how a real reviewer scores **borderline** cases,
to prevent the model from drifting lenient.
- Vague creativity claim ("developed innovative solutions") → **3–4**, not 5–6.
- Passive-voice achievement ("improvements were made") → **3–4** (unclear ownership).
- Stated motivation without proof → **2–3**.
- One solid, quantified example → **5–6** (needs multiple for higher).
- Generic supervisor praise → **2–3**.

The purpose is to make the AI's bar match an experienced human rater's bar on the
hard-to-call cases, not just the obvious ones.

---

## Summary of recommended follow-ups

| # | Item | Type |
|---|------|------|
| 1 | Replace generic program description with the real program description | Quick edit |
| 4 | Rename clinician `challenge_complexity` → "problem articulation"; add curiosity signal; treat acuity as unscored context | Prompt change |
| 5 | Decide role axis: activity-based (current) vs. academia/industry vs. hybrid | Design decision |
| 6 | Add dependent/multi-role prompting (router pre-pass or multi-role selection) | Feature |
| 7 | Promote problem-analysis into the common prompt for all applicants | Prompt refactor |
