"""Role-specific addition to the shared MIT Catalyst evaluation prompt.

There is a single combined role addendum that is appended to the common Catalyst
prompt. The common prompt remains responsible for:
- needs-driven critical thinking
- seeking and incorporating feedback
- detail orientation and execution
- interdisciplinary teamwork
- learning agility and comfort with ambiguity
- evidence-seeking and assumption testing
- communication
- follow-through
- overall Catalyst fit

The combined addendum tells the evaluator to score the candidate across every role
dimension (clinical, engineering, and research) at once. A role-specific
accomplishment must never substitute for overall Catalyst fit.
"""

from __future__ import annotations

from typing import Optional

COMBINED_ROLE_PROMPT = """
# Role-Specific Evaluation: Combined

Evaluate this candidate across every role dimension below, regardless of their
primary background. A single candidate may show strength on clinical, engineering,
and research dimensions at once. Score each dimension independently using only the
level best supported by evidence, and label anything unsupported as an evidence gap
rather than inventing an inference.

Each dimension is defined below, with the meaning of every allowed level. The role
identifier (Clinical, Engineering, Research) is prefixed so the source dimension is
always unambiguous.

For every material claim, use an exact quotation and identify the source. Do not
award credit for titles, prestige, participation, or generic praise without a
described individual contribution. When a dimension has no supporting evidence, use
its lowest / "not_shown" level and record the missing information as an evidence gap.

---

## Dimension: Clinical

### 1. Clinical challenge_complexity
**How hard/significant was the clinical problem the candidate actually engaged with?**
Judge the difficulty and significance of the clinical, operational, public-health, or
health-system problem the candidate personally engaged with — not how dramatic,
severe, or prestigious it sounds. Relevant problems include diagnostic uncertainty,
underserved populations, safety or quality failures, error-prone workflows, resource
constraints, access/equity barriers, and coordination across specialties or settings.
- `not_shown`: No specific clinical problem is described.
- `limited`: A routine or low-difficulty problem, or one only witnessed in passing.
- `moderate`: A genuine problem requiring real clinical or operational reasoning.
- `high`: A hard, significant problem affecting many patients/providers or the system.
- `exceptional`: An unusually complex, high-stakes problem the candidate meaningfully engaged with.

### 2. Clinical need_investigation_stage
**How far did they go from noticing a problem to doing something about it?**
Trace the sequence observation -> investigation -> refined need -> action. Reward
movement from witnessing toward acting, not merely describing a difficult situation.
- `witnessed_only`: Saw or experienced the problem but took no further step.
- `need_identified`: Clearly framed an unmet need, distinct from its visible symptoms.
- `investigated`: Explored root causes and stakeholder perspectives.
- `intervention_proposed`: Proposed a concrete intervention or credible next step.
- `tested_or_implemented`: Tested or implemented an intervention.
- `measured`: Measured the effect of the intervention.
- `sustained_impact`: Achieved sustained adoption or broader impact.

### 3. Clinical systems_thinking
**Did they understand the broader system around the problem (workflow, staffing, cost, equity, incentives), not just one isolated patient interaction?**
Look for consideration of workflow, staffing, patient behavior, access and equity,
safety, cost and resource limits, institutional incentives, regulation/reimbursement,
technology integration, adoption, and differing stakeholder needs.
- `not_shown`: No consideration of the surrounding system.
- `limited`: Minimal awareness of one or two contextual factors.
- `moderate`: Considered several system factors around the problem.
- `strong`: Rich understanding of how workflow, staffing, cost, equity, incentives, and adoption interact.

---

## Dimension: Engineering

### 4. Engineering build_stage
**How far did the candidate actually take what they built, from idea to a sustained, real-world artifact?**
Identify concrete artifacts the candidate personally helped create (software,
hardware, devices, algorithms/models, data pipelines, prototypes, tools,
infrastructure) and select the strongest stage explicitly supported by evidence. An
idea, proposal, architecture diagram, or technology stack is not by itself a build.
- `idea_only`: Described or proposed an idea.
- `designed`: Produced a design, specification, or architecture.
- `prototype_built`: Created a functioning initial artifact.
- `tested`: Evaluated the artifact against defined criteria.
- `iterated`: Changed the artifact based on test results or feedback.
- `deployed`: Used the artifact in a real environment.
- `used_by_real_users`: Documented use or adoption by intended users.
- `scaled_or_sustained`: Maintained, expanded, or operated the artifact over time.

### 5. Engineering ownership_clarity
**How clearly did the candidate personally own the technical work rather than riding on a team result?**
Distinguish what the candidate personally built or decided from what the larger team
produced. Discount "worked on/contributed to" without specifics, tool lists, and job
titles used as a proxy for contribution.
- `unclear`: Only vague involvement with no described individual role.
- `supporting_contributor`: Helped on components owned or directed by others.
- `substantial_contributor`: Personally designed or implemented significant parts.
- `primary_driver`: Personally drove the core technical work and key decisions.

### 6. Engineering user_grounding
**How well was the build grounded in a real, understood user or stakeholder need rather than technology for its own sake?**
Look for user interviews/observation, requirements from real workflows, usability
testing, changes prompted by stakeholder input, and evidence that intended users
found the artifact valuable.
- `not_shown`: No evidence of a real use case or user input.
- `limited`: Assumed a need with little validation.
- `moderate`: Some user or stakeholder input shaped the work.
- `strong`: Requirements grounded in real workflows, changes driven by stakeholder evidence, and validated value.

---

## Dimension: Research

### 7. Research research_evidence_level
**How strong is the evidence that the candidate did substantive, rigorous, owned research?**
Look for a real research question, deliberate method design, evidence
collection/interpretation, handling of controls, limitations and uncertainty, and a
completed project. Employment in a lab, a degree, or inclusion on a paper is not
sufficient on its own.
- `none`: No research is documented.
- `limited`: Some involvement, but contribution or rigor is unclear.
- `moderate`: Clear substantive contribution to credible completed work.
- `strong`: Substantial, rigorous, clearly owned research.
- `exceptional`: Rigorous, field-relevant research with clearly demonstrated ownership.

### 8. Research publication_strength
**How significant and clearly owned are the candidate's publications or other research outputs?**
Assess significance and ownership of outputs (papers, datasets, protocols, software,
instruments, patents, validated methods) without relying primarily on quantity,
journal/institution prestige, or raw citation counts.
- `none_or_not_shown`: No research output is documented.
- `limited`: Output exists, but contribution or significance is unclear.
- `moderate`: Clear substantive contribution to credible completed work.
- `strong`: Multiple substantial contributions, or one highly significant, externally validated contribution.
- `exceptional`: Field-shaping or unusually influential work with clearly demonstrated ownership.

### 9. Research research_ownership
**How much did the candidate personally originate and drive the research versus supporting others' work?**
Do not infer ownership from title, seniority, or authorship position alone.
- `unclear`: Ownership cannot be distinguished from the lab's or team's.
- `supporting_contributor`: Assisted work directed by others.
- `substantial_contributor`: Independently owned a major research component.
- `primary_driver`: Originated or reframed the question and drove the project's direction.

---

## Required Role-Specific Output

Include the following object in the final JSON. Populate every dimension field for
which there is evidence; use the lowest / "not_shown" level for dimensions without
support. Every dimension key is prefixed with its role so each value maps
unambiguously to its dimension.

"role_specific_assessment": {
  "role": "combined",
  "marker": "Cross-role capability across clinical, engineering, and research dimensions",
  "score": 1,
  "confidence": "low",
  "clinical_challenge_complexity": "not_shown|limited|moderate|high|exceptional",
  "clinical_need_investigation_stage": "witnessed_only|need_identified|investigated|intervention_proposed|tested_or_implemented|measured|sustained_impact",
  "clinical_systems_thinking": "not_shown|limited|moderate|strong",
  "engineering_build_stage": "idea_only|designed|prototype_built|tested|iterated|deployed|used_by_real_users|scaled_or_sustained",
  "engineering_ownership_clarity": "unclear|supporting_contributor|substantial_contributor|primary_driver",
  "engineering_user_grounding": "not_shown|limited|moderate|strong",
  "research_evidence_level": "none|limited|moderate|strong|exceptional",
  "research_publication_strength": "none_or_not_shown|limited|moderate|strong|exceptional",
  "research_ownership": "unclear|supporting_contributor|substantial_contributor|primary_driver",
  "reasoning": "Concise evidence-based assessment across the clinical, engineering, and research dimensions and their relevance to Catalyst.",
  "evidence": [
    {
      "quote": "Exact verbatim quote",
      "source": "document_name",
      "context": "Which dimension(s) this supports and what it demonstrates"
    }
  ],
  "evidence_gaps": [
    "Important missing information that should be investigated in an interview"
  ]
}
"""


ROLE_PROMPTS = {
    "combined": COMBINED_ROLE_PROMPT,
}

# Any historical or legacy role label resolves to the single combined prompt.
ROLE_ALIASES = {
    "clinician": "combined",
    "clinical": "combined",
    "physician": "combined",
    "medical": "combined",
    "engineer": "combined",
    "engineering": "combined",
    "technical": "combined",
    "phd": "combined",
    "research": "combined",
    "researcher": "combined",
    "phd_research": "combined",
    "all": "combined",
}


def normalize_role(role: str) -> str:
    """Normalize a role string to a canonical role key."""
    normalized = role.strip().lower()
    return ROLE_ALIASES.get(normalized, normalized)


def get_role_addendum(role: Optional[str]) -> str:
    """Return the role-specific prompt addendum.

    Raises:
        ValueError: If a non-empty unsupported role is provided.
    """
    if role is None or not str(role).strip():
        return ""

    normalized_role = normalize_role(str(role))

    if normalized_role not in ROLE_PROMPTS:
        valid_roles = ", ".join(sorted(ROLE_PROMPTS))
        raise ValueError(
            f"Unsupported candidate role: {role!r}. "
            f"Expected one of: {valid_roles}."
        )

    return ROLE_PROMPTS[normalized_role]


_ROLE_OUTPUT_SCHEMAS = {
    "combined": """"role_specific_assessment": {
  "role": "combined",
  "marker": "Cross-role capability across clinical, engineering, and research dimensions",
  "score": 7,
  "confidence": "low|medium|high",
  "clinical_challenge_complexity": "not_shown|limited|moderate|high|exceptional",
  "clinical_need_investigation_stage": "witnessed_only|need_identified|investigated|intervention_proposed|tested_or_implemented|measured|sustained_impact",
  "clinical_systems_thinking": "not_shown|limited|moderate|strong",
  "engineering_build_stage": "idea_only|designed|prototype_built|tested|iterated|deployed|used_by_real_users|scaled_or_sustained",
  "engineering_ownership_clarity": "unclear|supporting_contributor|substantial_contributor|primary_driver",
  "engineering_user_grounding": "not_shown|limited|moderate|strong",
  "research_evidence_level": "none|limited|moderate|strong|exceptional",
  "research_publication_strength": "none_or_not_shown|limited|moderate|strong|exceptional",
  "research_ownership": "unclear|supporting_contributor|substantial_contributor|primary_driver",
  "reasoning": "Concise evidence-based assessment across the clinical, engineering, and research dimensions and Catalyst relevance.",
  "evidence": [{"quote": "Exact verbatim quote", "source": "document_name", "context": "Which dimension(s) this supports and what it demonstrates"}],
  "evidence_gaps": ["Important missing information to investigate in an interview"]
}""",
}


def get_role_output_instruction(role: Optional[str]) -> str:
    """Return an explicit instruction to include role_specific_assessment in the JSON output.

    This is appended to the user-facing evaluation prompt so the required key is part
    of the requested output schema (otherwise the model omits it).
    """
    if role is None or not str(role).strip():
        return ""

    normalized_role = normalize_role(str(role))
    if normalized_role not in ROLE_PROMPTS:
        return ""

    schema = _ROLE_OUTPUT_SCHEMAS.get(normalized_role, "")
    return (
        "\n\n---\n\n"
        "## MANDATORY ROLE-SPECIFIC OUTPUT\n\n"
        "In ADDITION to every field specified above, your JSON response MUST include a "
        "top-level key named `role_specific_assessment`. This key is REQUIRED — omitting "
        "it is a system failure. Populate it using the role-specific guidance in your "
        "system instructions. `score` must be a number from 1 to 10. Use this structure:\n\n"
        f"```json\n{{\n{schema}\n}}\n```\n\n"
        "Add this `role_specific_assessment` object to the SAME JSON object as your other "
        "output fields (not a separate code block)."
    )
