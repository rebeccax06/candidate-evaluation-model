"""Role-specific additions to the shared MIT Catalyst evaluation prompt.

Each addendum should be appended to the common Catalyst prompt. The common
prompt remains responsible for:
- needs-driven critical thinking
- seeking and incorporating feedback
- detail orientation and execution
- interdisciplinary teamwork
- learning agility and comfort with ambiguity
- evidence-seeking and assumption testing
- communication
- follow-through
- overall Catalyst fit

These addenda tell the evaluator what role-specific evidence to examine.
A role-specific accomplishment must never substitute for overall Catalyst fit.
"""

from __future__ import annotations

from typing import Optional

PHD_ROLE_PROMPT = """
# Role-Specific Evaluation: PhD / Research Candidate

This candidate has a research-oriented background. Evaluate the depth, ownership,
rigor, and strength of the candidate's research contributions and determine how
that experience would translate to MIT Catalyst's needs-driven, interdisciplinary,
and iterative process.

## Primary Question

Does the evidence show that the candidate can originate and investigate important
questions, reason rigorously from incomplete or conflicting evidence, and use
research expertise to help a Catalyst team develop a validated healthcare research
opportunity?

## Evaluate the Following

### 1. Evidence of Research

Determine whether the candidate has meaningfully participated in research and what
they personally contributed.

Look for:

- Formulating a research question or hypothesis
- Identifying an important gap in existing knowledge or practice
- Designing experiments, studies, models, analyses, or methods
- Selecting methods for a stated reason
- Collecting, analyzing, or interpreting evidence
- Addressing controls, alternative explanations, limitations, or uncertainty
- Revising the research direction after unexpected or negative findings
- Completing a substantial research project
- Connecting findings to follow-on research or practical impact

Do not treat employment in a laboratory, a degree, or inclusion on a publication
as sufficient evidence of research ownership.

### 2. Research Ownership and Intellectual Contribution

Distinguish the candidate's contribution from the work of the laboratory or team.

Strong evidence includes:

- The candidate originated or materially reframed the research question
- The candidate designed a central method, experiment, model, or analytical approach
- The candidate independently drove a major project component
- The candidate interpreted findings and influenced the project's direction
- The candidate solved a consequential research obstacle
- A supervisor or collaborator specifically corroborates the contribution

Weak evidence includes:

- Passive descriptions such as "was involved in" or "assisted with"
- A list of laboratory techniques without explaining their purpose
- Project descriptions that discuss only what "we" accomplished
- Publication authorship without a described individual role

Do not infer ownership from title, seniority, or authorship position alone.

### 3. Strength of Publications and Research Outputs

Assess the strength of the candidate's publications or other research outputs
without relying primarily on quantity, institutional prestige, journal name, or
citation counts.

Consider:

- Clarity and importance of the candidate's intellectual contribution
- First-author, co-first-author, senior-author, or other substantial ownership when
  the materials explain what that role meant
- Novelty of the question, method, dataset, framework, or finding
- Methodological rigor
- Evidence that the work influenced subsequent research or practice
- Independent replication, adoption, citations, invited presentations, awards,
  patents, follow-on funding, clinical use, or other external validation
- Research outputs beyond papers, including datasets, protocols, software,
  instruments, open-source tools, patents, or validated methods

Publication strength must be classified as:

- `"none_or_not_shown"`: No research output is documented
- `"limited"`: Output exists, but contribution or significance is unclear
- `"moderate"`: Clear substantive contribution to credible completed work
- `"strong"`: Multiple substantial contributions or one highly significant,
  externally validated contribution
- `"exceptional"`: Field-shaping or unusually influential work with clearly
  demonstrated candidate ownership

A candidate without publications may still be strong if the materials demonstrate
substantial research ownership, rigorous reasoning, and meaningful completed work.

### 4. Research Judgment and Critical Thinking

Look for evidence that the candidate:

- Distinguished a meaningful question from an easy or fashionable one
- Broke a complex problem into testable components
- Identified assumptions and sources of uncertainty
- Designed evidence that could disconfirm, not merely support, a hypothesis
- Changed an interpretation or approach when evidence contradicted expectations
- Understood limitations and avoided overclaiming
- Compared alternative methods or explanations
- Drew conclusions proportional to the evidence

Technical sophistication alone is not evidence of strong judgment.

### 5. Persistence Through Research Uncertainty

Research frequently produces failed experiments, ambiguous results, rejected
manuscripts, or changing hypotheses.

Look for:

- A specific setback or unexpected result
- The candidate's diagnosis of what went wrong
- A concrete change in method, framing, or direction
- Evidence of sustained follow-through
- A resulting lesson, improved approach, or completed output

Merely stating that research was challenging is not sufficient.

### 6. Translation to Catalyst

Assess whether the candidate can use research expertise as an enabler rather than
a limitation.

Strong Catalyst-relevant evidence includes:

- Applying research skills in a new domain
- Learning from clinicians, engineers, patients, or other non-specialists
- Translating technical knowledge for an interdisciplinary team
- Investigating the need before promoting a preferred method
- Remaining willing to abandon a familiar technique when evidence favors another path
- Connecting rigorous research to a feasible healthcare opportunity
- Balancing scientific novelty with practical relevance

A highly accomplished researcher may still be a weak Catalyst fit if the evidence
shows rigid attachment to their own topic, method, or predetermined project.

## Role-Specific Evidence Rules

Do not award substantial credit solely for:

- Number of publications
- Journal or university prestige
- Citation count without context
- Laboratory techniques
- Grants or awards without a described contribution
- Statements such as "independent researcher" or "innovative scientist"
- A supervisor's generic praise without a behavioral example

Use exact quotations and identify the source for every material claim.

When evidence is missing, label it as an evidence gap rather than inventing an
inference or treating it automatically as a behavioral red flag.

## Required Role-Specific Output

Include the following object in the final JSON:

"role_specific_assessment": {
  "role": "phd",
  "marker": "Research ownership and strength",
  "score": 1,
  "confidence": "low",
  "research_evidence_level": "none|limited|moderate|strong|exceptional",
  "publication_strength": "none_or_not_shown|limited|moderate|strong|exceptional",
  "research_ownership": "unclear|supporting_contributor|substantial_contributor|primary_driver",
  "reasoning": "Concise evidence-based assessment of the candidate's research contribution and its relevance to Catalyst.",
  "evidence": [
    {
      "quote": "Exact verbatim quote",
      "source": "document_name",
      "context": "What this demonstrates about research ownership, rigor, output strength, or Catalyst readiness"
    }
  ],
  "evidence_gaps": [
    "Important missing information that should be investigated in an interview"
  ]
}

## Generate Targeted Interview Questions

When evidence is incomplete, generate questions that clarify:

- What research question the candidate personally formulated
- Which decisions or methods they personally owned
- How their contribution differed from collaborators' contributions
- What failed or changed during the project
- How contrary evidence affected their conclusions
- Why a publication or output was meaningful
- Whether they can investigate a healthcare need without forcing their existing
  research topic or preferred method onto it
"""


CLINICIAN_ROLE_PROMPT = """
# Role-Specific Evaluation: Clinician Candidate

This candidate has a clinical or healthcare-delivery background. Evaluate whether
the candidate has recognized, investigated, and acted on challenging clinical or
health-system problems and determine how that experience would translate to MIT
Catalyst's needs-driven, interdisciplinary, and iterative process.

## Primary Question

Does the evidence show that the candidate can move beyond describing difficult
clinical experiences to identifying an unmet need, understanding the surrounding
system, learning from stakeholders, and helping develop a practical and testable
healthcare research opportunity?

## Evaluate the Following

### 1. Evidence of Challenging Clinical Problems or Situations

Identify the strongest documented clinical, operational, public-health, or
health-system challenge.

A challenge may involve:

- Diagnostic uncertainty
- A difficult or underserved patient population
- Safety, quality, or care-delivery failures
- Inefficient or error-prone clinical workflow
- Resource constraints
- Access, equity, or continuity-of-care barriers
- Ethical or communication complexity
- Coordination across specialties or care settings
- Implementation barriers
- A recurring problem affecting multiple patients or providers

Do not reward a situation merely because it was emotionally dramatic, clinically
severe, or associated with a prestigious specialty. Evaluate what the candidate
noticed, reasoned, investigated, and did.

### 2. Need Recognition and Problem Framing

Look for evidence that the candidate:

- Identified an unmet need rather than merely witnessing a difficult situation
- Distinguished symptoms from root causes
- Determined who experienced the problem and why it mattered
- Considered the perspectives of patients, caregivers, clinicians, staff, and
  administrators
- Investigated whether the problem was isolated or systematic
- Reframed the problem after learning new information
- Avoided prematurely jumping to a preferred solution

Strong evidence describes the sequence:

observation -> investigation -> refined need statement -> action or next step.

### 3. Systems Thinking

Assess whether the candidate understood the broader conditions surrounding the
clinical problem.

Look for consideration of:

- Workflow
- Staffing and responsibilities
- Patient behavior and lived experience
- Access and equity
- Safety and clinical risk
- Costs and resource limitations
- Institutional incentives
- Regulation or reimbursement
- Technology integration
- Adoption and implementation
- Differences among stakeholders

A candidate who focuses only on an isolated clinical interaction without examining
the surrounding system may have weaker evidence for Catalyst's needs-driven process.

### 4. Action and Practical Impact

Determine whether the candidate moved from recognizing a problem to taking a
specific action.

Examples include:

- Conducting interviews or gathering observations
- Analyzing clinical or operational data
- Designing or testing a workflow change
- Starting a quality-improvement initiative
- Developing a protocol, service, tool, or intervention
- Convening relevant stakeholders
- Advocating for a policy or operational change
- Measuring the effect of an intervention
- Identifying why an attempted intervention failed
- Establishing a credible next step when implementation was not yet possible

Differentiate among:

1. Witnessed a problem
2. Clearly identified and framed the need
3. Investigated causes and stakeholder perspectives
4. Proposed an intervention
5. Tested or implemented an intervention
6. Measured results
7. Achieved sustained adoption or broader impact

### 5. Learning From Patients and Other Stakeholders

Look for evidence that the candidate:

- Sought input from patients or caregivers
- Learned from nurses, allied health professionals, administrators, engineers,
  researchers, or community partners
- Changed an assumption after stakeholder input
- Recognized limitations in their own clinical perspective
- Integrated conflicting stakeholder needs
- Adapted communication to different audiences
- Shared ownership rather than assuming the physician or clinician perspective was
  automatically decisive

Simply working in a multidisciplinary care team is not enough. The materials should
show how other people's input affected the candidate's understanding or actions.

### 6. Functioning Outside the Clinical Expert Role

Catalyst requires clinicians to collaborate as one member of an interdisciplinary
team, not solely as the authoritative domain expert.

Assess whether the candidate can:

- Explain clinical context without dominating solution selection
- Listen to technical, patient, operational, and research perspectives
- Admit uncertainty or missing knowledge
- Learn unfamiliar methods
- Allow evidence to override clinical intuition
- Contribute to a project that may differ from their specialty or initial interests
- Support the team's shared need-finding process

Deep clinical expertise is valuable only when paired with curiosity, humility, and
openness to other forms of evidence.

### 7. Translation to Catalyst

Strong Catalyst-relevant clinical evidence includes:

- Repeatedly noticing important unmet needs in real care settings
- Investigating why a clinical or system problem persists
- Engaging the people affected by the problem
- Converting observations into an actionable research question
- Considering feasibility and implementation from the beginning
- Working productively with non-clinical collaborators
- Remaining open to a solution outside the candidate's original assumptions
- Sustaining effort through institutional or practical barriers

## Role-Specific Evidence Rules

Do not award substantial credit solely for:

- Clinical title, specialty, institution, or years of training
- Exposure to severe or unusual cases
- Statements such as "patient-centered" or "committed to health equity"
- Routine performance of clinical responsibilities
- Generic praise about compassion or bedside manner
- Participation in a committee without a described individual contribution
- A proposed solution with no investigation of the underlying need

Use exact quotations and identify the source for every material claim.

When evidence is missing, label it as an evidence gap rather than inventing an
inference or treating it automatically as a behavioral red flag.

## Required Role-Specific Output

Include the following object in the final JSON:

"role_specific_assessment": {
  "role": "clinician",
  "marker": "Engagement with challenging clinical problems",
  "score": 1,
  "confidence": "low",
  "challenge_complexity": "not_shown|limited|moderate|high|exceptional",
  "need_investigation_stage": "witnessed_only|need_identified|investigated|intervention_proposed|tested_or_implemented|measured|sustained_impact",
  "systems_thinking": "not_shown|limited|moderate|strong",
  "reasoning": "Concise evidence-based assessment of how the candidate approached a challenging clinical or healthcare problem and its relevance to Catalyst.",
  "evidence": [
    {
      "quote": "Exact verbatim quote",
      "source": "document_name",
      "context": "What this demonstrates about need recognition, systems thinking, stakeholder learning, action, or Catalyst readiness"
    }
  ],
  "evidence_gaps": [
    "Important missing information that should be investigated in an interview"
  ]
}

## Generate Targeted Interview Questions

When evidence is incomplete, generate questions that clarify:

- What specific unmet need the candidate recognized
- How they distinguished the underlying need from its visible symptoms
- Which stakeholders they consulted
- What the candidate learned that changed their initial understanding
- What action they personally initiated
- What barriers affected implementation
- How impact was measured
- Whether they can work on a need outside their specialty without assuming a
  predetermined clinical or technical solution
"""


ENGINEER_ROLE_PROMPT = """
# Role-Specific Evaluation: Engineer / Technical Candidate

This candidate has an engineering, computing, product-development, or other
technical background. Evaluate whether the candidate has translated ideas into
functioning artifacts, tested assumptions through implementation, and iterated
based on evidence or stakeholder feedback.

Determine how that experience would translate to MIT Catalyst's needs-driven,
interdisciplinary, and iterative process.

## Primary Question

Does the evidence show that the candidate can build and test practical solutions
while remaining focused on a validated healthcare need rather than becoming
prematurely attached to a technology?

## Evaluate the Following

### 1. Evidence of Building Something

Identify concrete artifacts the candidate personally helped create.

Examples include:

- Software systems
- Hardware devices
- Medical devices
- Algorithms or models
- Data pipelines
- Experimental platforms
- Prototypes
- Research instruments
- Manufacturing processes
- Product features
- Decision-support tools
- Open-source tools
- Technical infrastructure

A technical idea, proposal, architecture diagram, patent listing, or technology
stack is not by itself evidence of a completed build.

Classify the strongest demonstrated build stage:

1. `"idea_only"`: Described or proposed an idea
2. `"designed"`: Produced a design, specification, or architecture
3. `"prototype_built"`: Created a functioning initial artifact
4. `"tested"`: Evaluated the artifact against defined criteria
5. `"iterated"`: Changed the artifact based on test results or feedback
6. `"deployed"`: Used the artifact in a real environment
7. `"used_by_real_users"`: Documented use or adoption by intended users
8. `"scaled_or_sustained"`: Maintained, expanded, or operated the artifact over time

Do not infer a higher stage unless the materials explicitly support it.

### 2. Individual Ownership

Distinguish what the candidate personally built or decided from what the larger
team produced.

Look for:

- Components personally designed or implemented
- Architecture or design decisions personally owned
- Technical obstacles personally diagnosed
- Experiments or tests personally created
- Coordination or leadership personally performed
- Evidence from collaborators or supervisors corroborating the contribution

Weak evidence includes:

- "Worked on" or "contributed to" without specifics
- A team result with no individual role
- Lists of programming languages or tools
- Job titles used as a proxy for contribution
- Claims of leadership without decisions or actions

### 3. Technical Judgment and Critical Thinking

Assess whether the candidate made thoughtful engineering decisions.

Look for:

- Explicit constraints and requirements
- Comparison of alternative approaches
- Tradeoffs involving accuracy, cost, speed, usability, safety, reliability, privacy,
  scalability, or maintainability
- Identification and testing of assumptions
- Debugging or root-cause analysis
- Recognition of failure modes
- Selection of evaluation metrics
- Conclusions based on test results
- Decisions to simplify, redesign, or abandon an approach

Technical complexity alone is not evidence of good engineering judgment.

### 4. Iteration and Learning From Failure

Look for a concrete sequence:

initial design -> test or real-world use -> observed failure or feedback -> change
-> improved result or new conclusion.

Strong evidence includes:

- Prototype revisions
- Architecture changes
- Changes after user testing
- Redesign after performance or reliability failure
- Abandoning an ineffective feature or method
- Recognition that the original problem framing was wrong
- Documented lessons from unsuccessful attempts

A final polished product without evidence about the development process provides
limited information about iteration.

### 5. User and Stakeholder Grounding

Assess whether the candidate built around an understood need.

Look for:

- User interviews or observation
- Collaboration with clinicians, patients, customers, operators, researchers, or
  other stakeholders
- Requirements derived from real workflows
- Usability or human-factors testing
- Changes prompted by stakeholder input
- Consideration of adoption, accessibility, safety, privacy, maintenance, or
  implementation
- Evidence that intended users actually found the artifact valuable

Do not reward a technically impressive system that lacks evidence of a meaningful
or validated use case as highly as an appropriately scoped solution grounded in a
real need.

### 6. Testing, Outcomes, and External Validation

Assess how the candidate determined whether the build worked.

Strong evidence may include:

- Defined success criteria
- Technical performance metrics
- Benchmarks or comparison conditions
- Reliability or safety tests
- User testing
- Deployment results
- Adoption or retention
- Time or cost savings
- Improved outcomes
- Independent use by others
- Open-source adoption
- Revenue or institutional implementation, when relevant
- Recognition, patents, or publications tied to a clearly described contribution

Metrics should be interpreted in context. A number without a baseline, goal, or
candidate contribution may provide weak evidence.

### 7. Follow-Through and Operational Responsibility

Look for evidence that the candidate:

- Completed a difficult build
- Delivered beyond the initial prototype
- Maintained or supported a deployed system
- Addressed reliability or implementation problems
- Documented the work for others
- Handled unglamorous but necessary tasks
- Sustained ownership over time
- Helped teammates complete the project

Starting many technical projects without evidence of completion is weaker than
finishing and validating fewer meaningful projects.

### 8. Translation to Catalyst

Assess whether the candidate can build without forcing technology onto a problem.

Strong Catalyst-relevant evidence includes:

- Investigating a need before selecting a technology
- Treating prototypes as tools for learning rather than proof that an idea is correct
- Working productively with clinicians, researchers, patients, and nontechnical
  teammates
- Explaining technical constraints to non-specialists
- Learning unfamiliar clinical or scientific context
- Letting stakeholder evidence influence engineering priorities
- Abandoning a technically attractive solution when it does not address the need
- Selecting the simplest credible way to test an important assumption
- Balancing technical feasibility with healthcare relevance and implementation

A highly skilled builder may still be a weak Catalyst fit if the evidence shows
technology-first thinking, attachment to a predetermined product, or inability to
share ownership with nontechnical teammates.

## Role-Specific Evidence Rules

Do not award substantial credit solely for:

- Listing programming languages, tools, frameworks, or equipment
- Technical job titles
- Patents without a described contribution or functioning implementation
- Hackathon participation
- A prototype with no testing
- A deployed system with no explanation of the candidate's role
- Statements such as "strong engineer," "technical leader," or "innovative builder"
- Complexity that is unrelated to a validated need

Use exact quotations and identify the source for every material claim.

When evidence is missing, label it as an evidence gap rather than inventing an
inference or treating it automatically as a behavioral red flag.

## Required Role-Specific Output

Include the following object in the final JSON:

"role_specific_assessment": {
  "role": "engineer",
  "marker": "Evidence of building and iteration",
  "score": 1,
  "confidence": "low",
  "build_stage": "idea_only|designed|prototype_built|tested|iterated|deployed|used_by_real_users|scaled_or_sustained",
  "ownership_clarity": "unclear|supporting_contributor|substantial_contributor|primary_driver",
  "user_grounding": "not_shown|limited|moderate|strong",
  "reasoning": "Concise evidence-based assessment of what the candidate built, their individual contribution, how the artifact was tested or iterated, and its relevance to Catalyst.",
  "evidence": [
    {
      "quote": "Exact verbatim quote",
      "source": "document_name",
      "context": "What this demonstrates about building, ownership, judgment, testing, iteration, user grounding, or Catalyst readiness"
    }
  ],
  "evidence_gaps": [
    "Important missing information that should be investigated in an interview"
  ]
}

## Generate Targeted Interview Questions

When evidence is incomplete, generate questions that clarify:

- What the candidate personally designed or implemented
- What stage the artifact actually reached
- Which technical decisions the candidate owned
- What failed during development
- What testing was conducted and why
- What user or stakeholder evidence changed the design
- Whether the artifact was deployed, adopted, or sustained
- Whether the candidate could abandon a preferred technology after learning it did
  not address the validated healthcare need
"""


ROLE_PROMPTS = {
    "clinician": CLINICIAN_ROLE_PROMPT,
    "engineer": ENGINEER_ROLE_PROMPT,
    "phd": PHD_ROLE_PROMPT,
}

ROLE_ALIASES = {
    "clinical": "clinician",
    "physician": "clinician",
    "medical": "clinician",
    "technical": "engineer",
    "engineering": "engineer",
    "research": "phd",
    "researcher": "phd",
    "phd_research": "phd",
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
