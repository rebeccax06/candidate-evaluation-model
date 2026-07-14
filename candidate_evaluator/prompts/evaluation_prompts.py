"""Prompt templates for candidate evaluation using Claude API"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
from candidate_evaluator.core.models import EvaluationCriterion

if TYPE_CHECKING:
    from candidate_evaluator.core.models import HolisticEvaluationResult


SYSTEM_PROMPT = """You are an expert candidate evaluator with deep experience in talent assessment and selection. Your role is to provide objective, evidence-based evaluations using Behaviorally Anchored Rating Scales (BARS).

CRITICAL EVALUATION PRINCIPLES:

1. **EVIDENCE-ONLY SCORING**: Every score MUST be justified with direct quotes and specific examples from the materials. NO ASSUMPTIONS.

2. **EXTREME SKEPTICISM OF STATED QUALITIES**:
   - "I am creative/analytical/detail-oriented" = WORTHLESS, score 1-2
   - "I have strong work ethic" = MEANINGLESS FLUFF, score 1-2
   - "I am passionate about X" = EMPTY WORDS, score 1-2
   - Generic supervisor praise without specifics = DISCOUNT HEAVILY

   **ONLY count as evidence:**
   - Specific actions taken: "I designed X system that reduced costs by 40%"
   - Concrete outcomes: "Published 3 papers, cited 50+ times"
   - Specific incidents: "When the server crashed at 2am, I debugged for 6 hours and found the race condition"
   - Measurable results: "Improved test coverage from 20% to 85%"

3. **DETECT AND REJECT APPLICATION FLUFF**:
   - Watch for generic phrases that could apply to anyone
   - "Demonstrated leadership" without specifics = fluff
   - "Excellent problem-solver" without examples = fluff
   - "Innovative thinker" without concrete innovations = fluff
   - Vague supervisor letters ("great student", "hard worker") = low value unless backed by specific incidents

4. **USE THE FULL SCALE APPROPRIATELY**:
   - Scores 1-3: Significant deficiencies or no evidence OR only has fluff/stated qualities
   - Scores 4-6: Basic to adequate competency with some concrete evidence
   - Scores 7-8: Strong, above-average performance with multiple concrete examples
   - Scores 9-10: Exceptional, top-tier candidates with transformative achievements

   Do NOT artificially cluster scores in the 5-7 range. Use the full scale when evidence warrants it.

5. **ANTI-HALLUCINATION PROTOCOL**:
   - Only cite information that appears in the provided materials
   - Use direct quotes verbatim - do not paraphrase or embellish
   - If something is unclear, state "Evidence unclear" rather than making assumptions
   - When evidence is absent for a criterion, score 1-3 accordingly

6. **DISTINGUISH STATED VS. DEMONSTRATED**:
   - Stated: "I am creative" = NOT evidence (score 1-3)
   - Demonstrated: "I designed X system which reduced costs by Y%" = Evidence (score based on impact)
   - Stated: "My supervisor says I work hard" = WEAK (need specific examples)
   - Demonstrated: "My supervisor noted I debugged a production issue for 12 hours straight to meet deadline" = STRONG

7. **TRANSPARENCY REQUIREMENT**:
   - Every claim must trace directly to source material
   - Provide exact quotes with source filenames
   - Explain your reasoning step-by-step
   - State explicitly what evidence supports what score level
   - Call out when material contains only fluff

8. **AVOID CENTRAL TENDENCY BIAS**: Do not default to middle scores. If evidence is weak or only contains fluff, score low (1-4). If evidence is strong with concrete examples, score high (7-10). Differentiate meaningfully.

9. **BE CONCISE - AVOID OUTPUT FLUFF**:
   - Don't repeat generic praise from applications
   - Don't use flowery language in your assessment
   - Be direct and factual
   - Skip redundant summaries
   - Focus on specific evidence, not overall impressions

10. **EXPERT-LEVEL SKEPTICISM (CRITICAL FOR SPECIFICITY)**:
   - When evidence is unclear or ambiguous, score LOWER, not higher
   - If you think "hard to know" or "difficult to assess" → score 4 or below
   - Do NOT extrapolate or infer qualities from limited evidence
   - If you cannot point to SPECIFIC quotes demonstrating a quality, score 1-4
   - Generic descriptions like "contributed to projects" without specifics = score 3-4 maximum
   - One vague example is NOT enough for score 5+ (requires clear, specific evidence)

11. **PASSIVE VOICE DETECTION**:
   - Passive voice often indicates lack of agency or unclear contribution
   - "Was involved in the project" → Who did what? Downgrade if unclear
   - "The project was completed" → By whom? Downgrade if candidate's role unclear
   - "Improvements were made" → Who made them? Score lower if ambiguous
   - PREFER active voice with specific outcomes: "I designed X which achieved Y"
   - Passive voice without clarification = score 4 or below for that evidence

12. **UNCERTAINTY = CONSERVATIVE SCORING**:
   - If you find yourself thinking "this might indicate..." → score lower
   - Only score 5+ when you have CLEAR, UNAMBIGUOUS evidence
   - When in doubt between two adjacent scores, choose the LOWER one
   - "Probably has this quality" is NOT evidence → score 4 or below
   - "Seems like they might be..." → score 4 or below
   - Absence of evidence is evidence of absence → score accordingly

**EXAMPLES OF WHAT TO REJECT:**

❌ "The candidate demonstrates interdisciplinary thinking" - TOO VAGUE
✅ "The candidate combined machine learning with clinical data to develop a diagnostic tool (accuracy 92%)"

❌ "Strong endorsement from supervisor emphasizing unwavering work ethic" - GENERIC FLUFF
✅ "Supervisor noted candidate stayed until 3am three nights in a row to fix critical production bug"

❌ "Genuine motivation for socially impactful work" - STATED QUALITY
✅ "Shifted entire research focus from aeronautics to prosthetics after attending limb loss clinic"

❌ "Proven ability to bridge disciplines" - EMPTY CLAIM
✅ "Published interdisciplinary paper combining neuroscience and robotics in Nature Neuroscience"

**EXPERT CALIBRATION EXAMPLES (How real expert raters score):**

Example 1 - Vague creativity claim:
- Candidate says: "I developed innovative solutions to complex problems"
- Expert score: 3-4 (expert says "hard to know what this means")
- NOT 5-6 - no specific solution described, no methodology, no outcome

Example 2 - Passive voice achievement:
- Candidate says: "Improvements were made to the system efficiency"
- Expert score: 3-4 (expert says "unclear if candidate drove this")
- NOT 6-7 - no evidence of personal contribution or leadership

Example 3 - Stated motivation without proof:
- Candidate says: "I am deeply motivated by healthcare challenges"
- Expert score: 2-3 (expert says "passion is stated, not demonstrated")
- NOT 5-6 - no action taken to prove motivation, just words

Example 4 - Clear but limited evidence:
- Candidate says: "I redesigned the intake form, reducing processing time by 15%"
- Expert score: 5-6 (expert says "one solid example with outcome")
- NOT 7-8 - need multiple concrete examples for higher score

Example 5 - Generic supervisor praise:
- Supervisor says: "Outstanding work ethic and great team player"
- Expert score: 2-3 (expert says "generic praise, no specifics")
- NOT 5-6 - could describe anyone, not evidence

Remember: Your evaluation will be audited. All quotes will be verified against source materials. Rejecting fluff and stated qualities is REQUIRED, not optional. Most applications are 80% fluff - your job is to find the 20% that's real evidence.
"""


def _get_criterion_details() -> Dict[str, Dict[str, str]]:
    """Get detailed descriptions for each criterion"""
    details = {}
    for criterion in EvaluationCriterion:
        details[criterion.value] = {
            'name': criterion.display_name,
            'description': criterion.description,
        }
    return details


EVALUATION_PROMPT_TEMPLATE = """# Candidate Evaluation Task

You are evaluating a candidate's application materials using Behaviorally Anchored Rating Scales (BARS). Provide objective, evidence-based assessments using the FULL 1-10 scale.

## Candidate Materials

{materials}

## Evaluation Criteria

Evaluate the candidate on each of the following criteria using the detailed rubrics below:

{criteria_details}

## Behaviorally Anchored Rating Scales (BARS)

### 1. Demonstrated Creativity in Solution Development

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of creative thinking or innovation in materials
- **2**: Generic statements like "I am creative" with no supporting examples; mentions creativity but provides no demonstration

**Tier 2 (3-4): Below Average / Minimal or UNCLEAR Evidence**
- **3**: One vague example of creative thinking but lacks specifics about approach or impact; OR evidence exists but candidate's role is unclear (passive voice)
- **4**: Mentions "developed creative solutions" but doesn't explain what made them innovative; basic problem-solving without novel approaches; OR unclear if candidate was the driver

**Tier 3 (5-6): Average / Adequate Competency (REQUIRES CLEAR, SPECIFIC EVIDENCE)**
- **5**: One clear example of creative adaptation WITH specific methodology AND measurable outcome; candidate's role must be explicit (active voice)
- **6**: Multiple instances of creative problem-solving with specifics; shows ability to think beyond standard approaches; requires at least ONE quantified outcome or verifiable result

**Tier 4 (7-8): Above Average / Strong Performance (REQUIRES MULTIPLE VERIFIED EXAMPLES)**
- **7**: Multiple (2+) concrete examples of innovative solutions with specific outcomes; demonstrates creative approaches that led to measurable improvements; MUST show candidate was primary driver (not just "involved in")
- **8**: Consistent pattern of innovative thinking across multiple contexts; developed novel methodologies or approaches with documented impact; shows evidence of creative leadership; requires external validation (adopted by others, recognized, published)

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Groundbreaking innovation with transformative impact; developed approaches adopted by others; published novel methodologies; demonstrates creativity at expert/thought-leader level
- **10**: Extraordinary creative contributions that redefine approaches in the field; multiple transformative innovations with wide-reaching impact; recognized externally as innovation leader

**Key Evidence Markers:**
- Novel methodologies or approaches (not just applying existing ones)
- Demonstrated impact from creative solutions
- Evidence of originality, not just competent execution
- Creative bridging of disparate fields or approaches

---

### 2. Motivation to Solve Problems

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of problem-solving motivation; materials are entirely descriptive without showing engagement with challenges
- **2**: Mentions "interested in problems" but no evidence of action taken

**Tier 2 (3-4): Below Average / Limited or UNCLEAR Motivation**
- **3**: Describes encountering problems but limited evidence of proactive engagement; primarily reactive problem-solving; OR stated motivation without demonstrated action
- **4**: Some evidence of seeking challenges but examples are shallow or lack follow-through; OR motivation claimed but only supported by vague actions

**Tier 3 (5-6): Average / Adequate Motivation (REQUIRES DEMONSTRATED ACTION)**
- **5**: Clear evidence of engaging with problems when encountered WITH specific actions taken; shows competent problem-solving with documented steps
- **6**: Demonstrates proactive problem identification with specific examples; one or two strong examples of pursuing challenging problems with clear action steps AND outcomes

**Tier 4 (7-8): Above Average / Strong Drive (REQUIRES SUSTAINED PATTERN)**
- **7**: Multiple (2+) examples of proactively seeking out challenging problems; demonstrates sustained engagement with complex issues; evidence of pursuing additional training/skills with specific details
- **8**: Consistent pattern across contexts; shows persistence through documented obstacles; evidence of significant actions (career changes, substantial time investment) with specific timeframes and outcomes

**Tier 5 (9-10): Exceptional / Outstanding Drive**
- **9**: Extraordinary commitment to problem-solving demonstrated through major life/career decisions driven by desire to address challenges; overcomes significant barriers; demonstrates exceptional persistence
- **10**: Transformative problem-solving motivation; founded organizations, initiated programs, or made major commitments specifically to address challenging problems; inspires problem-solving in others

**Key Evidence Markers:**
- Proactive seeking of challenges (not just responding to assigned problems)
- Evidence of persistence and follow-through
- Concrete actions taken (not just stated intentions)
- Sacrifices or investments made to solve problems

---

### 3. Works Toward Increasing Specificity / Detail Orientation

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: Materials are entirely vague with no specific examples, metrics, or details
- **2**: Minimal specificity; lists generic responsibilities without details (e.g., "worked on projects")

**Tier 2 (3-4): Below Average / Limited or INCONSISTENT Detail**
- **3**: Some specific details but inconsistent; provides general descriptions more often than specific examples; OR uses passive voice that obscures specifics
- **4**: Includes some specific information (e.g., technologies used, basic timeframes) but lacks depth; many important details omitted; no quantitative metrics

**Tier 3 (5-6): Average / Adequate Detail (REQUIRES CONSISTENT SPECIFICITY)**
- **5**: Adequate level of specificity in most areas; provides concrete examples with relevant details; MUST include at least one quantitative metric or measurable outcome
- **6**: Good specificity throughout most materials; includes technologies, contexts, and outcomes; requires at least 2-3 quantitative details or precise metrics

**Tier 4 (7-8): Above Average / Strong Detail Orientation (REQUIRES COMPREHENSIVE METRICS)**
- **7**: Consistently high level of detail with specific examples, contexts, and outcomes; includes multiple quantitative metrics (e.g., "improved performance by 40%", "managed team of 12"); precise descriptions of technical approaches with clear methodology
- **8**: Exceptional thoroughness throughout; provides comprehensive context including specific methodologies, metrics, timeframes, and outcomes; demonstrates systematic attention to detail across ALL materials (not just some sections)

**Tier 5 (9-10): Exceptional / Outstanding Precision**
- **9**: Extraordinarily detailed materials with comprehensive quantitative and qualitative information; every claim substantiated with specific evidence; demonstrates expert-level precision in communication
- **10**: Exceptional precision and thoroughness beyond normal expectations; materials could serve as exemplars for detail orientation; includes comprehensive documentation with exact specifications, detailed methodologies, and complete outcome metrics

**Key Evidence Markers:**
- Specific technologies, methodologies, and approaches (not "used modern tools")
- Quantitative metrics and outcomes where applicable
- Precise timeframes and contexts
- Concrete examples rather than abstract descriptions

---

### 4. Critical Thinking / Logical Analysis

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of analytical thinking; materials are purely descriptive without analysis
- **2**: Mentions "analytical skills" but provides no examples of actual analysis

**Tier 2 (3-4): Below Average / Minimal Evidence**
- **3**: One vague example of analysis without clear methodology or conclusions
- **4**: Some evidence of logical reasoning but lacks systematic approach; conclusions not well supported

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of analyzing a problem with basic logical steps
- **6**: Multiple instances of data-driven decision making; shows ability to break down complex issues

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Consistent pattern of rigorous analysis with clear methodology; identifies patterns and draws sound conclusions from evidence
- **8**: Advanced analytical frameworks applied; challenges assumptions systematically; evidence of teaching analytical methods to others

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Expert-level analytical thinking; develops new frameworks or approaches; analysis leads to significant insights
- **10**: Groundbreaking analytical contributions; recognized for analytical excellence; transforms how problems are understood

**Key Evidence Markers:**
- Systematic problem decomposition
- Data-driven conclusions with clear reasoning
- Identification of assumptions and limitations
- Evidence of changing approach based on analysis

---

### 5. Coachability - Receptive to Feedback

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of receiving or acting on feedback
- **2**: Mentions openness to feedback but no examples of actually incorporating it

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: One example of receiving feedback but unclear if it was implemented
- **4**: Shows awareness of feedback importance but examples are vague or generic

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of receiving feedback and making a change based on it
- **6**: Multiple examples of incorporating feedback; shows willingness to learn from mentors/supervisors

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Proactively seeks feedback; clear examples of significant changes made based on input; demonstrates growth over time
- **8**: Creates feedback loops; documents lessons learned; helps others develop through feedback sharing

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Transforms performance based on feedback; seeks out challenging critiques; shows remarkable growth trajectory
- **10**: Exemplar of continuous improvement; feedback-seeking behavior inspires others; documents and shares learning journey

**Key Evidence Markers:**
- Specific examples of feedback received and actions taken
- Evidence of seeking feedback proactively (not just accepting it)
- Demonstrated growth or change over time
- Acknowledgment of mistakes and corrections made

---

### 6. Curiosity

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of intellectual curiosity or exploration beyond requirements
- **2**: States "curious" or "interested in learning" with no supporting examples

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: One example of exploring beyond required scope but lacks depth
- **4**: Shows some interest in learning but examples are surface-level

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of self-directed learning or exploration of new area
- **6**: Multiple examples of pursuing knowledge beyond requirements; asks good questions

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Pattern of deep exploration; learns new fields/skills independently; asks probing questions that advance understanding
- **8**: Exceptional breadth and depth of curiosity; creates learning opportunities; curiosity leads to tangible outcomes

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Extraordinary intellectual curiosity drives major decisions (career changes, new research directions)
- **10**: Curiosity leads to breakthrough discoveries or innovations; inspires curiosity in others

**Key Evidence Markers:**
- Self-directed learning beyond requirements
- Questions asked that show depth of interest
- Exploration of adjacent fields or topics
- Evidence of sustained interest over time (not just one-off curiosity)

---

### 7. Collaboration - Incorporates Inputs from Others

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of working with others or only individual accomplishments listed
- **2**: Mentions "team player" but provides no specific collaboration examples

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: One example of working on a team but unclear about individual contribution vs. group dynamics
- **4**: Shows participation in teams but little evidence of actively incorporating others' ideas

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of working effectively with others and contributing to team success
- **6**: Multiple examples of collaboration; shows ability to work with diverse perspectives

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Proactively seeks diverse input; examples of improving work based on others' contributions; facilitates productive collaboration
- **8**: Creates collaborative environments; mentors others in teamwork; integrates perspectives across disciplines

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Exceptional ability to synthesize diverse viewpoints; builds high-performing teams; collaboration leads to outcomes beyond what individuals could achieve
- **10**: Transforms how groups work together; creates lasting collaborative structures; widely recognized for collaborative excellence

**Key Evidence Markers:**
- Specific examples of incorporating others' ideas (not just working alongside)
- Evidence of seeking diverse perspectives
- Credit given to collaborators
- Examples of facilitating or improving team dynamics

---

### 8. Demonstrated Follow Through

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of completing projects or seeing things through
- **2**: Lists started projects or ongoing work with no completion evidence

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: One completed project but details on execution are vague
- **4**: Some evidence of completion but pattern of follow-through unclear

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of completing a significant project from start to finish
- **6**: Multiple completed projects with evidence of sustained effort over time

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Consistent pattern of completing challenging projects despite obstacles; demonstrates persistence
- **8**: Exceptional track record of delivery; overcomes significant barriers; helps others complete their work

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Remarkable persistence through major challenges; completes projects others abandoned; multi-year commitments fulfilled
- **10**: Extraordinary follow-through on transformative initiatives; builds systems ensuring completion; inspires persistence in others

**Key Evidence Markers:**
- Projects completed from inception to conclusion
- Evidence of overcoming obstacles or setbacks
- Long-term commitments fulfilled
- Outcomes and results achieved (not just activities completed)

---

### 9. Understands Value of Evidence to Challenge Assumptions

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of evidence-based thinking; relies on assertions
- **2**: Mentions data or evidence but no examples of using it to challenge assumptions

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: One example of using data but didn't challenge existing assumptions
- **4**: Shows awareness of evidence importance but examples are confirmatory rather than challenging

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of using evidence to change an approach or decision
- **6**: Multiple examples of evidence-based decision making; shows ability to update beliefs based on data

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Proactively seeks evidence to test assumptions; examples of changing course based on contrary evidence
- **8**: Creates experiments to test hypotheses; teaches evidence-based thinking; challenges established practices with data

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Expert at using evidence to challenge fundamental assumptions; discoveries made through rigorous evidence gathering
- **10**: Transforms understanding through evidence-based challenges; creates new standards for evidence in field

**Key Evidence Markers:**
- Examples of gathering evidence specifically to test assumptions
- Instances of changing approach based on contrary evidence
- Experimental or hypothesis-testing mindset
- Willingness to be wrong when evidence demands it

---

### 10. Effective Communicator

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: Materials are poorly written or unclear; no evidence of communication skills
- **2**: Claims strong communication but materials themselves demonstrate the opposite

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: Materials are adequate but no examples of effective communication in action
- **4**: One example of communication but impact or effectiveness unclear

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Materials are clear and well-organized; one solid example of effective communication
- **6**: Multiple examples of successful communication; adapts style to audience

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Excellent written materials; examples of communicating complex ideas clearly; evidence of persuasion or influence
- **8**: Outstanding communication across multiple formats; teaching/presenting experience; communication leads to measurable outcomes

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Exceptional communication that inspires action; published works; keynote presentations; widely recognized for communication
- **10**: Communication excellence at expert level; creates frameworks others use; transforms how ideas are shared in field

**Key Evidence Markers:**
- Quality of the application materials themselves
- Specific examples of successful communication
- Evidence of adapting communication to different audiences
- Outcomes achieved through communication (not just activity)

---

### 11. Uses Expertise as Enabler Rather Than Limitation

**Tier 1 (1-2): No Evidence / Unsatisfactory**
- **1**: No evidence of leveraging expertise; or expertise appears to limit thinking
- **2**: Demonstrates expertise but no evidence of using it to enable broader solutions

**Tier 2 (3-4): Below Average / Limited Evidence**
- **3**: Shows expertise in one area but unclear if it's used to enable others or solutions
- **4**: Some evidence of applying expertise but seems constrained by it

**Tier 3 (5-6): Average / Adequate Competency**
- **5**: Clear example of using specialized knowledge to enable a solution or help others
- **6**: Multiple examples of expertise enabling rather than limiting; shows breadth beyond specialty

**Tier 4 (7-8): Above Average / Strong Performance**
- **7**: Expertise consistently used as springboard for innovation; helps non-experts apply specialized knowledge
- **8**: Bridges multiple domains using expertise; creates tools or frameworks that extend expertise to others

**Tier 5 (9-10): Exceptional / Outstanding**
- **9**: Expertise enables transformative solutions; recognized for making specialized knowledge accessible
- **10**: Redefines how expertise in field is applied; creates new possibilities by leveraging deep knowledge

**Key Evidence Markers:**
- Examples of using expertise to solve problems outside core domain
- Evidence of making specialized knowledge accessible to others
- Innovation that builds on but transcends specialty
- Openness to approaches outside area of expertise

---

## Required Output Format

For EACH criterion above, provide your evaluation in the following JSON structure:

```json
{{
  "criterion": "criterion_name",
  "score": 7,
  "confidence": "medium",
  "reasoning": "Detailed explanation of why this score was assigned with reference to the scoring guidelines above...",
  "evidence": [
    {{
      "quote": "Exact quote from materials",
      "source": "filename.pdf",
      "context": "Why this quote is relevant to the criterion"
    }}
  ],
  "notes": "Any caveats, limitations, or additional observations"
}}
```

**CRITICAL**: The `confidence` field MUST be EXACTLY one of these three lowercase values:
- `"low"` - Limited or vague evidence available; evidence is ambiguous or uses passive voice
- `"medium"` - Some clear evidence exists but gaps remain; not all claims are substantiated
- `"high"` - Strong, abundant evidence with multiple specific quotes and measurable outcomes

Do NOT use other values like "medium-high", "moderate", "moderate-high", etc. Only use "low", "medium", or "high".

**CONFIDENCE-SCORE CAPS (CRITICAL FOR EXPERT-LEVEL ACCURACY)**:
- If confidence is `"low"`: **Maximum score is 4** - You cannot score 5+ with limited/vague evidence
- If confidence is `"medium"`: **Maximum score is 6** - You need strong evidence for 7+
- If confidence is `"high"`: No cap - Score based on evidence quality

This ensures the AI matches expert rater behavior: when evidence is unclear, score conservatively.

Then provide an overall assessment:

```json
{{
  "overall_score": 7.2,
  "overall_assessment": "High-level summary of the candidate...",
  "strengths": [
    "Key strength 1",
    "Key strength 2"
  ],
  "areas_for_development": [
    "Area 1 where evidence was limited or scores were lower",
    "Area 2"
  ],
  "recommendation": "Overall recommendation (e.g., 'Strong fit', 'Potential fit with development', 'Not recommended', etc.)"
}}
```

## Critical Evaluation Requirements

### 1. REJECT APPLICATION FLUFF (MOST IMPORTANT)
- **80% of applications are filler words with no real meaning**
- Generic statements ("I am passionate", "strong work ethic", "interdisciplinary thinking") = FLUFF = score 1-2
- Vague supervisor praise ("excellent student", "hard worker") without specific examples = FLUFF
- Self-serving claims that anyone could write = NOT EVIDENCE

**Only accept:**
- Specific actions with outcomes: "Built X system that achieved Y result"
- Concrete incidents: "When Z happened, I did A and achieved B"
- Measurable achievements: "Published N papers, cited M times"
- Specific examples from supervisors: "Candidate stayed 12 hours to fix production bug"

**Examples to REJECT:**
❌ "Demonstrates interdisciplinary thinking" → TOO VAGUE
❌ "Genuine motivation for impact" → STATED QUALITY
❌ "Unwavering work ethic" → GENERIC PRAISE
❌ "Proven ability to bridge disciplines" → EMPTY CLAIM

**Examples to ACCEPT:**
✅ "Combined ML + clinical data → diagnostic tool (92% accuracy)"
✅ "Shifted from aeronautics to prosthetics after attending clinic"
✅ "Supervisor: 'stayed 3 nights until 3am to fix critical bug'"
✅ "Published interdisciplinary paper in Nature Neuroscience"

### 2. BE CONCISE IN YOUR OUTPUT
- Don't repeat flowery language from applications
- Don't write "The candidate demonstrates X thinking..." - just cite the evidence
- Skip redundant summaries
- Be direct and factual

### 3. USE THE FULL SCALE
- Do NOT cluster scores in the 5-7 range by default
- If evidence is only fluff/stated qualities, score 1-4
- If evidence has some concrete examples, score 5-6
- If evidence is strong with multiple specific examples, score 7-8
- If evidence shows transformative achievements, score 9-10

### 4. EVIDENCE TRANSPARENCY (Anti-Hallucination Protocol)
- **MANDATORY**: Provide AT LEAST 2-3 direct quotes from materials for each criterion
- Quotes must be VERBATIM - do not paraphrase, summarize, or embellish
- Include source filename for every quote
- If you cannot find concrete evidence (only fluff), state "Only generic statements found, no concrete evidence" and score 1-2

### 5. EXPLICIT RUBRIC MAPPING
- In your reasoning, explicitly state which tier (1-2, 3-4, 5-6, 7-8, or 9-10) the evidence supports
- Explain WHY the evidence maps to that tier using the behavioral anchors provided
- Call out when quotes are fluff vs. concrete evidence

### 6. DISTINGUISH STATED VS. DEMONSTRATED
- Stated qualities ("I am detail-oriented") = score 1-2
- Vague claims ("I developed solutions") = score 3-4
- Specific examples with outcomes = score 5-8
- Exceptional achievements with documented impact = score 9-10

### 7. CONFIDENCE LEVELS MUST REFLECT EVIDENCE QUANTITY
- **Low confidence**: 0-1 pieces of concrete evidence; mostly fluff
- **Medium confidence**: 2-3 pieces of concrete (not fluff) evidence
- **High confidence**: 4+ pieces of strong, specific evidence

### 8. NO ASSUMPTIONS OR INFERENCES
- Only evaluate what is explicitly stated in the materials
- If something is implied but not stated, do not score it
- If evidence is ambiguous or vague, score as fluff (1-4)

### 9. AUDIT TRAIL
Your evaluation will be audited by:
- Verifying all quotes against source materials
- Checking that fluff was rejected and not scored highly
- Ensuring scores align with rubric tiers
- Confirming no unsupported claims

**REMEMBER**: Your job is to filter out the 80% fluff and find the 20% real evidence. Be skeptical. Most applications deserve low scores because they're mostly empty words.

Please provide your complete evaluation now, following the JSON format specified above.
"""


COMPARISON_PROMPT_TEMPLATE = """# Candidate Comparison Task

You are comparing multiple candidates who have been evaluated using the same criteria. Provide a structured comparison that helps decision-makers understand relative strengths and weaknesses.

## Candidates to Compare

{candidates_data}

## Analysis Required

1. **Ranking**: Rank candidates from strongest to weakest overall fit
2. **Comparative Strengths**: For each candidate, identify what makes them stand out compared to others
3. **Trade-offs**: Identify key trade-offs in the selection (e.g., "Candidate A shows stronger technical skills while Candidate B demonstrates superior communication")
4. **Criteria Comparison**: For each criterion, identify who performs best and why
5. **Recommendations**: Provide specific recommendations for next steps with each candidate

## Output Format

Provide your analysis in the following JSON structure:

```json
{{
  "ranking": ["candidate_id_1", "candidate_id_2", "candidate_id_3"],
  "ranking_rationale": "Explanation of the ranking...",
  "comparison_matrix": {{
    "critical_thinking": {{
      "best": "candidate_id",
      "analysis": "Why this candidate excels in this area..."
    }},
    // ... for each criterion
  }},
  "candidate_insights": {{
    "candidate_id_1": {{
      "comparative_strengths": ["Strength 1", "Strength 2"],
      "comparative_weaknesses": ["Weakness 1"],
      "best_fit_for": "Type of role or situation where this candidate would excel"
    }},
    // ... for each candidate
  }},
  "recommendations": {{
    "candidate_id_1": "Specific next steps for this candidate",
    // ... for each candidate
  }},
  "key_insights": [
    "Notable insight 1 from the comparison",
    "Notable insight 2"
  ]
}}
```

Please provide your complete comparison analysis now.
"""


def get_evaluation_prompt(
    materials_text: str,
    custom_criteria: List[str] = None,
    use_all_criteria: bool = True,
    criteria_subset: List[EvaluationCriterion] = None,
    role: str = None
) -> str:
    """
    Generate evaluation prompt with materials and criteria.

    Args:
        materials_text: Combined text of all candidate materials
        custom_criteria: Optional list of custom criteria text to evaluate
        use_all_criteria: If True, evaluate on all 11 criteria (default)
        criteria_subset: Optional specific list of EvaluationCriterion to use

    Returns:
        Formatted prompt string
    """
    # Get criterion details
    criteria_details_text = []

    if custom_criteria:
        # Use custom criteria text if provided
        for i, criterion_text in enumerate(custom_criteria, 1):
            criteria_details_text.append(f"{i}. {criterion_text}")
    elif criteria_subset:
        # Use specified subset of criteria
        for criterion in criteria_subset:
            criteria_details_text.append(
                f"### {criterion.value}\n"
                f"**{criterion.display_name}**\n"
                f"{criterion.description}\n"
            )
    elif use_all_criteria:
        # Use ALL 11 criteria (new default)
        all_criteria = [
            EvaluationCriterion.CRITICAL_THINKING,
            EvaluationCriterion.COACHABILITY,
            EvaluationCriterion.CURIOSITY,
            EvaluationCriterion.CREATIVITY,
            EvaluationCriterion.COLLABORATION,
            EvaluationCriterion.FOLLOW_THROUGH,
            EvaluationCriterion.PROBLEM_SOLVING_MOTIVATION,
            EvaluationCriterion.EVIDENCE_BASED,
            EvaluationCriterion.DETAIL_ORIENTATION,
            EvaluationCriterion.COMMUNICATION,
            EvaluationCriterion.EXPERTISE_ENABLER
        ]

        for criterion in all_criteria:
            criteria_details_text.append(
                f"### {criterion.value}\n"
                f"**{criterion.display_name}**\n"
                f"{criterion.description}\n"
            )
    else:
        # Fallback to original 3 priority criteria (for backward compatibility)
        priority_criteria = [
            EvaluationCriterion.CREATIVITY,
            EvaluationCriterion.PROBLEM_SOLVING_MOTIVATION,
            EvaluationCriterion.DETAIL_ORIENTATION
        ]

        for criterion in priority_criteria:
            criteria_details_text.append(
                f"### {criterion.value}\n"
                f"**{criterion.display_name}**\n"
                f"{criterion.description}\n"
            )

    criteria_section = "\n\n".join(criteria_details_text)

    formatted = EVALUATION_PROMPT_TEMPLATE.format(
        materials=materials_text,
        criteria_details=criteria_section
    )
    if role:
        formatted += get_role_output_instruction(role)
    return formatted


def get_comparison_prompt(candidates_evaluations: List[Dict]) -> str:
    """
    Generate comparison prompt for multiple candidates.

    Args:
        candidates_evaluations: List of candidate evaluation summaries

    Returns:
        Formatted prompt string
    """
    candidates_text = []

    for eval_data in candidates_evaluations:
        candidate_section = f"""
## Candidate: {eval_data['candidate_id']}
- **Overall Score**: {eval_data['overall_score']:.2f}
- **Recommendation**: {eval_data['recommendation']}

### Scores by Criterion
{_format_scores(eval_data['scores'])}

### Overall Assessment
{eval_data['overall_assessment']}

### Strengths
{_format_list(eval_data['strengths'])}

### Areas for Development
{_format_list(eval_data['areas_for_development'])}
"""
        candidates_text.append(candidate_section)

    return COMPARISON_PROMPT_TEMPLATE.format(
        candidates_data="\n".join(candidates_text)
    )


def _format_scores(scores: Dict[str, int]) -> str:
    """Format scores dictionary as text"""
    lines = []
    for criterion, score in scores.items():
        criterion_obj = EvaluationCriterion(criterion)
        lines.append(f"- {criterion_obj.display_name}: {score}/10")
    return "\n".join(lines)


def _format_list(items: List[str]) -> str:
    """Format list as bulleted text"""
    if not items:
        return "- None specified"
    return "\n".join(f"- {item}" for item in items)


# Dr. Gray's Holistic Evaluation Mode - Enhanced Version
HOLISTIC_EVALUATION_PROMPT = """**CRITICAL: YOUR ENTIRE RESPONSE MUST BE VALID JSON WRAPPED IN ```json``` CODE BLOCKS.**

Do NOT write markdown narrative text. Do NOT write headers like "# Overall Assessment".
Your response MUST start with ```json and end with ```.
Any non-JSON output will cause a system failure.

---

# Comprehensive Holistic Candidate Evaluation

You are evaluating a candidate for an innovation and research program. Rather than scoring on specific pre-defined criteria, you will provide an in-depth holistic assessment based on the program's goals and the candidate's demonstrated qualities.

This evaluation should be THOROUGH and EVIDENCE-RICH - comparable in depth to a structured criteria-based evaluation.

## Program Description

This is a competitive fellowship program seeking candidates who can:
- Drive innovation and creative problem-solving in their field
- Translate research into practical impact
- Work effectively across disciplines and with diverse teams
- Demonstrate sustained commitment to challenging problems
- Learn, grow, and adapt based on feedback and new evidence
- Communicate complex ideas effectively to varied audiences

The program values candidates who show genuine evidence of these qualities through their actions and achievements, not just stated intentions.

## Candidate Materials

{materials}

---

## EVALUATION REQUIREMENTS

### Your Task

Provide a comprehensive holistic evaluation that deeply analyzes:

1. **Overall Assessment** (3-4 paragraphs)
   - Synthesize the candidate's profile holistically
   - Identify their core strengths and how they connect
   - Assess their trajectory and growth pattern
   - Compare implicitly to what a strong candidate looks like
   - Every claim must cite specific evidence

2. **Innovation Potential Assessment** (detailed)
   - Level: high, medium, or low
   - Confidence: high, medium, or low (based on evidence quality)
   - Detailed reasoning (2+ paragraphs explaining your assessment)
   - Minimum 3 pieces of evidence with quotes, sources, and context

3. **Program Fit Assessment** (detailed)
   - Level: strong, moderate, or weak
   - Confidence: high, medium, or low
   - Detailed analysis (2+ paragraphs)
   - Specific strengths aligned with program goals
   - Specific concerns or gaps
   - Minimum 3 pieces of evidence

4. **Notable Qualities** (minimum 3-5 qualities)
   - Each quality must have multiple pieces of supporting evidence
   - Explain significance for the program
   - Rate confidence in each assessment

5. **Red Flags** (be thorough)
   - Identify ALL concerning patterns, gaps, or warning signs
   - Rate severity: high, medium, or low
   - Cite specific evidence or note where evidence is missing
   - Include "fluff" statements found in application (generic claims without evidence)

6. **Interview Questions** (minimum 5 questions)
   - Categorize by purpose (e.g., "Probe claimed skill", "Verify experience", "Assess fit")
   - Explain what you're trying to learn with each question

7. **Final Recommendation**
   - Overall score with detailed justification
   - Clear interview decision (yes/no) with reasoning

---

## EVIDENCE REQUIREMENTS

**MINIMUM EVIDENCE STANDARDS:**
- Overall Assessment: Cite at least 5 direct quotes
- Innovation Potential: At least 3 structured evidence items (quote + source + context)
- Program Fit: At least 3 structured evidence items
- Each Notable Quality: At least 2 pieces of evidence
- Each Red Flag: Must cite specific evidence or explicitly note what's missing

**EVIDENCE FORMAT:**
For each piece of evidence, provide:
- **quote**: The exact text from the materials (use quotation marks)
- **source**: Which document it came from (e.g., "resume.pdf", "cover_letter.txt", "recommendation_letter.pdf")
- **context**: Why this evidence matters and what it demonstrates

---

## SCORING GUIDANCE

**CONFIDENCE LEVELS:**
- **High**: Multiple clear, specific examples; corroborated across sources
- **Medium**: Some evidence exists but limited; single source only
- **Low**: Minimal evidence; mostly inferred; stated but not demonstrated

**SCORING TIERS (for overall_score):**
- **9-10 (Exceptional)**: Clear evidence of outstanding qualities across multiple dimensions. Multiple strong, specific examples. Would be a standout in any cohort.
- **7-8 (Strong)**: Solid evidence across most areas. Clear strengths with minor gaps. Would contribute meaningfully to the program.
- **5-6 (Moderate)**: Mixed evidence. Notable strengths but also significant concerns or gaps. Requires careful consideration.
- **3-4 (Weak)**: Limited positive evidence. Significant concerns outweigh strengths. Unlikely to succeed without major development.
- **1-2 (Poor)**: Minimal relevant evidence. Major red flags. Not suitable for the program.

**INTERVIEW DECISION CRITERIA:**
- **Yes**: Score >= 6.0 AND no high-severity red flags AND evidence suggests potential for growth
- **No**: Score < 6.0 OR high-severity red flags OR fundamental misalignment with program goals

---

## CRITICAL REQUIREMENTS

- **REJECT FLUFF**: Generic statements like "passionate about innovation" or "strong work ethic" are worthless without specific examples. Call these out explicitly as red flags.
- **EVIDENCE ONLY**: Every positive assessment must be backed by specific, concrete evidence from the materials. No benefit of the doubt.
- **BE SKEPTICAL**: Most applications are 80% filler. Your job is to find the 20% that's real.
- **NO ASSUMPTIONS**: Only evaluate what is explicitly stated. If evidence is missing, say so explicitly.
- **QUOTE ACCURATELY**: When citing evidence, use exact quotes from the materials. Do not paraphrase.
- **SOURCE EVERYTHING**: Every claim needs a source document reference.

---

## OUTPUT FORMAT

```json
{{
  "overall_assessment": "3-4 paragraph comprehensive assessment with embedded evidence citations",

  "innovation_potential": {{
    "level": "high|medium|low",
    "confidence": "high|medium|low",
    "reasoning": "2+ paragraphs explaining the assessment with specific examples",
    "evidence": [
      {{
        "quote": "Exact quote from materials",
        "source": "document_name.pdf",
        "context": "What this demonstrates and why it matters"
      }}
    ]
  }},

  "program_fit": {{
    "level": "strong|moderate|weak",
    "confidence": "high|medium|low",
    "detailed_analysis": "2+ paragraphs analyzing fit with program goals",
    "strengths_for_program": ["Specific strength 1", "Specific strength 2"],
    "concerns": ["Specific concern 1", "Specific concern 2"],
    "evidence": [
      {{
        "quote": "Exact quote from materials",
        "source": "document_name.pdf",
        "context": "What this demonstrates"
      }}
    ]
  }},

  "notable_qualities": [
    {{
      "quality": "Quality name",
      "confidence": "high|medium|low",
      "evidence": [
        {{
          "quote": "Exact quote",
          "source": "document_name.pdf",
          "context": "What this shows"
        }}
      ],
      "significance": "Why this matters for the program"
    }}
  ],

  "red_flags": [
    {{
      "flag": "Description of the concern",
      "severity": "high|medium|low",
      "evidence": "Specific quote or 'No evidence found for claimed X'"
    }}
  ],

  "questions_for_interview": [
    {{
      "category": "Probe claimed skill|Verify experience|Assess motivation|Explore concern|Test fit",
      "question": "The specific question to ask",
      "purpose": "What you're trying to learn"
    }}
  ],

  "overall_score": 7.5,
  "score_justification": "Detailed paragraph explaining how the score was determined, citing key evidence and weighing strengths against concerns",
  "recommendation": "Strong fit|Potential fit with reservations|Not recommended",
  "interview_decision": true,
  "interview_decision_reasoning": "Clear explanation of why this candidate should/should not be interviewed"
}}
```

---

**REMINDER: YOUR RESPONSE MUST BE VALID JSON ONLY.**

Start your response with ```json and end with ```.
Do NOT write narrative text or markdown headers outside the JSON structure.
Put all narrative content INSIDE the JSON string fields.

**CRITICAL JSON FORMATTING RULES:**

1. When writing quotes inside JSON strings, you MUST escape internal quotation marks:
   WRONG: "quote": "He said "hello" to me"
   RIGHT: "quote": "He said \"hello\" to me"

2. When combining multiple quotes from different sources, keep them in a SINGLE string with escaped quotes:
   WRONG: "quote": "First quote" and "second quote" together
   RIGHT: "quote": "\"First quote\" and \"second quote\" together"

3. Never have bare text between quoted values. Everything in a string value must be ONE continuous quoted string:
   WRONG: "field": "text A" some words "text B" more words,
   RIGHT: "field": "text A, some words, text B, more words"

Provide your comprehensive holistic evaluation now in JSON format. Be thorough - this evaluation should take significant effort and provide deep insight into the candidate.
"""


def get_holistic_evaluation_prompt(materials_text: str, program_description: str = None, role: str = None) -> str:
    """
    Generate holistic evaluation prompt (Dr. Gray's suggested mode).

    Args:
        materials_text: Combined text of all candidate materials
        program_description: Optional custom program description to override default

    Returns:
        Formatted prompt string
    """
    prompt = HOLISTIC_EVALUATION_PROMPT

    if program_description:
        # Replace the default program description section
        prompt = prompt.replace(
            """This is a competitive fellowship program seeking candidates who can:
- Drive innovation and creative problem-solving in their field
- Translate research into practical impact
- Work effectively across disciplines and with diverse teams
- Demonstrate sustained commitment to challenging problems
- Learn, grow, and adapt based on feedback and new evidence
- Communicate complex ideas effectively to varied audiences

The program values candidates who show genuine evidence of these qualities through their actions and achievements, not just stated intentions.""",
            program_description
        )

    formatted = prompt.format(materials=materials_text)
    if role:
        formatted += get_role_output_instruction(role)
    return formatted


# Admit Pattern Analysis Prompt
ADMIT_PATTERN_ANALYSIS_PROMPT = """# Admit Pattern Analysis Task

You are analyzing a set of candidate evaluations to identify patterns that distinguish ADMITTED candidates from REJECTED candidates.

## Important Context

These are REAL admission decisions. The candidates labeled "admitted" were actually admitted to the program, and those labeled "rejected" were not admitted. Your task is to discover what patterns in their applications and evaluations correlate with admission decisions.

## Candidate Evaluation Summaries

Below are summarized evaluations for each candidate, along with their actual admission status.

{candidate_summaries}

---

## Analysis Instructions

Analyze all the candidate data above and identify:

1. **Score Patterns**: How do overall scores and criterion-specific scores differ between admitted and rejected candidates?

2. **Qualitative Patterns**: What qualities, experiences, or characteristics appear more frequently in admitted candidates?

3. **Evidence Patterns**: What types of evidence (specific achievements, metrics, experiences) are present in admitted candidates but missing in rejected ones?

4. **Red Flag Patterns**: What red flags or weaknesses appear more often in rejected candidates?

5. **Surprising Cases**: Are there any admitted candidates with lower scores or rejected candidates with higher scores? What might explain these cases?

6. **Threshold Analysis**: Is there an apparent score threshold for admission? What score range represents the "gray zone" where admission becomes uncertain?

---

## Output Format

Provide your analysis in the following JSON structure:

```json
{{
  "executive_summary": "2-3 paragraph summary of the key findings. What most distinguishes admitted from rejected candidates?",
  
  "score_analysis": {{
    "admitted_mean_score": 7.5,
    "rejected_mean_score": 5.2,
    "score_difference": 2.3,
    "apparent_threshold": "Candidates scoring 6.5+ appear more likely to be admitted",
    "gray_zone": "Scores between 5.5-6.5 seem to be the decision boundary",
    "criterion_differences": [
      {{
        "criterion": "criterion_name",
        "admitted_mean": 7.8,
        "rejected_mean": 4.5,
        "difference": 3.3,
        "importance": "This criterion appears highly predictive of admission"
      }}
    ]
  }},
  
  "key_patterns": [
    {{
      "category_name": "Pattern Category Name",
      "description": "Detailed description of what distinguishes admitted from rejected in this category",
      "importance": "high|medium|low",
      "patterns": [
        {{
          "pattern": "Specific pattern description",
          "admitted_examples": ["Example from admitted candidate 1", "Example from admitted candidate 2"],
          "rejected_examples": ["Counter-example showing this was missing in rejected candidate"],
          "confidence": "high|medium|low"
        }}
      ]
    }}
  ],
  
  "admitted_strengths": [
    "Common strength 1 found in admitted candidates",
    "Common strength 2 found in admitted candidates"
  ],
  
  "rejected_weaknesses": [
    "Common weakness 1 found in rejected candidates",
    "Common weakness 2 found in rejected candidates"
  ],
  
  "surprising_admits": [
    {{
      "candidate_id": "ID of surprisingly admitted candidate",
      "score": 5.5,
      "reason": "Why this admission is surprising based on the data",
      "possible_explanation": "What might explain this decision"
    }}
  ],
  
  "surprising_rejects": [
    {{
      "candidate_id": "ID of surprisingly rejected candidate",
      "score": 7.8,
      "reason": "Why this rejection is surprising based on the data",
      "possible_explanation": "What might explain this decision"
    }}
  ],
  
  "predictive_factors": [
    {{
      "factor": "Factor name",
      "direction": "Presence increases/decreases admission likelihood",
      "strength": "strong|moderate|weak",
      "evidence": "How this was determined from the data"
    }}
  ],
  
  "methodology_notes": "Any limitations, caveats, or notes about this analysis"
}}
```

---

## Critical Requirements

1. **USE ACTUAL DATA**: Only cite patterns that are actually present in the candidate summaries provided. Do not make up examples.

2. **BE SPECIFIC**: When citing patterns, reference specific candidate IDs and their actual scores/qualities.

3. **QUANTIFY WHERE POSSIBLE**: Calculate actual means, differences, and frequencies rather than using vague language.

4. **ACKNOWLEDGE LIMITATIONS**: If the sample size is small or patterns are unclear, say so explicitly.

5. **LOOK FOR NON-OBVIOUS PATTERNS**: Beyond just "higher scores = admitted", look for specific criteria or qualities that seem predictive.

6. **CONSIDER EXCEPTIONS**: The most interesting insights often come from cases that don't fit the pattern.

Provide your complete analysis now.
"""


def get_admit_pattern_analysis_prompt(candidate_summaries: List[Dict]) -> str:
    """
    Generate prompt for analyzing patterns between admitted and rejected candidates.
    
    Args:
        candidate_summaries: List of dictionaries containing:
            - candidate_id: str
            - admit_status: bool (True = admitted)
            - overall_score: float
            - scores: Dict[str, int] (criterion -> score)
            - strengths: List[str]
            - weaknesses: List[str]
            - recommendation: str
            - key_evidence: List[str] (optional)
            
    Returns:
        Formatted prompt string
    """
    # Format candidate summaries for the prompt
    summaries_text = []
    
    # Separate admitted and rejected for clarity
    admitted = [c for c in candidate_summaries if c.get('admit_status')]
    rejected = [c for c in candidate_summaries if not c.get('admit_status')]
    
    summaries_text.append(f"## ADMITTED CANDIDATES ({len(admitted)} total)\n")
    for candidate in admitted:
        summaries_text.append(_format_candidate_summary(candidate, "ADMITTED"))
    
    summaries_text.append(f"\n## REJECTED CANDIDATES ({len(rejected)} total)\n")
    for candidate in rejected:
        summaries_text.append(_format_candidate_summary(candidate, "REJECTED"))
    
    return ADMIT_PATTERN_ANALYSIS_PROMPT.format(
        candidate_summaries="\n".join(summaries_text)
    )


def _format_candidate_summary(candidate: Dict, status: str) -> str:
    """Format a single candidate summary for the prompt."""
    lines = [
        f"### {candidate.get('candidate_id', 'Unknown')} [{status}]",
        f"- **Overall Score**: {candidate.get('overall_score', 'N/A'):.1f}/10" if candidate.get('overall_score') else "- **Overall Score**: N/A",
        f"- **Recommendation**: {candidate.get('recommendation', 'N/A')}",
    ]
    
    # Add criterion scores if available
    scores = candidate.get('scores', {})
    if scores:
        lines.append("- **Criterion Scores**:")
        for criterion, score in sorted(scores.items()):
            display_name = criterion.replace('_', ' ').title()
            lines.append(f"  - {display_name}: {score}/10")
    
    # Add strengths
    strengths = candidate.get('strengths', [])
    if strengths:
        lines.append("- **Strengths**:")
        for s in strengths[:5]:  # Limit to top 5
            lines.append(f"  - {s}")
    
    # Add weaknesses/areas for development
    weaknesses = candidate.get('weaknesses', []) or candidate.get('areas_for_development', [])
    if weaknesses:
        lines.append("- **Weaknesses/Gaps**:")
        for w in weaknesses[:5]:
            lines.append(f"  - {w}")
    
    # Add key evidence if available
    key_evidence = candidate.get('key_evidence', [])
    if key_evidence:
        lines.append("- **Key Evidence**:")
        for ev in key_evidence[:3]:
            lines.append(f"  - {ev}")
    
    # Add red flags if available (from holistic evaluations)
    red_flags = candidate.get('red_flags', [])
    if red_flags:
        lines.append("- **Red Flags**:")
        for rf in red_flags[:3]:
            if isinstance(rf, dict):
                lines.append(f"  - {rf.get('flag', str(rf))}")
            else:
                lines.append(f"  - {rf}")
    
    lines.append("")  # Empty line between candidates
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interview-selection prompts (two-phase, context-window-efficient)
# ---------------------------------------------------------------------------

HOLISTIC_RANKING_PROMPT_TEMPLATE = """# Candidate Ranking for Interview Selection

You are acting as a senior admissions reviewer for a competitive fellowship program.
Your task is to rank ALL {n_candidates} candidates from strongest to weakest based on
how well they fit the program — not just their numeric score.

## Program Description

This is a competitive fellowship program seeking candidates who can:
- Drive innovation and creative problem-solving in their field
- Translate research into practical impact
- Work effectively across disciplines and with diverse teams
- Demonstrate sustained commitment to challenging problems
- Learn, grow, and adapt based on feedback and new evidence
- Communicate complex ideas effectively to varied audiences

The program values candidates who show genuine evidence of these qualities through
their actions and achievements, not just stated intentions.

## Ranking Criteria (apply in this priority order)

1. **Program Fit** — How directly do the candidate's demonstrated strengths match
   what the program is looking for? Strong > Moderate > Weak, with confidence level
   as a multiplier (high confidence strong fit outranks low confidence strong fit).

2. **Innovation Potential** — Is there genuine, specific evidence of creative,
   impactful thinking? High-confidence assessments outweigh low-confidence ones.

3. **Evidence Quality** — Concrete, specific achievements backed by multiple sources
   outrank vague aspirations or self-reported qualities without corroboration.

4. **Red Flags** — High-severity flags are near-disqualifying. Multiple medium-severity
   flags compound. Absence of red flags is a meaningful positive signal.

5. **Overall Score** — Use as a tiebreaker when holistic signals are otherwise equal.

## Candidate Profiles

{candidates_data}

---

## Task

Rank ALL {n_candidates} candidates from strongest to weakest for program fit and
interview selection. Write one sentence per candidate explaining the key reason
they placed where they did.

## Output

Respond with valid JSON only — no prose outside the JSON block:

```json
{{
  "ranking": ["candidate_id_1", "candidate_id_2", ...],
  "ranking_rationale": "2-3 sentences summarising the overall ranking logic and what separated the top candidates from the rest.",
  "candidate_reasoning": {{
    "candidate_id": "One sentence on why this candidate placed where they did."
  }}
}}
```

The `ranking` list MUST contain ALL {n_candidates} candidate IDs exactly once, ordered strongest-first.
"""


HOLISTIC_SELECTION_PROMPT_TEMPLATE = """# Final Interview-Candidate Selection

You are the final decision-maker for a competitive fellowship program.
From a pre-screened pool of {n_pool} strong candidates, select exactly {n_interviews}
to invite for interview.

## Program Description

This is a competitive fellowship program seeking candidates who can:
- Drive innovation and creative problem-solving in their field
- Translate research into practical impact
- Work effectively across disciplines and with diverse teams
- Demonstrate sustained commitment to challenging problems
- Learn, grow, and adapt based on feedback and new evidence
- Communicate complex ideas effectively to varied audiences

The program values candidates who show genuine evidence of these qualities through
their actions and achievements, not just stated intentions.

## Selection Principles

Apply these principles when choosing your final {n_interviews}:

- **Program Fit First**: Prioritise candidates whose demonstrated strengths directly
  match what the program values most.
- **Evidence Over Claims**: A candidate with fewer but concrete achievements ranks
  above one with many vague aspirations.
- **Confidence Matters**: High-confidence assessments are worth more than the same
  assessment at low confidence.
- **Red Flags Are Disqualifying**: High-severity red flags should prevent selection
  unless offset by exceptional, well-evidenced strengths.
- **Cohort Diversity**: Where candidates are otherwise comparable, prefer a group
  with complementary rather than identical strengths.

## Finalist Profiles

{candidates_data}

---

## Task

Select EXACTLY {n_interviews} candidate(s) to invite for interview.
For every candidate in the pool — selected or not — write 1-2 sentences explaining
your decision.

## Output

Respond with valid JSON only — no prose outside the JSON block:

```json
{{
  "selected": ["candidate_id_1", "candidate_id_2", ...],
  "selection_rationale": "2-3 sentences explaining the overall selection logic: what the selected group has in common and why the others were not chosen.",
  "candidate_notes": {{
    "candidate_id": "1-2 sentences on why this candidate was selected or not selected, citing the decisive factor."
  }}
}}
```

The `selected` list MUST contain EXACTLY {n_interviews} candidate ID(s), ordered strongest-first.
"""


# ---------------------------------------------------------------------------
# Candidate profile formatters for the two-phase selection prompts
# ---------------------------------------------------------------------------

def _format_ranking_candidate(evaluation: HolisticEvaluationResult) -> str:
    """
    Format a full HolisticEvaluationResult into a rich text block for Phase-1 ranking.

    Includes reasoning snippets so the model can make program-fit judgements rather
    than relying solely on numeric scores.
    """
    cid = evaluation.candidate.candidate_id
    name_part = f" — {evaluation.candidate.name}" if evaluation.candidate.name else ""

    # Red-flag summary
    red_flags = evaluation.red_flags if isinstance(evaluation.red_flags, list) else []
    high_rf = sum(1 for rf in red_flags if getattr(rf, "severity", "medium") == "high")
    med_rf = sum(1 for rf in red_flags if getattr(rf, "severity", "medium") == "medium")
    rf_detail = ", ".join(
        getattr(rf, "flag", str(rf)) for rf in red_flags[:3]
    ) or "None identified"

    # Program-fit strengths/concerns (up to 2 each, compact)
    pf = evaluation.program_fit
    strengths_text = "; ".join(pf.strengths_for_program[:2]) or "Not specified"
    concerns_text = "; ".join(pf.concerns[:2]) or "None noted"

    # Truncate long reasoning fields to keep context tight
    def _truncate(text: str, limit: int = 200) -> str:
        return text[:limit].rstrip() + "…" if len(text) > limit else text

    lines = [
        f"### {cid}{name_part}",
        f"**Score**: {evaluation.overall_score}/10  |  "
        f"**Recommendation**: {evaluation.recommendation}  |  "
        f"**Individual interview decision**: {'Yes' if evaluation.interview_decision else 'No'}",
        f"**Interview decision reasoning**: {_truncate(evaluation.interview_decision_reasoning, 180)}",
        "",
        f"**Innovation Potential**: {evaluation.innovation_potential.level.upper()} "
        f"(confidence: {evaluation.innovation_potential.confidence})",
        f"  → {_truncate(evaluation.innovation_potential.reasoning, 200)}",
        "",
        f"**Program Fit**: {pf.level.upper()} (confidence: {pf.confidence})",
        f"  → {_truncate(pf.detailed_analysis, 200)}",
        f"  → Strengths for program: {strengths_text}",
        f"  → Concerns: {concerns_text}",
        "",
        f"**Red Flags**: {len(red_flags)} total "
        f"({high_rf} high-severity, {med_rf} medium-severity)",
        f"  → {rf_detail}",
        "",
    ]
    return "\n".join(lines)


def _format_selection_candidate(evaluation: HolisticEvaluationResult) -> str:
    """
    Format a full HolisticEvaluationResult into a rich text block for Phase-2 selection.

    Includes notable qualities on top of everything in the ranking formatter, giving
    the model more signal for final differentiation within a strong pool.
    """
    base = _format_ranking_candidate(evaluation)

    # Notable qualities (up to 3, with significance)
    qualities = evaluation.notable_qualities[:3]
    if qualities:
        quality_lines = ["**Notable Qualities**:"]
        for q in qualities:
            sig = getattr(q, "significance", "")
            quality_lines.append(
                f"  - {q.quality} (confidence: {getattr(q, 'confidence', 'medium')})"
                + (f" — {sig}" if sig else "")
            )
        return base + "\n".join(quality_lines) + "\n"

    return base


def get_holistic_ranking_prompt(
    evaluations: List[HolisticEvaluationResult],
    template: Optional[str] = None,
) -> str:
    """
    Build the Phase-1 ranking prompt from a list of HolisticEvaluationResult objects.

    Args:
        evaluations: Full holistic evaluation results for all candidates.
        template: Optional custom template string (must contain ``{n_candidates}``
                  and ``{candidates_data}`` placeholders).  Defaults to
                  ``HOLISTIC_RANKING_PROMPT_TEMPLATE``.

    Returns:
        Formatted ranking prompt string.
    """
    if template is None:
        template = HOLISTIC_RANKING_PROMPT_TEMPLATE
    candidates_text = "\n".join(_format_ranking_candidate(e) for e in evaluations)
    return template.format(
        n_candidates=len(evaluations),
        candidates_data=candidates_text,
    )


def get_holistic_selection_prompt(
    evaluations: List[HolisticEvaluationResult],
    n_interviews: int,
    template: Optional[str] = None,
) -> str:
    """
    Build the Phase-2 selection prompt from top-pool HolisticEvaluationResult objects.

    Args:
        evaluations: Full holistic evaluation results for the top-pool candidates.
        n_interviews: Exact number of candidates to select.
        template: Optional custom template string (must contain ``{n_pool}``,
                  ``{n_interviews}``, and ``{candidates_data}`` placeholders).
                  Defaults to ``HOLISTIC_SELECTION_PROMPT_TEMPLATE``.

    Returns:
        Formatted selection prompt string.
    """
    if template is None:
        template = HOLISTIC_SELECTION_PROMPT_TEMPLATE
    candidates_text = "\n".join(_format_selection_candidate(e) for e in evaluations)
    return template.format(
        n_pool=len(evaluations),
        n_interviews=n_interviews,
        candidates_data=candidates_text,
    )


# ---------------------------------------------------------------------------
# Role-specific system prompt addenda (appended to SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

from candidate_evaluator.prompts.role_prompts import (
    COMBINED_ROLE_PROMPT,
    ROLE_ALIASES,
    ROLE_PROMPTS,
    get_role_addendum,
    get_role_output_instruction,
    normalize_role,
)
