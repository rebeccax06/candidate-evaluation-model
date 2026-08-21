"""Core candidate evaluation engine using Claude API"""

import json
import logging
import os
import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from candidate_evaluator.core.models import (
    EvaluationResult,
    CriterionScore,
    CandidateProfile,
    Evidence,
    EvaluationCriterion,
    ComparisonResult,
    HolisticEvaluationResult,
    InnovationPotential,
    ProgramFit,
    NotableQuality,
    HolisticEvidence,
    RedFlag,
    InterviewQuestion,
    InterviewSelectionResult,
    ScreeningResult,
    parse_role_specific_assessment,
)
from candidate_evaluator.core.research_generator import ResearchReportGenerator
from candidate_evaluator.core.research_models import ResearchEvaluationReport
from candidate_evaluator.utils.config import Config
from candidate_evaluator.utils.file_processor import FileProcessor
from candidate_evaluator.prompts.evaluation_prompts import (
    SYSTEM_PROMPT,
    get_evaluation_prompt,
    get_comparison_prompt,
    get_holistic_evaluation_prompt,
    get_holistic_ranking_prompt,
    get_holistic_selection_prompt,
    get_screening_prompt,
)
from candidate_evaluator.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class CandidateEvaluator:
    """Main class for evaluating candidates using Claude API"""

    def __init__(self, config: Config):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.client = Anthropic(api_key=config.api.anthropic_api_key)
        self.file_processor = FileProcessor(
            max_file_size_mb=config.processing.max_file_size_mb
        )
        self.research_generator = ResearchReportGenerator()
        self.prompt_manager = PromptManager()
        self._last_processed_files = None  # Store for research report generation

    def evaluate_candidate(
        self,
        candidate_id: str,
        material_paths: List[str],
        candidate_name: Optional[str] = None,
        custom_criteria: Optional[List[str]] = None,
        role: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a candidate based on their application materials.

        Args:
            candidate_id: Unique identifier for the candidate
            material_paths: List of paths to candidate materials
            candidate_name: Optional name of the candidate
            custom_criteria: Optional custom evaluation criteria
            role: Optional role-specific context (clinician, engineer, phd)

        Returns:
            EvaluationResult object

        Raises:
            ValueError: If materials cannot be processed or evaluation fails
        """
        logger.info(f"Starting evaluation for candidate: {candidate_id}" + (f" (role: {role})" if role else ""))
        start_time = time.time()

        # Process files
        logger.info(f"Processing {len(material_paths)} files...")
        processed_files = self.file_processor.process_multiple_files(material_paths)
        self._last_processed_files = processed_files  # Store for research report generation
        combined_materials = self.file_processor.combine_materials(processed_files)

        logger.info(f"Total materials length: {len(combined_materials)} characters")

        # Generate evaluation prompt
        prompt = get_evaluation_prompt(combined_materials, custom_criteria, role=role)

        # Call Claude API
        logger.info("Calling Claude API for evaluation...")
        response = self._call_claude_api(prompt, role=role)

        # Parse response
        logger.info("Parsing evaluation response...")
        evaluation_data = self._parse_evaluation_response(response)

        # Calculate overall score
        weights = self.config.criteria.weights
        weighted_sum = 0
        total_weight = 0

        scores = []
        for score_data in evaluation_data['criterion_scores']:
            # Skip items without a valid criterion (e.g. overall_assessment mixed in)
            criterion_value = score_data.get('criterion')
            if not criterion_value:
                continue
            try:
                criterion = EvaluationCriterion(criterion_value)
            except ValueError:
                logger.warning(f"Skipping invalid criterion: {criterion_value}")
                continue
            weight = getattr(weights, criterion.value, 10)

            weighted_sum += score_data['score'] * weight
            total_weight += weight

            # Create evidence objects
            evidence_list = [
                Evidence(**ev) for ev in score_data.get('evidence', [])
            ]

            # Normalize confidence to valid values (low, medium, high)
            confidence_raw = score_data.get('confidence', 'medium').lower()
            if 'high' in confidence_raw:
                confidence = 'high'
            elif 'low' in confidence_raw:
                confidence = 'low'
            else:
                confidence = 'medium'

            scores.append(CriterionScore(
                criterion=criterion,
                score=score_data['score'],
                reasoning=score_data['reasoning'],
                evidence=evidence_list,
                confidence=confidence,
                notes=score_data.get('notes')
            ))

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Create candidate profile
        candidate = CandidateProfile(
            candidate_id=candidate_id,
            name=candidate_name,
            materials=[str(Path(p).name) for p in material_paths],
            evaluation_date=datetime.now()
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create evaluation result
        result = EvaluationResult(
            candidate=candidate,
            scores=scores,
            overall_score=overall_score,
            overall_assessment=evaluation_data.get('overall_assessment', ''),
            strengths=evaluation_data.get('strengths', []),
            areas_for_development=evaluation_data.get('areas_for_development', []),
            recommendation=evaluation_data.get('recommendation', ''),
            role=role,
            role_specific_assessment=parse_role_specific_assessment(
                evaluation_data.get('role_specific_assessment')
            ),
            metadata={
                'model': self.config.api.model,
                'processing_time_seconds': processing_time,
                'materials_character_count': len(combined_materials),
                'timestamp': datetime.now().isoformat()
            }
        )

        logger.info(f"Evaluation completed in {processing_time:.2f} seconds")
        logger.info(f"Overall score: {overall_score:.2f}")

        return result

    def evaluate_batch(
        self,
        candidates: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple candidates in batch.

        Args:
            candidates: List of candidate dictionaries with keys:
                - candidate_id: str
                - material_paths: List[str]
                - candidate_name: Optional[str]
            batch_size: Optional batch size (defaults to config)

        Returns:
            List of EvaluationResult objects
        """
        if batch_size is None:
            batch_size = self.config.processing.batch_size

        logger.info(f"Starting batch evaluation of {len(candidates)} candidates")

        results = []
        for i, candidate_data in enumerate(candidates, 1):
            logger.info(f"Processing candidate {i}/{len(candidates)}: {candidate_data['candidate_id']}")

            try:
                result = self.evaluate_candidate(
                    candidate_id=candidate_data['candidate_id'],
                    material_paths=candidate_data['material_paths'],
                    candidate_name=candidate_data.get('candidate_name')
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to evaluate candidate {candidate_data['candidate_id']}: {e}")
                # Continue with other candidates
                continue

        logger.info(f"Batch evaluation completed: {len(results)}/{len(candidates)} successful")
        return results

    def compare_candidates(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> ComparisonResult:
        """
        Compare multiple evaluated candidates.

        Args:
            evaluation_results: List of EvaluationResult objects

        Returns:
            ComparisonResult object
        """
        logger.info(f"Comparing {len(evaluation_results)} candidates")

        # Prepare candidate data for comparison
        candidates_data = [
            result.to_summary_dict() for result in evaluation_results
        ]

        # Generate comparison prompt
        prompt = get_comparison_prompt(candidates_data)

        # Call Claude API
        logger.info("Calling Claude API for comparison...")
        response = self._call_claude_api(prompt)

        # Parse comparison response
        logger.info("Parsing comparison response...")
        comparison_data = self._parse_comparison_response(response)

        # Create comparison matrix
        comparison_matrix = comparison_data.get('comparison_matrix', {})

        result = ComparisonResult(
            candidates=evaluation_results,
            ranking=comparison_data.get('ranking', []),
            comparison_matrix=comparison_matrix,
            insights=comparison_data.get('key_insights', [])
        )

        logger.info("Comparison completed")
        return result

    def select_interviews_holistic(
        self,
        evaluations: List[HolisticEvaluationResult],
        n_interviews: int,
        pool_multiplier: float = 2.0,
    ) -> InterviewSelectionResult:
        """
        Two-phase, context-window-efficient interview selection over holistic evaluations.

        Phase 1 — Initial ranking (skipped when the candidate pool is small enough):
            All candidates are ranked using their compact ``to_summary_dict()``
            representations.  No raw materials are re-sent to the model.

        Phase 2 — Final selection:
            Only the top ``pool_size`` candidates (determined by ``pool_multiplier``)
            are passed to Claude for a deeper comparison, and exactly ``n_interviews``
            are selected.

        Args:
            evaluations:     List of already-computed HolisticEvaluationResult objects.
            n_interviews:    Number of interview slots to fill.
            pool_multiplier: How many times ``n_interviews`` to include in the Phase-2
                             pool (minimum pool size is ``n_interviews + 2``).

        Returns:
            InterviewSelectionResult with ranked + selected candidate IDs.

        Raises:
            ValueError: If ``n_interviews`` is less than 1 or exceeds the pool.
        """
        if n_interviews < 1:
            raise ValueError("n_interviews must be at least 1")
        if n_interviews > len(evaluations):
            raise ValueError(
                f"n_interviews ({n_interviews}) exceeds the number of candidates "
                f"({len(evaluations)})"
            )

        start_time = time.time()
        all_ids = [e.candidate.candidate_id for e in evaluations]

        # Determine intermediate pool size
        pool_size = min(
            len(evaluations),
            max(n_interviews + 2, int(n_interviews * pool_multiplier)),
        )

        # ------------------------------------------------------------------ #
        # Phase 1 – initial ranking across the full candidate set              #
        # (skipped when every candidate will already fit in the Phase-2 pool)  #
        # ------------------------------------------------------------------ #
        ranking_template = self.prompt_manager.get_ranking_template()

        if pool_size >= len(evaluations):
            # Pool covers all candidates — no LLM call needed; sort by score
            logger.info(
                "Pool size covers all candidates — skipping Phase-1 LLM ranking, "
                "using score-based order."
            )
            ranked_all = sorted(
                all_ids,
                key=lambda cid: next(
                    (e.overall_score for e in evaluations if e.candidate.candidate_id == cid),
                    0.0,
                ),
                reverse=True,
            )
            phase1_rationale = "Score-based pre-sort (Phase 1 LLM call skipped: pool covers all candidates)."
        else:
            logger.info(
                f"Phase 1: ranking all {len(evaluations)} candidates …"
            )
            ranking_prompt = get_holistic_ranking_prompt(evaluations, template=ranking_template)
            response1 = self._call_claude_api(ranking_prompt)
            ranking_data = self._parse_comparison_response(response1)
            ranked_all = ranking_data.get("ranking", [])
            phase1_rationale = ranking_data.get("ranking_rationale", "")

            # Guard against missing or extra IDs returned by the model
            ranked_set = set(ranked_all)
            missing = [cid for cid in all_ids if cid not in ranked_set]
            ranked_all = ranked_all + missing  # append any the model omitted

        # ------------------------------------------------------------------ #
        # Phase 2 – deep comparison within the top pool                        #
        # ------------------------------------------------------------------ #
        pool_ids = ranked_all[:pool_size]
        eval_by_id = {e.candidate.candidate_id: e for e in evaluations}
        pool_evals = [eval_by_id[cid] for cid in pool_ids if cid in eval_by_id]

        logger.info(
            f"Phase 2: selecting {n_interviews} from top-{len(pool_evals)} pool …"
        )
        selection_template = self.prompt_manager.get_selection_template()
        selection_prompt = get_holistic_selection_prompt(
            pool_evals, n_interviews, template=selection_template
        )
        response2 = self._call_claude_api(selection_prompt)
        selection_data = self._parse_comparison_response(response2)

        selected_ids = selection_data.get("selected", [])
        # Fallback: take the top-N from the pool if the model returns nothing
        if not selected_ids:
            selected_ids = pool_ids[:n_interviews]

        elapsed = time.time() - start_time
        logger.info(
            f"Interview selection completed in {elapsed:.2f}s — "
            f"selected {len(selected_ids)} of {len(evaluations)} candidates."
        )

        return InterviewSelectionResult(
            selected_candidate_ids=selected_ids,
            pool_candidate_ids=pool_ids,
            all_candidate_ids_ranked=ranked_all,
            n_interviews_requested=n_interviews,
            selection_rationale=selection_data.get("selection_rationale", ""),
            candidate_notes=selection_data.get("candidate_notes", {}),
            pool_multiplier=pool_multiplier,
            metadata={
                "model": self.config.api.model,
                "processing_time_seconds": elapsed,
                "n_total_candidates": len(evaluations),
                "pool_size": len(pool_evals),
                "phase1_rationale": phase1_rationale,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def evaluate_candidate_holistic(
        self,
        candidate_id: str,
        material_paths: List[str],
        candidate_name: Optional[str] = None,
        program_description: Optional[str] = None,
        role: Optional[str] = None
    ) -> HolisticEvaluationResult:
        """
        Evaluate a candidate using holistic (criteria-free) approach.

        This mode evaluates candidates based on overall program fit without
        using the predefined 11 evaluation criteria, allowing for more
        open-ended assessment.

        Args:
            candidate_id: Unique identifier for the candidate
            material_paths: List of paths to candidate materials
            candidate_name: Optional name of the candidate
            program_description: Optional custom program description
            role: Optional role-specific context (clinician, engineer, phd)

        Returns:
            HolisticEvaluationResult object

        Raises:
            ValueError: If materials cannot be processed or evaluation fails
        """
        logger.info(f"Starting holistic evaluation for candidate: {candidate_id}" + (f" (role: {role})" if role else ""))
        start_time = time.time()

        # Process files
        logger.info(f"Processing {len(material_paths)} files...")
        processed_files = self.file_processor.process_multiple_files(material_paths)
        self._last_processed_files = processed_files
        combined_materials = self.file_processor.combine_materials(processed_files)

        logger.info(f"Total materials length: {len(combined_materials)} characters")

        # Generate holistic evaluation prompt
        prompt = get_holistic_evaluation_prompt(combined_materials, program_description, role=role)

        # Call Claude API with retry logic for JSON parsing failures
        max_retries = 2
        last_error = None
        evaluation_data = None

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    logger.info("Calling Claude API for holistic evaluation...")
                    response = self._call_claude_api(prompt, role=role)
                else:
                    # Retry with explicit JSON correction request
                    logger.warning(f"Retry attempt {attempt} due to JSON parsing failure")
                    retry_prompt = f"""Your previous response could not be parsed as valid JSON.
Error: {last_error}

Please regenerate your ENTIRE evaluation as valid JSON only.
Start with ```json and end with ```.
Make sure all quotes inside string values are escaped with backslash: \\"
Do NOT include any text outside the JSON structure.

Original request:
{prompt}"""
                    response = self._call_claude_api(retry_prompt, role=role)

                # Parse response
                logger.info("Parsing holistic evaluation response...")
                evaluation_data = self._parse_holistic_response(response)
                break  # Success, exit retry loop

            except ValueError as e:
                last_error = str(e)
                if attempt < max_retries:
                    logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {e}")
                else:
                    logger.error(f"All {max_retries + 1} attempts failed to parse JSON")
                    raise

        if evaluation_data is None:
            raise ValueError("Failed to get valid evaluation data after retries")

        # Create candidate profile
        candidate = CandidateProfile(
            candidate_id=candidate_id,
            name=candidate_name,
            materials=[str(Path(p).name) for p in material_paths],
            evaluation_date=datetime.now()
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Helper to parse structured evidence
        def parse_evidence_list(evidence_data):
            """Parse evidence list - handles both old string format and new structured format."""
            if not evidence_data:
                return []
            evidence_list = []
            for ev in evidence_data:
                if isinstance(ev, dict):
                    evidence_list.append(HolisticEvidence(
                        quote=ev.get('quote', ''),
                        source=ev.get('source', ''),
                        context=ev.get('context', '')
                    ))
                elif isinstance(ev, str):
                    # Legacy string format
                    evidence_list.append(HolisticEvidence(quote=ev, source='', context=''))
            return evidence_list

        # Build innovation potential (enhanced format)
        innovation_data = evaluation_data.get('innovation_potential', {})
        innovation_potential = InnovationPotential(
            level=innovation_data.get('level', 'medium'),
            confidence=innovation_data.get('confidence', 'medium'),
            reasoning=innovation_data.get('reasoning', ''),
            key_evidence=innovation_data.get('key_evidence', []),  # Legacy format
            evidence=parse_evidence_list(innovation_data.get('evidence', []))
        )

        # Build program fit (enhanced format)
        fit_data = evaluation_data.get('program_fit', {})
        program_fit = ProgramFit(
            level=fit_data.get('level', 'moderate'),
            confidence=fit_data.get('confidence', 'medium'),
            detailed_analysis=fit_data.get('detailed_analysis', ''),
            strengths_for_program=fit_data.get('strengths_for_program', []),
            concerns=fit_data.get('concerns', []),
            evidence=parse_evidence_list(fit_data.get('evidence', []))
        )

        # Build notable qualities (enhanced format)
        notable_qualities = []
        for q in evaluation_data.get('notable_qualities', []):
            if isinstance(q, dict):
                # Handle evidence which can be string or list
                ev = q.get('evidence', '')
                if isinstance(ev, list):
                    ev = parse_evidence_list(ev)
                notable_qualities.append(NotableQuality(
                    quality=q.get('quality', ''),
                    confidence=q.get('confidence', 'medium'),
                    evidence=ev,
                    significance=q.get('significance', '')
                ))

        # Build red flags (can be strings or structured objects)
        raw_red_flags = evaluation_data.get('red_flags', [])
        red_flags = []
        for rf in raw_red_flags:
            if isinstance(rf, dict):
                red_flags.append(RedFlag(
                    flag=rf.get('flag', ''),
                    severity=rf.get('severity', 'medium'),
                    evidence=rf.get('evidence', '')
                ))
            elif isinstance(rf, str):
                red_flags.append(rf)  # Keep legacy string format

        # Build interview questions (can be strings or structured objects)
        raw_questions = evaluation_data.get('questions_for_interview', [])
        questions = []
        for q in raw_questions:
            if isinstance(q, dict):
                questions.append(InterviewQuestion(
                    category=q.get('category', 'General'),
                    question=q.get('question', ''),
                    purpose=q.get('purpose', '')
                ))
            elif isinstance(q, str):
                questions.append(q)  # Keep legacy string format

        # Create holistic evaluation result
        result = HolisticEvaluationResult(
            candidate=candidate,
            overall_assessment=evaluation_data.get('overall_assessment', ''),
            innovation_potential=innovation_potential,
            program_fit=program_fit,
            notable_qualities=notable_qualities,
            red_flags=red_flags,
            questions_for_interview=questions,
            overall_score=float(evaluation_data.get('overall_score', 5.0)),
            score_justification=evaluation_data.get('score_justification', ''),
            recommendation=evaluation_data.get('recommendation', ''),
            interview_decision=evaluation_data.get('interview_decision', False),
            interview_decision_reasoning=evaluation_data.get('interview_decision_reasoning', ''),
            role=role,
            role_specific_assessment=parse_role_specific_assessment(
                evaluation_data.get('role_specific_assessment')
            ),
            metadata={
                'model': self.config.api.model,
                'processing_time_seconds': processing_time,
                'materials_character_count': len(combined_materials),
                'timestamp': datetime.now().isoformat(),
                'evaluation_mode': 'holistic_enhanced'
            }
        )

        logger.info(f"Holistic evaluation completed in {processing_time:.2f} seconds")
        logger.info(f"Overall score: {result.overall_score:.2f}, Interview: {result.interview_decision}")

        return result

    def screen_candidate(
        self,
        candidate_id: str,
        material_paths: List[str],
        description: str,
        candidate_name: Optional[str] = None
    ) -> ScreeningResult:
        """
        Screen a candidate against a natural-language target profile description.

        Produces a binary match/no-match decision (with confidence and evidence)
        rather than a full evaluation. Useful for filtering large candidate pools.

        Args:
            candidate_id: Unique identifier for the candidate
            material_paths: List of paths to candidate materials
            description: Natural-language description of the target profile. May
                include both inclusion requirements (must have) and exclusion
                requirements (must not have).
            candidate_name: Optional name of the candidate

        Returns:
            ScreeningResult object

        Raises:
            ValueError: If materials cannot be processed or screening fails
        """
        logger.info(f"Starting screening for candidate: {candidate_id}")
        start_time = time.time()

        # Process files
        processed_files = self.file_processor.process_multiple_files(material_paths)
        self._last_processed_files = processed_files
        combined_materials = self.file_processor.combine_materials(processed_files)

        # Generate screening prompt
        prompt = get_screening_prompt(combined_materials, description)

        # Call Claude API with retry logic for JSON parsing failures
        max_retries = 2
        last_error = None
        screening_data = None

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    logger.info("Calling Claude API for screening...")
                    response = self._call_claude_api(prompt)
                else:
                    logger.warning(f"Retry attempt {attempt} due to JSON parsing failure")
                    retry_prompt = f"""Your previous response could not be parsed as valid JSON.
Error: {last_error}

Please regenerate your ENTIRE response as valid JSON only.
Start with ```json and end with ```.
Make sure all quotes inside string values are escaped with backslash: \\"
Do NOT include any text outside the JSON structure.

Original request:
{prompt}"""
                    response = self._call_claude_api(retry_prompt)

                # Reuse the holistic JSON extraction/repair logic (generic JSON parse)
                screening_data = self._parse_holistic_response(response)
                break

            except ValueError as e:
                last_error = str(e)
                if attempt < max_retries:
                    logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {e}")
                else:
                    logger.error(f"All {max_retries + 1} attempts failed to parse JSON")
                    raise

        if screening_data is None:
            raise ValueError("Failed to get valid screening data after retries")

        candidate = CandidateProfile(
            candidate_id=candidate_id,
            name=candidate_name,
            materials=[str(Path(p).name) for p in material_paths],
            evaluation_date=datetime.now()
        )

        processing_time = time.time() - start_time

        # Normalize confidence to a 0-1 float (models sometimes emit 0-100 or strings)
        raw_confidence = screening_data.get('confidence', 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))

        def _as_str_list(value):
            if not value:
                return []
            if isinstance(value, str):
                return [value]
            return [str(v) for v in value]

        result = ScreeningResult(
            candidate=candidate,
            description=description,
            matches=bool(screening_data.get('matches', False)),
            confidence=confidence,
            reasoning=screening_data.get('reasoning', ''),
            supporting_evidence=_as_str_list(screening_data.get('supporting_evidence', [])),
            disqualifiers=_as_str_list(screening_data.get('disqualifiers', [])),
            metadata={
                'model': self.config.api.model,
                'processing_time_seconds': processing_time,
                'materials_character_count': len(combined_materials),
                'timestamp': datetime.now().isoformat(),
                'evaluation_mode': 'screen'
            }
        )

        logger.info(
            f"Screening completed in {processing_time:.2f}s "
            f"(matches={result.matches}, confidence={result.confidence:.2f})"
        )

        return result

    def _parse_holistic_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Claude's holistic evaluation response.

        Args:
            response: Raw response text from Claude

        Returns:
            Parsed holistic evaluation data

        Raises:
            ValueError: If response cannot be parsed
        """
        # Save raw response for debugging
        debug_dir = os.path.expanduser("~/candidate_eval_debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"holistic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(debug_file, 'w') as f:
                f.write(response)
            logger.info(f"Saved holistic response to: {debug_file}")
        except Exception as e:
            logger.warning(f"Could not save debug file: {e}")

        try:
            # Extract JSON from response - try multiple methods
            json_str = None

            # Method 1: Look for ```json code block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    json_str = response[start:end].strip()

            # Method 2: Look for any ``` code block containing JSON
            if not json_str and '```' in response:
                start = response.find('```') + 3
                # Skip language identifier if present
                newline = response.find('\n', start)
                if newline > start:
                    start = newline + 1
                end = response.find('```', start)
                if end > start:
                    potential_json = response[start:end].strip()
                    # Only use if it looks like JSON
                    if potential_json.startswith('{'):
                        json_str = potential_json

            # Method 3: Find JSON object directly using brace matching
            if not json_str:
                brace_start = response.find('{')
                if brace_start >= 0:
                    # Find matching closing brace, handling strings properly
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    for i, char in enumerate(response[brace_start:], brace_start):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\' and in_string:
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response[brace_start:i+1]
                                    break

            if not json_str:
                # Log more context to help debug
                logger.error(f"No JSON found. Response length: {len(response)}")
                logger.error(f"Response starts with: {response[:500]}...")
                logger.error(f"Response ends with: ...{response[-500:]}")
                raise ValueError("No JSON found in holistic response")

            # Always apply repair function first (proactively fix common issues)
            repaired = self._repair_json_string(json_str)

            # Try to parse
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as e:
                # Log the specific error location
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Error at position {e.pos}, around: ...{repaired[max(0,e.pos-50):e.pos+50]}...")

                # Try additional repair: remove trailing commas before } or ]
                repaired2 = re.sub(r',(\s*[}\]])', r'\1', repaired)
                try:
                    data = json.loads(repaired2)
                except json.JSONDecodeError as e2:
                    logger.warning(f"Second parse attempt failed: {e2}")

                    # Third repair attempt: fix malformed string values with embedded quotes
                    # Pattern: "key": "text A" bare_text "text B" bare_text, → "key": "text A bare_text text B bare_text",
                    repaired3 = self._aggressive_json_repair(repaired2)
                    try:
                        data = json.loads(repaired3)
                    except json.JSONDecodeError as e3:
                        logger.error(f"Third parse attempt (aggressive repair) failed: {e3}")
                        raise ValueError(f"Failed to parse JSON: {e3}")

            return data

        except Exception as e:
            logger.error(f"Failed to parse holistic response: {e}")
            logger.error(f"Response preview: {response[:1000]}...")
            raise ValueError(f"Failed to parse holistic response: {e}")

    def _repair_json_string(self, text: str) -> str:
        """Repair JSON with unescaped quotes and control characters inside strings."""
        result = []
        in_string = False
        i = 0
        while i < len(text):
            char = text[i]

            if not in_string:
                result.append(char)
                if char == '"':
                    in_string = True
            else:
                if char == '\\':
                    result.append(char)
                    i += 1
                    if i < len(text):
                        result.append(text[i])
                elif char == '"':
                    # Check if this is the real end of the string
                    j = i + 1
                    while j < len(text) and text[j] in ' \t\n\r':
                        j += 1
                    if j >= len(text) or text[j] in ':,}]':
                        result.append(char)
                        in_string = False
                    else:
                        result.append('\\')
                        result.append(char)
                # Handle control characters inside strings (ASCII 0-31)
                elif ord(char) < 32:
                    if char == '\n':
                        result.append('\\n')
                    elif char == '\r':
                        result.append('\\r')
                    elif char == '\t':
                        result.append('\\t')
                    else:
                        # Use unicode escape for other control chars
                        result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)

            i += 1
        return ''.join(result)

    def _aggressive_json_repair(self, text: str) -> str:
        """
        Aggressive JSON repair for badly malformed strings.

        Handles patterns like: "key": "text A" bare words "text B" bare,
        by trying to merge them into: "key": "text A bare words text B bare",
        """

        # Pattern to find malformed string values:
        # "key": "value1" some_text "value2" more_text, (or } or ])
        # This is tricky because we need to be careful not to break valid JSON

        # Strategy: Look for lines where a string value seems to continue after a closing quote
        # with non-JSON text before the next quote

        lines = text.split('\n')
        repaired_lines = []

        for line in lines:
            # Check if this line has the problematic pattern:
            # A string that ends with ", then non-quote text, then another "
            # Pattern: "..." baretext "..." baretext,
            # We need to be careful: "key": "value", is valid
            # But "key": "val1" and "val2", is not

            # Look for pattern: ": "..." non-json-chars "..." ... ,
            # where non-json-chars doesn't include { } [ ]
            match = re.search(r':\s*"[^"]*"\s+[^",{}\[\]:]+\s*"[^"]*"', line)
            if match:
                # This line has the problematic pattern
                # Try to fix by removing the embedded quotes and joining the text
                # Find the key-value pair
                kv_match = re.match(r'^(\s*"[^"]+"\s*:\s*)"(.*)(",?\s*)$', line)
                if kv_match:
                    prefix = kv_match.group(1)  # "key":
                    value_part = kv_match.group(2)  # everything between outer quotes
                    suffix = kv_match.group(3)  # trailing comma or nothing

                    # Remove unescaped internal quotes and normalize spacing
                    # This is crude but may help
                    fixed_value = re.sub(r'"\s*([^"]*)\s*"', r' \1 ', value_part)
                    fixed_value = re.sub(r'\s+', ' ', fixed_value).strip()

                    repaired_line = f'{prefix}"{fixed_value}"{suffix}'
                    repaired_lines.append(repaired_line)
                    continue

            repaired_lines.append(line)

        return '\n'.join(repaired_lines)

    def _call_claude_api(self, prompt: str, role: Optional[str] = None) -> str:
        """
        Call Claude API with the given prompt.

        Args:
            prompt: User prompt
            role: Optional role-specific context to append to system prompt

        Returns:
            API response text

        Raises:
            Exception: If API call fails
        """
        try:
            system_prompt = self.prompt_manager.build_system_prompt(role)

            # No temperature: Claude Sonnet 5 rejects non-default sampling
            # parameters (400), and newer SDKs dropped the kwarg entirely.
            message = self.client.messages.create(
                model=self.config.api.model,
                max_tokens=self.config.api.max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract text from response. Claude Sonnet 5 runs adaptive
            # thinking, so content may start with thinking blocks — collect
            # only the text blocks instead of assuming content[0] is text.
            response_text = "".join(
                block.text for block in message.content
                if getattr(block, "type", None) == "text"
            )
            if not response_text:
                raise ValueError("Claude response contained no text blocks")
            return response_text

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Claude's evaluation response.

        Args:
            response: Raw response text from Claude

        Returns:
            Parsed evaluation data

        Raises:
            ValueError: If response cannot be parsed
        """
        # Save raw response for debugging
        debug_dir = os.path.expanduser("~/candidate_eval_debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(debug_file, 'w') as f:
                f.write(response)
            logger.info(f"Saved raw response to: {debug_file}")
        except Exception as e:
            logger.warning(f"Could not save debug file: {e}")

        try:
            # Method 1: Try to extract JSON code blocks (```json or just ```)
            json_blocks = []
            lines = response.split('\n')
            in_json_block = False
            current_block = []

            for line in lines:
                stripped = line.strip()
                # Start of code block - either ```json or just ``` followed by JSON
                if stripped.startswith('```json') or (stripped == '```' and not in_json_block):
                    if stripped.startswith('```json'):
                        in_json_block = True
                        current_block = []
                    elif stripped == '```':
                        # Check if this might be start of a code block
                        in_json_block = True
                        current_block = []
                elif stripped == '```' and in_json_block:
                    in_json_block = False
                    block_content = '\n'.join(current_block).strip()
                    # Only add if it looks like JSON (starts with { or [)
                    if block_content and (block_content.startswith('{') or block_content.startswith('[')):
                        json_blocks.append(block_content)
                    current_block = []
                elif in_json_block:
                    current_block.append(line)

            # Method 2: If no code blocks found, try to find raw JSON objects
            if not json_blocks:
                logger.warning("No JSON code blocks found, attempting to extract raw JSON")

                # Try to find JSON objects using bracket matching
                brace_count = 0
                current_obj = []
                in_object = False

                for char in response:
                    if char == '{':
                        if brace_count == 0:
                            in_object = True
                            current_obj = []
                        brace_count += 1
                        current_obj.append(char)
                    elif char == '}':
                        current_obj.append(char)
                        brace_count -= 1
                        if brace_count == 0 and in_object:
                            json_blocks.append(''.join(current_obj))
                            in_object = False
                            current_obj = []
                    elif in_object:
                        current_obj.append(char)

            # Method 3: Try to find JSON arrays (starting with [)
            if not json_blocks:
                bracket_count = 0
                current_arr = []
                in_array = False

                for char in response:
                    if char == '[':
                        if bracket_count == 0:
                            in_array = True
                            current_arr = []
                        bracket_count += 1
                        current_arr.append(char)
                    elif char == ']':
                        current_arr.append(char)
                        bracket_count -= 1
                        if bracket_count == 0 and in_array:
                            json_blocks.append(''.join(current_arr))
                            in_array = False
                            current_arr = []
                    elif in_array:
                        current_arr.append(char)

            if not json_blocks:
                logger.error(f"Response preview: {response[:1000]}...")
                raise ValueError("No JSON blocks found in response")

            # Parse criterion scores from individual blocks
            criterion_scores = []
            overall_data = None

            valid_criteria = {e.value for e in EvaluationCriterion}

            def _separate_items(items):
                """Separate criterion scores from overall assessment in a list."""
                scores = []
                ov_data = None
                for item in items:
                    if isinstance(item, dict):
                        crit = item.get('criterion')
                        if crit and crit in valid_criteria:
                            scores.append(item)
                        elif 'overall_score' in item or 'overall_assessment' in item or crit == 'overall_assessment':
                            ov_data = item
                return scores, ov_data

            def _repair_json(text):
                """Repair JSON with unescaped quotes and control characters inside strings."""
                result = []
                in_string = False
                i = 0
                while i < len(text):
                    char = text[i]

                    if not in_string:
                        result.append(char)
                        if char == '"':
                            in_string = True
                    else:
                        if char == '\\':
                            result.append(char)
                            i += 1
                            if i < len(text):
                                result.append(text[i])
                        elif char == '"':
                            # Check if this is the real end of the string
                            # by looking at the next non-whitespace character
                            j = i + 1
                            while j < len(text) and text[j] in ' \t\n\r':
                                j += 1
                            if j >= len(text) or text[j] in ':,}]':
                                result.append(char)
                                in_string = False
                            else:
                                result.append('\\')
                                result.append(char)
                        # Handle control characters inside strings (ASCII 0-31)
                        elif ord(char) < 32:
                            if char == '\n':
                                result.append('\\n')
                            elif char == '\r':
                                result.append('\\r')
                            elif char == '\t':
                                result.append('\\t')
                            else:
                                # Use unicode escape for other control chars
                                result.append(f'\\u{ord(char):04x}')
                        else:
                            result.append(char)

                    i += 1
                return ''.join(result)

            def _parse_json_block(block):
                """Parse JSON block, attempting repair if initial parse fails."""
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    repaired = _repair_json(block)
                    return json.loads(repaired)

            for block in json_blocks:
                try:
                    data = _parse_json_block(block)

                    # Check if data itself is a list (array that may mix criteria + overall)
                    if isinstance(data, list):
                        scores, ov = _separate_items(data)
                        criterion_scores.extend(scores)
                        if ov and not overall_data:
                            overall_data = ov
                    # Handle case where scores are in an 'evaluations' array (check BEFORE overall_score!)
                    elif 'evaluations' in data and isinstance(data['evaluations'], list):
                        scores, ov = _separate_items(data['evaluations'])
                        criterion_scores.extend(scores)
                        if 'overall_score' in data:
                            overall_data = data
                        elif ov and not overall_data:
                            overall_data = ov
                    # Handle case where all scores are in a 'scores' array
                    elif 'scores' in data and isinstance(data['scores'], list):
                        scores, ov = _separate_items(data['scores'])
                        criterion_scores.extend(scores)
                        if 'overall_score' in data:
                            overall_data = data
                        elif ov and not overall_data:
                            overall_data = ov
                    # Handle case where data has both criterion_scores and overall_assessment
                    elif 'criterion_scores' in data:
                        scores, ov = _separate_items(data['criterion_scores'])
                        criterion_scores.extend(scores)
                        overall_data = data
                    # Check if this is a criterion score object with valid criterion
                    elif 'criterion' in data and data['criterion'] in valid_criteria:
                        criterion_scores.append(data)
                    # Check if this is overall assessment (no arrays)
                    elif 'overall_score' in data or 'overall_assessment' in data:
                        overall_data = data

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON block: {e}")
                    logger.debug(f"Problematic block: {block[:200]}...")
                    continue

            if not criterion_scores:
                logger.error(f"Parsed {len(json_blocks)} JSON blocks but found no criterion scores")
                logger.error("JSON blocks found:")
                for i, block in enumerate(json_blocks):
                    logger.error(f"Block {i}: {block[:300]}...")
                raise ValueError("No criterion scores found in response")

            # Validate completeness - detect truncated responses
            expected_criteria = {
                'critical_thinking', 'coachability', 'curiosity', 'creativity',
                'collaboration', 'follow_through', 'problem_solving_motivation',
                'evidence_based', 'detail_orientation', 'communication', 'expertise_enabler'
            }
            found_criteria = {s.get('criterion') for s in criterion_scores if s.get('criterion')}
            missing = expected_criteria - found_criteria

            if missing:
                logger.warning(
                    f"Missing {len(missing)} criteria (likely output truncated): {missing}. "
                    f"Found {len(found_criteria)}/11. Consider increasing max_tokens."
                )

            role_specific_assessment = None
            if overall_data:
                role_specific_assessment = overall_data.get('role_specific_assessment')
            if role_specific_assessment is None:
                for block in json_blocks:
                    try:
                        data = json.loads(block)
                        if isinstance(data, dict) and 'role_specific_assessment' in data:
                            role_specific_assessment = data['role_specific_assessment']
                            break
                    except json.JSONDecodeError:
                        continue

            # Combine parsed data
            result = {
                'criterion_scores': criterion_scores,
                'overall_assessment': overall_data.get('overall_assessment', '') if overall_data else '',
                'strengths': overall_data.get('strengths', []) if overall_data else [],
                'areas_for_development': overall_data.get('areas_for_development', []) if overall_data else [],
                'recommendation': overall_data.get('recommendation', '') if overall_data else '',
                'role_specific_assessment': role_specific_assessment,
            }

            return result

        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            logger.error(f"Response length: {len(response)} characters")
            logger.error(f"Response preview: {response[:1000]}...")
            raise ValueError(f"Failed to parse evaluation response: {e}")

    def _parse_comparison_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Claude's comparison response.

        Args:
            response: Raw response text from Claude

        Returns:
            Parsed comparison data

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            return data

        except Exception as e:
            logger.error(f"Failed to parse comparison response: {e}")
            logger.debug(f"Response was: {response}")
            raise ValueError(f"Failed to parse comparison response: {e}")

    def generate_research_report(
        self,
        evaluation_result: EvaluationResult
    ) -> ResearchEvaluationReport:
        """
        Generate a research-style report with linguistic analysis.

        Args:
            evaluation_result: Standard evaluation result

        Returns:
            ResearchEvaluationReport with detailed analysis

        Raises:
            ValueError: If no processed files available
        """
        if self._last_processed_files is None:
            raise ValueError(
                "No processed files available. "
                "Must run evaluate_candidate() before generating research report."
            )

        logger.info("Generating research report with linguistic analysis...")

        research_report = self.research_generator.generate_research_report(
            evaluation_result,
            self._last_processed_files
        )

        logger.info("Research report generation complete")
        return research_report
