"""JSON export functionality"""

import json
from pathlib import Path
from typing import Union, List
from datetime import datetime

from candidate_evaluator.core.models import EvaluationResult, ComparisonResult


class JSONExporter:
    """Export evaluation results to JSON format"""

    @staticmethod
    def export_evaluation(
        result: EvaluationResult,
        output_path: Union[str, Path],
        pretty: bool = True
    ) -> str:
        """
        Export evaluation result to JSON file.

        Args:
            result: EvaluationResult object
            output_path: Path to output file
            pretty: Whether to pretty-print JSON

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = {
            'candidate': {
                'candidate_id': result.candidate.candidate_id,
                'name': result.candidate.name,
                'materials': result.candidate.materials,
                'evaluation_date': result.candidate.evaluation_date.isoformat()
            },
            'overall_score': result.overall_score,
            'overall_assessment': result.overall_assessment,
            'recommendation': result.recommendation,
            'strengths': result.strengths,
            'areas_for_development': result.areas_for_development,
            'role': getattr(result, 'role', None),
            'role_specific_assessment': (
                result.role_specific_assessment.model_dump()
                if getattr(result, 'role_specific_assessment', None) is not None
                else None
            ),
            'scores': [
                {
                    'criterion': score.criterion.value,
                    'criterion_name': score.criterion.display_name,
                    'score': score.score,
                    'confidence': score.confidence,
                    'reasoning': score.reasoning,
                    'evidence': [
                        {
                            'quote': ev.quote,
                            'source': ev.source,
                            'context': ev.context
                        }
                        for ev in score.evidence
                    ],
                    'notes': score.notes
                }
                for score in result.scores
            ],
            'metadata': result.metadata
        }

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        return str(output_path)

    @staticmethod
    def export_comparison(
        result: ComparisonResult,
        output_path: Union[str, Path],
        pretty: bool = True
    ) -> str:
        """
        Export comparison result to JSON file.

        Args:
            result: ComparisonResult object
            output_path: Path to output file
            pretty: Whether to pretty-print JSON

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = {
            'comparison_date': result.comparison_date.isoformat(),
            'ranking': result.ranking,
            'comparison_matrix': result.comparison_matrix,
            'insights': result.insights,
            'candidates': [
                {
                    'candidate_id': eval_result.candidate.candidate_id,
                    'name': eval_result.candidate.name,
                    'overall_score': eval_result.overall_score,
                    'recommendation': eval_result.recommendation
                }
                for eval_result in result.candidates
            ]
        }

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        return str(output_path)

    @staticmethod
    def export_batch(
        results: List[EvaluationResult],
        output_path: Union[str, Path],
        pretty: bool = True
    ) -> str:
        """
        Export multiple evaluation results to single JSON file.

        Args:
            results: List of EvaluationResult objects
            output_path: Path to output file
            pretty: Whether to pretty-print JSON

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert all results
        data = {
            'export_date': datetime.now().isoformat(),
            'total_candidates': len(results),
            'evaluations': [
                result.to_summary_dict() for result in results
            ]
        }

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        return str(output_path)
