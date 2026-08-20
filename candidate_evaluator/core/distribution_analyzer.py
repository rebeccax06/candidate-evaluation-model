"""Distribution analyzer for analyzing score patterns across candidate groups."""

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

from candidate_evaluator.core.models import (
    EvaluationResult,
    DistributionAnalysis,
    EvaluationCriterion
)


class DistributionAnalyzer:
    """Analyze score distributions and patterns across candidate groups."""

    def __init__(self, results: Optional[List[EvaluationResult]] = None):
        self.results = results or []

    def set_results(self, results: List[EvaluationResult]):
        """Set evaluation results for analysis."""
        self.results = results

    def segment_by_percentile(
        self,
        percentile: float = 50,
        by_criterion: Optional[str] = None
    ) -> Tuple[List[EvaluationResult], List[EvaluationResult]]:
        """
        Split candidates into top and bottom groups by percentile.

        Args:
            percentile: Percentile threshold (default 50 = median split)
            by_criterion: Optional criterion to sort by (default: overall_score)

        Returns:
            Tuple of (top_group, bottom_group)
        """
        if not self.results:
            return [], []

        # Get scores to sort by
        if by_criterion:
            scores = []
            for result in self.results:
                for score in result.scores:
                    if score.criterion.value == by_criterion:
                        scores.append((result, score.score))
                        break
                else:
                    scores.append((result, 0))
        else:
            scores = [(result, result.overall_score) for result in self.results]

        # Calculate threshold
        score_values = [s[1] for s in scores]
        threshold = np.percentile(score_values, percentile)

        # Split groups
        top_group = [r for r, s in scores if s >= threshold]
        bottom_group = [r for r, s in scores if s < threshold]

        return top_group, bottom_group

    def segment_by_quartile(self) -> Dict[str, List[EvaluationResult]]:
        """
        Split candidates into quartile groups.

        Returns:
            Dict with keys 'Q1' (bottom 25%), 'Q2', 'Q3', 'Q4' (top 25%)
        """
        if not self.results:
            return {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

        scores = [(result, result.overall_score) for result in self.results]
        score_values = [s[1] for s in scores]

        q1_threshold = np.percentile(score_values, 25)
        q2_threshold = np.percentile(score_values, 50)
        q3_threshold = np.percentile(score_values, 75)

        return {
            'Q1': [r for r, s in scores if s < q1_threshold],
            'Q2': [r for r, s in scores if q1_threshold <= s < q2_threshold],
            'Q3': [r for r, s in scores if q2_threshold <= s < q3_threshold],
            'Q4': [r for r, s in scores if s >= q3_threshold]
        }

    def analyze_group(
        self,
        group: List[EvaluationResult],
        group_name: str = "Group"
    ) -> DistributionAnalysis:
        """
        Calculate statistics for a group of candidates.

        Args:
            group: List of evaluation results
            group_name: Name for this group

        Returns:
            DistributionAnalysis with statistics
        """
        if not group:
            return DistributionAnalysis(
                group_name=group_name,
                candidate_count=0,
                candidate_ids=[],
                mean_overall_score=0,
                median_overall_score=0,
                std_overall_score=0,
                min_overall_score=0,
                max_overall_score=0,
                criterion_stats={},
                common_strengths=[],
                common_weaknesses=[]
            )

        # Overall score statistics
        overall_scores = [r.overall_score for r in group]

        # Per-criterion statistics
        criterion_scores: Dict[str, List[float]] = {}
        for result in group:
            for score in result.scores:
                criterion = score.criterion.value
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score.score)

        criterion_stats = {}
        for criterion, scores in criterion_scores.items():
            criterion_stats[criterion] = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }

        # Find common strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        for result in group:
            all_strengths.extend(result.strengths)
            all_weaknesses.extend(result.areas_for_development)

        # Get most common (simplified - just count occurrences)
        strength_counts = Counter(all_strengths)
        weakness_counts = Counter(all_weaknesses)

        common_strengths = [s for s, _ in strength_counts.most_common(5)]
        common_weaknesses = [w for w, _ in weakness_counts.most_common(5)]

        return DistributionAnalysis(
            group_name=group_name,
            candidate_count=len(group),
            candidate_ids=[r.candidate.candidate_id for r in group],
            mean_overall_score=float(np.mean(overall_scores)),
            median_overall_score=float(np.median(overall_scores)),
            std_overall_score=float(np.std(overall_scores)),
            min_overall_score=float(np.min(overall_scores)),
            max_overall_score=float(np.max(overall_scores)),
            criterion_stats=criterion_stats,
            common_strengths=common_strengths,
            common_weaknesses=common_weaknesses
        )

    def compare_groups(
        self,
        top_group: List[EvaluationResult],
        bottom_group: List[EvaluationResult]
    ) -> Dict:
        """
        Compare top and bottom groups to identify discriminating factors.

        Returns:
            Dict with comparison statistics and insights
        """
        top_analysis = self.analyze_group(top_group, "Top Group")
        bottom_analysis = self.analyze_group(bottom_group, "Bottom Group")

        # Find criteria with largest differences
        criterion_differences = []
        for criterion in top_analysis.criterion_stats:
            if criterion in bottom_analysis.criterion_stats:
                top_mean = top_analysis.criterion_stats[criterion]['mean']
                bottom_mean = bottom_analysis.criterion_stats[criterion]['mean']
                diff = top_mean - bottom_mean
                criterion_differences.append({
                    'criterion': criterion,
                    'top_mean': top_mean,
                    'bottom_mean': bottom_mean,
                    'difference': diff,
                    'discriminating_power': abs(diff)
                })

        # Sort by discriminating power
        criterion_differences.sort(key=lambda x: x['discriminating_power'], reverse=True)

        # Identify most discriminating criteria
        most_discriminating = criterion_differences[:3] if criterion_differences else []

        return {
            'top_group': top_analysis.model_dump(),
            'bottom_group': bottom_analysis.model_dump(),
            'criterion_differences': criterion_differences,
            'most_discriminating_criteria': most_discriminating,
            'overall_score_difference': top_analysis.mean_overall_score - bottom_analysis.mean_overall_score,
            'insights': self._generate_insights(top_analysis, bottom_analysis, criterion_differences)
        }

    def _generate_insights(
        self,
        top_analysis: DistributionAnalysis,
        bottom_analysis: DistributionAnalysis,
        criterion_differences: List[Dict]
    ) -> List[str]:
        """Generate human-readable insights from the comparison."""
        insights = []

        # Overall score insight
        diff = top_analysis.mean_overall_score - bottom_analysis.mean_overall_score
        insights.append(
            f"Top performers score {diff:.1f} points higher on average "
            f"({top_analysis.mean_overall_score:.1f} vs {bottom_analysis.mean_overall_score:.1f})"
        )

        # Most discriminating criteria
        if criterion_differences:
            top_criterion = criterion_differences[0]
            criterion_name = top_criterion['criterion'].replace('_', ' ').title()
            insights.append(
                f"'{criterion_name}' shows the largest difference between groups "
                f"({top_criterion['difference']:.1f} points)"
            )

        # Criteria where bottom group is close
        close_criteria = [c for c in criterion_differences if c['discriminating_power'] < 1.0]
        if close_criteria:
            names = [c['criterion'].replace('_', ' ').title() for c in close_criteria[:2]]
            insights.append(
                f"Groups are similar on: {', '.join(names)}"
            )

        return insights

    def identify_score_patterns(self) -> Dict:
        """
        Identify patterns in scoring across all candidates.

        Returns:
            Dict with pattern analysis
        """
        if not self.results:
            return {}

        # Score distribution by criterion
        criterion_distributions = {}
        for result in self.results:
            for score in result.scores:
                criterion = score.criterion.value
                if criterion not in criterion_distributions:
                    criterion_distributions[criterion] = []
                criterion_distributions[criterion].append(score.score)

        # Calculate statistics and patterns
        patterns = {}
        for criterion, scores in criterion_distributions.items():
            patterns[criterion] = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'skewness': self._calculate_skewness(scores),
                'score_distribution': self._get_score_distribution(scores),
                'tendency': self._identify_tendency(scores)
            }

        # Overall patterns
        overall_scores = [r.overall_score for r in self.results]
        patterns['overall'] = {
            'mean': float(np.mean(overall_scores)),
            'median': float(np.median(overall_scores)),
            'std': float(np.std(overall_scores)),
            'skewness': self._calculate_skewness(overall_scores),
            'score_distribution': self._get_score_distribution(overall_scores),
            'tendency': self._identify_tendency(overall_scores)
        }

        return patterns

    def _calculate_skewness(self, scores: List[float]) -> float:
        """Calculate skewness of score distribution."""
        if len(scores) < 3:
            return 0.0
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return 0.0
        return float(np.mean(((np.array(scores) - mean) / std) ** 3))

    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Get count of scores in each range."""
        ranges = {
            '1-2 (Low)': 0,
            '3-4 (Below Avg)': 0,
            '5-6 (Average)': 0,
            '7-8 (Above Avg)': 0,
            '9-10 (Exceptional)': 0
        }
        for score in scores:
            if score <= 2:
                ranges['1-2 (Low)'] += 1
            elif score <= 4:
                ranges['3-4 (Below Avg)'] += 1
            elif score <= 6:
                ranges['5-6 (Average)'] += 1
            elif score <= 8:
                ranges['7-8 (Above Avg)'] += 1
            else:
                ranges['9-10 (Exceptional)'] += 1
        return ranges

    def _identify_tendency(self, scores: List[float]) -> str:
        """Identify if there's central tendency bias or appropriate spread."""
        if not scores:
            return "No data"

        std = np.std(scores)
        mean = np.mean(scores)

        # Check for central tendency (clustering around middle)
        middle_count = sum(1 for s in scores if 4 <= s <= 7)
        middle_pct = middle_count / len(scores)

        if std < 1.5 and middle_pct > 0.7:
            return "Central tendency bias detected - scores cluster around middle"
        elif std < 1.0:
            return "Very low variance - may indicate scoring issue"
        elif mean < 4:
            return "Skewed low - strict scoring"
        elif mean > 7:
            return "Skewed high - lenient scoring"
        else:
            return "Good distribution spread"

    def analyze_top_performers(
        self,
        percentile: float = 50,
        detailed: bool = True
    ) -> Dict:
        """
        Analyze characteristics of top performers.

        Args:
            percentile: Percentile threshold for "top" performers
            detailed: Whether to include per-candidate details

        Returns:
            Analysis of top performer characteristics
        """
        top_group, bottom_group = self.segment_by_percentile(percentile)

        comparison = self.compare_groups(top_group, bottom_group)

        result = {
            'top_performer_count': len(top_group),
            'threshold_score': min(r.overall_score for r in top_group) if top_group else 0,
            'comparison': comparison,
            'discriminating_criteria': comparison['most_discriminating_criteria']
        }

        if detailed:
            result['top_performers'] = [
                {
                    'candidate_id': r.candidate.candidate_id,
                    'overall_score': r.overall_score,
                    'recommendation': r.recommendation,
                    'strengths': r.strengths
                }
                for r in sorted(top_group, key=lambda x: x.overall_score, reverse=True)
            ]

        return result

    def export_analysis_report(self, output_path: str) -> str:
        """Export comprehensive distribution analysis to markdown."""
        output_path = Path(output_path)

        # Run analyses
        patterns = self.identify_score_patterns()
        top_analysis = self.analyze_top_performers(percentile=50)

        report = f"""# Score Distribution Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Total Candidates: {len(self.results)}

## Overall Score Distribution

| Metric | Value |
|--------|-------|
| Mean | {patterns.get('overall', {}).get('mean', 'N/A'):.2f} |
| Median | {patterns.get('overall', {}).get('median', 'N/A'):.2f} |
| Std Dev | {patterns.get('overall', {}).get('std', 'N/A'):.2f} |
| Tendency | {patterns.get('overall', {}).get('tendency', 'N/A')} |

### Score Range Distribution

"""
        if 'overall' in patterns:
            dist = patterns['overall'].get('score_distribution', {})
            for range_name, count in dist.items():
                pct = count / len(self.results) * 100 if self.results else 0
                report += f"- **{range_name}**: {count} candidates ({pct:.1f}%)\n"

        report += f"""

## Per-Criterion Analysis

| Criterion | Mean | Std Dev | Tendency |
|-----------|------|---------|----------|
"""
        for criterion, stats in patterns.items():
            if criterion != 'overall':
                name = criterion.replace('_', ' ').title()
                report += f"| {name} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['tendency'][:30]}... |\n"

        report += f"""

## Top 50% vs Bottom 50% Comparison

### Most Discriminating Criteria

"""
        for i, crit in enumerate(top_analysis.get('discriminating_criteria', []), 1):
            name = crit['criterion'].replace('_', ' ').title()
            report += f"{i}. **{name}**: {crit['difference']:.1f} point difference (Top: {crit['top_mean']:.1f}, Bottom: {crit['bottom_mean']:.1f})\n"

        report += f"""

### Insights

"""
        for insight in top_analysis.get('comparison', {}).get('insights', []):
            report += f"- {insight}\n"

        with open(output_path, 'w') as f:
            f.write(report)

        return str(output_path)
