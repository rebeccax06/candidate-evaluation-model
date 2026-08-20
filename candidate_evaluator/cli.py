"""Command-line interface for candidate evaluator"""

import csv
import sys
import logging
from pathlib import Path
from shutil import copy
from typing import List, Optional
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from candidate_evaluator.core.evaluator import CandidateEvaluator
from candidate_evaluator.utils.config import load_config, get_default_config
from candidate_evaluator.utils.logger import setup_logger
from candidate_evaluator.exporters import (
    JSONExporter,
    MarkdownExporter,
    HTMLExporter,
    CSVExporter
)
from candidate_evaluator.exporters.research_exporter import ResearchPaperExporter
from candidate_evaluator.exporters.comparison_exporter import ComparisonExporter

console = Console()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """
    Candidate Evaluator - AI-powered candidate assessment tool

    Evaluate candidate application materials using Claude API against
    specific criteria with evidence-based scoring.
    """
    ctx.ensure_object(dict)

    # Load configuration
    try:
        if config:
            cfg = load_config(config)
        else:
            try:
                cfg = load_config()
            except ValueError:
                cfg = get_default_config()

        ctx.obj['config'] = cfg

        # Setup logger
        log_level = 'DEBUG' if verbose else cfg.logging.level
        logger = setup_logger(
            level=log_level,
            log_file=cfg.logging.log_file,
            console_logging=cfg.logging.console_logging
        )
        ctx.obj['logger'] = logger

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        console.print("\n[yellow]Tip: Set ANTHROPIC_API_KEY environment variable or create config.yaml[/yellow]")
        sys.exit(1)


@cli.command()
@click.argument('materials', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--candidate-id', '-id', required=True, help='Unique candidate identifier')
@click.option('--name', '-n', help='Candidate name (optional)')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.option('--format', '-f', 'formats', multiple=True,
              type=click.Choice(['json', 'markdown', 'html', 'csv'], case_sensitive=False),
              help='Output format(s)')
@click.option('--research', is_flag=True, help='Generate detailed research report with linguistic analysis')
@click.pass_context
def evaluate(ctx, materials, candidate_id, name, output_dir, formats, research):
    """
    Evaluate a single candidate based on their application materials.

    MATERIALS: One or more paths to candidate materials (PDF, DOCX, TXT, MD)

    Example:
        candidate-eval evaluate resume.pdf cover_letter.txt --candidate-id CAND001 --name "John Doe"
    """
    config = ctx.obj['config']
    logger = ctx.obj['logger']

    # Set output directory
    if not output_dir:
        output_dir = config.output.output_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set formats
    if not formats:
        formats = config.output.default_formats

    console.print(f"\n[bold blue]Evaluating candidate: {candidate_id}[/bold blue]")
    console.print(f"Materials: {', '.join(materials)}")
    console.print(f"Output directory: {output_path}\n")

    try:
        # Create evaluator
        evaluator = CandidateEvaluator(config)

        # Evaluate with progress spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Evaluating candidate...", total=None)

            result = evaluator.evaluate_candidate(
                candidate_id=candidate_id,
                material_paths=list(materials),
                candidate_name=name
            )

        # Display summary
        console.print(f"\n[bold green]✓ Evaluation completed![/bold green]\n")
        console.print(f"Overall Score: [bold]{result.overall_score:.2f}/10[/bold]")
        console.print(f"Recommendation: {result.recommendation}\n")

        # Display scores table
        table = Table(title="Criterion Scores")
        table.add_column("Criterion", style="cyan")
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("Confidence", justify="center")

        for score in sorted(result.scores, key=lambda s: s.score, reverse=True):
            table.add_row(
                score.criterion.display_name,
                f"{score.score}/10",
                score.confidence
            )

        console.print(table)

        # Export results
        console.print(f"\n[bold]Exporting results...[/bold]")

        exported_files = []

        if 'json' in formats:
            json_path = output_path / f"{candidate_id}_evaluation.json"
            JSONExporter.export_evaluation(result, json_path)
            exported_files.append(str(json_path))

        if 'markdown' in formats:
            md_path = output_path / f"{candidate_id}_evaluation.md"
            MarkdownExporter.export_evaluation(
                result, md_path,
                include_evidence=config.output.include_evidence
            )
            exported_files.append(str(md_path))

        if 'html' in formats:
            html_path = output_path / f"{candidate_id}_evaluation.html"
            HTMLExporter.export_evaluation(
                result, html_path,
                include_evidence=config.output.include_evidence
            )
            exported_files.append(str(html_path))

        if 'csv' in formats:
            csv_path = output_path / f"{candidate_id}_evaluation.csv"
            CSVExporter.export_evaluation(result, csv_path)
            exported_files.append(str(csv_path))

        console.print(f"\n[green]Results exported to:[/green]")
        for file in exported_files:
            console.print(f"  • {file}")

        # Generate research report if requested
        if research:
            console.print(f"\n[bold cyan]Generating research report with linguistic analysis...[/bold cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress.add_task(description="Analyzing linguistic patterns...", total=None)

                research_report = evaluator.generate_research_report(result)

            # Export research report
            research_path = output_path / f"{candidate_id}_research_report.md"
            ResearchPaperExporter.export_research_paper(
                research_report,
                research_path,
                format='markdown'
            )

            console.print(f"\n[bold green]✓ Research report generated![/bold green]")
            console.print(f"\n[yellow]Innovation Potential Assessment:[/yellow]")
            console.print(f"  • Innovation Score: {research_report.innovation_assessment.overall_innovation_score:.2f}/10")
            console.print(f"  • Potential Level: {research_report.innovation_assessment.innovation_potential_level.upper()}")
            console.print(f"  • Innovation Markers: {research_report.innovation_assessment.innovation_indicators_count}")
            console.print(f"\n[yellow]Linguistic Analysis Summary:[/yellow]")
            console.print(f"  • Total Markers Identified: {research_report.linguistic_patterns['total_markers']}")
            console.print(f"  • Words Analyzed: {research_report.statistical_summary['total_words_analyzed']:,}")
            console.print(f"\n[green]Research report: {research_path}[/green]")

    except Exception as e:
        console.print(f"\n[red]Error during evaluation: {e}[/red]")
        logger.exception("Evaluation failed")
        sys.exit(1)


@cli.command()
@click.argument('candidates-file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.option('--format', '-f', 'formats', multiple=True,
              type=click.Choice(['json', 'markdown', 'html', 'csv'], case_sensitive=False),
              help='Output format(s)')
@click.option('--compare', is_flag=True, help='Generate comparison report')
@click.pass_context
def batch(ctx, candidates_file, output_dir, formats, compare):
    """
    Evaluate multiple candidates from a CSV file.

    CANDIDATES_FILE: CSV file with columns: candidate_id, name (optional), material_paths (semicolon-separated)

    Example CSV format:
        candidate_id,name,material_paths
        CAND001,John Doe,resume1.pdf;cover1.txt
        CAND002,Jane Smith,resume2.pdf;cover2.txt

    Example:
        candidate-eval batch candidates.csv --compare
    """
    config = ctx.obj['config']
    logger = ctx.obj['logger']

    # Set output directory
    if not output_dir:
        output_dir = config.output.output_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set formats
    if not formats:
        formats = config.output.default_formats

    console.print(f"\n[bold blue]Batch Evaluation[/bold blue]")
    console.print(f"Input file: {candidates_file}")
    console.print(f"Output directory: {output_path}\n")

    try:
        # Load candidates from CSV
        candidates = []
        with open(candidates_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                material_paths = row['material_paths'].split(';')
                material_paths = [p.strip() for p in material_paths]

                candidates.append({
                    'candidate_id': row['candidate_id'],
                    'candidate_name': row.get('name'),
                    'material_paths': material_paths
                })

        console.print(f"Loaded {len(candidates)} candidates\n")

        # Create evaluator
        evaluator = CandidateEvaluator(config)

        # Evaluate all candidates
        results = []
        for i, candidate in enumerate(candidates, 1):
            console.print(f"[cyan]Evaluating {i}/{len(candidates)}: {candidate['candidate_id']}[/cyan]")

            try:
                result = evaluator.evaluate_candidate(
                    candidate_id=candidate['candidate_id'],
                    material_paths=candidate['material_paths'],
                    candidate_name=candidate.get('candidate_name')
                )
                results.append(result)
                console.print(f"  ✓ Score: {result.overall_score:.2f}/10\n")

            except Exception as e:
                console.print(f"  [red]✗ Error: {e}[/red]\n")
                logger.error(f"Failed to evaluate {candidate['candidate_id']}: {e}")
                continue

        if not results:
            console.print("[red]No candidates were successfully evaluated[/red]")
            sys.exit(1)

        console.print(f"\n[bold green]✓ Batch evaluation completed![/bold green]")
        console.print(f"Successfully evaluated: {len(results)}/{len(candidates)}\n")

        # Display summary table
        table = Table(title="Evaluation Summary")
        table.add_column("Rank", justify="right")
        table.add_column("Candidate ID", style="cyan")
        table.add_column("Name")
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("Recommendation")

        sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)
        for i, result in enumerate(sorted_results, 1):
            table.add_row(
                str(i),
                result.candidate.candidate_id,
                result.candidate.name or "-",
                f"{result.overall_score:.2f}",
                result.recommendation[:50] + "..." if len(result.recommendation) > 50 else result.recommendation
            )

        console.print(table)

        # Export individual results
        console.print(f"\n[bold]Exporting results...[/bold]")

        for result in results:
            candidate_id = result.candidate.candidate_id

            if 'json' in formats:
                json_path = output_path / f"{candidate_id}_evaluation.json"
                JSONExporter.export_evaluation(result, json_path)

            if 'markdown' in formats:
                md_path = output_path / f"{candidate_id}_evaluation.md"
                MarkdownExporter.export_evaluation(result, md_path)

            if 'html' in formats:
                html_path = output_path / f"{candidate_id}_evaluation.html"
                HTMLExporter.export_evaluation(result, html_path)

        # Export batch summary
        if 'csv' in formats:
            csv_path = output_path / "batch_results.csv"
            CSVExporter.export_batch(results, csv_path)
            console.print(f"  • Batch summary: {csv_path}")

        if 'markdown' in formats:
            md_path = output_path / "batch_summary.md"
            MarkdownExporter.export_batch_summary(results, md_path)
            console.print(f"  • Batch summary: {md_path}")

        # Generate comparison if requested
        if compare and len(results) > 1:
            console.print(f"\n[bold]Generating comparison report...[/bold]")

            comparison = evaluator.compare_candidates(results)

            if 'json' in formats:
                json_path = output_path / "comparison.json"
                JSONExporter.export_comparison(comparison, json_path)

            if 'markdown' in formats:
                md_path = output_path / "comparison.md"
                MarkdownExporter.export_comparison(comparison, md_path)

            if 'html' in formats:
                html_path = output_path / "comparison.html"
                HTMLExporter.export_comparison(comparison, html_path)

            if 'csv' in formats:
                csv_path = output_path / "comparison_matrix.csv"
                CSVExporter.export_comparison_matrix(comparison, csv_path)

            console.print("[green]✓ Comparison report generated[/green]")

        console.print(f"\n[green]All results exported to: {output_path}[/green]")

    except Exception as e:
        console.print(f"\n[red]Error during batch evaluation: {e}[/red]")
        logger.exception("Batch evaluation failed")
        sys.exit(1)


@cli.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for results')
@click.option('--format', '-f', 'formats', multiple=True,
              type=click.Choice(['json', 'markdown', 'html', 'csv'], case_sensitive=False),
              default=['json', 'markdown'],
              help='Output format(s)')
@click.option('--pattern', '-p', default='*.pdf', help='File pattern to match (default: *.pdf)')
@click.option('--compare', is_flag=True, help='Generate comparison report after batch evaluation')
@click.option('--research', is_flag=True, help='Generate research reports for each candidate')
@click.option('--max-tokens', type=int, default=8192, help='Max tokens for API calls (higher = more detailed, default: 8192)')
@click.pass_context
def batch_folder(ctx, folder, output_dir, formats, pattern, compare, research, max_tokens):
    """
    Evaluate all candidates from PDFs in a folder (QUALITY MODE).

    This command processes each PDF as a separate candidate with a dedicated API call
    for maximum evaluation depth and quality. Cost is not optimized - quality is.

    FOLDER: Directory containing candidate PDFs

    Each PDF should contain all candidate materials (resume, cover letter,
    interview responses, recommendation letters, etc.)

    Filename becomes the candidate ID (e.g., "candidate_001.pdf" -> "candidate_001")

    Example:
        candidate-evaluator batch-folder ./dropbox --output-dir ./results --compare
    """
    config = ctx.obj['config']
    logger = ctx.obj['logger']

    # Override max_tokens for quality
    config.api.max_tokens = max_tokens

    # Set output directory
    if not output_dir:
        output_dir = config.output.output_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    folder_path = Path(folder)
    pdf_files = sorted(folder_path.glob(pattern))

    if not pdf_files:
        console.print(f"[red]No files matching '{pattern}' found in {folder}[/red]")
        sys.exit(1)

    console.print(f"\n[bold blue]Batch Folder Evaluation (Quality Mode)[/bold blue]")
    console.print(f"Folder: {folder}")
    console.print(f"Pattern: {pattern}")
    console.print(f"Found: {len(pdf_files)} candidate(s)")
    console.print(f"Max tokens per candidate: {max_tokens}")
    console.print(f"Output directory: {output_path}\n")

    # Create evaluator
    evaluator = CandidateEvaluator(config)

    # Evaluate all candidates
    results = []
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        for i, pdf_path in enumerate(pdf_files, 1):
            # Extract candidate ID from filename (remove extension)
            candidate_id = pdf_path.stem

            task = progress.add_task(
                description=f"[cyan]Evaluating {i}/{len(pdf_files)}: {candidate_id}[/cyan]",
                total=None
            )

            try:
                # Evaluate with full context
                result = evaluator.evaluate_candidate(
                    candidate_id=candidate_id,
                    material_paths=[str(pdf_path)],
                    candidate_name=None
                )

                results.append(result)
                progress.update(task, description=f"[green]✓ {candidate_id}: {result.overall_score:.2f}/10[/green]")

                # Export individual results
                candidate_dir = output_path / candidate_id
                candidate_dir.mkdir(exist_ok=True)

                if 'json' in formats:
                    json_path = candidate_dir / f"{candidate_id}_evaluation.json"
                    JSONExporter.export_evaluation(result, json_path)

                if 'markdown' in formats:
                    md_path = candidate_dir / f"{candidate_id}_evaluation.md"
                    MarkdownExporter.export_evaluation(result, md_path, include_evidence=True)

                if 'html' in formats:
                    html_path = candidate_dir / f"{candidate_id}_evaluation.html"
                    HTMLExporter.export_evaluation(result, html_path, include_evidence=True)

                if 'csv' in formats:
                    csv_path = candidate_dir / f"{candidate_id}_evaluation.csv"
                    CSVExporter.export_evaluation(result, csv_path)

                # Generate research report if requested
                if research:
                    research_report = evaluator.generate_research_report(result)
                    research_path = candidate_dir / f"{candidate_id}_research_report.md"
                    ResearchPaperExporter.export_research_paper(
                        research_report,
                        research_path,
                        format='markdown'
                    )

            except Exception as e:
                progress.update(task, description=f"[red]✗ {candidate_id}: {str(e)[:50]}[/red]")
                logger.error(f"Failed to evaluate {candidate_id}: {e}", exc_info=True)
                failed.append({'candidate_id': candidate_id, 'error': str(e)})
                continue

    # Print summary
    console.print(f"\n[bold green]Batch Evaluation Complete[/bold green]")
    console.print(f"Successfully evaluated: {len(results)}/{len(pdf_files)}")

    if failed:
        console.print(f"[red]Failed: {len(failed)}[/red]")
        for fail in failed:
            console.print(f"  • {fail['candidate_id']}: {fail['error'][:80]}")

    if not results:
        console.print("[red]No candidates were successfully evaluated[/red]")
        sys.exit(1)

    # Display rankings
    console.print("\n[bold]Rankings:[/bold]")
    ranked = sorted(results, key=lambda r: r.overall_score, reverse=True)

    ranking_table = Table(title="Candidate Rankings")
    ranking_table.add_column("Rank", justify="right", style="cyan")
    ranking_table.add_column("Candidate ID", style="magenta")
    ranking_table.add_column("Overall Score", justify="right", style="green")
    ranking_table.add_column("Recommendation", style="yellow")

    for rank, result in enumerate(ranked, 1):
        ranking_table.add_row(
            str(rank),
            result.candidate.candidate_id,
            f"{result.overall_score:.2f}/10",
            result.recommendation[:50] + "..." if len(result.recommendation) > 50 else result.recommendation
        )

    console.print(ranking_table)

    # Generate comparison report if requested
    if compare and len(results) > 1:
        console.print(f"\n[bold cyan]Generating comparison report...[/bold cyan]")

        try:
            comparison = evaluator.compare_candidates(results)

            # Export comparison
            comparison_path = output_path / "comparison_report.md"
            ComparisonExporter.export_comparison(comparison, comparison_path)

            console.print(f"[green]✓ Comparison report saved to: {comparison_path}[/green]")

        except Exception as e:
            console.print(f"[red]Failed to generate comparison: {e}[/red]")
            logger.error(f"Comparison generation failed: {e}", exc_info=True)

    console.print(f"\n[bold green]All results saved to: {output_path}[/bold green]\n")


@cli.command()
@click.pass_context
def init(ctx):
    """
    Initialize configuration files in the current directory.

    Creates config.yaml and .env templates.
    """
    console.print("\n[bold blue]Initializing Candidate Evaluator[/bold blue]\n")

    # Copy template files
    try:
        # Get template paths
        template_dir = Path(__file__).parent.parent / "config_templates"

        # Copy config.yaml template
        config_template = template_dir / "config.yaml.template"
        config_dest = Path.cwd() / "config.yaml"

        if config_dest.exists():
            console.print(f"[yellow]config.yaml already exists, skipping[/yellow]")
        else:
            copy(config_template, config_dest)
            console.print(f"[green]✓ Created config.yaml[/green]")

        # Copy .env template
        env_template = template_dir / ".env.template"
        env_dest = Path.cwd() / ".env"

        if env_dest.exists():
            console.print(f"[yellow].env already exists, skipping[/yellow]")
        else:
            copy(env_template, env_dest)
            console.print(f"[green]✓ Created .env[/green]")

        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Add your Anthropic API key to .env or config.yaml")
        console.print("2. Customize criteria weights and settings in config.yaml")
        console.print("3. Run: candidate-eval evaluate <materials> --candidate-id <ID>\n")

    except Exception as e:
        console.print(f"[red]Error during initialization: {e}[/red]")
        sys.exit(1)


def main():
    """Entry point for CLI"""
    cli(obj={})


if __name__ == '__main__':
    main()
