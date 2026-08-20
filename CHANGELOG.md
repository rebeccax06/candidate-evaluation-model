# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Screening on the cloud web app
- **New "Screening" page** in the cloud app (`web_app_cloud.py`) sidebar,
  matching the local app's screening feature: upload candidate PDFs, describe
  a target profile in plain language, and screen each candidate against it
- Screening jobs run on the background (Railway) worker: `worker.py` now
  handles `evaluation_mode == "screen"` via `CandidateEvaluator.screen_candidate`
- Results are stored in Supabase as evaluations with `evaluation_type='screen'`
  and shown on the Screening page's **Results** tab (job filter, outcome
  filter, review-threshold slider, CSV export, evidence details)
- Batch Jobs result summaries show Outcome/Confidence for screening jobs; the
  Guide page and Help chatbot now document the Screening page

#### Three-way screening outcomes
- Screening decisions are now **Match / Needs review / No match** instead of a
  hard yes/no: any decision whose confidence is below the review threshold
  (default 0.7, adjustable in the results view) is surfaced for human review,
  since model confidence is not calibrated
- `ScreeningResult.outcome()` in `core/models.py` implements the rule and is
  used by both apps and both workers

#### Concurrent evaluations in both workers
- Workers now run up to 4 Claude calls at once per job (tunable via the
  `EVAL_CONCURRENCY` env var), cutting a 200-candidate screen from ~2-3 hours
  to ~30-45 minutes. Output quality is unchanged — every candidate gets the
  same prompt, model, and parameters as a serial run; only timing differs.
  File downloads, result saving, and DB writes stay on the main thread; only
  the API calls are parallel

### Fixed

- **All evaluations failed on newer `anthropic` SDK versions** with
  `Messages.create() got an unexpected keyword argument 'temperature'` —
  recent SDKs removed the parameter from the method signature. The value is
  now sent via `extra_body`, which produces an identical API request (same
  model behavior/output) and works on any SDK version. Affected the Railway
  worker (fresh installs get the latest SDK) and the Help chatbot.

### Changed

#### Model upgraded to Claude Sonnet 5
- Default model is now `claude-sonnet-5` (was `claude-sonnet-4-5-20250929`);
  the Admit Pattern Analyzer moves off the deprecated Sonnet 4 to the same
- `temperature` is no longer sent to the API (Sonnet 5 rejects non-default
  sampling parameters); the config field remains accepted but is deprecated,
  and the Settings page no longer displays it
- Note: Sonnet 5 runs adaptive thinking by default and uses a new tokenizer
  (~30% more tokens for the same text), so per-evaluation cost and latency
  baselines shift; `max_tokens` (16384) has adequate headroom

#### Shared screening/processing code (was duplicated per stack)
- New `core/processing.py`: `EvaluationMode` enum plus shared per-candidate
  dispatch/serialization (`evaluate_candidate_for_mode`, `serialize_result`,
  `result_filename`, `summary_entry`) used by **both** workers
  (`background_worker.py` local, `worker.py` cloud)
- New `screening_ui.py`: the screening submit form and results view used by
  **both** web apps (`web_app.py` local, `web_app_cloud.py` cloud)

#### Job config payload (jobs table)
- Mode-specific job parameters now live in a `jobs.config` JSONB column
  (`{"role": ..., "screen_description": ...}`) instead of one column per
  parameter — new modes need no schema migration. `Database.get_job_config()`
  reads it with a fallback to the legacy `role` column for old rows. See the
  MIGRATION section at the bottom of `supabase_schema.sql`

#### Browser tab icon
- The tab icon (`page_icon`) now uses a square ring mark with a transparent
  background (`assets/catalyst_icon.png`) instead of the wide wordmark logo,
  which browsers distorted when squeezing it into the square favicon slot.
  The sidebar keeps the wordmark

#### Code hygiene
- Moved function-local imports to module top across `core/evaluator.py`,
  `core/distribution_analyzer.py`, `core/pattern_analyzer.py`, `web_app.py`,
  `prompt_manager.py`, `job_manager.py`, and `cli.py` (deliberately lazy
  imports — optional deps like matplotlib/streamlit — are kept and commented);
  removed unused `pkg_resources` import and the unused `MAX_RETRIES` constant

## [1.1.0] - 2026-02-03

### Added

#### Admit Pattern Analysis Feature
- **New "Admit Patterns" tab** in the web interface for analyzing what distinguishes admitted from rejected candidates
- **Pattern discovery system** that:
  - Accepts batch upload of candidate application PDFs
  - Uses CSV mapping file to specify admit/reject labels for each candidate
  - Evaluates all candidates (using holistic or criteria-based mode)
  - Analyzes patterns distinguishing admitted from rejected candidates
  - Identifies predictive factors, common strengths/weaknesses, and surprising cases

#### New Components
- `AdmitPatternAnalyzer` class in `pattern_analyzer.py` for running pattern analysis
- `AdmitPatternAnalysisResult`, `AdmitPatternCategory`, `AdmitPatternEvidence` models
- `ADMIT_PATTERN_ANALYSIS_PROMPT` template for Claude-powered pattern discovery
- `get_admit_pattern_analysis_prompt()` function for generating analysis prompts
- Sample admit mapping CSV template in `examples/sample_admit_mapping.csv`

#### CSV Format for Admit Mapping
```csv
filename,admit_status
candidate_001_application.pdf,yes
candidate_002_application.pdf,no
```

### Technical Details
- Supports both holistic and criteria-based evaluation modes for candidate processing
- Calculates basic statistics without API calls for quick preview
- Full Claude-powered analysis for detailed pattern discovery
- Saves analysis results to JSON for later reference
- Handles large batches by processing candidates sequentially

## [1.0.0] - 2024-01-05

### Added

#### Core Features
- **Evidence-based evaluation system** using Claude API (Sonnet 4.5)
- **11 evaluation criteria** with customizable weights
- **Multi-format file processing**: PDF, DOCX, TXT, Markdown support
- **Batch processing** for evaluating multiple candidates
- **Comparison mode** for side-by-side candidate analysis

#### Interfaces
- **CLI tool** with comprehensive commands:
  - `evaluate` - Single candidate evaluation
  - `batch` - Batch processing from CSV
  - `init` - Configuration initialization
- **Streamlit web interface** with:
  - Single evaluation page
  - Batch evaluation page
  - Results viewer
  - Settings panel

#### Export Formats
- **JSON exporter** for structured data
- **Markdown exporter** for human-readable reports
- **HTML exporter** with styled templates
- **CSV exporter** for data analysis

#### Configuration
- YAML-based configuration system
- Environment variable support
- Customizable criteria weights
- Flexible output settings

#### Developer Tools
- Comprehensive test suite
- Type hints throughout
- Detailed logging system
- Error handling and validation

#### Documentation
- Complete README with quick start
- Detailed usage guide
- Contributing guidelines
- Example materials and templates

### Technical Details

- Python 3.8+ support
- Pydantic models for data validation
- Click for CLI framework
- Rich for terminal formatting
- Comprehensive error handling
- Production-ready logging

### Notes

Initial release providing a complete, production-ready candidate evaluation system with CLI and web interfaces, multiple export formats, and comprehensive documentation.
