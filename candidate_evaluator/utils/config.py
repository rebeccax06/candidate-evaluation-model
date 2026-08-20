"""Configuration management for candidate evaluator"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator


class APIConfig(BaseModel):
    """API configuration"""
    anthropic_api_key: str
    model: str = "claude-sonnet-5"
    max_tokens: int = 16384
    # DEPRECATED: not sent to the API. Claude Sonnet 5 rejects non-default
    # sampling parameters (400). Kept only so existing config files that set
    # `temperature` still parse.
    temperature: float = 0.3

    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v


class CriteriaWeights(BaseModel):
    """Weights for evaluation criteria"""
    critical_thinking: int = 10
    coachability: int = 10
    curiosity: int = 10
    creativity: int = 10
    collaboration: int = 10
    follow_through: int = 10
    problem_solving_motivation: int = 10
    evidence_based: int = 10
    detail_orientation: int = 10
    communication: int = 10
    expertise_enabler: int = 10


class CriteriaConfig(BaseModel):
    """Criteria configuration"""
    weights: CriteriaWeights = Field(default_factory=CriteriaWeights)


class OutputConfig(BaseModel):
    """Output configuration"""
    output_dir: str = "./results"
    default_formats: List[str] = ["json", "markdown"]
    include_evidence: bool = True
    include_quotes: bool = True


class ProcessingConfig(BaseModel):
    """Processing configuration"""
    max_file_size_mb: int = 10
    supported_formats: List[str] = ["pdf", "docx", "txt", "md"]
    batch_size: int = 5


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = "./candidate_evaluator.log"
    console_logging: bool = True


class Config(BaseModel):
    """Main configuration class"""
    api: APIConfig
    criteria: CriteriaConfig = Field(default_factory=CriteriaConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Environment variables take precedence over config file.

    Args:
        config_path: Path to YAML config file. If None, looks for config.yaml in current directory.

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file not found and API key not in environment
        ValueError: If configuration is invalid
    """
    # Load environment variables
    load_dotenv()

    # Try to load YAML config
    config_data: Dict[str, Any] = {}

    if config_path is None:
        # Look for config.yaml in current directory and parent directories
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            potential_config = parent / "config.yaml"
            if potential_config.exists():
                config_path = str(potential_config)
                break

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

    # Override with environment variables
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key and config_data.get('api', {}).get('anthropic_api_key') == 'your-api-key-here':
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables or config file. "
            "Please set ANTHROPIC_API_KEY environment variable or update config.yaml"
        )

    if api_key:
        if 'api' not in config_data:
            config_data['api'] = {}
        config_data['api']['anthropic_api_key'] = api_key

    # Override model if specified in environment
    model = os.getenv('MODEL')
    if model:
        config_data['api']['model'] = model

    # Override output directory if specified
    output_dir = os.getenv('OUTPUT_DIR')
    if output_dir:
        if 'output' not in config_data:
            config_data['output'] = {}
        config_data['output']['output_dir'] = output_dir

    # Override log level if specified
    log_level = os.getenv('LOG_LEVEL')
    if log_level:
        if 'logging' not in config_data:
            config_data['logging'] = {}
        config_data['logging']['level'] = log_level

    # Create config object
    try:
        return Config(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def get_default_config() -> Config:
    """Get default configuration with API key from environment"""
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables. "
            "Please set ANTHROPIC_API_KEY or create a config.yaml file."
        )

    return Config(
        api=APIConfig(anthropic_api_key=api_key)
    )
