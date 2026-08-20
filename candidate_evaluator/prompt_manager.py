"""Prompt manager for storing and retrieving custom evaluation prompts."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from candidate_evaluator.prompts.evaluation_prompts import (
    SYSTEM_PROMPT,
    EVALUATION_PROMPT_TEMPLATE,
    HOLISTIC_EVALUATION_PROMPT,
    HOLISTIC_RANKING_PROMPT_TEMPLATE,
    HOLISTIC_SELECTION_PROMPT_TEMPLATE,
    COMBINED_ROLE_PROMPT,
)
from candidate_evaluator.prompts.role_prompts import normalize_role


class PromptManager:
    """Manages custom evaluation prompts with persistent storage."""

    PROMPT_TYPES = ["system", "criteria", "holistic", "ranking", "selection", "combined"]
    ROLE_PROMPT_TYPES = ["combined"]

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Directory for storing custom prompts.
                        Defaults to ~/.candidate_evaluator/prompts/
        """
        if prompts_dir is None:
            prompts_dir = Path.home() / ".candidate_evaluator" / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self._normalize_role = normalize_role
        self._defaults = {
            "system": SYSTEM_PROMPT,
            "criteria": EVALUATION_PROMPT_TEMPLATE,
            "holistic": HOLISTIC_EVALUATION_PROMPT,
            "ranking": HOLISTIC_RANKING_PROMPT_TEMPLATE,
            "selection": HOLISTIC_SELECTION_PROMPT_TEMPLATE,
            "combined": COMBINED_ROLE_PROMPT,
        }

    def _get_prompt_path(self, prompt_type: str) -> Path:
        """Get the file path for a prompt type."""
        return self.prompts_dir / f"{prompt_type}_prompt.json"

    def _load_custom_prompt(self, prompt_type: str) -> Optional[Dict[str, Any]]:
        """Load a custom prompt from disk if it exists."""
        path = self._get_prompt_path(prompt_type)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_prompt(self, prompt_type: str, content: str, name: Optional[str] = None) -> None:
        """Save a custom prompt to disk."""
        data = {
            "name": name or f"Custom {prompt_type.title()} Prompt",
            "type": prompt_type,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_default": False
        }

        # Check if updating existing
        existing = self._load_custom_prompt(prompt_type)
        if existing:
            data["created_at"] = existing.get("created_at", data["created_at"])

        path = self._get_prompt_path(prompt_type)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_system_prompt(self) -> str:
        """Get the system prompt (custom or default)."""
        custom = self._load_custom_prompt("system")
        if custom and custom.get("content"):
            return custom["content"]
        return self._defaults["system"]

    def get_criteria_template(self) -> str:
        """Get the criteria-based evaluation template (custom or default)."""
        custom = self._load_custom_prompt("criteria")
        if custom and custom.get("content"):
            return custom["content"]
        return self._defaults["criteria"]

    def get_holistic_template(self) -> str:
        """Get the holistic evaluation template (custom or default)."""
        custom = self._load_custom_prompt("holistic")
        if custom and custom.get("content"):
            return custom["content"]
        return self._defaults["holistic"]

    def save_system_prompt(self, content: str, name: Optional[str] = None) -> None:
        """Save a custom system prompt."""
        self._save_prompt("system", content, name)

    def save_criteria_template(self, content: str, name: Optional[str] = None) -> None:
        """Save a custom criteria-based template."""
        self._save_prompt("criteria", content, name)

    def save_holistic_template(self, content: str, name: Optional[str] = None) -> None:
        """Save a custom holistic template."""
        self._save_prompt("holistic", content, name)

    def get_ranking_template(self) -> str:
        """Get the interview-ranking template (custom or default)."""
        custom = self._load_custom_prompt("ranking")
        if custom and custom.get("content"):
            return custom["content"]
        return self._defaults["ranking"]

    def save_ranking_template(self, content: str, name: Optional[str] = None) -> None:
        """Save a custom interview-ranking template."""
        self._save_prompt("ranking", content, name)

    def get_selection_template(self) -> str:
        """Get the interview-selection template (custom or default)."""
        custom = self._load_custom_prompt("selection")
        if custom and custom.get("content"):
            return custom["content"]
        return self._defaults["selection"]

    def save_selection_template(self, content: str, name: Optional[str] = None) -> None:
        """Save a custom interview-selection template."""
        self._save_prompt("selection", content, name)

    def get_combined_prompt(self) -> str:
        """Get the combined all-dimensions role prompt (custom or default)."""
        custom = self._load_custom_prompt("combined")
        if custom and custom.get("content"):
            return custom["content"]
        return self._defaults["combined"]

    def save_combined_prompt(self, content: str, name: Optional[str] = None) -> None:
        """Save a custom combined role prompt."""
        self._save_prompt("combined", content, name)

    def get_role_prompt(self, role: str) -> str:
        """Get the role-specific prompt addendum for a given role.

        All role labels (including legacy clinician/engineer/phd values) normalize to
        the single combined prompt.
        """
        normalized = self._normalize_role(role)
        if normalized == "combined":
            return self.get_combined_prompt()
        raise ValueError(f"Invalid role: {role}")

    def save_role_prompt(self, role: str, content: str, name: Optional[str] = None) -> None:
        """Save a custom role-specific prompt."""
        role = role.lower()
        if role not in self.ROLE_PROMPT_TYPES:
            raise ValueError(f"Invalid role: {role}")
        self._save_prompt(role, content, name)

    def build_system_prompt(self, role: Optional[str] = None) -> str:
        """Build the full system prompt, optionally appending a role-specific addendum."""
        system_prompt = self.get_system_prompt()
        if not role:
            return system_prompt
        role_addendum = self.get_role_prompt(role)
        return f"{system_prompt}\n\n---\n\n{role_addendum}"

    def reset_to_default(self, prompt_type: str) -> None:
        """Reset a prompt to its default value by deleting the custom file."""
        if prompt_type not in self.PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        path = self._get_prompt_path(prompt_type)
        if path.exists():
            path.unlink()

    def get_prompt_metadata(self, prompt_type: str) -> Dict[str, Any]:
        """Get metadata about a prompt (custom or default status, timestamps, etc.)."""
        if prompt_type not in self.PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        custom = self._load_custom_prompt(prompt_type)
        if custom:
            return {
                "is_custom": True,
                "name": custom.get("name", f"Custom {prompt_type.title()} Prompt"),
                "created_at": custom.get("created_at"),
                "updated_at": custom.get("updated_at"),
                "content_length": len(custom.get("content", ""))
            }
        else:
            return {
                "is_custom": False,
                "name": f"Default {prompt_type.title()} Prompt",
                "created_at": None,
                "updated_at": None,
                "content_length": len(self._defaults.get(prompt_type, ""))
            }

    def get_default_prompt(self, prompt_type: str) -> str:
        """Get the default prompt for a given type (for comparison/reset)."""
        if prompt_type not in self.PROMPT_TYPES:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        return self._defaults.get(prompt_type, "")

    def is_custom(self, prompt_type: str) -> bool:
        """Check if a prompt type has a custom version."""
        return self._get_prompt_path(prompt_type).exists()

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all prompt types."""
        return {
            prompt_type: self.get_prompt_metadata(prompt_type)
            for prompt_type in self.PROMPT_TYPES
        }
