from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

PROMPT_TEMPLATES = {
    "default": "macro photo of {token} steel defect",
    "detail": "detail photo of {token} steel surface with {cls}",
}


@dataclass
class PromptTemplateConfig:
    prompt_template: str = PROMPT_TEMPLATES["default"]


@dataclass
class PromptBuilder:
    cfg: PromptTemplateConfig = field(default_factory=PromptTemplateConfig)

    def build_prompts(self, class_tokens: Dict[str, str]) -> List[str]:
        prompts: List[str] = []
        for cls, token in class_tokens.items():
            prompts.append(self.cfg.prompt_template.format(token=token, cls=cls))
        return prompts
