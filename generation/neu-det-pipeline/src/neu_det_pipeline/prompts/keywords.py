"""关键词提取工具，用于构建论文风格的提示词。"""

from typing import List, Optional


class KeywordExtractor:
    """Extract high-frequency keywords for DRFT-v2 class-token prompts."""
    
    # Paper-specified keywords (top 40% frequency-based selection)
    # 注意：rolled-in scale 已移除，因为它是特定缺陷类别，不应作为通用关键词
    PAPER_KEYWORDS = [
        "grayscale", "greyscale", "hotrolled steel strip", "monochrome", "no humans",
        "surface defects", "texture"
    ]
    
    DEFECT_SPECIFIC_KEYWORDS = {
        "crazing": ["fine cracks", "crack networks", "fracture lines"],
        "inclusion": ["embedded particles", "inclusions", "dark clusters"],
        "patches": ["oxidized patches", "stain-like", "diffuse defects"],
        "pitted_surface": ["pitting", "corrosion pits", "cavities"],
        "rolled-in_scale": ["oxide scale", "scale streaks", "residue bands"],
        "scratches": ["scratch grooves", "abrasion", "scratch marks"],
    }
    
    @staticmethod
    def build_paper_style_prompt(
        class_name: str,
        token: str,
        base_keywords: Optional[List[str]] = None,
        defect_specific: Optional[List[str]] = None,
        lora_weight: float = 1.0,
    ) -> str:
        """Build paper-style prompt: keyword1, keyword2, ..., loRA:token:weight"""
        keywords = []
        if base_keywords:
            keywords.extend(base_keywords)
        if defect_specific:
            keywords.extend(defect_specific)
        
        unique_keywords = list(dict.fromkeys(keywords))
        lora_weight_int = int(lora_weight) if lora_weight == int(lora_weight) else lora_weight
        prompt = ", ".join(unique_keywords)
        prompt += f", loRA:{token}:{lora_weight_int}"
        return prompt
