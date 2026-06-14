from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from rich.progress import track
from transformers import CLIPModel, CLIPProcessor

from ..data import DefectSample
from .keywords import KeywordExtractor


# 延迟导入 BLIP-2 以避免循环导入
def _get_blip2_generator():
    """延迟导入 BLIP2CaptionGenerator"""
    from .blip2_caption import BLIP2CaptionGenerator
    return BLIP2CaptionGenerator


class CaptionGenerator:
    """Generate defect-generation prompts by ranking templates with CLIP vision-language similarity."""

    BASE_TEMPLATES: Sequence[str] = (
        "macro shot of {token} steel surface showing {descriptor}",
        "industrial close-up of {token} exhibiting {descriptor}",
        "{descriptor} on hot-rolled steel texture ({token})",
        "detail photo of {token} steel plate with {descriptor}",
    )

    DEFECT_DESCRIPTORS: Dict[str, List[str]] = {
        "crazing": ["fine crack networks", "crazing fracture lines", "micro-scale crack mesh"],
        "inclusion": ["embedded inclusions", "foreign particles trapped in steel", "dark inclusion clusters"],
        "patches": ["irregular oxidized patches", "stain-like surface patches", "diffuse defect patches"],
        "pitted_surface": ["pitting corrosion pits", "scattered cavities", "etched pit patterns"],
        "rolled-in_scale": ["rolled-in oxide scale", "compressed scale streaks", "oxide residue bands"],
        "scratches": ["linear scratch grooves", "abrasion streaks", "parallel scratch marks"],
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or self._require_discrete_cuda()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _require_discrete_cuda() -> torch.device:
        """Ensure CLIP scoring runs on a CUDA-capable discrete GPU."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用的 NVIDIA CUDA 设备。提示生成已强制使用独立显卡，不再回落 CPU/核显。"
            )
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"Using CUDA device {idx}: {props.name} ({props.total_memory / (1024 ** 3):.1f} GB)")
        return torch.device(f"cuda:{idx}")
    
    def _build_candidates(self, cls_name: str, token: str) -> List[str]:
        descriptors = self.DEFECT_DESCRIPTORS.get(
            cls_name, ["steel surface defect texture", cls_name.replace("_", " ")]
        )
        prompts: List[str] = []
        for template in self.BASE_TEMPLATES:
            for descriptor in descriptors:
                prompts.append(template.format(token=token, descriptor=descriptor, cls=cls_name.replace("_", " ")))
        # Ensure at least one fallback template
        prompts.append(f"macro photo of {token} steel surface highlighting {cls_name.replace('_', ' ')}")
        return list(dict.fromkeys(prompt.strip() for prompt in prompts))  # deduplicate while preserving order

    def _select_prompt(self, image: Image.Image, candidates: List[str]) -> str:
        inputs = self.processor(text=candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
        best_idx = int(torch.argmax(logits))
        return candidates[best_idx]

    def generate_with_token(
        self,
        samples: List[DefectSample],
        token_map: Dict[str, str],
        output_file: Optional[Path] = None,
        use_paper_keywords: bool = True,
        lora_weight: float = 1.0,
        use_clip_selection: bool = False,
    ) -> Dict[str, str]:
        """
        Generate captions using paper-style keyword format with LoRA weights.
        
        Args:
            samples: List of defect samples
            token_map: Mapping from class name to learned token
            output_file: Optional path to save caption mapping
            use_paper_keywords: If True, use paper-specified keywords
            lora_weight: LoRA weight in prompt (typically 1.0)
            use_clip_selection: If True, use CLIP to select best template for each image
                               (slower but more accurate). If False, just concatenate keywords.
            
        Returns:
            Dictionary mapping image stem to paper-style prompts
        """
        keyword_extractor = KeywordExtractor()
        captions: Dict[str, str] = {}
        
        # 如果使用 CLIP 选择，需要加载图像
        if use_clip_selection:
            from .data import load_image
        
        for sample in track(samples, description="Generating paper-style captions"):
            stem = sample.image_path.stem
            cls_name = sample.cls_name
            token = token_map.get(cls_name, cls_name.replace("_", " "))
            
            base_keywords = keyword_extractor.PAPER_KEYWORDS
            defect_keywords = keyword_extractor.DEFECT_SPECIFIC_KEYWORDS.get(
                cls_name, [cls_name.replace("_", " ")]
            )
            
            if use_clip_selection:
                # 使用 CLIP 选择最佳模板
                try:
                    # 加载图像
                    image_array = load_image(sample.image_path)
                    # 转换为 PIL Image
                    import numpy as np
                    if len(image_array.shape) == 2:
                        image = Image.fromarray(image_array.astype(np.uint8)).convert("RGB")
                    else:
                        image = Image.fromarray(image_array[:, :, :3].astype(np.uint8))
                    
                    # 构建候选模板（结合关键词）
                    candidates = []
                    for descriptor in defect_keywords:
                        for template in self.BASE_TEMPLATES:
                            template_prompt = template.format(
                                token=token, 
                                descriptor=descriptor,
                                cls=cls_name.replace("_", " ")
                            )
                            # 添加基础关键词
                            full_prompt = f"{template_prompt}, {', '.join(base_keywords)}"
                            candidates.append(full_prompt)
                    
                    # 如果没有候选，使用回退
                    if not candidates:
                        candidates = [f"macro photo of {token} steel surface"]
                    
                    # 使用 CLIP 选择最佳提示词
                    best_template = self._select_prompt(image, candidates)
                    
                    # 添加 LoRA 权重
                    lora_weight_int = int(lora_weight) if lora_weight == int(lora_weight) else lora_weight
                    prompt = f"{best_template}, loRA:{token}:{lora_weight_int}"
                    
                except Exception as e:
                    # 如果 CLIP 选择失败，回退到关键词拼接
                    print(f"Warning: CLIP selection failed for {stem}: {e}, falling back to keyword concatenation")
                    prompt = keyword_extractor.build_paper_style_prompt(
                        class_name=cls_name,
                        token=token,
                        base_keywords=base_keywords,
                        defect_specific=defect_keywords,
                        lora_weight=lora_weight,
                    )
            else:
                # 简单拼接关键词（当前默认行为）
                prompt = keyword_extractor.build_paper_style_prompt(
                    class_name=cls_name,
                    token=token,
                    base_keywords=base_keywords,
                    defect_specific=defect_keywords,
                    lora_weight=lora_weight,
                )
            
            captions[stem] = prompt
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
        
        return captions
    
    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_captions_with_blip2(
    samples: List[DefectSample],
    token_map: Dict[str, str],
    output_file: Optional[Path] = None,
    use_paper_keywords: bool = True,
    lora_weight: float = 1.0,
    model_name: str = "Salesforce/blip2-opt-2.7b",
    combine_with_keywords: bool = True,
) -> Dict[str, str]:
    """
    使用 BLIP-2 生成提示词的便捷函数。
    
    Args:
        samples: 缺陷样本列表
        token_map: 类别到 token 的映射
        output_file: 输出文件路径（可选）
        use_paper_keywords: 是否结合论文关键词
        lora_weight: LoRA 权重
        model_name: BLIP-2 模型名称
        combine_with_keywords: 是否将 BLIP-2 描述与关键词结合
        
    Returns:
        字典：图像名称 -> 提示词
    """
    # 延迟导入以避免循环导入
    from .blip2_caption import BLIP2CaptionGenerator
    
    generator = BLIP2CaptionGenerator(model_name=model_name)
    try:
        captions = generator.generate_with_token(
            samples=samples,
            token_map=token_map,
            output_file=output_file,
            use_paper_keywords=use_paper_keywords,
            lora_weight=lora_weight,
            combine_with_keywords=combine_with_keywords,
        )
        return captions
    finally:
        generator.cleanup()


def load_captions_from_file(caption_file: Path) -> Dict[str, str]:
    """Load pre-generated captions from JSON file."""
    import json
    with open(caption_file, "r", encoding="utf-8") as f:
        return json.load(f)
