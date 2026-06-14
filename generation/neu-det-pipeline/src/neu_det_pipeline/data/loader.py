from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np
from rich.progress import track
from sklearn.model_selection import train_test_split

ANNOT_SUFFIX = ".xml"
IMG_SUFFIX = ".jpg"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class DefectObject:
    cls_name: str
    bbox: tuple[int, int, int, int]
    index: int


@dataclass
class DefectSample:
    image_path: Path
    annotation_path: Path
    cls_name: str
    bbox: tuple[int, int, int, int]
    object_index: int = 0
    objects: tuple[DefectObject, ...] = ()
    target_key_override: Optional[str] = None

    @property
    def source_stem(self) -> str:
        return self.image_path.stem

    @property
    def target_key(self) -> str:
        return self.target_key_override or self.image_path.stem


@dataclass
class DatasetSplits:
    train: List[DefectSample]
    val: List[DefectSample]


def load_pascal_voc_objects(xml_path: Path) -> list[DefectObject]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    objects: list[DefectObject] = []
    for idx, obj in enumerate(tree.findall("object")):
        cls = obj.findtext("name")
        bbox_el = obj.find("bndbox")
        if not cls or bbox_el is None:
            continue
        bbox = tuple(int(bbox_el.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax"))
        objects.append(DefectObject(cls, bbox, idx))
    return objects


def load_pascal_voc_annotation(xml_path: Path) -> tuple[str, tuple[int, int, int, int]]:
    objects = load_pascal_voc_objects(xml_path)
    if not objects:
        raise ValueError(f"No Pascal VOC object found in {xml_path}")
    first = objects[0]
    return first.cls_name, first.bbox


def _candidate_image_paths(root: Path, stem: str, split_name: str | None = None) -> Iterator[Path]:
    image_roots = [root / "IMAGES", root / "images"]
    for image_root in image_roots:
        search_roots = []
        if split_name:
            search_roots.append(image_root / split_name)
        search_roots.append(image_root)
        for search_root in search_roots:
            for suffix in IMAGE_SUFFIXES:
                yield search_root / f"{stem}{suffix}"


def _voc_annotation_image_pairs(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield Pascal VOC annotation/image pairs in flat or split layouts.

    Supported layouts:
    - ``root/ANNOTATIONS/*.xml`` with ``root/IMAGES/*.jpg``
    - ``root/annotations/{train,val,test}/*.xml`` with matching
      ``root/images/{train,val,test}/*``.
    """

    annotation_roots = [root / "ANNOTATIONS", root / "annotations"]
    seen: set[Path] = set()
    for ann_root in annotation_roots:
        if not ann_root.exists():
            continue
        for xml_path in sorted(ann_root.rglob(f"*{ANNOT_SUFFIX}")):
            if xml_path in seen:
                continue
            seen.add(xml_path)
            split_name = xml_path.parent.name if xml_path.parent != ann_root else None
            for img_path in _candidate_image_paths(root, xml_path.stem, split_name):
                if img_path.exists():
                    yield xml_path, img_path
                    break


def collect_dataset(root: Path) -> List[DefectSample]:
    samples: List[DefectSample] = []
    for xml_path, img_path in _voc_annotation_image_pairs(root):
        cls, bbox = load_pascal_voc_annotation(xml_path)
        samples.append(DefectSample(img_path, xml_path, cls, bbox))
    return samples


def collect_dataset_images(root: Path) -> List[DefectSample]:
    """Collect one sample per image while retaining all annotated objects."""

    samples: List[DefectSample] = []
    for xml_path, img_path in _voc_annotation_image_pairs(root):
        objects = tuple(load_pascal_voc_objects(xml_path))
        if not objects:
            continue

        primary = max(
            objects,
            key=lambda obj: max(0, obj.bbox[2] - obj.bbox[0]) * max(0, obj.bbox[3] - obj.bbox[1]),
        )
        union_bbox = (
            min(obj.bbox[0] for obj in objects),
            min(obj.bbox[1] for obj in objects),
            max(obj.bbox[2] for obj in objects),
            max(obj.bbox[3] for obj in objects),
        )
        samples.append(
            DefectSample(
                img_path,
                xml_path,
                primary.cls_name,
                union_bbox,
                object_index=primary.index,
                objects=objects,
                target_key_override=xml_path.stem,
            )
        )
    return samples


def collect_dataset_instances(root: Path) -> List[DefectSample]:
    """Collect one sample per annotated object while keeping image-level context."""

    samples: List[DefectSample] = []
    for xml_path, img_path in _voc_annotation_image_pairs(root):
        objects = tuple(load_pascal_voc_objects(xml_path))
        if not objects:
            continue
        for obj in objects:
            samples.append(
                DefectSample(
                    img_path,
                    xml_path,
                    obj.cls_name,
                    obj.bbox,
                    object_index=obj.index,
                    objects=objects,
                    target_key_override=f"{xml_path.stem}_o{obj.index:02d}",
                )
            )
    return samples


def split_dataset(samples: Sequence[DefectSample], test_size: float = 0.1, seed: int = 42) -> DatasetSplits:
    train, val = train_test_split(samples, test_size=test_size, random_state=seed, stratify=[s.cls_name for s in samples])
    return DatasetSplits(list(train), list(val))


def export_metadata(samples: Sequence[DefectSample], out_path: Path) -> None:
    data = [
        {
            "image": str(sample.image_path),
            "annotation": str(sample.annotation_path),
            "class": sample.cls_name,
            "bbox": sample.bbox,
            "object_index": sample.object_index,
            "target_key": sample.target_key,
            "object_count": len(sample.objects) if sample.objects else 1,
        }
        for sample in samples
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))


def copy_assets(samples: Sequence[DefectSample], dest_root: Path) -> None:
    img_out = dest_root / "images"
    ann_out = dest_root / "annotations"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)
    for sample in track(samples, description="Copying assets"):
        shutil.copy2(sample.image_path, img_out / sample.image_path.name)
        shutil.copy2(sample.annotation_path, ann_out / sample.annotation_path.name)


def compute_class_counts(samples: Sequence[DefectSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.cls_name] = counts.get(sample.cls_name, 0) + 1
    return counts


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dataset_iterator(samples: Sequence[DefectSample]) -> Iterator[tuple[np.ndarray, DefectSample]]:
    for sample in samples:
        yield load_image(sample.image_path), sample

