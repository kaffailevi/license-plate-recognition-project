import glob
import os
import random
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset


class CarPlateDataset(Dataset):
    """Dataset for the Kaggle car-plate-detection annotations."""

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        train: bool = True,
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.image_dir = os.path.join(root, "images")
        annotation_dir = os.path.join(root, "annotations")
        annotation_files = sorted(glob.glob(os.path.join(annotation_dir, "*.xml")))

        if not annotation_files:
            raise FileNotFoundError(
                f"No annotation files were found in {annotation_dir}. "
                "Download and extract the Kaggle dataset into this folder."
            )

        indices = list(range(len(annotation_files)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        split_idx = max(1, int(len(indices) * split_ratio))
        selected = indices[:split_idx] if train else indices[split_idx:]
        if not selected:
            selected = indices

        self.annotation_files = [annotation_files[i] for i in selected]

    def __len__(self) -> int:
        return len(self.annotation_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ann_path = self.annotation_files[idx]
        image_id = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = self._resolve_image_path(image_id)

        image = Image.open(img_path).convert("RGB")
        boxes: List[List[float]] = []
        areas: List[float] = []

        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax - xmin) * (ymax - ymin))

        if not boxes:
            raise ValueError(f"No bounding boxes found for {ann_path}")

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target

    def _resolve_image_path(self, image_id: str) -> str:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find image for annotation {image_id} in {self.image_dir}"
        )


def collate_fn(batch):
    return tuple(zip(*batch))
