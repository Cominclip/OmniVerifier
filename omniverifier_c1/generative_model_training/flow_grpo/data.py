import json
from pathlib import Path

from PIL import Image, ImageOps


class PromptImageDataset:
    def __init__(self, dataset, resolution=512, split="train", mode="pointwise"):
        if mode not in ("pointwise", "pairwise"):
            raise ValueError(f"Unknown verification mode: {mode}")
        self.dataset = Path(dataset).expanduser().resolve()
        self.resolution = resolution
        self.mode = mode
        self.file_path = self.dataset / f"{split}_metadata.jsonl"
        self.metadatas = []
        with self.file_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    raise ValueError(f"{self.file_path}:{line_number}: prompt must be a nonempty editing instruction")
                for field in ("initial_image", "reference_image") if mode == "pairwise" else ("initial_image",):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"{self.file_path}:{line_number}: {mode} requires {field}")
                    if not self.image_path(row[field]).is_file():
                        raise FileNotFoundError(self.image_path(row[field]))
                self.metadatas.append(row)
        if not self.metadatas:
            raise ValueError(f"Dataset is empty: {self.file_path}")
        self.prompts = [row["prompt"] for row in self.metadatas]

    def image_path(self, value):
        return self.dataset / Path(value).expanduser()

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, index):
        # Padded sampler entries keep every GPU's batch shape equal; they never train.
        is_padding = isinstance(index, tuple)
        if is_padding:
            index, is_padding = index
        row = self.metadatas[index]
        with Image.open(self.image_path(row["initial_image"])) as source:
            initial_image = ImageOps.fit(
                source.convert("RGB"), (self.resolution, self.resolution), method=Image.Resampling.LANCZOS
            )
        reference_image = None
        if self.mode == "pairwise":
            with Image.open(self.image_path(row["reference_image"])) as source:
                reference_image = source.convert("RGB")
                reference_image.thumbnail((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        return {
            "prompt": row["prompt"],
            "metadata": row,
            "initial_image": initial_image,
            "reference_image": reference_image,
            "group_id": -1 if is_padding else index,
        }

    @staticmethod
    def collate_fn(examples):
        return (
            [example["prompt"] for example in examples],
            [example["metadata"] for example in examples],
            [example["initial_image"] for example in examples],
            [example["reference_image"] for example in examples],
            [example["group_id"] for example in examples],
        )
