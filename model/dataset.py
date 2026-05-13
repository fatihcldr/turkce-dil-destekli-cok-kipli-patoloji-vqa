from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch
import io


ANSWER_TO_IDX = {"yes": 0, "no": 1}


class PathVQADataset(Dataset):
    def __init__(self, split="train", image_processor=None, tokenizer=None, max_length=128):
        raw = load_dataset("flaviagiammarino/path-vqa", split=split)
        self.data = [
            item for item in raw
            if str(item["answer"]).strip().lower() in ANSWER_TO_IDX
        ]
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["image"]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        question = str(item["question"])
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = ANSWER_TO_IDX[str(item["answer"]).strip().lower()]

        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
