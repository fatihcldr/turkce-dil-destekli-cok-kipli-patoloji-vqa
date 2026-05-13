import torch
import io
from PIL import Image
from transformers import ViTImageProcessor, BertTokenizer
from transformers import MarianMTModel, MarianTokenizer

from model.model import PathVQAModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IDX_TO_ANSWER = {0: "yes", 1: "no"}
TR_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-tr-en"


class VQAInference:
    def __init__(self, model_path: str):
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.translator_tokenizer = MarianTokenizer.from_pretrained(TR_EN_MODEL)
        self.translator = MarianMTModel.from_pretrained(TR_EN_MODEL).to(DEVICE)
        self.translator.eval()

        self.model = PathVQAModel().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    def translate_tr_to_en(self, text: str) -> str:
        inputs = self.translator_tokenizer([text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            translated = self.translator.generate(**inputs)
        return self.translator_tokenizer.decode(translated[0], skip_special_tokens=True)

    def predict(self, image_bytes: bytes, question: str) -> dict:
        en_question = self.translate_tr_to_en(question)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)

        encoding = self.tokenizer(
            en_question,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = self.model(pixel_values, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(1).item()
            confidence = probs[0][pred_idx].item()

        return {
            "answer": IDX_TO_ANSWER[pred_idx],
            "confidence": round(confidence, 4),
            "translated_question": en_question,
        }
