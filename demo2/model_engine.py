import torch
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from config import MODEL_DIR, DEVICE


class LayoutLMv3Predictor:
    def __init__(self, model_dir=MODEL_DIR, device=DEVICE):
        if not model_dir.exists():
            raise FileNotFoundError(f"Khong tim thay model tai: {model_dir}")

        print(f"Dang tai LayoutLMv3 tu {model_dir} tren {device}...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            str(model_dir)
        )
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    def predict_words(self, image, records):
        if not records:
            return records

        words = [record["word"] for record in records]
        boxes = [record["box"] for record in records]

        encoding = self.processor(
            images=[image],
            text=[words],
            boxes=[boxes],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "bbox", "pixel_values"}
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        predictions = outputs.logits.argmax(dim=-1)[0].cpu().tolist()

        seen = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)

            label_id = predictions[token_index]
            records[word_id]["label"] = self.id2label.get(label_id, str(label_id))
            records[word_id]["model_confidence"] = float(
                probabilities[token_index, label_id].item()
            )

        for record in records:
            record.setdefault("label", "NOT_PROCESSED")
            record.setdefault("model_confidence", 0.0)

        return records