"""
BLIP VQA Fine-Tuning Script for BoneFractureQA
================================================
Fine-tunes Salesforce/blip-vqa-base on bone fracture X-ray QA data.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🚀 Training BLIP on Device: {device}")

JSONL_PATH = "/Users/saimohith/Documents/Sem-5/DL/BoneFractureQA/metadata.jsonl"
BLIP_SAVE_DIR = "/Users/saimohith/Documents/Sem-5/DL/BoneFractureQA/fine_tuned_blip"


class BoneFractureVQADataset(Dataset):
    def __init__(self, jsonl_path, processor, split='train'):
        self.dataset = []
        self.processor = processor
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['split'] == split:
                    self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            image = Image.open(item['image']).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='white')

        inputs = self.processor(images=image, text=item['question'], padding="max_length", return_tensors="pt")
        labels = self.processor(text=item['answer'], padding="max_length", return_tensors="pt").input_ids

        # 🛠️ THE CRITICAL FIX: Tell the loss function to ignore all padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels.squeeze(0)
        return inputs


if __name__ == "__main__":
    print("🧠 Initializing Salesforce BLIP VQA Architecture...")
    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

    print("📦 Loading Datasets...")
    train_dataset = BoneFractureVQADataset(JSONL_PATH, processor, split='train')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("🔥 Starting BLIP Fine-Tuning...")
    model.train()
    epochs = 1
    for epoch in range(epochs):
        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0:
                print(f"   Batch: {idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    model.save_pretrained(BLIP_SAVE_DIR)
    processor.save_pretrained(BLIP_SAVE_DIR)
    print(f"✅ BLIP VQA Weights Saved to {BLIP_SAVE_DIR}")
