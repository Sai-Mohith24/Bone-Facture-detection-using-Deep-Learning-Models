"""
CNN Training Script for BoneFractureQA — with Full Metrics
===========================================================
Includes: per-batch & per-epoch accuracy, validation evaluation,
AUC-ROC curve, and confusion matrix plots.
"""

# ──────────────────────────────────────────────
# 0. SSL Fix (MUST be before any network calls)
# ──────────────────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ──────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ──────────────────────────────────────────────
# 2. Device Selection (Apple Silicon / CUDA / CPU)
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🚀 Training CNN on Device: {device}")

# ──────────────────────────────────────────────
# 3. Paths
# ──────────────────────────────────────────────
BASE_DIR = "/Users/saimohith/Documents/Sem-5/DL/BoneFractureQA"
JSONL_PATH = os.path.join(BASE_DIR, "metadata.jsonl")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "efficientnet_fracture.pth")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 4. Transforms
# ──────────────────────────────────────────────
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────
# 5. Dataset
# ──────────────────────────────────────────────
class CNNFractureDataset(Dataset):
    def __init__(self, jsonl_path, split='train', transform=None):
        self.dataset = []
        self.transform = transform
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['split'] == split:
                    self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = 1.0 if "Yes" in item['answer'] else 0.0
        try:
            image = Image.open(item['image']).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='white')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)

# ──────────────────────────────────────────────
# 6. Model
# ──────────────────────────────────────────────
class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.model(x)

# ──────────────────────────────────────────────
# 7. Data Loading
# ──────────────────────────────────────────────
print("📦 Loading Datasets...")
train_dataset = CNNFractureDataset(JSONL_PATH, split='train', transform=cnn_transform)
val_dataset   = CNNFractureDataset(JSONL_PATH, split='val',   transform=cnn_transform)
test_dataset  = CNNFractureDataset(JSONL_PATH, split='test',  transform=cnn_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ──────────────────────────────────────────────
# 8. Model, Loss, Optimizer
# ──────────────────────────────────────────────
print("🏗️ Building EfficientNet-B0...")
model_cnn = EfficientNetClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_cnn.model.classifier.parameters(), lr=1e-3)

# ──────────────────────────────────────────────
# 9. Training Loop — with Accuracy per batch/epoch
# ──────────────────────────────────────────────
epochs = 3
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print(f"🔥 Starting CNN Training for {epochs} Epochs...\n")
for epoch in range(epochs):
    # ── Training Phase ──
    model_cnn.train()
    total_loss = 0
    correct = 0
    total = 0

    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Batch accuracy
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if idx % 10 == 0:
            batch_acc = (preds == labels).float().mean().item() * 100
            print(f"   Epoch {epoch+1}/{epochs} | Batch {idx:>3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.1f}%")

    epoch_train_loss = total_loss / len(train_loader)
    epoch_train_acc  = correct / total * 100
    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)

    # ── Validation Phase ──
    model_cnn.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_cnn(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc  = val_correct / val_total * 100
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)

    print(f"\n   📊 Epoch {epoch+1}/{epochs} Summary:")
    print(f"      Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
    print(f"      Val   Loss: {epoch_val_loss:.4f} | Val   Acc: {epoch_val_acc:.2f}%\n")

# ──────────────────────────────────────────────
# 10. Save Model
# ──────────────────────────────────────────────
torch.save(model_cnn.state_dict(), MODEL_SAVE_PATH)
print(f"✅ CNN Backbone Saved to {MODEL_SAVE_PATH}\n")

# ──────────────────────────────────────────────
# 11. Plot Training History (Loss + Accuracy)
# ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(range(1, epochs+1), history["train_loss"], 'o-', label="Train Loss", color="#FF6B6B")
ax1.plot(range(1, epochs+1), history["val_loss"],   'o-', label="Val Loss",   color="#4ECDC4")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2.plot(range(1, epochs+1), history["train_acc"], 'o-', label="Train Acc", color="#FF6B6B")
ax2.plot(range(1, epochs+1), history["val_acc"],   'o-', label="Val Acc",   color="#4ECDC4")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training & Validation Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
plt.show()
print(f"📈 Training history plot saved to {PLOTS_DIR}/training_history.png")

# ──────────────────────────────────────────────
# 12. Evaluate on Val/Test — AUC-ROC + Confusion Matrix
# ──────────────────────────────────────────────
def evaluate_and_plot(model, loader, split_name, device):
    """Run inference, compute metrics, plot AUC-ROC & confusion matrix."""
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    # ── Metrics ──
    acc  = accuracy_score(all_labels, all_preds) * 100
    auc  = roc_auc_score(all_labels, all_probs)
    print(f"\n{'='*50}")
    print(f"📋 {split_name.upper()} SET RESULTS")
    print(f"{'='*50}")
    print(f"   Accuracy : {acc:.2f}%")
    print(f"   AUC-ROC  : {auc:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['No Fracture', 'Fracture'])}")

    # ── Plot: AUC-ROC Curve ──
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fpr, tpr, color="#FF6B6B", lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random (AUC = 0.5)")
    ax1.fill_between(fpr, tpr, alpha=0.15, color="#FF6B6B")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"AUC-ROC Curve — {split_name}")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Plot: Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Fracture", "Fracture"])
    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
    ax2.set_title(f"Confusion Matrix — {split_name}")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"auc_cm_{split_name.lower()}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Plots saved to {save_path}")

    return {"accuracy": acc, "auc": auc}

# Run evaluation on validation and test sets
print("\n" + "🔍 Running Final Evaluation...")
val_metrics  = evaluate_and_plot(model_cnn, val_loader,  "Validation", device)
test_metrics = evaluate_and_plot(model_cnn, test_loader, "Test",       device)

print(f"\n{'='*50}")
print(f"🏁 ALL DONE!")
print(f"{'='*50}")
print(f"   Model      : {MODEL_SAVE_PATH}")
print(f"   Val  Acc    : {val_metrics['accuracy']:.2f}%  |  AUC: {val_metrics['auc']:.4f}")
print(f"   Test Acc    : {test_metrics['accuracy']:.2f}%  |  AUC: {test_metrics['auc']:.4f}")
print(f"   Plots saved : {PLOTS_DIR}/")
