# ============================================
# EfficientNet_CBAM Final Version (v1)
# ============================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm.auto import tqdm
import gc

# ------------------ SETTINGS ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

GLOBAL_EPOCH = 256
GLOBAL_Lr = 5e-4
BATCH_SIZE = 128
MODEL_NAME = "efficientnet_cbam_final"
SAVE_DIR = f"./results/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, f"{MODEL_NAME}_training_log.txt")
EVAL_FILE = os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_results.txt")

# ------------------ DATA ------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=r"D:\Data\NYC\Retina\Data")
print(f"Class to index mapping: {dataset.class_to_idx}")

train_idx, temp_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.3,
    stratify=dataset.targets,
    random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=[dataset.targets[i] for i in temp_idx],
    random_state=42
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = test_transforms
test_dataset.dataset.transform = test_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Total images: {len(dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ------------------ LABEL SMOOTHING LOSS ------------------
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(inputs, targets)

# ------------------ CBAM Module ------------------
# ===========================================
# EfficientNet_CBAM_Transformer_v3 (FINAL)
# ===========================================

import math
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

# ----------- CBAM -----------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# ----------- Positional Encoding -----------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

# ----------- EfficientNet_CBAM_Transformer_v3 -----------

class EfficientNet_CBAM_Transformer_v3(nn.Module):
    def __init__(self, num_transformer_layers=1, num_heads=8):
        super(EfficientNet_CBAM_Transformer_v3, self).__init__()

        # Load pretrained EfficientNet-B0
        efficientnet = efficientnet_b0(pretrained=True)

        # Keep everything except classifier
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.hidden_dim = 1280  # output of EfficientNet-B0

        # CBAM
        self.cbam = CBAM(in_planes=self.hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=self.hidden_dim, max_len=49)  # 7x7 feature map

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Final FC head (higher capacity + GELU)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # EfficientNet features
        x = self.features(x)  # (B, C, H, W)

        # CBAM
        x = self.cbam(x)

        # Flatten to sequence
        B, C, H, W = x.shape
        x_seq = x.flatten(2).permute(2, 0, 1)  # (HW, B, C)

        # Positional encoding
        x_seq = self.pos_encoding(x_seq)

        # Transformer encoder
        x_seq = self.transformer_encoder(x_seq)

        # Global average pooling over sequence
        x_seq = x_seq.mean(dim=0)  # (B, C)

        # Final FC head
        out = self.fc(x_seq)
        return out

# ------------------ Training ------------------
def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, wd=1e-4, save_path="best_model.pth"):
    model = model.to(device)
    criterion = BCEWithLogitsLossWithLabelSmoothing(smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 5

    with open(LOG_FILE, "w") as log_f:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            log_f.write(f"\nEpoch {epoch + 1}/{num_epochs}\n")

            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for inputs, labels in tqdm(train_loader, desc="Training"):
                inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / train_total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / val_total

            scheduler.step(epoch)

            log_f.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}\n")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print("Validation loss improved. Model saved.")
                log_f.write("Validation loss improved. Model saved.\n")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                log_f.write("Early stopping triggered.\n")
                break

# ------------------ Evaluation ------------------
def evaluate_model(model, test_loader, criterion, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / test_total

    class_names = ['Diabetic', 'Normal']
    all_labels = [int(x[0]) for x in all_labels]
    all_preds = [int(x[0]) for x in all_preds]

    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    with open(EVAL_FILE, "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{MODEL_NAME} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{MODEL_NAME}_confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.title(f"{MODEL_NAME} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{MODEL_NAME}_roc_curve.png"))
    plt.close()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    efficientnet_cbam_transformer_v3 = EfficientNet_CBAM_Transformer_v3(num_transformer_layers=1, num_heads=8)

    train_model(
        model=efficientnet_cbam_transformer_v3,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=GLOBAL_EPOCH,
        lr=GLOBAL_Lr,
        wd=1e-4,
        save_path=os.path.join(SAVE_DIR, f"efficientnet_cbam_transformer_v3_best.pth")
    )

    evaluate_model(
        model=efficientnet_cbam_transformer_v3,
        test_loader=test_loader,
        criterion=BCEWithLogitsLossWithLabelSmoothing(smoothing=0.1),
        checkpoint_path=os.path.join(SAVE_DIR, f"efficientnet_cbam_transformer_v3_best.pth")
    )


    del efficientnet_cbam_transformer_v3
    torch.cuda.empty_cache()
    gc.collect()
