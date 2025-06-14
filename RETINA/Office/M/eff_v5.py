# ============================================
# EfficientNetV2_CBAM_Transformer_v5 FINAL (OFFLINE READY, IMPROVED)
# ============================================

import os
import math
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm.auto import tqdm
import gc

# SETTINGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

GLOBAL_EPOCH = 256
GLOBAL_Lr = 5e-4
BATCH_SIZE = 128
MODEL_NAME = "efficientnetv2_cbam_transformer_v5"
SAVE_DIR = f"./results/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, f"{MODEL_NAME}_training_log.txt")
EVAL_FILE = os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_results.txt")

# DATA
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

# LOSS
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(inputs, targets)

# CBAM MODULE
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

# Positional Encoding
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

# EfficientNetV2_CBAM_Transformer_v5
class EfficientNetV2_CBAM_Transformer_v5(nn.Module):
    def __init__(self, num_transformer_layers=8, num_heads=8):
        super(EfficientNetV2_CBAM_Transformer_v5, self).__init__()

        self.efficientnet = timm.create_model(
            'tf_efficientnetv2_s.in21k_ft_in1k',
            pretrained=False,
            drop_path_rate=0.2
        )

        self.hidden_dim = self.efficientnet.num_features
        self.cbam = CBAM(in_planes=self.hidden_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_encoding = PositionalEncoding(d_model=self.hidden_dim, max_len=50)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.efficientnet.forward_features(x)
        x = self.cbam(x)

        B, C, H, W = x.shape
        x_seq = x.flatten(2).permute(2, 0, 1)

        cls_tokens = self.cls_token.expand(-1, B, -1)
        x_seq = torch.cat((cls_tokens, x_seq), dim=0)

        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer_encoder(x_seq)

        x_cls = x_seq[0]
        out = self.fc(x_cls)
        return out

# Training
def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, wd=1e-4, save_path="best_model.pth"):
    model = model.to(device)
    criterion = BCEWithLogitsLossWithLabelSmoothing(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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

# MAIN
if __name__ == "__main__":
    model = EfficientNetV2_CBAM_Transformer_v5(num_transformer_layers=8, num_heads=8)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=GLOBAL_EPOCH,
        lr=GLOBAL_Lr,
        wd=1e-4,
        save_path=os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth")
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()
