import os
import math
import gc
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm
import torch.optim as optim

# SETTINGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

GLOBAL_EPOCH = 256
GLOBAL_Lr = 5e-4
BATCH_SIZE = 128
MODEL_NAME = "convnext_transformer_v5"
MODEL_WEIGHT_PATH = r"D:\Data\NYC\Retina\model_weight\convnext_tiny-983f1562.pth"
SAVE_DIR = f"./results/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, f"{MODEL_NAME}_training_log.txt")
EVAL_FILE = os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_results.txt")

# DATA
dataset = datasets.ImageFolder(root=r"D:\Data\NYC\Retina\Data")
print(f"Class to index mapping: {dataset.class_to_idx}")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_idx, temp_idx = train_test_split(
    list(range(len(dataset))), test_size=0.3, stratify=dataset.targets, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=[dataset.targets[i] for i in temp_idx], random_state=42
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

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# MODEL
class ConvNeXt_Transformer_v5(nn.Module):
    def __init__(self, convnext_variant='convnext_tiny', num_transformer_layers=8, num_heads=8, transformer_dropout=0.1):
        super().__init__()
        self.convnext = timm.create_model(convnext_variant, pretrained=False, features_only=True)
        self.convnext.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location='cpu'), strict=False)
        self.hidden_dim = self.convnext.feature_info.channels()[-1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads,
            dim_feedforward=2048, dropout=transformer_dropout,
            activation='gelu', batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        f = self.convnext(x)[-1]  # (B, C, H, W)
        B, C, H, W = f.shape
        seq = f.flatten(2).permute(2, 0, 1)
        cls = self.cls_token.expand(-1, B, -1)
        seq = torch.cat([cls, seq], dim=0)
        seq = self.pos_encoding(seq)
        enc = self.transformer_encoder(seq)
        cls_out = enc[0]
        gp = self.global_pool(f).view(B, C)
        return self.fc(torch.cat([cls_out, gp], dim=1))

# TRAINING

def train_model(model, train_loader, val_loader, num_epochs, lr, wd, save_path):
    model.to(device)
    criterion = BCEWithLogitsLossWithLabelSmoothing()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    best_loss = float('inf')
    patience, no_improve = 5, 0
    with open(LOG_FILE, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x, y = x.to(device), y.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                pred = (torch.sigmoid(out) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc = correct / total
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, None, validate_only=True)
            scheduler.step(epoch)
            log.write(f"Epoch {epoch+1}: TrainLoss={total_loss/total:.4f}, TrainAcc={acc:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}\n")
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break

# EVALUATION

def evaluate_model(model, loader, criterion, ckpt_path, validate_only=False):
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model.to(device)
    total_loss, correct, total = 0, 0, 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            out = model(x)
            loss = criterion(out, y)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
            total_loss += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)
    if validate_only:
        return total_loss / total, correct / total
    from sklearn.metrics import classification_report, confusion_matrix
    all_labels = [int(x[0]) for x in all_labels]
    all_preds = [int(x[0]) for x in all_preds]
    report = classification_report(all_labels, all_preds, target_names=["Diabetic", "Normal"])
    cm = confusion_matrix(all_labels, all_preds)
    with open(EVAL_FILE, "w") as f:
        f.write(report)
        f.write(str(cm))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig(os.path.join(SAVE_DIR, f"{MODEL_NAME}_confusion_matrix.png"))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.savefig(os.path.join(SAVE_DIR, f"{MODEL_NAME}_roc_curve.png"))
    return total_loss / total, correct / total

# MAIN
if __name__ == "__main__":
    model = ConvNeXt_Transformer_v5()
    train_model(model, train_loader, val_loader, GLOBAL_EPOCH, GLOBAL_Lr, 1e-4, os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth"))
    evaluate_model(model, test_loader, BCEWithLogitsLossWithLabelSmoothing(), os.path.join(SAVE_DIR, f"{MODEL_NAME}_best.pth"), validate_only=False)
    del model
    gc.collect()
    torch.cuda.empty_cache()
