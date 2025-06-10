import os
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from torchvision import models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to dataset folders
ddiabetic_folder = r"D:\Data\NYC\Retina\Data\Diabetic"
normal_folder = r"D:\Data\NYC\Retina\Data\Diabetic\Normal"

# Transformations for training data (with augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Transformations for validation and test data (without augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Combine both folders into a single dataset
dataset = datasets.ImageFolder(
    root=r"D:\Data\NYC\Retina\Data"
)

# Explicitly verify class names and indices
print(f"Class to index mapping: {dataset.class_to_idx}")

# Splitting indices for train, validation, and test sets
train_idx, temp_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.3,  # 30% for validation and test
    stratify=dataset.targets,  # Maintain class balance
    random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,  # Split remaining 30% equally into validation and test
    stratify=[dataset.targets[i] for i in temp_idx],  # Maintain class balance
    random_state=42
)

# Create subsets for DataLoader
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Apply transformations to subsets
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = test_transforms
test_dataset.dataset.transform = test_transforms

# DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Verify dataset sizes
print(f"Total images: {len(dataset)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")


GLOBAL_EPOCH = 250          # or whatever value you want
GLOBAL_Lr = 5e-4           # or any learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eva_type = "full"          # (example) naming the evaluation file



import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # For logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.cuda.amp as amp  # For mixed precision training

def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, wd=1e-4,
                log_dir="./logs", save_path="best_model.pth", patience=5,
                use_scheduler=True, use_mixed_precision=False):
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # For binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    scaler = amp.GradScaler() if use_mixed_precision else None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()

            with amp.autocast(enabled=use_mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation
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
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Validation loss improved. Model weights saved!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if use_scheduler:
            scheduler.step(val_loss)

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs! Best Val Loss: {best_val_loss:.4f}")
            break

    print("Training Complete!")

    # Plotting
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    


import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # <-- For saving CSV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tqdm.auto import tqdm  # <-- Smoother tqdm for all environments (optional but better)

def evaluate_model(model, test_loader, criterion, model_name,
                   checkpoint_path=None, output_file="evaluation_results.txt",
                   save_dir="./results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If checkpoint is provided, load it
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs, labels = inputs.to(device), labels.float().to(device)

            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            outputs = model(inputs)

            # Handle output if model outputs a dict (like ViT sometimes)
            if isinstance(outputs, dict) and "logits" in outputs:
                outputs = outputs["logits"]

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

    # Explicit class names + class order handling
    class_names = ['Diabetic', 'Normal']
    class_labels = [0, 1]

    # Ensure labels and preds are flattened properly
    all_labels = [int(x[0]) for x in all_labels]  # Flatten nested arrays
    all_preds = [int(x[0]) for x in all_preds]

    # Classification report (handle missing classes safely)
    report = classification_report(
        all_labels, all_preds,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )

    # Confusion matrix (explicit label order)
    cm = confusion_matrix(
        all_labels, all_preds,
        labels=class_labels
    )

    # Save text results
    output_path = os.path.join(save_dir, output_file)
    with open(output_path, "w") as f:
        results = (
            f"Model: {model_name}\n"
            f"Test Loss: {test_loss:.4f}\n"
            f"Test Accuracy: {test_accuracy:.4f}\n"
            f"Classification Report:\n{report}\n"
            f"Confusion Matrix:\n{cm}\n"
            "----------------------------------------\n"
        )
        print(results)
        f.write(results)

    # -------- Confusion Matrix Plot --------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_save_path)
    plt.show()
    plt.close()

    # -------- ROC Curve --------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    roc_save_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_save_path)
    plt.show()
    plt.close()
    return test_loss, test_accuracy


# ======================= Swin Transformer Model =======================

class SwinTransformerBinaryClassifier(nn.Module):
    def __init__(self, swin_variant='swin_tiny_patch4_window7_224'):
        super(SwinTransformerBinaryClassifier, self).__init__()

        # Load Swin Transformer pretrained
        self.backbone = timm.create_model(swin_variant, pretrained=True)

        # Modify head for binary classification
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Binary output
        )

    def forward(self, x):
        return self.backbone(x)

# ======================= EfficientNet + Transformer Patch =======================

class EfficientNet_TransformerPatch(nn.Module):
    def __init__(self, effnet_variant='efficientnet_b0', num_transformer_layers=3, num_heads=8):
        super(EfficientNet_TransformerPatch, self).__init__()

        # Load EfficientNet backbone
        self.backbone = timm.create_model(effnet_variant, pretrained=True, features_only=True)

        self.hidden_dim = self.backbone.feature_info.channels()[-1]  # 1280 for b0
        self.patch_H = 7  # EfficientNetB0 final feature map 7x7
        self.patch_W = 7

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=1024, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Binary output
        )

    def forward(self, x):
        # EfficientNet feature map: (B, C, 7, 7)
        features = self.backbone(x)[-1]
        B, C, H, W = features.shape

        # Reshape to (H*W, B, C) → sequence
        features = features.flatten(2).permute(2, 0, 1)

        # Transformer
        features = self.transformer_encoder(features)

        # Global average pooling over tokens
        features = features.mean(dim=0)

        # Classifier
        out = self.fc(features)
        return out

# ======================= MAIN SCRIPT =======================

if __name__ == "__main__":
    # 1 Swin Transformer
    swin_model = SwinTransformerBinaryClassifier()

    save_path = "swin_transformer_best.pth"
    log_dir = "./logs/swin_transformer"

    train_model(
        model=swin_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=GLOBAL_EPOCH,
        lr=GLOBAL_Lr,
        wd=1e-4,
        log_dir=log_dir,
        save_path=save_path,
        patience=5,
        use_scheduler=True,
        use_mixed_precision=True
    )

    swin_model.load_state_dict(torch.load(save_path))
    evaluate_model(
        model=swin_model,
        test_loader=test_loader,
        criterion=nn.BCEWithLogitsLoss(),
        model_name="swin_transformer",
        output_file="swin_transformer_eval.txt",
        save_dir="./results/swin_transformer"
    )
    del swin_model
    torch.cuda.empty_cache()
    gc.collect()

    # 2 EfficientNet + TransformerPatch
    effnet_transformer_patch_model = EfficientNet_TransformerPatch()

    save_path = "effnet_transformer_patch_best.pth"
    log_dir = "./logs/effnet_transformer_patch"

    train_model(
        model=effnet_transformer_patch_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=GLOBAL_EPOCH,
        lr=GLOBAL_Lr,
        wd=1e-4,
        log_dir=log_dir,
        save_path=save_path,
        patience=5,
        use_scheduler=True,
        use_mixed_precision=True
    )

    effnet_transformer_patch_model.load_state_dict(torch.load(save_path))
    evaluate_model(
        model=effnet_transformer_patch_model,
        test_loader=test_loader,
        criterion=nn.BCEWithLogitsLoss(),
        model_name="effnet_transformer_patch",
        output_file="effnet_transformer_patch_eval.txt",
        save_dir="./results/effnet_transformer_patch"
    )
    del effnet_transformer_patch_model
    torch.cuda.empty_cache()
    gc.collect()

    # 3️⃣ ConvNeXt + Transformer (your original)
    convnext_transformer = ConvNeXt_Transformer()

    save_path = "convnext_transformer.pth"
    log_dir = "./logs/convnext_transformer"

    train_model(
        model=convnext_transformer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=GLOBAL_EPOCH,
        lr=GLOBAL_Lr,
        wd=1e-4,
        log_dir=log_dir,
        save_path=save_path,
        patience=5,
        use_scheduler=True,
        use_mixed_precision=True
    )

    convnext_transformer.load_state_dict(torch.load(save_path))
    evaluate_model(
        model=convnext_transformer,
        test_loader=test_loader,
        criterion=nn.BCEWithLogitsLoss(),
        model_name="convnext_transformer",
        output_file="convnext_transformer_eval.txt",
        save_dir="./results/convnext_transformer"
    )
    del convnext_transformer
    torch.cuda.empty_cache()
    gc.collect()
