import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import os
import itertools

# ========== Step 1: Load and Clean Data ==========
print("# ========== Step 1: Load and Clean Data ==========")
raw_csv_path = r"D:\Data\NYC\KINZ\KINECT_ACC_dataset_with_qor15_2025-05-27_14-29PM.csv"
df = pd.read_csv(raw_csv_path)
joint_columns = [col for col in df.columns if any(j in col for j in ['_X', '_Y', '_Z'])]
df_clean = df.dropna(subset=joint_columns + ['QoR_class'])

# ========== Step 2: Patient-wise Split ==========
patient_class_map = df_clean.groupby('patientID')['QoR_class'].first()
class_0_patients = patient_class_map[patient_class_map == 0.0].index.tolist()
class_1_patients = patient_class_map[patient_class_map == 1.0].index.tolist()

def split_patients(patient_ids):
    train, temp = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test

train_0, val_0, test_0 = split_patients(class_0_patients)
train_1, val_1, test_1 = split_patients(class_1_patients)
train_patients = train_0 + train_1
val_patients = val_0 + val_1
test_patients = test_0 + test_1

train_df = df_clean[df_clean['patientID'].isin(train_patients)]
val_df = df_clean[df_clean['patientID'].isin(val_patients)]
test_df = df_clean[df_clean['patientID'].isin(test_patients)]

# ========== Step 3: Temporal Dataset Class ==========
class KinectTemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_len=16, stride=1):
        self.graph_sequences = []
        self.labels = []

        joints = [
            'FOOT_RIGHT', 'FOOT_LEFT', 'ANKLE_RIGHT', 'ANKLE_LEFT', 'KNEE_RIGHT', 'KNEE_LEFT',
            'HIP_RIGHT', 'HIP_LEFT', 'PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST',
            'CLAVICLE_RIGHT', 'CLAVICLE_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT',
            'ELBOW_RIGHT', 'ELBOW_LEFT', 'WRIST_RIGHT', 'WRIST_LEFT', 'HAND_RIGHT',
            'HAND_LEFT', 'HANDTIP_RIGHT', 'HANDTIP_LEFT', 'THUMB_RIGHT', 'THUMB_LEFT',
            'NECK', 'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT']

        edges = torch.tensor([
            [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 8], [8, 9],
            [9, 10], [10, 11], [10, 12], [11, 13], [12, 14], [13, 15], [14, 16],
            [15, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [17, 23],
            [18, 24], [10, 25], [25, 26], [26, 27], [26, 28], [26, 29], [26, 30], [26, 31]
        ]).t().contiguous()

        grouped = df.groupby('patientID')

        for _, patient_df in tqdm(grouped, desc="Creating temporal graph dataset"):
            patient_df = patient_df.sort_values('t_uniform').reset_index(drop=True)
            num_frames = len(patient_df)

            for start in range(0, num_frames - seq_len + 1, stride):
                sequence = []
                for i in range(seq_len):
                    row = patient_df.iloc[start + i]
                    node_features = [[row[f'{joint}_X'], row[f'{joint}_Y'], row[f'{joint}_Z']] for joint in joints]
                    x = torch.tensor(node_features, dtype=torch.float)
                    data = Data(x=x, edge_index=edges.clone())
                    sequence.append(data)

                label_row = patient_df.iloc[start + seq_len // 2]
                label = torch.tensor([label_row['QoR_class']], dtype=torch.float)
                self.graph_sequences.append(sequence)
                self.labels.append(label)

    def __len__(self):
        return len(self.graph_sequences)

    def __getitem__(self, idx):
        return self.graph_sequences[idx], self.labels[idx]

# ========== Step 4: Collate Function ==========

def temporal_collate(batch):
    sequences, labels = zip(*batch)
    return list(sequences), torch.tensor(labels, dtype=torch.float)

# ========== Step 5: Model ==========

class GCN_GRU_QoR(nn.Module):
    def __init__(self, gcn_hidden=64, gru_hidden=128, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(3, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.fc = nn.Linear(gru_hidden, 1)
        self.dropout = dropout

    def forward(self, sequence):
        embedded = []
        device = next(self.parameters()).device
        for data in sequence:
            data = data.to(device)
            x = F.relu(self.gcn1(data.x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.gcn2(x, data.edge_index))
            pooled = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(device))
            embedded.append(pooled)
        sequence_tensor = torch.stack(embedded, dim=1)
        _, h_n = self.gru(sequence_tensor)
        out = self.fc(h_n.squeeze(0))
        return out.squeeze()

# ========== Step 6: Training & Evaluation Functions ==========
def train_temporal_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    device = next(model.parameters()).device
    for sequences, labels in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        preds = [model(sequence) for sequence in sequences]
        logits = torch.stack(preds).to(device)
        labels = labels.to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

def eval_temporal_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for sequences, labels in loader:
            preds = [model(sequence) for sequence in sequences]
            logits = torch.stack(preds).to(device)
            labels = labels.to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_bin)
    prec = precision_score(all_labels, preds_bin, zero_division=0)
    rec = recall_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)
    return total_loss / len(loader.dataset), acc, prec, rec, f1

# ========== Step 7: Tuning Loop ==========

def run_tuning(train_df, val_df, test_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_path = "tuning_results.txt"
    with open(results_path, 'w') as f:
        f.write("Hyperparameter Tuning Results\n")

    # Grid Search Hyperparameters
    param_grid = {
        'seq_len': [32, 64, 128],
        'batch_size': [8, 16, 64],
        'dropout': [0.3, 0.5],
        'gcn_hidden': [64, 128],
        'gru_hidden': [128, 256]
    }

    for params in tqdm(itertools.product(*param_grid.values())):
        param_dict = dict(zip(param_grid.keys(), params))
        seq_len = param_dict['seq_len']
        batch_size = param_dict['batch_size']
        dropout = param_dict['dropout']
        gcn_hidden = param_dict['gcn_hidden']
        gru_hidden = param_dict['gru_hidden']

        train_dataset = KinectTemporalGraphDataset(train_df, seq_len=seq_len)
        val_dataset = KinectTemporalGraphDataset(val_df, seq_len=seq_len)
        test_dataset = KinectTemporalGraphDataset(test_df, seq_len=seq_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate)

        model = GCN_GRU_QoR(gcn_hidden=gcn_hidden, gru_hidden=gru_hidden, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        labels = [label.item() for _, label in train_dataset]
        class_weights = compute_class_weight('balanced', classes=np.array([0., 1.]), y=labels)
        pos_weight_tensor = torch.tensor([class_weights[1]], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(1, 51):
            train_loss = train_temporal_epoch(model, train_loader, optimizer, criterion)
            val_loss, acc, prec, rec, f1 = eval_temporal_epoch(model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        model.load_state_dict(torch.load("best_model.pt"))
        model.eval()
        test_loss, acc, prec, rec, f1 = eval_temporal_epoch(model, test_loader, criterion)

        result_str = f"Params: {param_dict}\nTest Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n{'='*50}\n"
        with open(results_path, 'a') as f:
            f.write(result_str)
        print(result_str)

if __name__ == '__main__':
    print("# ========== Step 7: Tuning Loop ==========")
    run_tuning(train_df, val_df, test_df)
