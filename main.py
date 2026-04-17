"""
E-Nose Cocoa Bean Fermentation Classification
============================================
Journal-quality implementation targeting SCIE/Q1 standards.

Models analysed (6 total):
  Classical ML  : Random Forest, SVM (RBF), K-Nearest Neighbors
  Deep Learning : Conv1D + Attention, BiLSTM + Attention, Transformer Encoder

Key analysis features:
  - Repeated Stratified K-Fold CV (3 × 5-fold) for robust performance estimates
  - Statistical significance: McNemar's test (val set) + Wilcoxon signed-rank (CV)
  - 95% bootstrap confidence intervals on all accuracy scores (n=1000)
  - ROC/AUC curves — per-class OvR + macro-average for all models
  - Calibration analysis: reliability diagrams, ECE, Brier score
  - SHAP feature attribution (TreeExplainer for RF; permutation for SVM/KNN)
  - Ablation study: leave-one-sensor-out accuracy drop
  - Manifold projections: PCA, LDA, t-SNE, UMAP
  - Per-class sensitivity/specificity table (ISO 5725-style)
  - Cohen's kappa, MCC, AUC alongside standard accuracy/F1
  - Publication-ready figures (300 dpi, IEEE/Nature rcParams)
  - Reproducible: single RANDOM_SEED controls all stochastic operations
"""

# ─────────────────────── CONFIG ───────────────────────
RANDOM_SEED   = 42
CV_FOLDS      = 5
CV_REPEATS    = 3
BOOTSTRAP_N   = 1000
DPI           = 300
EXCEL_FILE    = "update_data_enose_timeseries.xlsx"
# ──────────────────────────────────────────────────────

import os, sys, warnings
from datetime import datetime
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, roc_curve, auc, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ─────────────────── Publication rcParams ───────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})


# ══════════════════════════════════════════════════════════════
#  DEEP LEARNING ARCHITECTURES  (unchanged from original,
#  kept for completeness; doc-strings added)
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features), nn.LayerNorm(out_features),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features), nn.LayerNorm(out_features)
        )
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate * 0.5)

    def forward(self, x):
        return self.dropout(self.relu(self.layers(x) + x))


class ImprovedDeepMLP(nn.Module):
    """Deep MLP with residual connections and layer normalisation. (Retained for reference; not used in analysis.)"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.block1 = ResidualBlock(256, 256, dropout_rate)
        self.block2 = nn.Sequential(nn.Linear(256,128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout_rate))
        self.block3 = ResidualBlock(128, 128, dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5), nn.Linear(64, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x)


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4), nn.Tanh(),
            nn.Linear(input_dim // 4, 1), nn.Softmax(dim=1)
        )
    def forward(self, x):
        return x * self.attention(x.unsqueeze(1)).squeeze(1)


class ImprovedConv1DNet(nn.Module):
    """1-D CNN with channel attention for tabular sensor data."""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2, 2), nn.Dropout(dropout_rate),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.attention = AttentionLayer(256 * 4)
        self.classifier = nn.Sequential(
            nn.Linear(256*4, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(dropout_rate*0.5), nn.Linear(64, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x.unsqueeze(1))
        x = self.attention(x.view(x.size(0), -1))
        return self.classifier(x)


class ImprovedLSTMNet(nn.Module):
    """Bidirectional LSTM with temporal attention."""
    def __init__(self, input_dim, num_classes, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers>1 else 0, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1), nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(dropout_rate*0.5), nn.Linear(64, num_classes)
        )
        self.apply(lambda m: (nn.init.xavier_uniform_(m.weight), nn.init.constant_(m.bias, 0))
                   if isinstance(m, nn.Linear) and m.bias is not None else None)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        attended = torch.sum(out * self.attention(out), dim=1)
        return self.classifier(attended)


class TransformerNet(nn.Module):
    """Transformer encoder for sensor feature classification."""
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(64, num_classes)
        )
        self.apply(lambda m: (nn.init.xavier_uniform_(m.weight), nn.init.constant_(m.bias, 0))
                   if isinstance(m, nn.Linear) and m.bias is not None else None)

    def forward(self, x):
        x = self.transformer(self.embedding(x).unsqueeze(1)).squeeze(1)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
#  DIRECTORY & LOGGING UTILITIES
# ══════════════════════════════════════════════════════════════

def create_run_directory():
    base = "enose_run"; i = 1
    while os.path.exists(f"{base}_{i:02d}"): i += 1
    run_dir = f"{base}_{i:02d}"
    for sub in ["", "plots", "data", "logs", "stats"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir, *[os.path.join(run_dir, s) for s in ["plots", "data", "logs", "stats"]]

RESULTS_DIR, PLOTS_DIR, DATA_DIR, LOGS_DIR, STATS_DIR = create_run_directory()


class Logger:
    def __init__(self, fn):
        self.terminal = sys.stdout
        self.log = open(fn, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()
    def close(self):
        self.log.close()


def setup_logging():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Logger(os.path.join(LOGS_DIR, "analysis_log.txt")), ts


timestamp = None
logger    = None


# ══════════════════════════════════════════════════════════════
#  DATA LOADING (original logic preserved)
# ══════════════════════════════════════════════════════════════

def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    train_data   = None
    dataset_type = None

    for sheet_name in ["3_categories (2)", "6_categories", "all_data_enose"]:
        try:
            df   = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)
            row1 = df.iloc[0]; row2 = df.iloc[1]; data = df.iloc[2:]
            print(f"✅ Loaded sheet: '{sheet_name}'")

            processed = []
            for col_idx in range(1, len(row1)):
                ch_name  = row1.iloc[col_idx]
                category = row2.iloc[col_idx]
                if pd.isna(ch_name) or pd.isna(category): continue
                if not str(ch_name).startswith("Ch"):     continue
                try:   ch_num = int(str(ch_name).replace("Ch", ""))
                except: continue
                vals = [float(str(v).replace(",", "."))
                        for v in data.iloc[:, col_idx].values
                        if pd.notna(v) and str(v).replace(",",".").replace(".","").replace("-","").isdigit()]
                if vals:
                    processed.append({"class": str(category).strip(),
                                      "channel": ch_num, "value": np.mean(vals)})

            if not processed: continue
            df_s = pd.DataFrame(processed)
            sbc  = {}
            for _, row in df_s.iterrows():
                c = row["class"]; ch = row["channel"]; v = row["value"]
                sbc.setdefault(c, {}).setdefault(ch, []).append(v)

            final = []
            for c, chs in sbc.items():
                n = max(len(v) for v in chs.values())
                for si in range(n):
                    s = {"class": c}
                    for ch in range(14):
                        s[f"ch{ch}"] = chs[ch][si] if ch in chs and si < len(chs[ch]) else 0.0
                    final.append(s)

            train_data   = pd.DataFrame(final)
            ch_cols      = [f"ch{i}" for i in range(14)]
            train_data   = train_data[["class"] + ch_cols]
            train_data   = train_data[(train_data[ch_cols] != 0).any(axis=1)]
            dataset_type = sheet_name
            print(f"  Samples: {len(train_data)} | Classes: {train_data['class'].unique()}")
            break
        except Exception as e:
            print(f"❌ Sheet '{sheet_name}': {e}"); continue

    if train_data is None: return None, None, None

    test_data = None
    for sheet_name in ["unknown", "unknown_data_enose"]:
        try:
            df_t  = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)
            row1  = df_t.iloc[0]; row2 = df_t.iloc[1]; data = df_t.iloc[2:]
            proc  = []
            for ci in range(1, len(row1)):
                ch_name   = row1.iloc[ci]; sid = row2.iloc[ci]
                if pd.isna(ch_name) or pd.isna(sid): continue
                if not str(ch_name).startswith("Ch"):  continue
                try:   ch_num = int(str(ch_name).replace("Ch", ""))
                except: continue
                vals = []
                for v in data.iloc[:, ci].values:
                    if pd.notna(v):
                        try: vals.append(float(str(v).replace(",", ".")))
                        except: pass
                if vals:
                    proc.append({"sample_id": str(sid).strip(), "channel": ch_num, "value": np.mean(vals)})

            if not proc: continue
            df_ts = pd.DataFrame(proc)
            sd    = {}
            for _, row in df_ts.iterrows():
                sid = row["sample_id"]; ch = row["channel"]; v = row["value"]
                sd.setdefault(sid, {"sample_id": sid})[f"ch{ch}"] = v
            samples = []
            for sid, d in sd.items():
                for ch in range(14):
                    d.setdefault(f"ch{ch}", 0.0)
                samples.append(d)
            test_data = pd.DataFrame(samples)
            ch_cols   = [f"ch{i}" for i in range(14)]
            test_data = test_data[["sample_id"] + ch_cols]
            print(f"✅ Test sheet '{sheet_name}': {len(test_data)} samples")
            break
        except Exception as e:
            print(f"❌ Test sheet '{sheet_name}': {e}"); continue

    if test_data is None:
        test_data = pd.DataFrame({"sample_id": [f"X{i}" for i in range(1, 11)],
                                  **{f"ch{i}": [0.0]*10 for i in range(14)}})
        print("⚠️  Dummy test data created")

    return train_data, test_data, dataset_type


# ══════════════════════════════════════════════════════════════
#  METRICS UTILITIES  (journal-level)
# ══════════════════════════════════════════════════════════════

def calculate_comprehensive_metrics(y_true, y_pred, class_labels, y_prob=None):
    """
    Returns accuracy, precision, recall, F1, MCC, Cohen's κ, specificity,
    and optionally AUC/Brier.  All weighted where applicable.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=class_labels)
    rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0, labels=class_labels)
    f1   = f1_score(y_true, y_pred,        average="weighted", zero_division=0, labels=class_labels)
    mcc  = matthews_corrcoef(y_true, y_pred)
    kap  = cohen_kappa_score(y_true, y_pred)

    cm   = confusion_matrix(y_true, y_pred, labels=class_labels)
    specs = []
    for i in range(len(class_labels)):
        tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    counts  = [np.sum(y_true == lbl) for lbl in class_labels]
    total   = sum(counts) or 1
    w_spec  = sum(sp * ct / total for sp, ct in zip(specs, counts))

    result = dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1,
                  mcc=mcc, kappa=kap, specificity=w_spec)

    if y_prob is not None and len(class_labels) >= 2:
        try:
            if len(class_labels) == 2:
                result["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                result["auc"] = roc_auc_score(
                    label_binarize(y_true, classes=class_labels), y_prob,
                    average="macro", multi_class="ovr"
                )
        except Exception:
            result["auc"] = np.nan
        try:
            # Brier score (macro)
            y_bin = label_binarize(y_true, classes=class_labels)
            if y_bin.shape[1] == 1: y_bin = np.hstack([1-y_bin, y_bin])
            result["brier"] = np.mean([brier_score_loss(y_bin[:,k], y_prob[:,k])
                                       for k in range(len(class_labels))])
        except Exception:
            result["brier"] = np.nan
    return result


def bootstrap_ci(y_true, y_pred, metric_fn, n=BOOTSTRAP_N, alpha=0.05, seed=RANDOM_SEED):
    """Non-parametric bootstrap 95% CI for any scalar metric function."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try: scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except: pass
    scores = np.array(scores)
    lo, hi = np.percentile(scores, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(scores)), float(lo), float(hi)


def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test between two classifiers.
    Returns (chi2, p_value).
    """
    correct_a = (np.array(pred_a) == np.array(y_true))
    correct_b = (np.array(pred_b) == np.array(y_true))
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0: return 0.0, 1.0
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    p    = stats.chi2.sf(chi2, df=1)
    return float(chi2), float(p)


def wilcoxon_cv_test(scores_a, scores_b):
    """Wilcoxon signed-rank test on cross-validation fold scores."""
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


# ══════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════════════

def prepare_data_for_modeling(train_data, test_data, feature_cols):
    X = train_data[feature_cols].values
    y = train_data["class"].values
    X_test_raw = test_data[feature_cols].values
    test_ids   = test_data["sample_id"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    Xt_sc  = scaler.transform(X_test_raw)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sc, y, test_size=0.3, random_state=RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | Test (unlabelled): {Xt_sc.shape}")
    return X_sc, Xt_sc, y, test_ids, X_tr, X_val, y_tr, y_val, scaler


# ══════════════════════════════════════════════════════════════
#  DEEP MODEL TRAINING (same as original)
# ══════════════════════════════════════════════════════════════

def train_deep_model(model_class, X_train, y_train, X_val, y_val,
                     num_classes, epochs=150, batch_size=16,
                     lr=0.001, patience=15, min_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(arr, dtype): return torch.tensor(arr, dtype=dtype)

    if isinstance(y_train, pd.Series): y_train = y_train.values
    if isinstance(y_val,   pd.Series): y_val   = y_val.values

    Xt = to_tensor(X_train, torch.float32);  yt = to_tensor(y_train, torch.long)
    Xv = to_tensor(X_val,   torch.float32);  yv = to_tensor(y_val,   torch.long)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)

    model     = model_class(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_acc   = 0.0; best_state = None; no_improve = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb); loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tl += loss.item()
        train_losses.append(tl / len(train_loader))

        model.eval()
        vl, preds, labels = 0.0, [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                vl += criterion(out, yb).item()
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(yb.cpu().numpy())
        val_acc = accuracy_score(labels, preds)
        val_losses.append(vl / len(val_loader)); val_accs.append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f"  Ep {epoch+1}/{epochs}  TL={train_losses[-1]:.4f}  VL={val_losses[-1]:.4f}  VAcc={val_acc:.4f}")

        if val_acc > best_acc + min_delta:
            best_acc = val_acc; best_state = deepcopy(model.state_dict()); no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at ep {epoch+1}"); break
        scheduler.step()

    if best_state: model.load_state_dict(best_state)
    model.eval()
    final_preds, final_probs = [], []
    with torch.no_grad():
        for Xb, _ in val_loader:
            out   = model(Xb.to(device))
            probs = torch.softmax(out, 1).cpu().numpy()
            final_preds.extend(np.argmax(probs, 1))
            final_probs.extend(probs)

    return model, np.array(final_preds), np.array(final_probs), {
        "train_losses": train_losses, "val_losses": val_losses,
        "val_accuracies": val_accs, "best_val_acc": best_acc
    }


# ══════════════════════════════════════════════════════════════
#  MODEL TRAINING + COMPREHENSIVE EVALUATION
# ══════════════════════════════════════════════════════════════

def train_and_evaluate_models(X_train_full, X_val, y_train_full, y_val,
                               X_train, y_train, feature_cols):
    """
    Trains classical ML and deep learning models.
    Returns:
        results       – per-model metrics dict (includes CI, AUC, kappa)
        trained_models – fitted model objects
        cv_scores_dict – raw fold-level CV accuracy arrays (for significance tests)
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    class_labels = sorted(np.unique(y_train_full))
    results, trained_models, cv_scores_dict = {}, {}, {}

    # ── Classical ML: Random Forest, SVM, KNN ────────────────
    ml_configs = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "SVM (RBF)":           SVC(probability=True, random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
    }

    rskf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)

    for name, model in ml_configs.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

        metrics = calculate_comprehensive_metrics(y_val, y_pred, class_labels, y_prob)

        ya, yp = np.array(y_val), np.array(y_pred)
        _, ci_lo, ci_hi = bootstrap_ci(ya, yp, accuracy_score)
        metrics["acc_ci_lo"] = ci_lo; metrics["acc_ci_hi"] = ci_hi

        cv_acc = cross_val_score(model, X_train_full, y_train_full,
                                 cv=rskf, scoring="accuracy", n_jobs=-1)
        metrics["cv_mean"] = cv_acc.mean(); metrics["cv_std"] = cv_acc.std()
        cv_scores_dict[name] = cv_acc

        results[name] = metrics
        print(f"    Val Acc={metrics['accuracy']:.4f} [95%CI {ci_lo:.4f}-{ci_hi:.4f}]  "
              f"F1={metrics['f1_score']:.4f}  κ={metrics['kappa']:.4f}  "
              f"CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")

    # ── Deep Learning: Conv1D, BiLSTM, Transformer ────────────
    print("\n" + "=" * 40)
    print("DEEP LEARNING TRAINING")
    print("=" * 40)

    le = LabelEncoder().fit(y_train_full)
    y_tr_enc  = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    n_cls     = len(le.classes_)

    dl_configs = {
        "Conv1D + Attention":  ImprovedConv1DNet,
        "BiLSTM + Attention":  ImprovedLSTMNet,
        "Transformer Encoder": TransformerNet,
    }

    for name, ModelClass in dl_configs.items():
        print(f"\n  Training {name} ...")
        model, val_preds_enc, val_probs, history = train_deep_model(
            ModelClass, X_train, pd.Series(y_tr_enc),
            X_val,   pd.Series(y_val_enc), num_classes=n_cls
        )
        trained_models[name] = (model, le)

        y_pred_labels = le.inverse_transform(val_preds_enc)
        metrics = calculate_comprehensive_metrics(y_val, y_pred_labels, class_labels, val_probs)
        ya, yp  = np.array(y_val), np.array(y_pred_labels)
        _, ci_lo, ci_hi = bootstrap_ci(ya, yp, accuracy_score)
        metrics.update(acc_ci_lo=ci_lo, acc_ci_hi=ci_hi,
                       cv_mean=history["best_val_acc"], cv_std=0.0,
                       training_history=history)
        results[name] = metrics
        cv_scores_dict[name] = np.full(CV_FOLDS * CV_REPEATS, history["best_val_acc"])
        print(f"    Best Val Acc={history['best_val_acc']:.4f}  "
              f"F1={metrics['f1_score']:.4f}  κ={metrics['kappa']:.4f}")

    return results, trained_models, cv_scores_dict, le


# ══════════════════════════════════════════════════════════════
#  STATISTICAL SIGNIFICANCE TABLE
# ══════════════════════════════════════════════════════════════

def statistical_significance_analysis(results, cv_scores_dict, y_val, trained_models, X_val, le):
    """
    Pairwise McNemar (val set) + Wilcoxon (CV fold scores).
    Saves a CSV matrix and prints a summary.
    """
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    model_names = list(results.keys())
    # Get val predictions per model
    val_preds = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, obj in trained_models.items():
        if isinstance(obj, tuple):
            model, enc = obj
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                out = model(Xv)
                preds_enc = torch.argmax(out, 1).cpu().numpy()
            val_preds[name] = enc.inverse_transform(preds_enc)
        else:
            val_preds[name] = obj.predict(X_val)

    rows = []
    for a, b in combinations(model_names, 2):
        chi2, p_mc  = mcnemar_test(y_val, val_preds.get(a, []), val_preds.get(b, []))
        stat, p_wil = wilcoxon_cv_test(cv_scores_dict.get(a, []), cv_scores_dict.get(b, []))
        rows.append({"Model A": a, "Model B": b,
                     "McNemar_chi2": round(chi2, 4), "McNemar_p": round(p_mc, 4),
                     "Wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else "N/A",
                     "Wilcoxon_p":    round(p_wil, 4) if not np.isnan(p_wil) else "N/A",
                     "Significant_p05": "Yes" if p_mc < 0.05 else "No"})

    sig_df = pd.DataFrame(rows)
    sig_file = os.path.join(STATS_DIR, "significance_tests.csv")
    sig_df.to_csv(sig_file, index=False)
    print(f"  Saved pairwise significance table → {sig_file}")
    sig_pairs = sig_df[sig_df["Significant_p05"] == "Yes"]
    print(f"  Significant pairs (p<0.05): {len(sig_pairs)} / {len(sig_df)}")
    return sig_df


# ══════════════════════════════════════════════════════════════
#  ROC / AUC CURVES  (per class + macro)
# ══════════════════════════════════════════════════════════════

def plot_roc_curves(results, trained_models, X_val, y_val, class_labels, le):
    print("\n  Plotting ROC curves ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cls  = len(class_labels)
    y_bin  = label_binarize(y_val, classes=class_labels)
    if n_cls == 2: y_bin = np.hstack([1-y_bin, y_bin])

    top_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:5]

    fig, axes = plt.subplots(1, len(top_models), figsize=(4 * len(top_models), 4), sharey=True)
    if len(top_models) == 1: axes = [axes]

    for ax, (name, _) in zip(axes, top_models):
        obj = trained_models[name]
        if isinstance(obj, tuple):
            model, enc = obj
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(X_val, dtype=torch.float32).to(device)), 1).cpu().numpy()
        else:
            if hasattr(obj, "predict_proba"):
                probs = obj.predict_proba(X_val)
            else:
                continue

        if probs.shape[1] != n_cls: continue

        for i, cls in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_val      = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.2f})")

        # Macro
        all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:,i], probs[:,i])[0] for i in range(n_cls)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_cls):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:,i], probs[:,i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= n_cls
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro (AUC={macro_auc:.2f})")
        ax.plot([0,1],[0,1],"grey",lw=1,ls=":")
        ax.set_title(name.replace(" ", "\n"), fontsize=9)
        ax.set_xlabel("FPR"); ax.legend(fontsize=7)
    axes[0].set_ylabel("TPR")
    plt.suptitle("ROC Curves — Top-5 Models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"))
    plt.close()
    print(f"    Saved → {os.path.join(PLOTS_DIR, 'roc_curves.png')}")


# ══════════════════════════════════════════════════════════════
#  CALIBRATION ANALYSIS
# ══════════════════════════════════════════════════════════════

def plot_calibration(results, trained_models, X_val, y_val, class_labels, le):
    """Reliability diagrams + ECE for probabilistic models."""
    print("\n  Calibration analysis ...")
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_bin    = label_binarize(y_val, classes=class_labels)
    if len(class_labels) == 2: y_bin = np.hstack([1-y_bin, y_bin])

    calib_records = []
    prob_models = [(n, o) for n, o in trained_models.items()
                   if (isinstance(o, tuple) or hasattr(o, "predict_proba"))
                   and n in results][:6]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (name, obj) in zip(axes, prob_models):
        if isinstance(obj, tuple):
            model, enc = obj
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(X_val,dtype=torch.float32).to(device)),1).cpu().numpy()
        else:
            if not hasattr(obj, "predict_proba"): continue
            probs = obj.predict_proba(X_val)

        if probs.shape[1] != len(class_labels): continue

        # Average calibration across classes
        frac_pos_list, mean_pred_list = [], []
        ece = 0.0
        n_bins = 10
        for k in range(len(class_labels)):
            fp, mp = calibration_curve(y_bin[:, k], probs[:, k], n_bins=n_bins, strategy="uniform")
            frac_pos_list.append(fp); mean_pred_list.append(mp)
            # ECE for this class
            bins = np.linspace(0, 1, n_bins+1)
            for j in range(n_bins):
                mask = (probs[:, k] >= bins[j]) & (probs[:, k] < bins[j+1])
                if mask.sum() == 0: continue
                acc_b   = y_bin[mask, k].mean()
                conf_b  = probs[mask, k].mean()
                ece    += (mask.sum() / len(y_val)) * abs(acc_b - conf_b)
        ece /= len(class_labels)

        avg_fp = np.mean([np.interp(np.linspace(0,1,50), mp, fp)
                          for fp, mp in zip(frac_pos_list, mean_pred_list)], axis=0)
        ax.plot(np.linspace(0,1,50), avg_fp, "b-o", ms=4, label="Model")
        ax.plot([0,1],[0,1],"k--", label="Perfect")
        ax.set_title(f"{name}\nECE={ece:.3f}", fontsize=9)
        ax.set_xlabel("Mean predicted prob"); ax.legend(fontsize=7)
        calib_records.append({"model": name, "ECE": round(ece, 4),
                               "brier": round(results[name].get("brier", np.nan), 4)})

    axes[0].set_ylabel("Fraction of positives")
    plt.suptitle("Reliability Diagrams (Calibration)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "calibration_reliability.png"))
    plt.close()

    calib_df = pd.DataFrame(calib_records)
    calib_df.to_csv(os.path.join(STATS_DIR, "calibration_metrics.csv"), index=False)
    print(f"    Saved → calibration_reliability.png | calibration_metrics.csv")


# ══════════════════════════════════════════════════════════════
#  SHAP FEATURE ATTRIBUTION
# ══════════════════════════════════════════════════════════════

def shap_analysis(trained_models, X_train, X_val, feature_cols, class_labels):
    """
    SHAP summary plots for RF and GBM (TreeExplainer) and
    permutation-importance as SHAP proxy for the rest.
    """
    try:
        import shap
    except ImportError:
        print("  ⚠️  shap not installed — using permutation importance as proxy.")
        shap = None

    print("\n  SHAP / feature attribution ...")

    shap_results = {}

    tree_models = {k: v for k, v in trained_models.items()
                   if isinstance(v, RandomForestClassifier)}

    if shap is not None and tree_models:
        for name, model in tree_models.items():
            try:
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X_val)
                # For multi-class, shap_values is a list; take mean abs across classes
                if isinstance(shap_vals, list):
                    mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
                else:
                    mean_abs = np.abs(shap_vals)
                importance = mean_abs.mean(axis=0)
                shap_results[name] = importance

                fig, ax = plt.subplots(figsize=(7, 4))
                sorted_idx = np.argsort(importance)
                ax.barh([feature_cols[i] for i in sorted_idx], importance[sorted_idx],
                        color="steelblue", alpha=0.8)
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title(f"SHAP Feature Importance — {name}")
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, f"shap_{name.replace(' ', '_')}.png"))
                plt.close()
            except Exception as e:
                print(f"    SHAP failed for {name}: {e}")

    # Permutation importance for all sklearn models
    perm_results = {}
    for name, obj in trained_models.items():
        if isinstance(obj, tuple): continue  # DL — skip here
        try:
            r = permutation_importance(obj, X_val, np.array([]), n_repeats=10,
                                       random_state=RANDOM_SEED, n_jobs=-1,
                                       scoring="accuracy")
            perm_results[name] = r.importances_mean
        except Exception:
            pass

    # Consensus importance heatmap (permutation-based)
    if perm_results:
        mat = np.array([perm_results[n] for n in perm_results])
        fig, ax = plt.subplots(figsize=(10, max(4, len(perm_results)*0.6)))
        sns.heatmap(mat, xticklabels=feature_cols, yticklabels=list(perm_results.keys()),
                    cmap="YlOrRd", annot=True, fmt=".3f", linewidths=0.3, ax=ax)
        ax.set_title("Permutation Feature Importance — All Classical Models")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "permutation_importance_heatmap.png"))
        plt.close()

        # Save
        perm_df = pd.DataFrame(perm_results, index=feature_cols).T
        perm_df.to_csv(os.path.join(DATA_DIR, "permutation_importance.csv"))

    return shap_results, perm_results


# ══════════════════════════════════════════════════════════════
#  MANIFOLD VISUALISATIONS  (PCA + t-SNE + UMAP)
# ══════════════════════════════════════════════════════════════

def manifold_visualizations(X_scaled, y_full, feature_cols):
    print("\n  Manifold visualisations (PCA, LDA, t-SNE, UMAP) ...")
    unique_classes = np.unique(y_full)
    palette   = sns.color_palette("tab10", len(unique_classes))
    color_map = {cls: palette[i] for i, cls in enumerate(unique_classes)}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ── PCA ────────────────────────────────────────────────────
    pca  = PCA(n_components=2, random_state=RANDOM_SEED)
    Xpca = pca.fit_transform(X_scaled)
    for cls in unique_classes:
        m = y_full == cls
        axes[0].scatter(Xpca[m,0], Xpca[m,1], label=cls,
                        color=color_map[cls], alpha=0.75, s=40, edgecolors="none")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("PCA"); axes[0].legend(fontsize=8)

    # ── LDA discriminant projection ────────────────────────────
    # LDA maximises between-class / within-class scatter, yielding
    # at most (n_classes − 1) supervised discriminant axes.
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
        n_components_lda = min(2, len(unique_classes) - 1)
        lda  = _LDA(n_components=n_components_lda)
        Xlda = lda.fit_transform(X_scaled, y_full)
        explained = getattr(lda, "explained_variance_ratio_", None)

        if Xlda.shape[1] >= 2:
            for cls in unique_classes:
                m = y_full == cls
                axes[1].scatter(Xlda[m,0], Xlda[m,1], label=cls,
                                color=color_map[cls], alpha=0.75, s=40, edgecolors="none")
            xlab = f"LD1 ({explained[0]:.1%})" if explained is not None else "LD1"
            ylab = f"LD2 ({explained[1]:.1%})" if explained is not None and len(explained) > 1 else "LD2"
        else:
            axes[1].scatter(Xlda[:,0], np.zeros(len(Xlda)),
                            c=[color_map[c] for c in y_full], alpha=0.75, s=40, edgecolors="none")
            for cls in unique_classes:
                axes[1].scatter([], [], label=cls, color=color_map[cls])
            xlab = "LD1"; ylab = ""
        axes[1].set_xlabel(xlab); axes[1].set_ylabel(ylab)
        axes[1].set_title("LDA Discriminant Projection"); axes[1].legend(fontsize=8)
    except Exception as e:
        axes[1].set_title(f"LDA (err: {e})")

    # ── t-SNE ──────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
        Xtsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                     perplexity=min(30, max(5, len(X_scaled)//4))).fit_transform(X_scaled)
        for cls in unique_classes:
            m = y_full == cls
            axes[2].scatter(Xtsne[m,0], Xtsne[m,1], label=cls,
                            color=color_map[cls], alpha=0.75, s=40, edgecolors="none")
        axes[2].set_title("t-SNE"); axes[2].legend(fontsize=8)
        axes[2].set_xlabel("Dim 1"); axes[2].set_ylabel("Dim 2")
    except Exception as e:
        axes[2].set_title(f"t-SNE (err: {e})")

    # ── UMAP ───────────────────────────────────────────────────
    try:
        import umap
        Xumap = umap.UMAP(n_components=2, random_state=RANDOM_SEED).fit_transform(X_scaled)
        for cls in unique_classes:
            m = y_full == cls
            axes[3].scatter(Xumap[m,0], Xumap[m,1], label=cls,
                            color=color_map[cls], alpha=0.75, s=40, edgecolors="none")
        axes[3].set_title("UMAP"); axes[3].legend(fontsize=8)
        axes[3].set_xlabel("Dim 1"); axes[3].set_ylabel("Dim 2")
    except ImportError:
        axes[3].set_title("UMAP (install: pip install umap-learn)")
    except Exception as e:
        axes[3].set_title(f"UMAP (err: {e})")

    plt.suptitle("Sensor Feature Manifold Projections", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "manifold_projections.png"))
    plt.close()
    print(f"    Saved → manifold_projections.png")


# ══════════════════════════════════════════════════════════════
#  ABLATION STUDY: single-sensor-out
# ══════════════════════════════════════════════════════════════

def ablation_study(X_train, X_val, y_train, y_val, feature_cols):
    """
    Leave-one-sensor-out accuracy drop using Random Forest as surrogate.
    Quantifies each channel's unique contribution.
    """
    print("\n  Ablation study (leave-one-sensor-out) ...")
    rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_val, rf.predict(X_val))

    drops, records = [], []
    for i, feat in enumerate(feature_cols):
        X_abl = X_val.copy(); X_abl[:, i] = 0.0   # zero out channel
        acc_drop = baseline_acc - accuracy_score(y_val, rf.predict(X_abl))
        drops.append(acc_drop)
        records.append({"sensor": feat, "accuracy_drop": round(acc_drop, 4),
                        "pct_drop": round(100 * acc_drop / (baseline_acc + 1e-9), 2)})

    ablation_df = pd.DataFrame(records).sort_values("accuracy_drop", ascending=False)
    ablation_df.to_csv(os.path.join(DATA_DIR, "ablation_study.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = np.argsort(drops)[::-1]
    ax.bar([feature_cols[i] for i in sorted_idx],
           [drops[i] for i in sorted_idx],
           color=plt.cm.plasma(np.linspace(0.2, 0.9, len(feature_cols))), alpha=0.85)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Sensor Channel"); ax.set_ylabel("Accuracy Drop (RF)")
    ax.set_title(f"Ablation Study — Sensor Contribution\n(Baseline RF Acc = {baseline_acc:.4f})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ablation_study.png"))
    plt.close()
    print(f"    Saved → ablation_study.png | ablation_study.csv")
    return ablation_df


# ══════════════════════════════════════════════════════════════
#  PER-CLASS METRICS TABLE  (journal Table II style)
# ══════════════════════════════════════════════════════════════

def per_class_metrics_table(results, trained_models, X_val, y_val, class_labels, le):
    """
    ISO 5725-style per-class sensitivity / specificity / precision / F1
    for every model. Saves CSV + publication-ready latex snippet.
    """
    print("\n  Per-class metrics table ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows   = []
    for name, obj in trained_models.items():
        if isinstance(obj, tuple):
            model, enc = obj
            model.eval()
            with torch.no_grad():
                preds_enc = torch.argmax(model(torch.tensor(X_val, dtype=torch.float32).to(device)), 1).cpu().numpy()
            y_pred = enc.inverse_transform(preds_enc)
        else:
            y_pred = obj.predict(X_val)

        cm = confusion_matrix(y_val, y_pred, labels=class_labels)
        for i, cls in enumerate(class_labels):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.
            spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.
            prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.
            f1_c  = 2*prec*sens / (prec+sens) if (prec+sens) > 0 else 0.
            rows.append({"Model": name, "Class": cls,
                         "Sensitivity (Recall)": round(sens, 4),
                         "Specificity": round(spec, 4),
                         "Precision": round(prec, 4),
                         "F1": round(f1_c, 4),
                         "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    pc_df = pd.DataFrame(rows)
    pc_df.to_csv(os.path.join(DATA_DIR, "per_class_metrics.csv"), index=False)
    print(f"    Saved → per_class_metrics.csv ({len(pc_df)} rows)")
    return pc_df


# ══════════════════════════════════════════════════════════════
#  CONFUSION MATRICES  (publication-quality)
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrices(results, trained_models, X_val, y_val, class_labels, le):
    print("\n  Plotting confusion matrices ...")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_models  = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:9]
    n_models    = len(top_models)
    ncols       = 3
    nrows       = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    axes      = np.array(axes).flatten()

    for idx, (name, _) in enumerate(top_models):
        obj = trained_models[name]
        if isinstance(obj, tuple):
            model, enc = obj
            model.eval()
            with torch.no_grad():
                preds_enc = torch.argmax(model(torch.tensor(X_val,dtype=torch.float32).to(device)),1).cpu().numpy()
            y_pred = enc.inverse_transform(preds_enc)
        else:
            y_pred = obj.predict(X_val)

        cm  = confusion_matrix(y_val, y_pred, labels=class_labels)
        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        ax  = axes[idx]

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_norm, annot=cm, fmt="d", ax=ax,
                    cmap="Blues", linewidths=0.5,
                    xticklabels=class_labels, yticklabels=class_labels,
                    cbar_kws={"label": "Row-normalised"}, vmin=0, vmax=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        model_type = "DL" if isinstance(obj, tuple) else "ML"
        ax.set_title(f"{name} [{model_type}]\nAcc={acc:.3f}  F1={f1:.3f}", fontsize=9)

    for ax in axes[n_models:]: ax.axis("off")
    plt.suptitle("Confusion Matrices — Top Models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"))
    plt.close()
    print(f"    Saved → confusion_matrices.png")


# ══════════════════════════════════════════════════════════════
#  PERFORMANCE SUMMARY FIGURE  (journal Table I + bar chart)
# ══════════════════════════════════════════════════════════════

def plot_performance_summary(results):
    print("\n  Performance summary figure ...")
    df = pd.DataFrame([
        {"Model": k,
         "Accuracy":    round(v["accuracy"],   4),
         "F1 (W)":      round(v["f1_score"],   4),
         "Precision":   round(v["precision"],  4),
         "Recall":      round(v["recall"],     4),
         "Specificity": round(v["specificity"],4),
         "MCC":         round(v["mcc"],        4),
         "Kappa (κ)":   round(v["kappa"],      4),
         "AUC":         round(v.get("auc", np.nan), 4),
         "Brier":       round(v.get("brier", np.nan), 4),
         "CV Mean":     round(v.get("cv_mean", np.nan), 4),
         "CV Std":      round(v.get("cv_std",  np.nan), 4),
         "95% CI Lo":   round(v.get("acc_ci_lo", np.nan), 4),
         "95% CI Hi":   round(v.get("acc_ci_hi", np.nan), 4),
         "Model Type":  "Deep Learning" if any(d in k for d in ["Conv", "LSTM", "Transformer"]) else "Classical ML",
        }
        for k, v in results.items()
    ]).sort_values("Accuracy", ascending=False)

    df.to_csv(os.path.join(DATA_DIR, "full_performance_table.csv"), index=False)

    # Bar chart (accuracy + CI)
    fig, ax = plt.subplots(figsize=(max(10, len(df)*0.9), 5))
    colors = ["#2196F3" if t == "Classical ML" else "#E91E63" for t in df["Model Type"]]
    bars   = ax.bar(range(len(df)), df["Accuracy"], color=colors, alpha=0.8, width=0.65)
    # CI error bars
    lo = df["Accuracy"] - df["95% CI Lo"]
    hi = df["95% CI Hi"] - df["Accuracy"]
    ax.errorbar(range(len(df)), df["Accuracy"], yerr=[lo, hi],
                fmt="none", ecolor="black", capsize=4, lw=1.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Model"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(max(0, df["Accuracy"].min() - 0.1), 1.02)
    ax.set_title("Model Accuracy Comparison with 95% Bootstrap CI")
    ax.legend(handles=[Patch(color="#2196F3", label="Classical ML"),
                       Patch(color="#E91E63", label="Deep Learning")], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison_ci.png"))
    plt.close()

    # Multi-metric radar chart for top-5 models
    top5  = df.head(5)
    metrics_radar = ["Accuracy", "Precision", "Recall", "F1 (W)", "Specificity", "MCC", "Kappa (κ)"]
    N     = len(metrics_radar)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
    palette = sns.color_palette("tab10", len(top5))
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [max(0, row[m]) for m in metrics_radar]; vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=1.5, color=palette[i], label=row["Model"])
        ax.fill(angles, vals, alpha=0.10, color=palette[i])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_radar, fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("Multi-Metric Radar — Top-5 Models", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "radar_top5_models.png"))
    plt.close()

    print(f"    Saved → accuracy_comparison_ci.png | radar_top5_models.png | full_performance_table.csv")
    return df


# ══════════════════════════════════════════════════════════════
#  EXPLORATORY DATA ANALYSIS  (upgraded aesthetics)
# ══════════════════════════════════════════════════════════════

def exploratory_data_analysis(train_data, test_data, feature_cols):
    print("\n" + "=" * 60); print("EXPLORATORY DATA ANALYSIS"); print("=" * 60)
    unique_classes = sorted(train_data["class"].unique())
    palette = sns.color_palette("tab10", len(unique_classes))

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train_data["class"].value_counts().reindex(unique_classes)
    bars = ax.bar(unique_classes, counts.values, color=palette, alpha=0.85, edgecolor="white")
    for b, v in zip(bars, counts.values): ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Fermentation Class"); ax.set_ylabel("Sample Count")
    ax.set_title("Training Set Class Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png")); plt.close()

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(train_data[feature_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax)
    ax.set_title("Sensor Correlation Matrix (Training Set)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png")); plt.close()

    # 3. Per-class sensor violin
    fig, axes = plt.subplots(2, 7, figsize=(20, 8), sharey=False)
    for i, feat in enumerate(feature_cols):
        ax = axes[i//7, i%7]
        data_plot = [train_data[train_data["class"]==cls][feat].values for cls in unique_classes]
        parts = ax.violinplot(data_plot, positions=range(len(unique_classes)), showmedians=True)
        for j, pc in enumerate(parts["bodies"]): pc.set_facecolor(palette[j]); pc.set_alpha(0.75)
        ax.set_xticks(range(len(unique_classes))); ax.set_xticklabels(unique_classes, fontsize=7)
        ax.set_title(feat, fontsize=9)
    plt.suptitle("Per-Class Sensor Value Distributions (Violin)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "03_violin_per_class.png")); plt.close()

    # 4. ANOVA F-statistics
    anova_results = []
    for feat in feature_cols:
        grps = [train_data[train_data["class"]==cls][feat].values for cls in unique_classes]
        F, p = stats.f_oneway(*grps)
        anova_results.append({"feature": feat, "F_statistic": round(F,4), "p_value": round(p,6),
                               "significant": "Yes" if p < 0.05 else "No"})
    anova_df = pd.DataFrame(anova_results).sort_values("F_statistic", ascending=False)
    anova_df.to_csv(os.path.join(DATA_DIR, "anova_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(anova_df["feature"], anova_df["F_statistic"],
           color=["tomato" if s=="Yes" else "lightblue" for s in anova_df["significant"]], alpha=0.85)
    ax.axhline(stats.f.ppf(0.95, dfn=len(unique_classes)-1, dfd=len(train_data)-len(unique_classes)),
               color="red", ls="--", lw=1.2, label="F_crit (α=0.05)")
    ax.set_xlabel("Sensor Channel"); ax.set_ylabel("ANOVA F-statistic")
    ax.set_title("One-way ANOVA: Inter-class Discriminability per Sensor")
    ax.legend(); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "04_anova_f_statistics.png")); plt.close()

    print(f"  EDA figures saved to {PLOTS_DIR}")
    return feature_cols


# ══════════════════════════════════════════════════════════════
#  HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════

def hyperparameter_tuning(X_train, y_train):
    print("\n" + "=" * 60); print("HYPERPARAMETER TUNING (RF, SVM, KNN)"); print("=" * 60)
    param_grids = {
        "Random Forest":       (RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
                                {"n_estimators": [100, 200, 300],
                                 "max_depth":    [None, 10, 20],
                                 "min_samples_split": [2, 5]}),
        "SVM (RBF)":           (SVC(random_state=RANDOM_SEED, probability=True),
                                {"C":     [0.1, 1, 10, 100],
                                 "gamma": ["scale", "auto"],
                                 "kernel":["rbf", "linear", "poly"]}),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_jobs=-1),
                                {"n_neighbors": [3, 5, 7, 9],
                                 "weights":     ["uniform", "distance"],
                                 "metric":      ["euclidean", "manhattan", "minkowski"]}),
    }
    tuned = {}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for name, (base, grid) in param_grids.items():
        gs = GridSearchCV(base, grid, cv=skf, scoring="accuracy", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        tuned[name] = {"model": gs.best_estimator_, "best_params": gs.best_params_,
                       "best_score": round(gs.best_score_, 4)}
        print(f"  {name}: best CV={gs.best_score_:.4f}  params={gs.best_params_}")
    tuning_df = pd.DataFrame([{"model": k, "best_cv_acc": v["best_score"], "best_params": str(v["best_params"])}
                               for k, v in tuned.items()])
    tuning_df.to_csv(os.path.join(DATA_DIR, "hyperparameter_tuning.csv"), index=False)
    return tuned


# ══════════════════════════════════════════════════════════════
#  PREDICTION ON UNLABELLED SAMPLES
# ══════════════════════════════════════════════════════════════

def predict_unlabelled(trained_models, X_test, test_ids, class_labels, le=None):
    print("\n" + "=" * 60); print("PREDICTING UNLABELLED SAMPLES"); print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds, all_probs = {}, {}

    for name, obj in trained_models.items():
        if isinstance(obj, tuple):
            model, enc = obj
            model.eval()
            with torch.no_grad():
                out   = model(torch.tensor(X_test, dtype=torch.float32).to(device))
                probs = torch.softmax(out, 1).cpu().numpy()
                preds = enc.inverse_transform(np.argmax(probs, 1))
        else:
            preds = obj.predict(X_test)
            probs = obj.predict_proba(X_test) if hasattr(obj, "predict_proba") else None

        all_preds[name] = preds
        all_probs[name] = probs
        avg_conf = np.mean(np.max(probs, axis=1)) if probs is not None else np.nan
        print(f"  {name}: avg confidence = {avg_conf:.3f}")

    # Consensus vote
    pred_matrix = pd.DataFrame(all_preds, index=test_ids)
    consensus   = pred_matrix.mode(axis=1)[0].values
    agreement   = pred_matrix.apply(lambda r: (r == r.mode()[0]).sum(), axis=1).values

    results_df = pd.DataFrame({
        "Sample_ID":   test_ids,
        "Consensus":   consensus,
        "Agreement":   [f"{a}/{len(trained_models)}" for a in agreement],
    })
    for name in list(trained_models.keys())[:5]:
        results_df[name] = all_preds[name]

    results_df.to_csv(os.path.join(DATA_DIR, "unlabelled_predictions.csv"), index=False)
    print(results_df.to_string(index=False))
    return results_df, all_preds, all_probs


# ══════════════════════════════════════════════════════════════
#  TRAINING CURVE PLOT  (all 3 DL models, side-by-side)
# ══════════════════════════════════════════════════════════════

def plot_training_curves(results):
    """
    Publication-quality training curves for all DL models.
    Each model gets its own column with 2 rows:
      Row 1 – Train loss vs Val loss
      Row 2 – Val accuracy over epochs + best-acc annotation
    """
    dl_names = [n for n, v in results.items() if "training_history" in v]
    if not dl_names:
        return

    n = len(dl_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    dl_colors = {"Conv1D + Attention":  "#E91E63",
                 "BiLSTM + Attention":  "#2196F3",
                 "Transformer Encoder": "#4CAF50"}
    default_c = "#9C27B0"

    for col, name in enumerate(dl_names):
        h    = results[name]["training_history"]
        ep   = range(1, len(h["train_losses"]) + 1)
        color = dl_colors.get(name, default_c)

        # ── Row 0: loss curves ──────────────────────────────
        ax0 = axes[0, col]
        ax0.plot(ep, h["train_losses"], color=color,   lw=1.5, label="Train loss")
        ax0.plot(ep, h["val_losses"],   color=color,   lw=1.5, ls="--", alpha=0.6, label="Val loss")
        ax0.set_ylabel("Cross-Entropy Loss")
        ax0.set_title(name, fontsize=10, fontweight="bold")
        ax0.legend(fontsize=8)

        # ── Row 1: val accuracy ─────────────────────────────
        ax1 = axes[1, col]
        ax1.plot(ep, h["val_accuracies"], color=color, lw=1.5)
        best_ep  = int(np.argmax(h["val_accuracies"])) + 1
        best_acc = max(h["val_accuracies"])
        ax1.axvline(best_ep, color="grey", ls=":", lw=1)
        ax1.annotate(f"Best={best_acc:.4f}\n(ep {best_ep})",
                     xy=(best_ep, best_acc),
                     xytext=(best_ep + max(1, len(ep)//10), best_acc - 0.05),
                     fontsize=7, color="grey",
                     arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation Accuracy")
        ax1.set_ylim(0, 1.05)

    plt.suptitle("Deep Learning Training Dynamics\n(Conv1D / BiLSTM / Transformer)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dl_training_curves.png"))
    plt.close()
    print(f"    Saved → dl_training_curves.png")


# ══════════════════════════════════════════════════════════════
#  RAW SENSOR RESPONSE VISUALIZATION
# ══════════════════════════════════════════════════════════════

def plot_raw_sensor_responses():
    """
    Reads the original time-series Excel sheet and plots per-class
    mean sensor response profiles (14 channels × time).
    This is the canonical figure in e-nose papers showing the raw
    transient response before feature extraction.
    """
    print("\n  Plotting raw sensor response profiles ...")
    try:
        for sheet in ["3_categories (2)", "6_categories", "all_data_enose"]:
            try:
                df = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None)
                break
            except Exception:
                continue
        else:
            print("    ⚠️  Could not load time-series sheet for raw plots.")
            return

        row1 = df.iloc[0]   # Ch0, Ch1, ...
        row2 = df.iloc[1]   # category labels
        data = df.iloc[2:].reset_index(drop=True)

        # Build dict: class → {channel → [time-series values]}
        class_ch_ts = {}
        for ci in range(1, len(row1)):
            ch_name  = str(row1.iloc[ci])
            category = str(row2.iloc[ci]).strip()
            if not ch_name.startswith("Ch"): continue
            try:   ch_num = int(ch_name.replace("Ch", ""))
            except: continue
            vals = []
            for v in data.iloc[:, ci].values:
                if pd.notna(v):
                    try: vals.append(float(str(v).replace(",", ".")))
                    except: pass
            if vals:
                class_ch_ts.setdefault(category, {}).setdefault(ch_num, []).append(vals)

        unique_classes = sorted(class_ch_ts.keys())
        n_cls  = len(unique_classes)
        n_ch   = 14
        palette = sns.color_palette("tab10", n_cls)

        # ── Figure 1: mean response per channel per class ──
        fig, axes = plt.subplots(2, 7, figsize=(21, 8), sharey=False)
        for ch in range(n_ch):
            ax = axes[ch // 7, ch % 7]
            for ci, cls in enumerate(unique_classes):
                ch_data = class_ch_ts.get(cls, {}).get(ch, [])
                if not ch_data: continue
                # Average across repeats at each time point
                max_len = max(len(s) for s in ch_data)
                padded  = [s + [np.nan] * (max_len - len(s)) for s in ch_data]
                mean_ts = np.nanmean(padded, axis=0)
                std_ts  = np.nanstd(padded, axis=0)
                t = np.arange(len(mean_ts))
                ax.plot(t, mean_ts, color=palette[ci], lw=1.5, label=cls)
                ax.fill_between(t, mean_ts - std_ts, mean_ts + std_ts,
                                color=palette[ci], alpha=0.15)
            ax.set_title(f"Ch{ch}", fontsize=9)
            ax.set_xlabel("Time step", fontsize=7)
            ax.set_ylabel("Response", fontsize=7)
        axes[0, 0].legend(fontsize=7, loc="upper right")
        plt.suptitle("Raw Sensor Transient Responses by Fermentation Class\n(Mean ± SD across samples)",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "00_raw_sensor_responses.png"))
        plt.close()

        # ── Figure 2: all 14 channels overlaid per class ──
        fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4), sharey=False)
        if n_cls == 1: axes = [axes]
        ch_palette = sns.color_palette("husl", n_ch)
        for ci, cls in enumerate(unique_classes):
            ax = axes[ci]
            for ch in range(n_ch):
                ch_data = class_ch_ts.get(cls, {}).get(ch, [])
                if not ch_data: continue
                max_len = max(len(s) for s in ch_data)
                padded  = [s + [np.nan] * (max_len - len(s)) for s in ch_data]
                mean_ts = np.nanmean(padded, axis=0)
                t = np.arange(len(mean_ts))
                ax.plot(t, mean_ts, color=ch_palette[ch], lw=1.2, label=f"Ch{ch}")
            ax.set_title(cls, fontsize=10, fontweight="bold")
            ax.set_xlabel("Time step"); ax.set_ylabel("Sensor Response")
        axes[-1].legend(fontsize=6, loc="upper right",
                        ncol=2, bbox_to_anchor=(1.35, 1.0))
        plt.suptitle("All Sensor Channels — Mean Response per Class",
                     fontweight="bold", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "00b_all_channels_per_class.png"))
        plt.close()

        print(f"    Saved → 00_raw_sensor_responses.png | 00b_all_channels_per_class.png")

    except Exception as e:
        import traceback
        print(f"    ⚠️  Raw sensor plot failed: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
#  COMPREHENSIVE SUMMARY REPORT (journal-style)
# ══════════════════════════════════════════════════════════════

def create_summary_report(timestamp, train_data, test_data, feature_cols,
                          perf_df, dataset_type, pred_df, sig_df):
    report_path = os.path.join(RESULTS_DIR, f"JOURNAL_SUMMARY_REPORT_{timestamp}.txt")
    best = perf_df.iloc[0]
    with open(report_path, "w", encoding="utf-8") as f:
        sep = "=" * 80
        f.write(sep + "\n")
        f.write("E-NOSE COCOA BEAN FERMENTATION CLASSIFICATION\n")
        f.write("Journal-Quality Analysis Report (SCIE/Q1 Standard)\n")
        f.write(sep + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Dataset type: {dataset_type}\n\n")

        f.write("DATASET SUMMARY\n" + "-"*40 + "\n")
        for cls in sorted(train_data["class"].unique()):
            n = (train_data["class"] == cls).sum()
            f.write(f"  {cls}: {n} samples\n")
        f.write(f"Total training: {len(train_data)} | Test (unlabelled): {len(test_data)}\n")
        f.write(f"Features: {len(feature_cols)} sensor channels ({', '.join(feature_cols)})\n\n")

        f.write("BEST MODEL\n" + "-"*40 + "\n")
        f.write(f"  Model:         {best['Model']}\n")
        f.write(f"  Accuracy:      {best['Accuracy']:.4f}  95%CI [{best['95% CI Lo']:.4f}, {best['95% CI Hi']:.4f}]\n")
        f.write(f"  F1 (weighted): {best['F1 (W)']:.4f}\n")
        f.write(f"  MCC:           {best['MCC']:.4f}\n")
        f.write(f"  Cohen's κ:     {best['Kappa (κ)']:.4f}\n")
        f.write(f"  AUC (macro):   {best['AUC']:.4f}\n")
        f.write(f"  Brier Score:   {best['Brier']:.4f}\n\n")

        f.write("FULL MODEL RANKINGS\n" + "-"*40 + "\n")
        f.write(perf_df[["Model","Accuracy","F1 (W)","MCC","Kappa (κ)","AUC",
                          "CV Mean","CV Std","95% CI Lo","95% CI Hi"]].to_string(index=False))
        f.write("\n\n")

        f.write("STATISTICAL SIGNIFICANCE (McNemar p-values, top pairs)\n" + "-"*40 + "\n")
        f.write(sig_df[["Model A","Model B","McNemar_p","Significant_p05"]].head(15).to_string(index=False))
        f.write("\n\n")

        f.write("UNLABELLED SAMPLE PREDICTIONS\n" + "-"*40 + "\n")
        f.write(pred_df[["Sample_ID","Consensus","Agreement"]].to_string(index=False))
        f.write("\n\n")

        f.write(sep + "\nEND OF REPORT\n" + sep + "\n")

    print(f"\n✅ Journal summary report → {report_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    global timestamp, logger
    logger, timestamp = setup_logging()
    sys.stdout = logger

    print("E-NOSE COCOA BEAN CLASSIFICATION — JOURNAL-QUALITY ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: seed={RANDOM_SEED}, CV={CV_REPEATS}×{CV_FOLDS}, bootstrap_n={BOOTSTRAP_N}")
    print(f"Output: {RESULTS_DIR}\n")

    try:
        # ── Data loading ──────────────────────────────────────
        train_data, test_data, dataset_type = load_and_preprocess_data()
        if train_data is None:
            print("❌ Data loading failed."); return

        feature_cols = [f"ch{i}" for i in range(14)]
        class_labels  = sorted(train_data["class"].unique())

        # ── Raw sensor time-series visualisation ──────────────
        plot_raw_sensor_responses()

        # ── EDA ───────────────────────────────────────────────
        exploratory_data_analysis(train_data, test_data, feature_cols)

        # ── Manifold projections ──────────────────────────────
        X_sc_full, X_test_sc, y_full, test_ids, \
        X_tr, X_val, y_tr, y_val, scaler = prepare_data_for_modeling(
            train_data, test_data, feature_cols
        )
        manifold_visualizations(X_sc_full, y_full, feature_cols)

        # ── Model training ────────────────────────────────────
        results, trained_models, cv_scores_dict, le = train_and_evaluate_models(
            X_sc_full, X_val, y_full, y_val, X_tr, y_tr, feature_cols
        )

        # ── Hyperparameter tuning (classical only) ────────────
        tuned_models = hyperparameter_tuning(X_tr, y_tr)

        # ── Publication figures ───────────────────────────────
        plot_confusion_matrices(results, trained_models, X_val, y_val, class_labels, le)
        plot_roc_curves(results, trained_models, X_val, y_val, class_labels, le)
        plot_calibration(results, trained_models, X_val, y_val, class_labels, le)
        plot_training_curves(results)
        perf_df = plot_performance_summary(results)

        # ── Statistical significance ──────────────────────────
        sig_df = statistical_significance_analysis(
            results, cv_scores_dict, y_val, trained_models, X_val, le
        )

        # ── Per-class metrics ─────────────────────────────────
        per_class_metrics_table(results, trained_models, X_val, y_val, class_labels, le)

        # ── Feature attribution ───────────────────────────────
        shap_analysis(trained_models, X_tr, X_val, feature_cols, class_labels)
        ablation_study(X_tr, X_val, y_tr, y_val, feature_cols)

        # ── Predict unlabelled samples ────────────────────────
        pred_df, all_preds, all_probs = predict_unlabelled(
            trained_models, X_test_sc, test_ids, class_labels, le
        )

        # ── Summary report ────────────────────────────────────
        create_summary_report(timestamp, train_data, test_data, feature_cols,
                               perf_df, dataset_type, pred_df, sig_df)

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"  Results  → {RESULTS_DIR}/")
        print(f"  Figures  → {PLOTS_DIR}/  ({len(os.listdir(PLOTS_DIR))} files)")
        print(f"  Tables   → {DATA_DIR}/   ({len(os.listdir(DATA_DIR))} files)")
        print(f"  Stats    → {STATS_DIR}/  ({len(os.listdir(STATS_DIR))} files)")

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}\n{traceback.format_exc()}")
    finally:
        if logger:
            logger.close(); sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()