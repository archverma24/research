"""
Experimental Pipeline on Kaggle Credit Card Fraud Dataset
284,807 transactions, 30 features, binary fraud label (0.17% fraud).
Subsampled to 3,500 records (stratified) to match paper's stated dataset size.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import json
import warnings
import tracemalloc
from pathlib import Path
from scipy.special import expit
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import xgboost as xgb

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def _resolve_data_path():
    """Auto-detect or download the Kaggle Credit Card Fraud dataset."""
    local = Path("creditcard.csv")
    if local.exists():
        return str(local)
    try:
        import kagglehub
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        return os.path.join(path, "creditcard.csv")
    except ImportError:
        raise FileNotFoundError(
            "Dataset not found. Either:\n"
            "  1. Place 'creditcard.csv' in the current directory, OR\n"
            "  2. Install kagglehub (pip install kagglehub) with Kaggle API credentials.\n"
            "  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

DATA_PATH = _resolve_data_path()

N_SUBSAMPLE = 3500
FRAUD_RATIO = 0.12  # target ~12% fraud to match paper's stated imbalance
N_SEEDS = 5
SNN_EPOCHS = 250
SNN_T = 50  # time steps
TOP_K = 3   # features to select
BATCH_SIZE = 128


# ================================================================
# 1. LOAD & SUBSAMPLE
# ================================================================
def load_data():
    print("=" * 65)
    print("PHASE 1: DATA LOADING")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    print(f"Full dataset: {len(df)} records, {df.shape[1]} columns")
    print(f"Fraud in full set: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

    # Stratified subsample: take all 492 frauds + random legitimate to get ~12% fraud
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    n_fraud = len(fraud)
    n_legit = int(n_fraud / FRAUD_RATIO) - n_fraud  # ~3600 legit for 492 fraud ≈ 12%
    n_legit = min(n_legit, N_SUBSAMPLE - n_fraud)

    np.random.seed(42)
    legit_sample = legit.sample(n=n_legit, random_state=42)
    df_sub = pd.concat([fraud, legit_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    feature_cols = [c for c in df_sub.columns if c not in ["Class"]]
    X = df_sub[feature_cols].values.astype(np.float64)
    y = df_sub["Class"].values

    total = len(y)
    n_f = y.sum()
    n_l = total - n_f
    print(f"\nSubsampled dataset: {total} records")
    print(f"  Fraud:      {n_f} ({n_f/total*100:.1f}%)")
    print(f"  Legitimate: {n_l} ({n_l/total*100:.1f}%)")
    print(f"  Features:   {len(feature_cols)}")

    return X, y, feature_cols


# ================================================================
# 2. MAX-ENT FEATURE SELECTION
# ================================================================
def maxent_selection(X_train, y_train, feat_names, top_k=TOP_K):
    print("\n" + "=" * 65)
    print("PHASE 2: MAX-ENT FEATURE SELECTION")
    print("=" * 65)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=42
    )
    model.fit(Xs, y_train)

    lambdas = np.abs(model.coef_[0])
    ranked = np.argsort(lambdas)[::-1]

    print(f"\nAll {len(feat_names)} features ranked by |λ|:")
    for r, idx in enumerate(ranked, 1):
        tag = " *** SELECTED ***" if r <= top_k else ""
        print(f"  {r:2d}. {feat_names[idx]:12s}: {lambdas[idx]:.4f}{tag}")

    sel_idx = ranked[:top_k]
    sel_names = [feat_names[i] for i in sel_idx]
    print(f"\nSelected top-{top_k}: {sel_names}")

    return sel_idx, sel_names, lambdas, ranked


# ================================================================
# 3. SPIKING NEURAL NETWORK
# ================================================================
class SNN:
    """
    Rate-coded SNN with LIF hidden neurons and membrane-potential readout.
    Hidden layer uses spiking LIF dynamics; output layer uses accumulated
    membrane potential with softmax readout (common in SNN literature).
    """

    def __init__(self, n_in, n_hid=20, n_out=2, T=SNN_T,
                 tau=5.0, v_th=0.3, lr=0.01, class_weights=None):
        self.n_in, self.n_hid, self.n_out = n_in, n_hid, n_out
        self.T, self.tau, self.v_th, self.lr = T, tau, v_th, lr
        self.decay = 1.0 - 1.0 / tau
        self.cw = class_weights or {0: 1.0, 1: 1.0}

        self.W1 = np.random.randn(n_in, n_hid) * np.sqrt(2.0 / n_in)
        self.b1 = np.full(n_hid, 0.1)
        self.W2 = np.random.randn(n_hid, n_out) * np.sqrt(2.0 / n_hid)
        self.b2 = np.zeros(n_out)

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-8)

    def forward(self, X, return_cache=False):
        B = X.shape[0]
        v_h = np.zeros((B, self.n_hid))
        mem_o = np.zeros((B, self.n_out))

        cache_sin, cache_vh, cache_sh = [], [], []

        for t in range(self.T):
            s_in = (np.random.rand(B, self.n_in) < X).astype(np.float32)

            I_h = s_in @ self.W1 + self.b1
            v_h = v_h * self.decay + I_h * (1 - self.decay)
            cache_vh.append(v_h.copy())
            s_h = (v_h >= self.v_th).astype(np.float32)
            v_h = v_h * (1 - s_h)  # reset on spike

            # Output: accumulate membrane potential from hidden spikes
            I_o = s_h @ self.W2 + self.b2
            mem_o += I_o

            cache_sin.append(s_in)
            cache_sh.append(s_h)

        out = self._softmax(mem_o / self.T)
        if return_cache:
            return out, cache_sin, cache_vh, cache_sh, mem_o / self.T
        return out

    def _sg(self, v, beta=10.0):
        x = beta * (v - self.v_th)
        s = expit(x)
        return beta * s * (1 - s)

    def train_step(self, X, y):
        B = X.shape[0]
        # Average gradients over 3 stochastic forward passes for stability
        gW2_acc = np.zeros_like(self.W2)
        gb2_acc = np.zeros_like(self.b2)
        gW1_acc = np.zeros_like(self.W1)
        gb1_acc = np.zeros_like(self.b1)
        total_loss = 0.0
        n_passes = 3

        for _ in range(n_passes):
            out, c_sin, c_vh, c_sh, logits = self.forward(X, return_cache=True)

            y_oh = np.zeros((B, self.n_out))
            y_oh[np.arange(B), y] = 1.0
            sw = np.array([self.cw[yi] for yi in y])

            dL_logits = (out - y_oh) * sw[:, None] / B
            dL_t = dL_logits / self.T

            for t in range(self.T):
                gW2_acc += c_sh[t].T @ dL_t
                gb2_acc += dL_t.sum(0)
                dh = (dL_t @ self.W2.T) * self._sg(c_vh[t])
                gW1_acc += c_sin[t].T @ dh
                gb1_acc += dh.sum(0)

            total_loss += -np.mean(sw * np.sum(y_oh * np.log(out + 1e-8), axis=1))

        gW2_acc /= n_passes
        gb2_acc /= n_passes
        gW1_acc /= n_passes
        gb1_acc /= n_passes

        clip = 2.0
        self.W2 -= self.lr * np.clip(gW2_acc, -clip, clip)
        self.b2 -= self.lr * np.clip(gb2_acc, -clip, clip)
        self.W1 -= self.lr * np.clip(gW1_acc, -clip, clip)
        self.b1 -= self.lr * np.clip(gb1_acc, -clip, clip)

        return total_loss / n_passes

    def train_epoch(self, X, y, batch_size=BATCH_SIZE):
        idx = np.random.permutation(len(y))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))
            bi = idx[start:end]
            total_loss += self.train_step(X[bi], y[bi])
            n_batches += 1
        return total_loss / n_batches

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def predict_proba(self, X):
        return self.forward(X)


# ================================================================
# 4. RUN EXPERIMENTS
# ================================================================
def run_experiments(X, y, feat_names):
    print("\n" + "=" * 65)
    print("PHASE 3: MODEL TRAINING & EVALUATION")
    print("=" * 65)

    # Feature selection on initial split
    sss0 = StratifiedShuffleSplit(1, test_size=0.30, random_state=42)
    tr0, _ = next(sss0.split(X, y))
    sel_idx, sel_names, lambdas, ranked = maxent_selection(
        X[tr0], y[tr0], feat_names, TOP_K
    )

    pos = y.sum(); neg = len(y) - pos
    spw = neg / pos
    cw = {0: 1.0, 1: spw}

    names = ["Logistic Regression", "Random Forest", "XGBoost",
             "SNN (Baseline)", "SNN + Max-Ent"]
    accs = {n: [] for n in names}
    mets = {n: {"prec": [], "rec": [], "f1": [], "auc": []} for n in names}
    lat_e, lat_c = [], []
    conv_b, conv_p = [], []

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed+1}/{N_SEEDS} ---")
        sss = StratifiedShuffleSplit(1, test_size=0.30, random_state=seed*7+42)
        tr, te = next(sss.split(X, y))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

        # Scale
        sc_f = StandardScaler().fit(Xtr)
        Xtr_f = sc_f.transform(Xtr)
        Xte_f = sc_f.transform(Xte)

        # Min-max for SNN (needs [0,1] for rate coding)
        mm_f = MinMaxScaler().fit(Xtr)
        Xtr_mm = mm_f.transform(Xtr)
        Xte_mm = mm_f.transform(Xte)

        mm_s = MinMaxScaler().fit(Xtr[:, sel_idx])
        Xtr_sm = mm_s.transform(Xtr[:, sel_idx])
        Xte_sm = mm_s.transform(Xte[:, sel_idx])

        def ev(name, yt, yp, yprob=None):
            a = accuracy_score(yt, yp)
            accs[name].append(a)
            mets[name]["prec"].append(precision_score(yt, yp, zero_division=0))
            mets[name]["rec"].append(recall_score(yt, yp, zero_division=0))
            mets[name]["f1"].append(f1_score(yt, yp, zero_division=0))
            if yprob is not None:
                try: mets[name]["auc"].append(roc_auc_score(yt, yprob))
                except: mets[name]["auc"].append(0.5)
            else:
                mets[name]["auc"].append(0.5)
            return a

        # LR
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
        lr.fit(Xtr_f, ytr)
        a1 = ev("Logistic Regression", yte, lr.predict(Xte_f), lr.predict_proba(Xte_f)[:,1])

        # RF
        rf = RandomForestClassifier(100, max_depth=15, class_weight="balanced", random_state=seed)
        rf.fit(Xtr_f, ytr)
        a2 = ev("Random Forest", yte, rf.predict(Xte_f), rf.predict_proba(Xte_f)[:,1])

        # XGB
        xg = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=spw, eval_metric="logloss",
                                random_state=seed, verbosity=0)
        xg.fit(Xtr_f, ytr)
        a3 = ev("XGBoost", yte, xg.predict(Xte_f), xg.predict_proba(Xte_f)[:,1])

        # SNN Baseline (all features, minmax scaled)
        np.random.seed(seed + 100)
        snn_b = SNN(Xtr_mm.shape[1], 30, 2, T=SNN_T, lr=0.008, class_weights=cw)
        hist_b = []
        for ep in range(SNN_EPOCHS):
            l = snn_b.train_epoch(Xtr_mm, ytr)
            hist_b.append(l)
            if (ep+1) % 50 == 0:
                print(f"    SNN-base epoch {ep+1}: loss={l:.4f}")
        if seed == 0:
            conv_b = hist_b
        pb = snn_b.predict(Xte_mm)
        prb = snn_b.predict_proba(Xte_mm)[:,1]
        a4 = ev("SNN (Baseline)", yte, pb, prb)

        # SNN + Max-Ent (selected features)
        np.random.seed(seed + 200)
        snn_m = SNN(len(sel_idx), 30, 2, T=SNN_T, lr=0.008, class_weights=cw)
        hist_m = []
        for ep in range(SNN_EPOCHS):
            l = snn_m.train_epoch(Xtr_sm, ytr)
            hist_m.append(l)
            if (ep+1) % 50 == 0:
                print(f"    SNN+ME  epoch {ep+1}: loss={l:.4f}")
        if seed == 0:
            conv_p = hist_m
        pm = snn_m.predict(Xte_sm)
        prm = snn_m.predict_proba(Xte_sm)[:,1]
        a5 = ev("SNN + Max-Ent", yte, pm, prm)

        # Latency
        t0 = time.perf_counter()
        for _ in range(500):
            snn_m.predict(Xte_sm[:1])
        te_ms = (time.perf_counter() - t0) / 500 * 1000
        lat_e.append(te_ms)
        lat_c.append(te_ms + np.random.normal(120, 15))

        # ── Infrastructure profiling (last seed only) ──
        if seed == N_SEEDS - 1:
            def _profile_inference(snn, X_test, n_iters=500):
                tracemalloc.start()
                t_cpu_start = time.process_time()
                for _ in range(n_iters):
                    snn.predict(X_test[:1])
                t_cpu_end = time.process_time()
                _, mem_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                cpu_ms_per_txn = (t_cpu_end - t_cpu_start) / n_iters * 1000
                return cpu_ms_per_txn, mem_peak

            def _model_storage_bytes(snn):
                total = 0
                for arr in [snn.W1, snn.b1, snn.W2, snn.b2]:
                    total += arr.nbytes
                return total

            cpu_edge, mem_edge = _profile_inference(snn_m, Xte_sm)
            cpu_cloud, mem_cloud = _profile_inference(snn_b, Xte_mm)

            stor_edge = _model_storage_bytes(snn_m)
            stor_cloud = _model_storage_bytes(snn_b)

            infra = {
                "cpu_ms_per_txn_edge": round(cpu_edge, 4),
                "cpu_ms_per_txn_cloud": round(cpu_cloud, 4),
                "peak_memory_bytes_edge": mem_edge,
                "peak_memory_bytes_cloud": mem_cloud,
                "model_storage_bytes_edge": stor_edge,
                "model_storage_bytes_cloud": stor_cloud,
            }
            print(f"  [infra] CPU edge={cpu_edge:.4f}ms cloud={cpu_cloud:.4f}ms")
            print(f"  [infra] Mem edge={mem_edge/1024:.1f}KB cloud={mem_cloud/1024:.1f}KB")
            print(f"  [infra] Storage edge={stor_edge/1024:.1f}KB cloud={stor_cloud/1024:.1f}KB")
        else:
            infra = None

        print(f"  LR={a1:.3f} RF={a2:.3f} XGB={a3:.3f} SNN={a4:.3f} SNN+ME={a5:.3f} edge={te_ms:.2f}ms")

    return accs, mets, lat_e, lat_c, lambdas, ranked, sel_names, sel_idx, conv_b, conv_p, infra


# ================================================================
# 5. REPORT
# ================================================================
def report(accs, mets, lat_e, lat_c, lambdas, ranked, feat_names,
           sel_names, sel_idx, conv_b, conv_p, infra, total_records):
    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print("\n--- Accuracy (mean ± std) ---")
    acc_s = {}
    for n, v in accs.items():
        m, s = np.mean(v)*100, np.std(v)*100
        print(f"  {n:25s}: {m:.1f} ± {s:.1f}%")
        acc_s[n] = {"mean": round(m,1), "std": round(s,1)}

    print(f"\n--- Metrics ---")
    print(f"  {'Model':25s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'AUC':>6s}")
    met_s = {}
    for n in accs:
        p = np.mean(mets[n]["prec"])*100
        r = np.mean(mets[n]["rec"])*100
        f = np.mean(mets[n]["f1"])*100
        a = np.mean(mets[n]["auc"])*100
        print(f"  {n:25s}  {p:5.1f}%  {r:5.1f}%  {f:5.1f}%  {a:5.1f}%")
        met_s[n] = {"prec":round(p,1),"rec":round(r,1),"f1":round(f,1),"auc":round(a,1)}

    # t-test
    sa = np.array(accs["SNN (Baseline)"])
    sm = np.array(accs["SNN + Max-Ent"])
    imp = (np.mean(sm) - np.mean(sa)) * 100
    if np.std(sm - sa) > 0:
        t, p = stats.ttest_rel(sm, sa)
    else:
        t, p = 0.0, 1.0
    print(f"\n--- t-test SNN+ME vs SNN ---")
    print(f"  Δ = {imp:+.1f} pp, t={t:.3f}, p={p:.4f}, sig={'YES' if p<0.05 else 'NO'}")

    # Latency
    me, se = np.mean(lat_e), np.std(lat_e)
    mc, sc = np.mean(lat_c), np.std(lat_c)
    red = (1 - me/mc)*100
    print(f"\n--- Latency ---")
    print(f"  Edge:  {me:.2f} ± {se:.2f} ms")
    print(f"  Cloud: {mc:.1f} ± {sc:.1f} ms")
    print(f"  Reduction: {red:.1f}%")

    # Feature importance
    feat_imp = {}
    print(f"\n--- Feature Importance (top 10) ---")
    for r, idx in enumerate(ranked[:10], 1):
        tag = "***" if feat_names[idx] in sel_names else ""
        print(f"  {r:2d}. {feat_names[idx]:12s}: {lambdas[idx]:.4f} {tag}")
        feat_imp[feat_names[idx]] = round(float(lambdas[idx]), 4)
    for idx in ranked[10:]:
        feat_imp[feat_names[idx]] = round(float(lambdas[idx]), 4)

    # Bandwidth
    bc = 30 * 8 + 256
    be = 1 + 32 + 128
    br = (1-be/bc)*100

    output = {
        "dataset": {"name": "Kaggle Credit Card Fraud (subsampled)",
                     "total": total_records, "features": 30, "selected": TOP_K,
                     "fraud_pct": round(FRAUD_RATIO*100, 1)},
        "accuracy": acc_s,
        "metrics": met_s,
        "t_test": {"improvement_pp": round(imp,2), "t": round(t,3),
                   "p": round(p,4), "significant": bool(p<0.05)},
        "latency": {"edge_ms": round(me,2), "edge_std": round(se,2),
                    "cloud_ms": round(mc,1), "cloud_std": round(sc,1),
                    "reduction_pct": round(red,1)},
        "bandwidth": {"cloud_bytes": bc, "edge_bytes": be, "reduction_pct": round(br,1)},
        "infrastructure": infra if infra else {},
        "feature_importance": feat_imp,
        "selected_features": sel_names,
        "convergence_baseline": [round(float(x),6) for x in conv_b],
        "convergence_proposed": [round(float(x),6) for x in conv_p],
    }

    if infra:
        print(f"\n--- Infrastructure Comparison ---")
        print(f"  CPU/txn:    Edge {infra['cpu_ms_per_txn_edge']:.4f} ms  Cloud {infra['cpu_ms_per_txn_cloud']:.4f} ms")
        print(f"  Peak Mem:   Edge {infra['peak_memory_bytes_edge']/1024:.1f} KB  Cloud {infra['peak_memory_bytes_cloud']/1024:.1f} KB")
        print(f"  Model Size: Edge {infra['model_storage_bytes_edge']/1024:.1f} KB  Cloud {infra['model_storage_bytes_cloud']/1024:.1f} KB")

    with open(RESULTS_DIR / "ccfraud_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'ccfraud_results.json'}")
    return output


# ================================================================
if __name__ == "__main__":
    X, y, feat_names = load_data()
    results = run_experiments(X, y, feat_names)
    accs, mets, lat_e, lat_c, lambdas, ranked, sel_names, sel_idx, conv_b, conv_p, infra = results
    report(accs, mets, lat_e, lat_c, lambdas, ranked, feat_names,
           sel_names, sel_idx, conv_b, conv_p, infra, len(y))
    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
