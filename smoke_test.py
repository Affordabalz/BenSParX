"""Fast CI check: format is valid and a small grouped CV runs. Not the full nested CV."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

path = sys.argv[1] if len(sys.argv) > 1 else "BenSParX_synthetic.csv"
df = pd.read_csv(path)
assert {"class", "id"} <= set(df.columns), "missing class/id"
assert set(df["class"].unique()) == {0, 1}, "class must be binary 0/1"
X = df.drop(columns=["class", "id"])
assert X.select_dtypes(exclude="number").empty, "non-numeric feature column"
assert not X.isna().any().any(), "NaNs present"
assert "hba1c_pct" not in X.columns, "label leakage"

y, g, Xv = df["class"].values, df["id"].values, X.values
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in {"LR": LogisticRegression(max_iter=1000),
                "RF": RandomForestClassifier(n_estimators=100, random_state=42)}.items():
    acc, auc = [], []
    for tr, te in cv.split(Xv, y, g):
        p = Pipeline([("s", StandardScaler()), ("m", m)]).fit(Xv[tr], y[tr])
        acc.append(accuracy_score(y[te], p.predict(Xv[te])))
        auc.append(roc_auc_score(y[te], p.predict_proba(Xv[te])[:, 1]))
    print(f"{name}: accuracy {np.mean(acc):.3f}  AUC {np.mean(auc):.3f}")
    assert np.mean(auc) > 0.6, f"{name} AUC too low - data relationships may have collapsed"
print("SMOKE TEST PASSED")
