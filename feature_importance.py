import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from prepare_for_modeling import X, y  # X: DataFrame, y: Series

# ── Fit RandomForestClassifier ────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# ── Feature importances ───────────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\n" + "=" * 50)
print("Feature Importance (RandomForestClassifier)")
print("=" * 50)
for rank, (feat, score) in enumerate(importances_sorted.items(), start=1):
    print(f"  {rank:>2}. {feat:<35s}  {score:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn(importances_sorted.values / importances_sorted.values.max())
importances_sorted.plot(kind="barh", ax=ax, color=colors)

ax.invert_yaxis()  # most important feature at the top
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance – RandomForestClassifier\n(target: churnlabel)", fontsize=13)
ax.axvline(x=importances_sorted.mean(), color="steelblue", linestyle="--",
           linewidth=1.2, label=f"Mean importance ({importances_sorted.mean():.4f})")
ax.legend(fontsize=10)
plt.tight_layout()

output_path = "feature_importance.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")
