import pandas as pd
df = pd.read_csv("runs/pt_svmcal_rf60_C3_ng37/por_test_predictions.csv")
pos = "por_bully"

fns = df[(df.y_true==pos)&(df.y_pred!=pos)].sort_values("proba_pos").head(3)
fps = df[(df.y_true!=pos)&(df.y_pred==pos)].sort_values("proba_pos", ascending=False).head(3)
print("FNs:\n", fns[["text","proba_pos"]], "\n")
print("FPs:\n", fps[["text","proba_pos"]])
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/pt_svmcal_rf60_C3_ng37/por_test_predictions.csv")
pos = "por_bully"
df["y_true_bin"] = (df["y_true"]==pos).astype(int)

plt.figure()
df[df.y_true_bin==1]["proba_pos"].hist(alpha=0.6, bins=20, label="bully (verdadeiro)")
df[df.y_true_bin==0]["proba_pos"].hist(alpha=0.6, bins=20, label="nonbully (verdadeiro)")
plt.axvline(0.33, linestyle="--")
plt.title("Distribuição de probabilidades (TEST) — linha = threshold 0.33")
plt.xlabel("proba_pos"); plt.ylabel("contagem"); plt.legend(); plt.tight_layout()
plt.savefig("runs/pt_svmcal_rf60_C3_ng37/proba_hist_test.png", dpi=150)
print("salvo proba_hist_test.png")
