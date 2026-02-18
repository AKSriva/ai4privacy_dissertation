import matplotlib.pyplot as plt

clean = {"Precision": 0.9672, "Recall": 0.9642, "F1": 0.9657}
perturbed = {"Precision": 0.6384, "Recall": 0.7836, "F1": 0.7036}

metrics = list(clean.keys())

clean_vals = [clean[m] for m in metrics]
pert_vals = [perturbed[m] for m in metrics]

x = range(len(metrics))

plt.figure(figsize=(8,5))
plt.bar([i - 0.2 for i in x], clean_vals, width=0.4, label="Clean")
plt.bar([i + 0.2 for i in x], pert_vals, width=0.4, label="Perturbed")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("DistilBERT Performance: Clean vs Perturbed")
plt.ylim(0,1.05)
plt.legend()

plt.tight_layout()
plt.savefig("outputs/fig_robustness_clean_vs_perturbed.png", dpi=300)
plt.show()
