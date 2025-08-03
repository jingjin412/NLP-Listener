from scipy.stats import ttest_rel
import json

def collect_per_sample_metrics(true_labels, pred_labels):
    per_sample_f1 = []
    per_sample_em = []
    for true, pred in zip(true_labels, pred_labels):
        true_set, pred_set = set(true), set(pred)
        common = true_set & pred_set
        p = len(common) / len(pred_set) if pred_set else 0
        r = len(common) / len(true_set) if true_set else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        em = 1 if true_set == pred_set else 0
        per_sample_f1.append(f1)
        per_sample_em.append(em)
    return per_sample_f1, per_sample_em

file_a = "../results/qwen/relation_name_noinfo_pre.json"
file_b = "../results/qwen/relation_name_position_pre.json"

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    true_labels, pred_labels = [], []
    for i in results:
        if i[0] == "less than 3 people":
            continue
        for j in i:
            true_labels.append(j["listener ans"].split(","))
            pred_labels.append(j["listener"].split(","))
    return true_labels, pred_labels

true_a, pred_a = load_labels(file_a)
true_b, pred_b = load_labels(file_b)

assert true_a == true_b

f1_a, em_a = collect_per_sample_metrics(true_a, pred_a)
f1_b, em_b = collect_per_sample_metrics(true_b, pred_b)

t_stat_f1, p_value_f1 = ttest_rel(f1_a, f1_b)
t_stat_em, p_value_em = ttest_rel(em_a, em_b)

print(f"File A: {file_a}, F1 = {sum(f1_a) / len(f1_a):.4f}, EM = {sum(em_a) / len(em_a):.4f}")
print(f"File B: {file_b}, F1 = {sum(f1_b) / len(f1_b):.4f}, EM = {sum(em_b) / len(em_b):.4f}")

print(f"\nPaired t-test on F1 scores: t = {t_stat_f1:.4f}, p = {p_value_f1:.10f}")
diff = sum(f1_a) / len(f1_a) - sum(f1_b) / len(f1_b)
if p_value_f1 < 0.05:
    print("Difference is statistically significant (p < 0.05)")
    print("\colordiff{", f"{diff:.4f}", "}$^\dag$", sep="")
else:
    print("No significant difference (p ≥ 0.05)")
    print("\colordiff{", f"{diff:.4f}", sep="")
    
print(f"\nPaired t-test on Exact Match: t = {t_stat_em:.4f}, p = {p_value_em:.10f}")
diff = (sum(em_a) / len(em_a) - sum(em_b) / len(em_b))
if p_value_em < 0.05:
    print("Difference is statistically significant (p < 0.05)")
    print("\colordiff{", f"{diff:.4f}", "}$^\dag$", sep="")
else:
    print("No significant difference (p ≥ 0.05)")
    print("\colordiff{", f"{diff:.4f}", "}", sep="")