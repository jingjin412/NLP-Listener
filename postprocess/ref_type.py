import json
import pandas as pd
import matplotlib.pyplot as plt

final_results = []

from collections import defaultdict

def calculate_exact_match(true_labels, pred_labels):
    exact_match = 0
    for true, pred in zip(true_labels, pred_labels):
        if set(true) == set(pred):
            exact_match += 1
    return exact_match / len(true_labels) if true_labels else 0

grouped_true_labels = defaultdict(list)
grouped_pred_labels = defaultdict(list)

with open("../mpdd/dialogue_chinese.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    
with open("../classification_results.json", "r", encoding="utf-8") as f:
    categories = json.load(f)

model_list = {"qwen": "Qwen2.5", "llama": "Llama-3.1", "gpt": "GPT-4o-mini", "R1_llama": "DeepSeek-R1-Distill-Llama"}
info_list = ["noinfo", "position", "relation"]
name_list = ["name"]
case_list = ["pre"]
type_dict = {"A": "Name", "B": "Role", "C": "Implicit"}

for model in model_list:
    for case in case_list:
        for name in name_list:
            for info in info_list:
                print(f"\nCalculating F1 and Exact Match by character count for {name} {info} {case}...")
                with open(f"../results/{model}/relation_{name}_{info}_{case}.json", "r", encoding="utf-8") as f:
                    results = json.load(f)

                index = 0
                grouped_true_labels.clear()
                grouped_pred_labels.clear()

                for d in dataset:
                    dialogue = dataset[d]
                    
                    result = results[index]
                    category = categories[index]
                    index += 1
                    if result[0] == "less than 3 people":
                        continue
                    for i in range(len(result)):
                        grouped_true_labels[category[i]["predicted_label"] + model_list[model]].append(result[i]["listener ans"].split(","))
                        grouped_pred_labels[category[i]["predicted_label"] + model_list[model]].append(result[i]["listener"].split(","))

                for group_key in grouped_true_labels.keys():
                    true_group = grouped_true_labels[group_key]
                    pred_group = grouped_pred_labels[group_key]
                    exact = calculate_exact_match(true_group, pred_group)
                    print(f"Character Count = {group_key}: Exact Match = {exact:.4f} ({len(true_group)} samples)")
                    
                    final_results.append((group_key[1:], type_dict[group_key[0]], info, exact))
     
df = pd.DataFrame(final_results, columns=["Model", "RefType", "Info", "Accuracy"])

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
colors = {"noinfo": "#66c2a5", "position": "#fc8d62", "relation": "#8da0cb"}

for i, model in enumerate(df["Model"].unique()):
    if i > 0:
        ax.legend_.remove()
    ax = axes[i]
    sub = df[df["Model"] == model]
    pivot = sub.pivot(index="RefType", columns="Info", values="Accuracy")
    pivot = pivot[["noinfo", "position", "relation"]]
    pivot = pivot.reindex(["Name", "Role", "Implicit"])
    pivot.plot(
        kind="bar",
        ax=ax,
        width=0.7,
        color=[colors[col] for col in pivot.columns]
    )
    ax.set_title(model, fontsize=14)
    ax.set_xlabel("Reference Type", fontsize=13)
    ax.tick_params(axis='x', rotation=0, labelsize=13)
    ax.set_ylim(0.4, 0.9)
    ax.grid(axis='y', linestyle='-', alpha=0.5)
    if i == 0:
        ax.set_ylabel("Exact Match Accuracy", fontsize=13)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("reftype_exact_match.png", dpi=300)