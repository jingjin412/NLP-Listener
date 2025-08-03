import json
import pandas as pd
import matplotlib.pyplot as plt

final_results = []

def process_characters(dialogue):
    all_char_set = set()

    for utterance in dialogue:
        speaker = utterance["speaker"]
        if speaker not in all_char_set:
            all_char_set.add(speaker)

        for listener in utterance["listener"]:
            listener_name = listener["name"]
            if listener_name not in all_char_set:
                all_char_set.add(listener_name)

    return len(all_char_set)
from collections import defaultdict

def calculate_exact_match(true_labels, pred_labels):
    exact_match = 0
    for true, pred in zip(true_labels, pred_labels):
        if set(true) == set(pred):
            exact_match += 1
    return exact_match / len(true_labels) if true_labels else 0

# 用於分類儲存不同角色數量下的資料
grouped_true_labels = defaultdict(list)
grouped_pred_labels = defaultdict(list)

with open("../mpdd/dialogue_chinese.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    
with open("classification_results.json", "r", encoding="utf-8") as f:
    categories = json.load(f)

info_list = ["noinfo", "position", "relation"]
name_list = ["name"]
case_list = ["pre"]

for case in case_list:
    for name in name_list:
        for info in info_list:
            print(f"\nCalculating F1 and Exact Match by character count for {name} {info} {case}...")
            with open(f"../results/qwen/relation_{name}_{info}_{case}.json", "r", encoding="utf-8") as f:
                results = json.load(f)

            index = 0
            grouped_true_labels.clear()
            grouped_pred_labels.clear()

            for d in dataset:
                dialogue = dataset[d]
                chars_num = process_characters(dialogue)
                group_key = ">5" if 5 <= chars_num <= 8 else str(chars_num)
                
                result = results[index]
                category = categories[index]
                index += 1
                if result[0] == "less than 3 people":
                    continue
                for i in range(len(result)):
                    grouped_true_labels[group_key + category[i]["predicted_label"]].append(result[i]["listener ans"].split(","))
                    grouped_pred_labels[group_key + category[i]["predicted_label"]].append(result[i]["listener"].split(","))

            # 分別計算每一角色數量群組的 exact match
            for group_key in grouped_true_labels.keys():
                true_group = grouped_true_labels[group_key]
                pred_group = grouped_pred_labels[group_key]
                exact = calculate_exact_match(true_group, pred_group)
                print(f"Character Count = {group_key}: Exact Match = {exact:.4f} ({len(true_group)} samples)")
                
                final_results.append((group_key[:-1], group_key[-1], info, exact))
                
df = pd.DataFrame(final_results, columns=["Characters", "RefType", "Info", "Accuracy"])

ref_map = {"A": "Name", "B": "Role", "C": "Implicit"}
df["RefType"] = df["RefType"].map(ref_map)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
ref_types = ["Name", "Role", "Implicit"]
colors = {"noinfo": "#66c2a5", "position": "#fc8d62", "relation": "#8da0cb"}

for i, ref in enumerate(ref_types):
    if i > 0:
        ax.legend_.remove() 
    ax = axes[i]
    subset = df[df["RefType"] == ref]
    pivot_df = subset.pivot(index="Characters", columns="Info", values="Accuracy")
    pivot_df = pivot_df[["noinfo", "position", "relation"]] 

    pivot_df.plot(
        kind="bar",
        ax=ax,
        width=0.7,  
        color=[colors[col] for col in pivot_df.columns]
    )

    ax.tick_params(axis='x', rotation=0, labelsize=13)  
    ax.set_title(ref, fontsize=14)
    ax.set_xlabel("Number of Characters", fontsize=14)
    ax.grid(axis='y', linestyle='-', alpha=0.5)

    if i == 0:
        ax.set_ylabel("Exact Match Accuracy", fontsize=14)
    else:
        ax.set_ylabel("")
    ax.set_ylim(0.2, 0.9)

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig("character_count_exact_match.png", dpi=300)
