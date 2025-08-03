import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def calculate_exact_match(true_labels, pred_labels):
    exact_match = 0
    for true, pred in zip(true_labels, pred_labels):
        if set(true) == set(pred):
            exact_match += 1
    return exact_match / len(true_labels) if true_labels else 0

final_results = []

# You can adjust the path to your dataset and results files as needed
model_list = {
    "qwen": "Qwen2.5",
    "llama": "LLaMA-3.1",
    "gpt": "GPT-4o-mini",
    "R1_llama": "DeepSeek-R1-Distill-Llama"
}
info_map = {"noinfo": "None", "position": "Position", "relation": "Relation"}
name_map = {"name": "Name", "noname": "Alias"}
case_list = ["pre", "whole"]
name_list = ["name", "noname"]
info_list = ["noinfo", "position", "relation"]

with open("../mpdd/dialogue_chinese.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for model in model_list:
    for case in case_list:
        for name in name_list:
            for info in info_list:
                print(f"Running {model_list[model]} - {case} / {name} / {info} ...")
                try:
                    with open(f"../results/{model}/relation_{name}_{info}_{case}.json", "r", encoding="utf-8") as f:
                        results = json.load(f)
                except FileNotFoundError:
                    print("File not found, skipping.")
                    continue

                true_labels = []
                pred_labels = []
                for i, d in enumerate(dataset):
                    dialogue = dataset[d]
                    result = results[i]
                    if result[0] == "less than 3 people":
                        continue
                    for utt in result:
                        true_labels.append(utt["listener ans"].split(","))
                        pred_labels.append(utt["listener"].split(","))

                exact = calculate_exact_match(true_labels, pred_labels)
                prompt = f"{case.capitalize()} / {name_map[name]}"
                final_results.append((model_list[model], info_map[info], prompt, exact))

df = pd.DataFrame(final_results, columns=["Model", "Character Info", "Prompt Setting", "Accuracy"])

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

models = df["Model"].unique()
fig, axes = plt.subplots(1, len(models), figsize=(16, 4.3), sharey=True)

from matplotlib.cm import ScalarMappable
import matplotlib as mpl

vmin, vmax = 0.52, 0.70
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(norm=norm, cmap="YlGnBu")
sm.set_array([])

for i, model in enumerate(models):
    ax = axes[i]
    subset = df[df["Model"] == model]
    pivot_df = subset.pivot(index="Character Info", columns="Prompt Setting", values="Accuracy")
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar=False,  
        ax=ax,
        vmin=vmin, vmax=vmax,
        annot_kws={"fontsize": 10}
    )
    ax.tick_params(axis='x', rotation=0, labelsize=8.5)
    ax.set_title(model, fontsize=14)
    ax.set_xlabel("Prompt Setting", fontsize=12)
    if i == 0:
        ax.set_ylabel("Character Info", fontsize=12)
    else:
        ax.set_ylabel("")

cbar_ax = fig.add_axes([0.95, 0.14, 0.015, 0.75]) 
fig.colorbar(sm, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.95, 1])  
plt.savefig("model_prompt_heatmap_wide.png", dpi=300)
