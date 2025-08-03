import json

def calculate_f1_score(true_labels, pred_labels):
    precision, recall, f1 = 0, 0, 0
    exact_match = 0
    for true, pred in zip(true_labels, pred_labels):
        true_set, pred_set = set(true), set(pred)
        common = true_set & pred_set
        if pred_set:
            precision_tmp = len(common)
            total_pred = len(pred_set)
        if true_set:
            recall_tmp = len(common)
            total_true = len(true_set)
        if true_set == pred_set:
            exact_match += 1
        precision_tmp /= total_pred if total_pred > 0 else 1
        recall_tmp /= total_true if total_true > 0 else 1
        precision += precision_tmp
        recall += recall_tmp
        f1 += 2 * precision_tmp * recall_tmp / (precision_tmp + recall_tmp) if (precision_tmp + recall_tmp) > 0 else 0
    return precision / len(true_labels), recall / len(true_labels), f1 / len(true_labels), exact_match / len(true_labels)

info_list = ["noinfo", "position", "relation"]
name_list = ["name", "noname"]
case_list = ["pre", "whole"]


for case in case_list:
    for name in name_list:
        for info in info_list:
            print(f"Calculating F1 score for {name} {info} {case}...")
            with open(f"../results/R1_llama/relation_{name}_{info}_{case}.json", "r", encoding="utf-8") as f:
                results = json.load(f)

            true_labels = []
            pred_labels = []
            for i in results:
                if i[0] == "less than 3 people":
                    continue
                else:
                    for j in i:
                        true_labels.append(j["listener ans"].split(","))
                        pred_labels.append(j["listener"].split(","))
                        
            precision, recall, f1, exact_match = calculate_f1_score(true_labels, pred_labels)
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Exact Match: {exact_match:.4f}")
