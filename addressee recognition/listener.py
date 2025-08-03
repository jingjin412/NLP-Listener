import json

from utils import (
    parse_args, load_all_data, load_prompts, load_model, ask_LLaMA, ask_GPT,
    ask_qwen, calculate_f1_score, relation_to_chinese, relation_to_position, relation_to_field
)

def process_characters(dialogue, anonymity, info_level):
    all_char_set = set()
    relation_set = set()
    char_id = {}

    for utterance in dialogue:
        speaker = utterance["speaker"]
        if speaker not in all_char_set:
            char_id[speaker] = f"P{len(all_char_set)+1}"
            all_char_set.add(speaker)

        for listener in utterance["listener"]:
            listener_name = listener["name"]
            if listener_name not in all_char_set:
                char_id[listener_name] = f"P{len(all_char_set)+1}"
                all_char_set.add(listener_name)

            relation_chinese = relation_to_chinese.get(listener["relation"], "未知")
            if info_level == "relation":
                relation = f"{speaker} 是 {listener_name} 的 {relation_chinese}"
            elif info_level == "position":
                position = relation_to_position.get(relation_chinese, "平輩")
                relation = f"{speaker} 是 {listener_name} 的 {position}"
            else:
                continue
            relation_set.add(" - " + relation)

    if anonymity == "anon":
        all_char = ", ".join(char_id[k] for k in all_char_set)
        relation_text = "\n".join(
            rel.replace(speaker, char_id.get(speaker, speaker)).replace(listener_name, char_id.get(listener_name, listener_name))
            for rel in relation_set
        )
    else:
        all_char = ", ".join(all_char_set)
        relation_text = "\n".join(relation_set)

    return char_id, all_char, relation_text

def build_context(dialogue_data, char_id, context_mode, whole_dialogue_data, current_utterance_index):
    dialogue = ""
    realdialogue = ""

    if context_mode == "whole":
        realdialogue = whole_dialogue_data['real_whole_dialogue']
        for j in range(len(whole_dialogue_data['each_utterance'])):
            dialogue += f'{char_id[dialogue_data[j]["speaker"]]}: {dialogue_data[j]["utterance"]}\n'
    else: 
        for j in range(len(whole_dialogue_data['each_utterance'])):
            if j > current_utterance_index:
                break
            dialogue += f'{char_id[dialogue_data[j]["speaker"]]}: {dialogue_data[j]["utterance"]}\n'
            realdialogue += f'{dialogue_data[j]["speaker"]}: {dialogue_data[j]["utterance"]}\n'
    return dialogue, realdialogue

def build_prompt(template, utt, dialogue_str, realdialogue_str, relation_text, char_id, info_level, all_char, anonymity):
    prompt = template
    speaker = utt["real speaker"]
    utterance = utt["utterance"]
    relation = utt["relation"]
    field = relation_to_field.get(relation, "非學校、公司、家庭的地方")
    position = relation_to_position.get(relation, "平輩")

    person_id = char_id.get(speaker, speaker) if anonymity == "anon" else speaker

    prompt = prompt.replace("<dialogue>", dialogue_str)
    prompt = prompt.replace("<realdialogue>", realdialogue_str)
    prompt = prompt.replace("<realperson>", speaker)
    prompt = prompt.replace("<person>", person_id)
    prompt = prompt.replace("<utterance>", utterance)
    prompt = prompt.replace("<relation>", relation)
    prompt = prompt.replace("<position>", position)
    prompt = prompt.replace("<field>", field)
    prompt = prompt.replace("<relation_list>", relation_text)
    prompt = prompt.replace("<all_char>", all_char)
    prompt = prompt.replace("<all_char_noname>", all_char)
    prompt = prompt.replace("<relation_list_noname>", relation_text)
    return prompt

def write_results(filename, results):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    index = 0
    args = parse_args()
    # TODO: Choose the model you want to use
    model, tokenizer = load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    # model, tokenizer = load_model("Qwen/Qwen2.5-32B-Instruct")
    # model, tokenizer, llm = load_model("meta-llama/Llama-3.1-8B-Instruct")
    prompts = load_prompts()

    meta_data, dialogue_data, whole_dialogue_data = load_all_data()
    final_results = []
    pred_labels, true_labels = [], []

    for i in whole_dialogue_data:
        index += 1
        if index <= args.start:
            continue
        
        whole_dialogue = whole_dialogue_data[i]

        char_map, all_char, relation_info = process_characters(dialogue_data[i], args.anonymity, args.info_level)

        if len(char_map) < 3:
            final_results.append(["less than 3 people"])
            continue

        
        
        utterance_results = []
        for j, utt in enumerate(whole_dialogue["each_utterance"]):
            dialogue_str, realdialogue_str = build_context(dialogue_data[i], char_map, args.context_mode, whole_dialogue, j)
            prompt_template = prompts[args.prompt_mode]
            prompt = build_prompt(prompt_template, utt, dialogue_str, realdialogue_str, relation_info, char_map,
                                  args.info_level, all_char, args.anonymity)
            # print(prompt)

            # TODO: Choose the model's function you want to use
            for attempt in range(2):
                content, prediction = ask_qwen(model, tokenizer, prompt)
                # content, prediction = ask_LLaMA(llm, tokenizer, prompt)
                # content, prediction = ask_GPT(prompt)
                if prediction:
                    break

            if args.anonymity == "anon":
                gold = ",".join(char_map[listener["name"]] for listener in dialogue_data[i][j]["listener"])
            else:
                gold = ",".join(listener["name"] for listener in dialogue_data[i][j]["listener"])

            pred_labels.append(prediction.split(","))
            true_labels.append(gold.split(","))

            utterance_results.append({
                "content": content,
                "listener": prediction,
                "listener ans": gold
            })

        

        final_results.append(utterance_results)
        # break

        if (index+1) % 20 == 0:
            write_results(args.output_file, final_results)
            

    print(len(pred_labels), len(true_labels))
    print(calculate_f1_score(true_labels, pred_labels))
    write_results(args.output_file, final_results)

if __name__ == "__main__":
    main()
