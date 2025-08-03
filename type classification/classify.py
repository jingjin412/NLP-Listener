import json
import torch
import opencc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, StoppingCriteria
from openai import OpenAI

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16
)

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence):
        self.eos_sequence = eos_sequence
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[0][-len(self.eos_sequence):].tolist()
        return last_ids == self.eos_sequence

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all_data():
    base_path = "/tmp2/b10204022/NLP lab/mpdd/"
    meta_data = load_json(f"{base_path}metadata.json")
    dialogue_data = load_json(f"{base_path}dialogue_chinese.json")
    whole_dialogue_data = load_json(f"{base_path}whole_dialogue_emotion.json")
    return meta_data, dialogue_data, whole_dialogue_data

converter1 = opencc.OpenCC('t2s') 
converter2 = opencc.OpenCC('s2t') 

def traditional_to_simplified(text):
    return converter1.convert(text)

def simplified_to_traditional(text):
    return converter2.convert(text)

def load_model(model_name="Qwen/Qwen2.5-32B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
sys = ("你是一个高效且精确的对话分析 AI，專門負責對話語進行分類。"
"你的任务是將拿到的句子分類，總共有A, B, C三種選擇。"
"A: 說話者明確叫出聆聽者的名字或綽號。 例如：「小左，你覺得這個怎麼樣？」、「麗華，今天你到底要我怎樣？」"
"B: 說話者沒有叫出聆聽者的名字或綽號，但有提及聆聽者的身份。 例如：「老婆子，你怎麼這樣想呢？」、「爸、媽，我愛你們」"
"C: 說話者沒有叫出聆聽者的名字或綽號，也沒有提及聆聽者的身份。 例如：「你還好嗎？」"
"你只需要回答 A, B, C 其中一個選項，不需要多餘的解釋。"
"回答格式：「答案：A」或「答案：B」或「答案：C」。"
)

def ask_qwen(model, tokenizer, prompt):
    content = traditional_to_simplified(sys)
    prompt = traditional_to_simplified(prompt)
    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    content = simplified_to_traditional(content)
    if "答案：" in content:
        answer = content.split("答案：", 1)[1].strip()
        return content, answer
    return content, ""

def write_results(filename, results):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    model, tokenizer = load_model()
    _, dialogue_data, whole_dialogue_data = load_all_data()

    final_results = []

    for idx, dialogue_id in enumerate(whole_dialogue_data):
        dialogue = whole_dialogue_data[dialogue_id]
        utterances = dialogue["each_utterance"]
        original_dialogue = dialogue_data[dialogue_id]

        all_char_set = set()
        for utt in original_dialogue:
            all_char_set.add(utt["speaker"])
            for listener in utt["listener"]:
                all_char_set.add(listener["name"])
        all_char = "、".join(all_char_set)

        dialogue_result = []

        for j, utt in enumerate(utterances):
            utterance_text = utt["utterance"]
            listener_names = [l["name"] for l in original_dialogue[j]["listener"]]
            gold = "、".join(listener_names)

            # 構建 prompt
            prompt = (
                f"你要判斷的句子是：{utterance_text}。\n"
                f"{utterance_text} 這句話的聆聽者是 {gold}。\n"
                f"{all_char} 是這個對話中所有人物的名字或綽號。"
            )

            # 詢問模型
            for attempt in range(2):
                full_response, answer = ask_qwen(model, tokenizer, prompt)
                if answer:
                    break

            dialogue_result.append({
                "utterance": utterance_text,
                "model_response": full_response,
                "predicted_label": answer
            })

        final_results.append(dialogue_result)

        if (idx + 1) % 10 == 0:
            write_results("classification_results.json", final_results)
            print(f"已儲存至第 {idx+1} 筆對話")

    write_results("classification_results.json", final_results)