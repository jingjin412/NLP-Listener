import yaml
import json
import torch
import opencc
import argparse
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

relation_to_position = {
    "父母": "長輩", "岳父/岳母": "長輩", "祖父母": "長輩", "上級": "長輩", "老師": "長輩", "老闆": "長輩",
    "孩子": "晚輩", "女婿/媳婦": "晚輩", "孫子/孫女": "晚輩", "下級": "晚輩", "學生": "晚輩", "下屬": "晚輩"
}

relation_to_field = {
    "老師": "學校", "學生": "學校", "同學": "學校",
    "老闆": "公司", "同事": "公司", "下屬": "公司", "夥伴": "公司",
    "父母": "家庭", "岳父/岳母": "家庭", "祖父母": "家庭", "上級": "家庭",
    "配偶": "家庭", "兄弟姐妹": "家庭", "同儕": "家庭", "孩子": "家庭",
    "女婿/媳婦": "家庭", "孫子/孫女": "家庭", "下級": "家庭"
}

relation_to_chinese = {
    "parent": "父母","parent-in-law": "岳父/岳母","grandparent": "祖父母","other superior": "上級",
    "spouse": "配偶","brothers and sisters": "兄弟姐妹","other peer": "同儕","child": "孩子",
    "son/daughter-in-law": "女婿/媳婦","grandchild": "孫子/孫女","other inferior": "下級","teacher": "老師",
    "classmate": "同學","student": "學生","boss": "老闆","colleague": "同事","subordinate": "下屬",
    "partner": "夥伴","couple": "情侶","friend": "朋友","enemy": "敵人","consignee": "受托人",
    "consignor": "委託者","stranger": "陌生人","unknown": "未知" 
}

def parse_args():
    parser = argparse.ArgumentParser(description="Addressee prediction with prompt-based LLM.")
    parser.add_argument("--output_file", type=str, required=True, help="The output file name for results.")
    parser.add_argument("--prompt_mode", type=str, required=True, help="The type of prompt to use.")
    parser.add_argument("--start", type=int, required=True, help="Start index for processing dialogues.")
    parser.add_argument("--anonymity", type=str, choices=["anon", "name"], required=True, help="Use anonymized names or real names.")
    parser.add_argument("--info_level", type=str, choices=["none", "position", "relation"], required=True, help="Level of relationship information to include.")
    parser.add_argument("--context_mode", type=str, choices=["whole", "pre"], required=True, help="Context type: whole dialogue or prefix only")
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompts(prompt_file="prompt.yaml"):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_all_data():
    base_path = "mpdd/"
    meta_data = load_json(f"{base_path}metadata.json")
    dialogue_data = load_json(f"{base_path}dialogue_chinese.json")
    whole_dialogue_data = load_json(f"{base_path}whole_dialogue_prepro.json")
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
    # llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    # return model, tokenizer, llm

def ask_qwen(model, tokenizer, prompt):
    prompt = traditional_to_simplified(prompt)
    messages = [
        {"role": "system", "content": "你是一个高效且精确的对话分析 AI，专门负责判断对话中的 聆听者（即说话者的对象）。除了按照格式回答问题以外，你不会说出多余的话。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    content = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    content = simplified_to_traditional(content)
    if "答案：" in content:
        answer = content.split("答案：", 1)[1].strip()
        return content, answer
    return content, ""

def ask_LLaMA(llm, tokenizer, prompt):
    prompt_s = traditional_to_simplified(prompt)

    chat = [
        {"role": "system", "content": "你是一個高效且精確的對話分析 AI，專門負責判斷對話中的 聆聽者（即說話者的對象）。"},
        {"role": "user", "content": prompt_s}
    ]
    
    flatten_chat_for_generation = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # calculate the length of the input
    # tokenized_input = tokenizer(flatten_chat_for_generation, return_tensors="pt", add_special_tokens=False)
    
    output = llm(
        flatten_chat_for_generation, 
        return_full_text=False, 
        max_new_tokens=1024, 
        stopping_criteria=[EosListStoppingCriteria([tokenizer.eos_token_id])]
    )
        
    generated_text = output[0]['generated_text']

    response_trad = simplified_to_traditional(generated_text)

    if "答案：" in response_trad:
        answer = response_trad.split("答案：", 1)[1].strip()
    else:
        answer = ""

    return response_trad, answer
    
client = OpenAI(api_key = "") # TODO: set your OpenAI API key here
def ask_GPT(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一個高效且精確的對話分析 AI，專門負責判斷對話中的 聆聽者（即說話者的對象）。"},
            {"role": "user",
             "content": [
                {"type": "text", "text": f"{prompt}"}
             ]
            }
        ],
        temperature=0.1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    generated_text = response.choices[0].message.content

    response_trad = simplified_to_traditional(generated_text)
    if "答案：" in response_trad:
        answer = response_trad.split("答案：", 1)[1].strip()
    else:
        answer = ""

    return response_trad, answer

def calculate_f1_score(true_labels, pred_labels):
    precision, recall, f1 = 0, 0, 0
    total_pred, total_true = 0, 0
    for true, pred in zip(true_labels, pred_labels):
        true_set, pred_set = set(true), set(pred)
        common = true_set & pred_set
        if pred_set:
            precision += len(common)
            total_pred += len(pred_set)
        if true_set:
            recall += len(common)
            total_true += len(true_set)
    precision /= total_pred
    recall /= total_true
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1