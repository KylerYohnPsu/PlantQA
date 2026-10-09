from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
import bitsandbytes
import torch

SYSTEM_PROMPT = ("You are a plant expert. You are to use the referenced chunks provided in the user input"
                " to help users with their questions. If the chunks are not relevant, do not use them. Do not invent facts")
class ResponseModel:
    def __init__(self, model_name = 'Qwen/Qwen2.5-3B-Instruct', lora = None, adapter = None):
        self.model_name = model_name
        self.model = self.build_model(model_name, lora = lora, adapter = adapter)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_answer(self, question, chunks, max_tokens = 220, visual_predictions = None):
        prompt = self.build_prompt(question, chunks, visual_predictions)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens = max_tokens)
        completion = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    def build_model(self, model_name = "Qwen/Qwen2.5-7B-Instruct", lora = None, adapter = None):
        # source for later https://huggingface.co/blog/4bit-transformers-bitsandbytes & huggingface.co/docs/transformers/quantization/bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            torch_dtype="auto",
            device_map="auto"
        )
        if adapter:
            return PeftModel.from_pretrained(model, adapter)
        if lora:
            model = prepare_model_for_kbit_training(model)
            return get_peft_model(model, LoraConfig(
                r=16, lora_alpha=32, task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        return model

    def build_prompt(self, question, chunks, visual_predictions = None):
        context = "\n".join(c["body"][:400] for c in chunks) if chunks else "(none)"
        prompt = (f"plant question:\nquestion:{question}"
                  f"\nretrieved context:\n{context}"
                  f"\nvisual model's predictions:\n{visual_predictions or '(none)'}")
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)

    def build_training_data(self, data, retriever = None):
        training_data = []
        for row in data.itertuples():
            chunks = retriever.retrieve(row.question_text) if retriever else None
            prompt = self.build_prompt(question=row.question_text, chunks=chunks, visual_predictions=f"{row.crop}, {row.disease}, {row.severity}")

            prompt_ids = self.tokenizer(prompt, truncation= True, max_len=1024)["input_ids"]
            answer_ids = self.tokenizer(row.answer, truncation= True, max_len=1024)["input_ids"]
            training_data.append({"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids})
        return training_data

    def train_model(self, data, retriever=None, epochs=1, num_results=None,
                    out="./models/response_models/qwen_lora"):
        training_data = self.build_training_data(data, retriever)
        Trainer(
            model=self.model,
            args=TrainingArguments(
                out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                num_train_epochs=epochs, learning_rate=2e-4, bf16=True,
                gradient_checkpointing=True, optim="paged_adamw_8bit", report_to=[]
            ),
            train_dataset=training_data,
            data_collator=lambda b: {k: torch.tensor([v]) for k, v in b[0].items()},
        ).train()

        self.save_model(out)

    def save_model(self, path="./models/response_models/qwen_lora"):
        self.model.save_pretrained(path)

    def load_model(self, path="./models/response_models/qwen_lora"):
        self.model = PeftModel.from_pretrained(self.model, path)
        return self.model


"""
    @misc{qwen2.5,
        title = {Qwen2.5: A Party of Foundation Models},
        url = {https://qwenlm.github.io/blog/qwen2.5/},
        author = {Qwen Team},
        month = {September},
        year = {2024}
    }

    @article{qwen2,
          title={Qwen2 Technical Report},
          author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
          journal={arXiv preprint arXiv:2407.10671},
          year={2024}
    }
    

"""