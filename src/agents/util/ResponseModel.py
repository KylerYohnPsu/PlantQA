from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)
import torch
SYSTEM_PROMPT = ("You are a plant expert. You are to use the referenced chunks provided in the user input"
                 "to help users with their questions. If the chunks are not relevant, do not use them. Do not invent facts")
class ResponseModel:
    def __init__(self, model_name = 'Qwen/Qwen2.5-7B-Instruct'):
        self.model_name = model_name
        self.model = self.build_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_answer(self, question, chunks):
        prompt = self.build_prompt(question, chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs["input_ids"], max_length = 200)
        completion = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    def build_model(self, model_name = "Qwen/Qwen2.5-7B-Instruct"):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def build_prompt(self, question, chunks):
        prompt = f"plant question:\nquestion:{question}\nretrieved context:\n{chunks}"
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": "\n\n".join(prompt)}],
            tokenize=False, add_generation_prompt=True)


    def train_model(self, data, retriever=None, epochs=1, num_results=None,
                    out="./models/response_models/qwen_lora"):

        rows = data.head(num_results) if num_results else data
        training_data = []
        for row in rows.itertuples():
            chunks = retriever.retrieve(row.question_text) if retriever else None
            prompt = self.build_prompt(question= row.question_text, chunks=chunks)
            full = self.tokenizer(prompt + row.answer + self.tokenizer.eos_token)["input_ids"]
            n = len(self.tokenizer(prompt)["input_ids"])
            training_data.append({"input_ids": full,
                             "labels": [-100] * n + full[n:]})

        Trainer(
            model=self.model,
            args=TrainingArguments(
                out, per_device_train_batch_size=1, gradient_accumulation_steps=16,
                num_train_epochs=epochs, learning_rate=2e-4, bf16=True,
                gradient_checkpointing=True, optim="paged_adamw_8bit", report_to=[]),
            train_dataset=training_data,
            data_collator=lambda b: {k: torch.tensor([v]) for k, v in b[0].items()},
        ).train()

        self.save_model(out)

    def save_model(self, path="./models/response_models/qwen_lora"):
        self.model.save_pretrained(path)

    def load_model(self, path="./models/response_models/qwen_lora"):
        return self.build_model(adapter=path)


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