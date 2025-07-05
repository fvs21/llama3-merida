from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os
from functools import lru_cache

PAD_TOKEN = "<|pad|>"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL = os.environ.get("NEW_MODEL")
MODEL_REPO = os.environ.get("NEW_MODEL_REPO")

@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if torch.cuda.is_available() else {"": "mps" if torch.mps.is_available() else "cpu"},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    model = PeftModel.from_pretrained(model, MODEL_REPO)

    return Llama3Merida(model, tokenizer)


class Llama3Merida:
    SYSTEM_PROMPT = "Only answer questions about Mérida, Yucatán or participate in friendly conversation."

    def __init__(self, model: PeftModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "assistant" in decoded:
            return decoded.split("assistant")[-1].strip()
        else:
            return decoded.split()