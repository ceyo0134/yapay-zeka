import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
EPOCHS = float(os.environ.get("EPOCHS", "3"))
RANK = int(os.environ.get("RANK", "16"))
LR, MAXLEN = 2e-4, 256
SYS = "Sen Türkçe konuşan, yardımsever bir asistansın."

pairs, u = [], None
for line in open("veri/egitim.txt", encoding="utf-8").read().splitlines():
    line = line.strip()
    if line.startswith("Kullanıcı:"): u = line[len("Kullanıcı:"):].strip()
    elif line.startswith("AI:") and u:
        pairs.append((u, line[3:].strip())); u = None
print(len(pairs), "soru-cevap bulundu", flush=True)
assert len(pairs) >= 10, "veri/egitim.txt icinde en az 10 soru-cevap olmali"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2*RANK, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()

class DS(torch.utils.data.Dataset):
    def __init__(s):
        s.items = [tok(tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":q},{"role":"assistant","content":a}], tokenize=False), truncation=True, max_length=MAXLEN) for q, a in pairs]
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS, learning_rate=LR, logging_steps=5, save_strategy="no", report_to="none",
    use_cpu=True, lr_scheduler_type="cosine")
Trainer(model=model, args=args, train_dataset=DS(), data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

model.eval()
ids = tok(tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":"Merhaba, adın ne?"}], tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids
print("TEST:", tok.decode(model.generate(ids, max_new_tokens=60)[0][ids.shape[1]:], skip_special_tokens=True), flush=True)
model = model.merge_and_unload().to(torch.float16)
model.save_pretrained("merged"); tok.save_pretrained("merged")

# MLC eski config alanini (rope_theta) bekliyor, yeni transformers farkli yaziyor
import json
cp = "merged/config.json"; c = json.load(open(cp))
rp = c.get("rope_parameters") or c.get("rope_scaling") or {}
c.setdefault("rope_theta", rp.get("rope_theta", 1000000.0))
c["torch_dtype"] = "float16"; c["dtype"] = "float16"
json.dump(c, open(cp, "w"), indent=1)
print("config yamalandi, rope_theta =", c["rope_theta"], flush=True)
