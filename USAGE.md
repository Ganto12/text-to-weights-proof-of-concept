# Description in, LoRA out

1. Write one sentence that describes a task.
2. The meta-model turns it into a rank-8 update for all 186 projection matrices of Qwen3.5-2B (12 projection types in each of 24 layers).
3. `export_peft.py` saves it as a standard PEFT LoRA adapter: `adapter_model.safetensors` (16.9 MB) and `adapter_config.json` (r 8, lora_alpha 16).
4. Load it like any other LoRA, or merge it for zero extra inference cost.

```bash
python scripts/export_peft.py --meta <meta-model checkpoint>.pt --description "<one sentence task description>" --out peft_export/adapter
```

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(base, "peft_export/adapter")  # no custom code
model = model.merge_and_unload()                                 # same cost as the plain base model
```

## Timing (RTX 3090, base model and meta-model already in memory)

| Step | Time |
|---|---|
| Description → update | 0.21 s |
| Update → adapter files | 0.04 s |
| Load adapter with peft | 1.56 s |
| Merge into the weights | 0.06 s |
| **Description → merged model** | **1.87 s** |
