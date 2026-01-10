import threading
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

app = FastAPI(title="Nemotron Nano v2 Inference")

_model = None
_tokenizer = None
_lock = threading.Lock()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7

def _cuda_status():
    if not torch.cuda.is_available():
        return {"cuda": False}
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "free_gib": round(free_b / 1024**3, 2),
        "total_gib": round(total_b / 1024**3, 2),
    }

def _load_once():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )

    # Prefer 8-bit on 16GB devices if bitsandbytes is usable.
    use_bnb = False
    try:
        import bitsandbytes  # noqa: F401
        use_bnb = True
    except Exception:
        use_bnb = False

    if use_bnb:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16

    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    _model.eval()

@app.get("/health")
def health():
    return {"ok": True, **_cuda_status()}

@app.post("/generate")
def generate(req: GenerateRequest):
    with _lock:
        _load_once()

        messages = [
            {"role": "system", "content": "/think"},
            {"role": "user", "content": req.prompt},
        ]

        inputs = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if hasattr(_model, "device") and isinstance(_model.device, torch.device):
            inputs = inputs.to(_model.device)

        gen_kwargs = dict(
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            eos_token_id=_tokenizer.eos_token_id,
        )
        if req.do_sample:
            gen_kwargs["temperature"] = req.temperature

        with torch.inference_mode():
            out = _model.generate(inputs, **gen_kwargs)

        text = _tokenizer.decode(out[0], skip_special_tokens=False)
        return {"text": text}
