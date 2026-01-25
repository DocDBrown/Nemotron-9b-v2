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

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # Prefer 8-bit if bitsandbytes is available.
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        pass

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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": req.prompt},
        ]

        inputs = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        gen_kwargs = dict(
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature if req.do_sample else None,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )
        # remove None temperature when not sampling
        if not req.do_sample:
            gen_kwargs.pop("temperature", None)

        with torch.inference_mode():
            out = _model.generate(inputs, **gen_kwargs)

        # Return only newly generated tokens (not the prompt)
        new_tokens = out[0][inputs.shape[-1]:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"text": text}
