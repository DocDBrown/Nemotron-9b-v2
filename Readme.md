# NVIDIA-Nemotron-Nano-9B-v2

```
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

```
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Give me one practical DevOps tip for running LLMs on edge devices.","max_new_tokens":128,"do_sample":false}' \
  | python3 -m json.tool
```


