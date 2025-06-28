# Distributed Inference Platform

A **minimal, reproducible** distributed‑LLM serving stack that focuses on *one thing*: **low‑latency generation at scale** using [Ray Serve] and [vLLM].


* **Ray Serve deployment graphs** for request routing.
* `scripts/build_trt_engine.py` stub that really builds engines with TensorRT‑LLM.
* Simplified Dockerfile and `requirements.txt`.
* Added Prometheus + Grafana example configs for observability.

## Quick start (single node, 1 GPU)

```bash
pip install -r requirements.txt
python -m inference_platform.serve --model meta-llama/Llama-3-8b-instruct
curl -X POST localhost:8000/generate -d '{"prompt":"Hello"}'
```

See `docs/README.md` for multi‑GPU and Kubernetes guides.
## Repository layout

```
inference_platform/
    serve.py             # Ray Serve deployment graph wrapping vLLM LLMServer
    engine_builder.py    # Convert HF checkpoint ➜ TensorRT‑LLM engine
k8s/
    rayserve-deployment.yaml
grafana/
    dashboards.json
scripts/
    build_trt_engine.py
Dockerfile
requirements.txt
```

[Ray Serve]: https://docs.ray.io/en/latest/serve/index.html
[vLLM]: https://github.com/vllm-project/vllm
