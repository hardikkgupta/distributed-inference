"""Ray Serve deployment graph wrapping vLLM's LLMServer."""

import argparse
import ray
from ray import serve
from ray.serve.llm import LLMServer

def create_app(model_name: str, gpu_per_replica: float = 1) -> serve.Deployment:
    llm_server = LLMServer.bind(model_name=model_name)

    @serve.deployment(
        num_replicas=1,
        ray_actor_options={"num_gpus": gpu_per_replica}
    )
    class Entrypoint:
        def __init__(self):
            self.backend = llm_server

        async def __call__(self, request):
            data = await request.json()
            prompt = data.get("prompt", "")
            return await self.backend.generate.remote(prompt)

    return Entrypoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Start Ray locally if not already inside a cluster
    if not ray.is_initialized():
        ray.init()

    serve.run(create_app(args.model).bind(), name="distributed-inference")
    print(f"🚀 HTTP server running on 0.0.0.0:{args.port}")
    serve.start(detached=False, http_options={"host": "0.0.0.0", "port": args.port})

if __name__ == "__main__":
    main()
