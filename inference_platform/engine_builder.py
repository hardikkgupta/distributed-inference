"""Script to build TensorRT‑LLM engine from a HuggingFace checkpoint.

Uses nvidia tensorrt_llm python API. Requires GPUs with sufficient VRAM.

Example:
    python -m inference_platform.engine_builder --model meta-llama/Llama-3-8b-instruct --precision fp8
"""

import argparse
from tensorrt_llm import Builder

def build_engine(model_name: str, precision: str = "fp16", seq_len: int = 4096, out_dir: str = "engines"):
    builder = Builder(model_name)
    builder.precision(precision).seq_len(seq_len)
    builder.build(save_dir=out_dir)
    print(f"Engine saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", default="fp16", choices=["fp8", "fp16", "int8"], help="Numeric precision")
    args = parser.parse_args()
    build_engine(args.model, args.precision)

if __name__ == "__main__":
    main()
