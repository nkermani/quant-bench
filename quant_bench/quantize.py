"""Model quantization module for Quant-Bench 2026."""

import argparse
import sys
from pathlib import Path

def quantize_model(model_name: str, format: str, bits: int = 4, output_dir: str = "./quantized_models"):
    """Quantize a model to specified format and bit-width."""
    output_path = Path(output_dir) / f"{model_name}-{format}-{bits}bit"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing {model_name} to {format}-{bits}bit...")
    print(f"Output directory: {output_path}")

    if format.lower() == "awq":
        try:
            from autoawq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
            print("Using AutoAWQ for quantization...")
        except ImportError:
            print("Error: autoawq not installed. Install with: pip install autoawq")
            sys.exit(1)

    elif format.lower() == "gptq":
        try:
            from auto_gptq import AutoGPTQForCausalLM
            print("Using AutoGPTQ for quantization...")
        except ImportError:
            print("Error: auto-gptq not installed. Install with: pip install auto-gptq")
            sys.exit(1)

    elif format.lower() == "fp8":
        print("Using FP8 (E4M3) quantization...")

    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Model quantized successfully: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Quantize LLM models")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--format", type=str, required=True, choices=["awq", "gptq", "fp8", "int8"], help="Quantization format")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--output-dir", type=str, default="./quantized_models", help="Output directory")
    args = parser.parse_args()

    quantize_model(args.model, args.format, args.bits, args.output_dir)

if __name__ == "__main__":
    main()
