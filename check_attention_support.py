import argparse
import torch
from transformers import AutoModelForCausalLM


def check_attention_implementation(model_id):
    """
    Check if a model can be loaded with different attention implementations.

    Args:
        model_id: HuggingFace model ID or local path
    """
    print(f"Testing model: {model_id}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("NO CUDA AVAILABLE")
        return
    print("-" * 80)

    device = torch.device("cuda")

    # Test Flash Attention 2
    print("\n[1/2] Testing Flash Attention 2")
    try:
        model_fa2 = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("✓ Flash Attention 2: SUCCESS")
        del model_fa2
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Flash Attention 2: FAILED")
        print(f"  Error: {str(e)}")

    # Test SDPA (Scaled Dot Product Attention)
    print("\n[2/2] Testing SDPA")
    try:
        model_sdpa = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("✓ SDPA: SUCCESS")
        del model_sdpa
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ SDPA: FAILED")
        print(f"  Error: {str(e)}")

    print("\n" + "-" * 80)
    print("Test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if a model supports Flash Attention 2 and SDPA"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID or local path (default: microsoft/phi-2)",
    )

    args = parser.parse_args()
    check_attention_implementation(args.model_id)
