"""Shared constants for Modal deployments."""

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
GPU_TYPE = "a10g"
VLLM_MAX_MODEL_LEN = 4096
VLLM_DTYPE = "half"
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# Cost tracking (for provider comparison report)
# Modal A10G: ~$0.000361/sec (~$1.30/hr)
MODAL_A10G_COST_PER_SEC = 0.000361
