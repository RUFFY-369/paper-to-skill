# SKILL: FlashAttention-3
## Description
An highly optimized, exact-attention mechanism utilizing asynchronous execution on Hopper GPUs.

## Schema
```json
{
  "input_tensors": {
    "Q": {
      "shape": [
        "B",
        "S",
        "H",
        "D"
      ],
      "dtype": "float16"
    },
    "K": {
      "shape": [
        "B",
        "S",
        "H",
        "D"
      ],
      "dtype": "float16"
    },
    "V": {
      "shape": [
        "B",
        "S",
        "H",
        "D"
      ],
      "dtype": "float16"
    }
  },
  "output_tensors": {
    "O": {
      "shape": [
        "B",
        "S",
        "H",
        "D"
      ],
      "dtype": "float16"
    }
  },
  "hardware_requirements": {
    "cuda_capability": "Hopper SM90a or newer",
    "min_vram": "80GB",
    "framework": "PyTorch 2.3+ with CUDA 12.x",
    "sram_tiling_blocks": {
      "Br": 64,
      "Bc": 128
    }
  },
  "objective_function": "Block-wise exact attention using SRAM tiling"
}
```

## Status
Successfully crystallized and tested against Atropos harness.
