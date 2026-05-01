# SKILL: SageAttention-2
## Description
An optimized, exact-attention mechanism utilizing outlier smoothing and per-thread/per-warp INT8/FP8 quantization on Ampere or newer GPUs.

## Level 2: API / Usage
### Usage Signature
```python
import torch
from generated_sage2_skill import candidate_function

inputs = {
    "Q": torch.ones((1, 1024, 8, 64), dtype=torch.float16),
    "K": torch.ones((1, 1024, 8, 64), dtype=torch.float16),
    "V": torch.ones((1, 1024, 8, 64), dtype=torch.float16)
}

output = candidate_function(**inputs)
print(f"Computed output shape: {output.shape}")
```

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
    "cuda_capability": "Ampere SM86 or newer",
    "min_vram": "24GB",
    "framework": "PyTorch 2.4+ with Triton 3.0, CUDA 12.x",
    "quantization": {
      "qk_precision": "INT8 per-warp",
      "pv_precision": "FP8 E4M3",
      "accumulator": "FP32 two-level"
    }
  },
  "objective_function": "O = softmax(smooth(Q) * smooth(K)^T / sqrt(d)) * V with INT8 QK and FP8 PV quantization"
}
```

## Status
Successfully crystallized and tested against Atropos harness.
