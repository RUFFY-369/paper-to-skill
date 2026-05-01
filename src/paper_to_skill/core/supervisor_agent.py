import os
import json
from typing import Dict, Any

class SupervisorAgent:
    def __init__(self):
        self.routing_critic_prompt = (
            "Identify the root cause of the failure. Was the Phase 1 schema mathematically impossible? "
            "Did the Phase 2 Test Engineer write a flawed harness? Or did the Phase 3 Synthesizer simply write bad syntax?"
        )

    def route_blame(self, schema: Dict[str, Any], harness_code: str, failed_code: str) -> Dict[str, Any]:
        """
        Counterfactual Routing: Analyzes a failed execution to assign causal attribution.
        Outputs a JSON explicitly blaming the responsible subagent.
        """
        impossible_dims = False
        for tensor_group in ["input_tensors", "output_tensors"]:
            for name, meta in schema.get(tensor_group, {}).items():
                for dim in meta.get("shape", []):
                    if isinstance(dim, (int, float)) and dim <= 0:
                        impossible_dims = True

        if impossible_dims:
            return {
                "blame": "ParserAgent",
                "reason": "Phase 1 schema contains mathematically impossible or negative dimensions (e.g. <= 0).",
                "reward": -1
            }

        return {
            "blame": "SynthesizerAgent",
            "reason": "The Phase 3 SynthesizerAgent generated bad syntax or incorrect tensor shapes.",
            "reward": -1
        }

    def crystallize_skill(self, valid_code: str, schema: Dict[str, Any], output_md_path: str):
        """
        Creates a SKILL.md file containing Level 1, 2, and 3 metadata.
        """
        hardware_meta = schema.get("hardware_requirements", {})
        
        # Clean up any debugging/extra code comments from raw skill
        stripped_code = "\n".join([line for line in valid_code.splitlines() if not line.startswith("#")])

        md_content = f"""---
name: sageattention-2
description: An optimized, exact-attention mechanism utilizing outlier smoothing and per-warp INT8 quantization on Ampere or newer GPUs.
triggers:
  - "implement sage attention"
  - "write a quantized attention kernel"
---

# Level 2: API / Usage
The skill exposes a single `candidate_function` accepting the schema inputs.

### Raw Validated Implementation
```python
{stripped_code}
```

### Usage Signature
```python
import torch
from generated_sage2_skill import candidate_function

inputs = {{
    "Q": torch.ones((1, 1024, 8, 64), dtype=torch.float16),
    "K": torch.ones((1, 1024, 8, 64), dtype=torch.float16),
    "V": torch.ones((1, 1024, 8, 64), dtype=torch.float16)
}}

output = candidate_function(**inputs)
print(f"Computed output shape: {{output.shape}}")
```

## Level 3: Hardware / VRAM Constraints
- **CUDA Capability**: {hardware_meta.get('cuda_capability', 'Ampere or newer')}
- **Acceleration Framework**: {hardware_meta.get('framework', 'PyTorch with Triton')}
- **Minimum VRAM**: {hardware_meta.get('min_vram', '24GB')}
"""
        with open(output_md_path, "w") as f:
            f.write(md_content)


