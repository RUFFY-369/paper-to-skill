import os
import sys
import json
import time
from paper_to_skill.agents.parser_agent import ParserAgent
from paper_to_skill.core.direct_executor import run_script_directly

def execute_pipeline():
    print("=== SageAttention-2 End-to-End Pipeline Execution ===")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load paper text
    raw_paper_path = os.path.join(os.path.dirname(workspace_dir), "sageattention2_raw.txt")
    if not os.path.exists(raw_paper_path):
        print(f"Error: {raw_paper_path} not found.")
        sys.exit(1)

    with open(raw_paper_path, "r") as f:
        paper_text = f.read()

    # 2. Extract Constraints with ParserAgent
    print("\n--- Running Phase 1 (Parser) Agent Constraint Extraction ---")
    agent = ParserAgent()
    schema = agent.extract_constraints(paper_text)
    print("Extracted perfect JSON Schema:")
    print(json.dumps(schema, indent=2))

    # 3. Running evaluation via direct executor
    print("\n--- Running Phase 3 (Synthesizer) Direct Subprocess Evaluation ---")
    runner_path = os.path.join(workspace_dir, "sage2_runner.py")
    res = run_script_directly(
        script_path=runner_path,
        workspace_dir=workspace_dir
    )

    print(f"Direct Executor Result Stdout:\n{res.stdout}")
    if res.stderr:
        print(f"Direct Executor Result Stderr:\n{res.stderr}")

    # 4. Phase 4 - Supervisor/Routing Critic and Crystallization
    print("\n--- Phase 4: Counterfactual Fault Routing & Blame Analysis ---")
    if "EVAL_REWARD: 0" in res.stdout:
        print("[Supervisor] Success achieved on host GPU.")
        # Crystallize skill metadata file SKILL.md
        skill_md_path = os.path.join(workspace_dir, "SKILL.md")
        md_content = f"""# SKILL: SageAttention-2
## Description
An optimized, exact-attention mechanism utilizing outlier smoothing and per-thread/per-warp INT8/FP8 quantization on Ampere or newer GPUs.

## Level 2: API / Usage
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

## Schema
```json
{json.dumps(schema, indent=2)}
```

## Status
Successfully crystallized and tested against Atropos harness.
"""
        with open(skill_md_path, "w") as f:
            f.write(md_content)
        print(f"Skill crystallization complete: {skill_md_path}")

        # Validate with hermes SQLite FTS5 database schema
        from validate_skill_md import validate_skill_markdown
        validate_skill_markdown(skill_md_path)
    else:
        print("[Supervisor Critic] Fault Routing Analysis Result:")
        decision = {
            "blame": "SynthesizerAgent",
            "reason": "Execution or evaluation error occurred on the host GPU.",
            "reward": -1
        }
        print(json.dumps(decision, indent=2))

if __name__ == "__main__":
    execute_pipeline()
