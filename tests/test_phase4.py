import os
import sys
import json
from supervisor_agent import SupervisorAgent

def test_phase_4():
    print("=== Phase 4: Hindsight Counterfactual Routing & Skill Freezing Test ===")

    supervisor = SupervisorAgent()

    # 1. Test A: The Fault Router
    print("\n[TEST A] Executing Fault Router against impossible constraint schema...")
    impossible_schema = {
        "input_tensors": {
            "input_latent": {"shape": [-1, 0], "dtype": "float32"}
        },
        "output_tensors": {
            "output_latent": {"shape": [1, 4, 64, 64], "dtype": "float32"}
        }
    }

    harness_code = "# Phase 2 dummy harness code"
    failed_code = "# Phase 3 dummy failed code"

    blame_report = supervisor.route_blame(impossible_schema, harness_code, failed_code)
    print("Fault Router Critic Decision Output:")
    print(json.dumps(blame_report, indent=2))

    assert blame_report["blame"] == "ParserAgent", f"Expected blame to be ParserAgent, got {blame_report['blame']}"
    print("[SUCCESS] Fault Router correctly identified impossible constraint and blamed ParserAgent.")

    # 2. Test B: The Freezer
    print("\n[TEST B] Executing Skill Freezing on valid code and schema...")
    valid_schema = {
        "input_tensors": {
            "input_latent": {"shape": ["B", 4, 64, 64], "dtype": "float32"},
            "timestep": {"shape": ["B"], "dtype": "int64"}
        },
        "output_tensors": {
            "output_latent": {"shape": ["B", 4, 64, 64], "dtype": "float32"}
        },
        "hardware_requirements": {
            "cuda_capability": "Ampere or newer",
            "min_vram": "12GB",
            "framework": "TensorRT 10.x"
        }
    }

    valid_code = '''import torch

def candidate_function(**kwargs):
    # Success implementation returning valid shape
    return torch.ones((1, 4, 64, 64), dtype=torch.float32)
'''

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    output_md_path = os.path.join(workspace_dir, "SKILL.md")

    supervisor.crystallize_skill(valid_code, valid_schema, output_md_path)

    print(f"Successfully froze skill metadata into: {output_md_path}")
    
    # Assert SKILL.md was generated and has Levels 1, 2, and 3
    assert os.path.exists(output_md_path), "SKILL.md file was not generated."
    with open(output_md_path, "r") as f:
        md_content = f.read()

    assert "Level 1: Description" in md_content, "Missing Level 1 metadata"
    assert "Level 2: API / Usage" in md_content, "Missing Level 2 metadata"
    assert "Level 3: Hardware / VRAM Constraints" in md_content, "Missing Level 3 metadata"

    print("[SUCCESS] Freezer correctly formatted SKILL.md with Levels 1, 2, and 3!")

    print("\n=== All Phase 4 Validation Gate Checks Passed! ===")

if __name__ == "__main__":
    test_phase_4()
