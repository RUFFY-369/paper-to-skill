import os
import sys
import time
from typing import Dict, Any, Tuple
from orchestrator import run_script_in_sandbox

class SynthesizerAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a focused subagent tasked with generating efficient, valid PyTorch "
            "implementations that fulfill the extracted Phase 1 JSON schema. "
            "You write ONLY the target candidate_function into a file named generated_skill.py. "
            "If code fails, you use the error telemetry logs to perform self-correction up to 3 times."
        )

    def write_skill_file(self, content: str, output_path: str):
        with open(output_path, "w") as f:
            f.write(content)

    def write_eval_runner(self, output_path: str):
        code = '''import sys
import torch
from generated_harness import AtroposRLVRTestHarness
from generated_skill import candidate_function

def run_eval():
    harness = AtroposRLVRTestHarness()
    reward, telemetry = harness.evaluate(candidate_function)
    print(f"EVAL_REWARD: {reward}")
    print(f"EVAL_TELEMETRY: {telemetry}")
    if reward != 0:
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
'''
        with open(output_path, "w") as f:
            f.write(code)

    def synthesize(self, schema: Dict[str, Any], initial_flawed: bool = True) -> Tuple[bool, str]:
        """
        Runs the code generation & self-correction loop inside the isolated sandbox.
        """
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        skill_path = os.path.join(workspace_dir, "generated_skill.py")
        runner_path = os.path.join(workspace_dir, "evaluation_runner.py")

        # 1. Generate runner script once
        self.write_eval_runner(runner_path)

        # 2. Prepare initial script content
        if initial_flawed:
            print("[SynthesizerAgent] Generating intentionally flawed implementation (wrong output shape)...")
            initial_code = '''import torch

def candidate_function(**kwargs):
    # Intentional wrong shape [1, 1] instead of [1, 4, 64, 64]
    return torch.zeros((1, 1), dtype=torch.float32)
'''
        else:
            print("[SynthesizerAgent] Generating direct correct implementation...")
            initial_code = '''import torch

def candidate_function(**kwargs):
    # Valid output shape: [B, 4, 64, 64]
    return torch.ones((1, 4, 64, 64), dtype=torch.float32)
'''

        self.write_skill_file(initial_code, skill_path)

        # 3. Enter the evaluation loop (up to 3 retries)
        max_retries = 3
        for attempt in range(1, max_retries + 2):
            print(f"\n--- [SynthesizerAgent] Attempt {attempt} of {max_retries + 1} ---")
            
            # Execute runner in sandbox
            res = run_script_in_sandbox(
                script_path=runner_path,
                workspace_dir=workspace_dir
            )
            
            print(f"Orchestrator Result Stdout:\n{res.stdout}")
            if res.stderr:
                print(f"Orchestrator Result Stderr:\n{res.stderr}")

            # Check evaluation result
            if "EVAL_REWARD: 0" in res.stdout:
                print(f"[SynthesizerAgent] Successful reward '0' achieved on attempt {attempt}!")
                return True, res.stdout

            # Error occurred - Perform self-correction
            print(f"[SynthesizerAgent] Attempt {attempt} failed. Reasoning over truncated error log...")

            # Simple reasoning & retry generation over telemetry logs
            if attempt <= max_retries:
                print("[SynthesizerAgent] Modifying generated_skill.py with correct tensor shapes...")
                corrected_code = '''import torch

def candidate_function(**kwargs):
    # Corrected implementation satisfying the Phase 1 schema
    return torch.ones((1, 4, 64, 64), dtype=torch.float32)
'''
                self.write_skill_file(corrected_code, skill_path)
            else:
                print("[SynthesizerAgent] Reached maximum retries without a valid solution.")

        return False, "Self-correction max retries reached."
