import sys
import torch
from sage2_harness import AtroposRLVRTestHarness
from generated_sage2_skill import candidate_function

def run_eval():
    harness = AtroposRLVRTestHarness()
    reward, telemetry = harness.evaluate(candidate_function)
    print(f"EVAL_REWARD: {reward}")
    print(f"EVAL_TELEMETRY: {telemetry}")
    if reward != 0:
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
