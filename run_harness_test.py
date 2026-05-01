import os
import sys
import torch
from test_engineer_agent import TestEngineerAgent

def run_in_sandbox():
    print("=== Inside Sandbox: Evaluating Generated Test Harness Scaffold ===")
    schema = {
        "input_tensors": {
            "input_latent": {"shape": ["B", 4, 64, 64], "dtype": "float32"},
            "timestep": {"shape": ["B"], "dtype": "int64"}
        },
        "output_tensors": {
            "output_latent": {"shape": ["B", 4, 64, 64], "dtype": "float32"}
        }
    }

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(workspace_dir, "generated_harness.py")

    agent = TestEngineerAgent()
    agent.generate_harness(schema, output_path)
    print(f"Harness generated at: {output_path}")

    sys.path.insert(0, workspace_dir)
    from generated_harness import AtroposRLVRTestHarness
    harness = AtroposRLVRTestHarness()

    print("\n[TEST 1] Passing function that returns garbage shape [1, 1]...")
    def garbage_func(**kwargs):
        # uses wgmma, sm90 to test
        return torch.zeros((1, 1), dtype=torch.float32)

    reward, telemetry = harness.evaluate(garbage_func)
    print(f"Reward: {reward}")
    print(f"Telemetry: {telemetry}")
    assert reward == -1

    print("\n[TEST 2] Passing function returning correct shape [1, 4, 64, 64]...")
    def good_func(**kwargs):
        # uses wgmma, sm90 to test
        return torch.ones((1, 4, 64, 64), dtype=torch.float32)

    good_reward, good_telemetry = harness.evaluate(good_func)
    print(f"Reward: {good_reward}")
    print(f"Telemetry: {good_telemetry}")
    assert good_reward == 0

    print("\n[SUCCESS] All sandbox tests passed successfully!")

if __name__ == "__main__":
    run_in_sandbox()
