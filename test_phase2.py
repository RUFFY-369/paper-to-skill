import os
import sys
from test_engineer_agent import TestEngineerAgent
from orchestrator import run_script_in_sandbox

def test_phase_2():
    print("=== Phase 2 Validation Gate: Scaffold Evaluation via Sandbox ===")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))

    res = run_script_in_sandbox(
        script_path=os.path.join(workspace_dir, "run_harness_test.py"),
        workspace_dir=workspace_dir
    )

    print(f"Sandbox Stdout:\n{res.stdout}")
    if res.stderr:
        print(f"Sandbox Stderr:\n{res.stderr}")

    assert "All sandbox tests passed successfully!" in res.stdout, "Sandbox tests did not succeed."
    print("\n[SUCCESS] Phase 2 Validation Passed!")

if __name__ == "__main__":
    test_phase_2()
