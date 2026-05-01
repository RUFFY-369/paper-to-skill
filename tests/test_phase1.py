import os
import sys
import json
from parser_agent import ParserAgent

def test_phase_1():
    print("=== Phase 1: The Parser Subagent Validation Gate ===")
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    flawed_paper_path = os.path.join(workspace_dir, "dummy_paper_flawed.txt")
    fixed_paper_path = os.path.join(workspace_dir, "dummy_paper_fixed.txt")
    
    # Check that test files exist
    assert os.path.exists(flawed_paper_path), f"{flawed_paper_path} missing"
    assert os.path.exists(fixed_paper_path), f"{fixed_paper_path} missing"

    with open(flawed_paper_path, "r") as f:
        flawed_text = f.read()

    with open(fixed_paper_path, "r") as f:
        fixed_text = f.read()

    agent = ParserAgent()

    # Step 1: Run against the flawed paper extract
    print("\n[TEST 1] Processing dummy_paper_flawed.txt (Expected to fail)...")
    try:
        agent.extract_constraints(flawed_text)
        print("[ERROR] Agent unexpectedly succeeded on the flawed paper.")
        sys.exit(1)
    except ValueError as e:
        print(f"[SUCCESS] Agent correctly failed and requested clarification: {e}")

    # Step 2: Run against the fixed paper extract
    print("\n[TEST 2] Processing dummy_paper_fixed.txt (Expected to succeed)...")
    try:
        schema = agent.extract_constraints(fixed_text)
        print("[SUCCESS] Agent successfully output the perfect JSON schema:")
        print(json.dumps(schema, indent=2))
        
        # Verify specific keys
        mandatory_keys = ["input_tensors", "output_tensors", "hardware_requirements", "objective_function"]
        for key in mandatory_keys:
            assert key in schema, f"Missing key: {key}"
        
        print("\nAll schema checks passed successfully.")
    except Exception as e:
        print(f"[ERROR] Agent unexpectedly failed on the fixed paper: {e}")
        sys.exit(1)

    print("\n=== All Phase 1 Validation Gate Checks Passed! ===")

if __name__ == "__main__":
    test_phase_1()
