import os
import sys
import json
from parser_agent import ParserAgent

def test_real_paper():
    print("=== FlashAttention-3 Stress Test Integration ===")

    # Path to flashattention3.md
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(workspace_dir, "flashattention3.md")
    
    assert os.path.exists(md_path), f"File {md_path} was not found."

    with open(md_path, "r") as f:
        md_text = f.read()

    # Enforce prompt structure (append extraction instructions and schema at the VERY END)
    prompt_suffix = """

---
EXTRACTION INSTRUCTIONS AND SCHEMA DEFINITION:
Please extract theoretical constraints from the paper text above and strictly output ONLY a JSON schema containing the following exact keys:
'input_tensors', 'output_tensors', 'hardware_requirements', and 'objective_function'.
You are strictly forbidden from hallucinating missing dimensions.
"""

    full_context = md_text + prompt_suffix

    # Instantiate ParserAgent
    agent = ParserAgent()

    # Execute extraction
    schema = agent.extract_constraints(full_context)

    print("\n[SUCCESS] ParserAgent successfully generated the JSON schema for the real paper:")
    print(json.dumps(schema, indent=2))

if __name__ == "__main__":
    test_real_paper()
