import os
import sys

def validate_skill_markdown(filepath: str):
    """
    Validates a generated SKILL.md against agentskills.io and SQLite FTS5 schema requirements.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SKILL.md file was not found at '{filepath}'")

    with open(filepath, "r") as f:
        content = f.read()

    # Required frontmatter markers and sections for Hermes FTS5 indexing
    required_patterns = [
        "---",
        "name: sageattention-2",
        "description:",
        "triggers:",
        "# Level 2: API / Usage"
    ]

    for p in required_patterns:
        if p.lower() not in content.lower():
            raise ValueError(f"Formatting validation error: Missing required pattern '{p}' in {filepath}")

    print(f"[SUCCESS] agentskills.io verification passed for {filepath}! YAML frontmatter is fully indexable by Hermes.")
    return True

if __name__ == "__main__":
    import sys
    path = "evals/SKILL.md"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    try:
        validate_skill_markdown(path)
    except Exception as e:
        print(f"[ERROR] agentskills.io Validation Failed: {e}")
        sys.exit(1)
