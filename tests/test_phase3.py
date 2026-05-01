import os
import sys
from synthesizer_agent import SynthesizerAgent

def test_phase_3():
    print("=== Phase 3: The Synthesizer Self-Correction Test ===")

    # Use Phase 1 schema
    schema = {
        "input_tensors": {
            "input_latent": {"shape": ["B", 4, 64, 64], "dtype": "float32"},
            "timestep": {"shape": ["B"], "dtype": "int64"}
        },
        "output_tensors": {
            "output_latent": {"shape": ["B", 4, 64, 64], "dtype": "float32"}
        }
    }

    agent = SynthesizerAgent()
    success, logs = agent.synthesize(schema, initial_flawed=True)

    print("\n--- Synthesis Final Logs ---")
    print(logs)

    assert success, "Synthesizer failed to self-correct and return success."
    print("\n[SUCCESS] Phase 3 Validation Gate Passed: Self-Correction Loop successfully recovered code!")

if __name__ == "__main__":
    test_phase_3()
