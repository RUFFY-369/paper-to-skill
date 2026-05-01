import os
import subprocess
import sys
from orchestrator import run_script_in_sandbox

def test_phase_0():
    print("=== Phase 0: Infrastructure & Ephemeral Sandboxing Test ===")
    
    # Step 1: Build the docker image
    print("Building base CUDA-enabled Docker image...")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    dockerfile_path = os.path.join(workspace_dir, "Dockerfile")
    
    build_cmd = [
        "docker", "build",
        "-t", "synthesizer-base:latest",
        "-f", dockerfile_path,
        workspace_dir
    ]
    print(f"Executing: {' '.join(build_cmd)}")
    build_proc = subprocess.run(build_cmd, capture_output=True, text=True)
    if build_proc.returncode != 0:
        print("Failed to build Docker image.")
        print("Stdout:", build_proc.stdout)
        print("Stderr:", build_proc.stderr)
        sys.exit(1)
    print("Successfully built synthesizer-base:latest")

    # Step 2: Run the dummy script that triggers CUDA OOM
    print("\nExecuting OOM script in the sandbox orchestrator...")
    dummy_script = os.path.join(workspace_dir, "test_oom.py")
    
    result = run_script_in_sandbox(
        script_path=dummy_script,
        workspace_dir=workspace_dir,
        image_name="synthesizer-base:latest",
        memory_limit="4g",
        cpu_limit="2.0",
        timeout_seconds=120
    )
    
    print("\n--- Execution Summary ---")
    print(f"Exit Code: {result.exit_code}")
    print(f"Timed out: {result.timed_out}")
    print(f"Peak VRAM captured (MB): {result.peak_vram_mb}")
    print(f"OOM Triggered detected by orchestrator: {result.oom_triggered}")
    
    print("\n--- Full Stdout ---")
    print(result.stdout)
    print("\n--- Full Stderr ---")
    print(result.stderr)
    
    # Validate result
    if result.oom_triggered:
        print("\n[SUCCESS] Sandbox caught CUDA OOM gracefully without crashing the host!")
    else:
        print("\n[ERROR] Sandbox failed to detect OOM condition.")
        sys.exit(1)

if __name__ == "__main__":
    test_phase_0()
