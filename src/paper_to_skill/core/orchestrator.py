import os
import time
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional
import pynvml

@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    peak_vram_mb: float
    oom_triggered: bool

class VRAMMonitor(threading.Thread):
    def __init__(self, gpu_id: int = 0, interval: float = 0.05):
        super().__init__()
        self.gpu_id = gpu_id
        self.interval = interval
        self.peak_vram = 0.0
        self._stop_event = threading.Event()

    def run(self):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            while not self._stop_event.is_set():
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    val = info.used / (1024 ** 2)
                    if val > self.peak_vram:
                        self.peak_vram = val
                except Exception:
                    pass
                time.sleep(self.interval)
        except Exception:
            # Fallback to subprocess in case of unexpected NVML initialization failure
            while not self._stop_event.is_set():
                try:
                    res = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                            f"--id={self.gpu_id}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    val = float(res.stdout.strip().split("\n")[0])
                    if val > self.peak_vram:
                        self.peak_vram = val
                except Exception:
                    pass
                time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def run_script_in_sandbox(
    script_path: str,
    workspace_dir: str,
    image_name: str = "synthesizer-base:latest",
    memory_limit: str = "4g",
    cpu_limit: str = "2.0",
    timeout_seconds: int = 120,
    gpu_id: int = 0,
) -> SandboxResult:
    """
    Executes a script in a sandboxed Docker container with strict CPU, RAM, and time limits.
    Monitors peak VRAM usage and catches CUDA Out Of Memory conditions using pynvml.
    """
    if not os.path.isabs(script_path):
        script_path = os.path.abspath(script_path)
    if not os.path.isabs(workspace_dir):
        workspace_dir = os.path.abspath(workspace_dir)

    # Ensure the script is inside the workspace_dir
    rel_script_path = os.path.relpath(script_path, workspace_dir)

    container_name = f"synthesizer_sandbox_{int(time.time())}_{os.getpid()}"

    # Record baseline VRAM before launching
    baseline_vram = 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        baseline_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
    except Exception:
        pass

    # Set up VRAM monitoring
    monitor = VRAMMonitor(gpu_id=gpu_id, interval=0.05)
    monitor.start()

    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = -1

    env_file_path = os.path.join(os.path.dirname(workspace_dir), ".env")
    cmd = [
        "docker", "run",
        "--name", container_name,
        "--device", "/dev/nvidia0",
        "--device", "/dev/nvidiactl",
        "--device", "/dev/nvidia-uvm",
        "-v", "/home/ruffy-369/NousResearch/driver_libs:/usr/lib/host-libs",
        "-v", f"{workspace_dir}:/workspace",
        "-e", "LD_LIBRARY_PATH=/usr/lib/host-libs",
    ]
    if os.path.exists(env_file_path):
        cmd.extend(["--env-file", env_file_path])
    cmd.extend([
        "--memory", memory_limit,
        "--cpus", cpu_limit,
        "--rm",
        image_name,
        "python", f"/workspace/{rel_script_path}"
    ])


    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        # Force kill container
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        # Force remove container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        exit_code = 124
    except Exception as e:
        stderr = str(e)
        exit_code = 1
    finally:
        # Give monitor a moment to poll any final peak VRAM
        time.sleep(0.1)
        monitor.stop()
        monitor.join()

        # Production Guard: Verify VRAM returned to baseline, clear zombie processes if stranded
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            current_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
            if current_vram > baseline_vram + 100:  # Allow 100MB margin
                print(f"[Orchestrator Guard] Alert: Stranded VRAM detected. Baseline {baseline_vram:.1f}MB, Current {current_vram:.1f}MB. Clearing orphaned contexts...")
                # Kill compute processes on device directly
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    try:
                        os.kill(p.pid, 9)
                    except Exception:
                        pass
        except Exception:
            pass

    # Determine if OOM occurred
    oom_triggered = (
        "OutOfMemoryError" in stderr
        or "OutOfMemoryError" in stdout
        or "out of memory" in stderr.lower()
        or "out of memory" in stdout.lower()
        or exit_code == 137
    )

    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        peak_vram_mb=monitor.peak_vram,
        oom_triggered=oom_triggered,
    )
