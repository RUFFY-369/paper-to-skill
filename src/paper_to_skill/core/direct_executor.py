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


def run_script_directly(
    script_path: str,
    workspace_dir: str,
    timeout_seconds: int = 120,
    gpu_id: int = 0,
) -> SandboxResult:
    """
    Executes a script directly (without Docker) via a subprocess.
    Still captures VRAM metrics, timeout, and OOM.
    """
    if not os.path.isabs(script_path):
        script_path = os.path.abspath(script_path)
    if not os.path.isabs(workspace_dir):
        workspace_dir = os.path.abspath(workspace_dir)

    # Hard-stop security guardrail requiring authorization for unsafe host execution
    if os.environ.get("ALLOW_DANGER_RUN_BARE_METAL", "").lower() != "true":
        raise PermissionError(
            "Security Guardrail blocked execution: Unsafe host execution requires setting "
            "the ALLOW_DANGER_RUN_BARE_METAL=true environment variable."
        )

    # Validate code content before direct execution
    from paper_to_skill.core.restricted_sandbox import validate_script_safety
    if os.path.exists(script_path):
        with open(script_path, "r") as f:
            code_str = f.read()
        validate_script_safety(code_str)



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

    cmd = ["python3", script_path]

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        exit_code = 124
    except Exception as e:
        stderr = str(e)
        exit_code = 1
    finally:
        # Give monitor a moment to poll any final peak VRAM
        time.sleep(0.1)
        monitor.stop()
        monitor.join()

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
