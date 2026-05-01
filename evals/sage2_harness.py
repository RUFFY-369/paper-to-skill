import torch
import time
import inspect
import math
from typing import Dict, Any, Tuple

class AtroposRLVRTestHarness:
    def __init__(self):
        self.schema = {
            "input_tensors": {
                "Q": {"shape": ["B", "S", "H", "D"], "dtype": "float16"},
                "K": {"shape": ["B", "S", "H", "D"], "dtype": "float16"},
                "V": {"shape": ["B", "S", "H", "D"], "dtype": "float16"}
            },
            "output_tensors": {
                "O": {"shape": ["B", "S", "H", "D"], "dtype": "float16"}
            },
            "hardware_requirements": {
                "cuda_capability": "Ampere SM86 or newer",
                "min_vram": "24GB",
                "framework": "PyTorch 2.4+ with Triton 3.0, CUDA 12.x",
                "quantization": {
                    "qk_precision": "INT8 per-warp",
                    "pv_precision": "FP8 E4M3",
                    "accumulator": "FP32 two-level"
                }
            },
            "objective_function": "O = softmax(smooth(Q) * smooth(K)^T / sqrt(d)) * V with INT8 QK and FP8 PV quantization"
        }

    def evaluate(self, candidate_function: callable) -> Tuple[int, Dict[str, Any]]:
        """
        Runs candidate_function against valid input tensors generated from the schema.
        Accurately profiles execution using PyTorch SOTA CUDA Events.
        Validates the output tensor against the baseline exact-attention implementation.
        Enforces source code inspection to prevent generic fallback reward hacking.
        Returns exact reward values (0 for Success, -1 for Failure).
        """
        telemetry = {"error": None, "latency_ms": 0.0, "rmse": 0.0}

        try:
            # Source Code Enforcement to prevent fallback reward hacking
            try:
                source = inspect.getsource(candidate_function)
            except Exception:
                source = ""

            valid_primitives = ["wgmma", "tma", "sm90", "triton", "hopper", "tl.", "quantize", "int8", "fp8"]
            if not any(kw in source.lower() for kw in valid_primitives):
                raise RuntimeError("Hardware bypass detected: Missing GPU/Triton primitives.")

            # 1. Setup dimensions for testing SageAttention-2
            B, S, H, D = 2, 1024, 8, 64
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            # 2. Create test inputs using realistic distributions
            Q = torch.randn(B, S, H, D, dtype=dtype, device=device)
            K = torch.randn(B, S, H, D, dtype=dtype, device=device)
            V = torch.randn(B, S, H, D, dtype=dtype, device=device)

            # 3. Standard PyTorch exact-attention baseline to compute numerical parity
            scale = 1.0 / (D ** 0.5)
            # Permute to [B, H, S, D]
            Q_p = Q.transpose(1, 2)
            K_p = K.transpose(1, 2)
            V_p = V.transpose(1, 2)

            scores = torch.matmul(Q_p, K_p.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype)
            baseline_O_p = torch.matmul(attn_weights, V_p)
            # Permute back to [B, S, H, D]
            baseline_O = baseline_O_p.transpose(1, 2)

            # 4. Call generated function with accurate timing
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                # Warmup
                for _ in range(3):
                    _ = candidate_function(Q, K, V)
                torch.cuda.synchronize()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output_O = candidate_function(Q, K, V)
                end_event.record()
                torch.cuda.synchronize()
                telemetry["latency_ms"] = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                output_O = candidate_function(Q, K, V)
                telemetry["latency_ms"] = (time.perf_counter() - start_time) * 1000

            if not isinstance(output_O, torch.Tensor):
                raise TypeError(f"Output must be a torch.Tensor, got {type(output_O)}")

            # 5. Verify numerical parity (RMSE)
            rmse = torch.sqrt(torch.mean((output_O.float() - baseline_O.float()) ** 2))
            telemetry["rmse"] = float(rmse)

            if rmse > 5e-2:  # Tolerable threshold for INT8/FP8 quantized attention
                raise ValueError(f"Numerical parity check failed: RMSE is too high ({rmse:.6f}).")

            return 0, telemetry

        except Exception as e:
            telemetry["error"] = str(e)
            return -1, telemetry
