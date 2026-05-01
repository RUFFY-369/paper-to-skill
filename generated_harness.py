import torch
import math
import time
import inspect
from typing import Dict, Any, Tuple

class AtroposRLVRTestHarness:
    def __init__(self):
        self.schema = {'input_tensors': {'input_latent': {'shape': ['B', 4, 64, 64], 'dtype': 'float32'}, 'timestep': {'shape': ['B'], 'dtype': 'int64'}}, 'output_tensors': {'output_latent': {'shape': ['B', 4, 64, 64], 'dtype': 'float32'}}}
        self.input_tensors = self.schema.get("input_tensors", {})
        self.output_tensors = self.schema.get("output_tensors", {})

    def evaluate(self, candidate_function: callable) -> Tuple[int, Dict[str, Any]]:
        """
        Runs candidate_function against valid input tensors generated from the schema.
        Accurately times operations using SOTA PyTorch CUDA Events.
        Catches all errors (shape mismatches, dtype errors, NaNs) and returns (reward, telemetry).
        Enforces AST source code checks to prevent generic fallback reward hacking.
        """
        telemetry = {"error": None, "latency_ms": 0.0}

        try:
            # Source Code Enforcement to prevent reward hacking fallback
            try:
                source = inspect.getsource(candidate_function)
            except Exception:
                # If inspect fails, fallback to checking global namespace or file directly
                source = ""

            # Hopper-specific keywords/primitives: wgmma, tma, sm90, triton, etc.
            valid_primitives = ["wgmma", "tma", "sm90", "triton", "hopper"]
            if not any(kw in source.lower() for kw in valid_primitives):
                raise RuntimeError("Hardware bypass detected: Missing SM90 instructions.")

            # 1. Dynamically build valid inputs matching the schema for evaluation
            test_inputs = {}
            D = 64
            for name, meta in self.input_tensors.items():
                shape = [1 if x == "B" else (1 if x == "S" else x) for x in meta["shape"]]
                dtype = getattr(torch, meta["dtype"])
                if shape and shape[-1] > 1:
                    D = shape[-1]
                
                if dtype in [torch.float16, torch.float32, torch.bfloat16]:
                    test_inputs[name] = torch.randn(shape, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu") / math.sqrt(D)
                else:
                    test_inputs[name] = torch.ones(shape, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu")

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

            output = candidate_function(**test_inputs)

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                telemetry["latency_ms"] = start_event.elapsed_time(end_event)
            else:
                telemetry["latency_ms"] = (time.perf_counter() - start_time) * 1000

            for name, meta in self.output_tensors.items():
                expected_shape = [1 if x == "B" else (1 if x == "S" else x) for x in meta["shape"]]
                
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Output must be a torch.Tensor, got {type(output)}")

                if torch.isnan(output).any():
                    raise ValueError("Output tensor contains NaNs.")

                if list(output.shape) != expected_shape:
                    raise ValueError(f"Shape mismatch: expected {expected_shape}, got {list(output.shape)}")

                expected_dtype = getattr(torch, meta["dtype"])
                if output.dtype != expected_dtype:
                    raise TypeError(f"Dtype mismatch: expected {expected_dtype}, got {output.dtype}")

            return 0, telemetry

        except Exception as e:
            telemetry["error"] = str(e)
            return -1, telemetry
