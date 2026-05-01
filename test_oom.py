import torch
import time

def main():
    print("Starting intensive GPU memory allocation loop in test_oom.py...")
    tensors = []
    try:
        while True:
            # Continuously allocate massive tensors (e.g., 25000 x 25000 floats ~= 2.5 GB)
            # This rapidly and reliably triggers a fatal CUDA OOM exception.
            tensors.append(torch.empty((25000, 25000), dtype=torch.float32, device="cuda"))
            print(f"Allocated tensor. Total accumulated: {len(tensors)}")
            time.sleep(0.1)
    except RuntimeError as e:
        print(f"Caught expected CUDA RuntimeError inside test_oom.py: {e}")
        raise e

if __name__ == "__main__":
    main()
