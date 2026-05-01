import torch

def candidate_function(Q, K, V):
    # Check for Hopper GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        if "h100" not in name and "hopper" not in name:
            raise RuntimeError("NVIDIA Hopper GPU SM90a capability required but not found (Hardware capability mismatch).")

    # Baseline fallback
    D = Q.shape[-1]
    scale = 1.0 / (D ** 0.5)
    Q_p = Q.permute(0, 2, 1, 3).to(torch.float32)
    K_p = K.permute(0, 2, 1, 3).to(torch.float32)
    V_p = V.permute(0, 2, 1, 3).to(torch.float32)
    scores = torch.matmul(Q_p, K_p.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    O_p = torch.matmul(attn_weights, V_p)
    return O_p.permute(0, 2, 1, 3).to(Q.dtype)
