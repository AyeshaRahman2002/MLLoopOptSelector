# MLLoopOptSelector/src/features.py
from typing import Dict, List
import math

KERNELS: List[str] = ["matmul", "conv1d", "stencil2d"]
KERNEL_TO_DEPTH = {"matmul": 3, "conv1d": 2, "stencil2d": 2}

def arithmetic_intensity(kernel: str, sizes: Dict[str, int]) -> float:
    if kernel == "matmul":
        N, M, K = sizes["N"], sizes["M"], sizes["K"]
        flops = 2.0 * N * M * K
        bytes_acc = 4.0 * (N*K + K*M + N*M)
        return flops / max(bytes_acc, 1.0)
    elif kernel == "conv1d":
        N, K = sizes["N"], sizes["K"]
        O = max(1, N - K + 1)
        flops = 2.0 * O * K
        bytes_acc = 4.0 * (N + K + O)
        return flops / max(bytes_acc, 1.0)
    elif kernel == "stencil2d":
        N, M = sizes["N"], sizes["M"]
        flops = 7.0 * max(0, (N-2)) * max(0, (M-2))
        bytes_acc = 4.0 * (2*N*M)
        return flops / max(bytes_acc, 1.0)
    else:
        raise ValueError("unknown kernel")

def _safe_log(x: float) -> float:
    return math.log(max(1.0, float(x)))

def _tile_ratio(n: int, t: int) -> float:
    return float(n) / float(max(1, t))

def _tile_misalignment(n: int, t: int) -> float:
    return (n % max(1, t)) / float(max(1, t))

def build_feature_row(kernel: str, sizes: Dict[str, int], choice: str,
                      tile_i: int, tile_j: int, tile_k: int) -> Dict[str, float]:
    ai = arithmetic_intensity(kernel, sizes)
    depth = KERNEL_TO_DEPTH[kernel]
    kvec = [int(kernel == k) for k in KERNELS]

    N = sizes.get("N", 0)
    M = sizes.get("M", 0)
    K = sizes.get("K", 0)

    # conv output length
    O = max(1, N - K + 1) if kernel == "conv1d" else (N if kernel == "stencil2d" else N)

    logN = _safe_log(N); logM = _safe_log(M); logK = _safe_log(K); logO = _safe_log(O)

    ti_r = _tile_ratio(N, tile_i)
    tj_r = _tile_ratio(M or N, tile_j)
    tk_r = _tile_ratio(K or N, tile_k)

    ti_m = _tile_misalignment(N, tile_i)
    tj_m = _tile_misalignment(M or N, tile_j)
    tk_m = _tile_misalignment(K or N, tile_k)

    conv_ratio = (K / max(1.0, N)) if kernel == "conv1d" else 0.0
    o_over_n   = (O / max(1.0, N)) if N else 0.0  # NEW

    return {
        "kernel_matmul": kvec[0],
        "kernel_conv1d": kvec[1],
        "kernel_stencil2d": kvec[2],
        "loop_depth": depth,
        "ai": ai,
        "N": N, "M": M, "K": K,
        "logN": logN, "logM": logM, "logK": logK,
        "O": O, "logO": logO, "ratio_O_over_N": o_over_n,
        "tile_i": tile_i, "tile_j": tile_j, "tile_k": tile_k,
        "ratio_i": ti_r, "ratio_j": tj_r, "ratio_k": tk_r,
        "mis_i": ti_m, "mis_j": tj_m, "mis_k": tk_m,
        "conv_ratio_K_over_N": conv_ratio,
        "choice": choice,
    }

FEATURE_COLUMNS: List[str] = [
    "kernel_matmul","kernel_conv1d","kernel_stencil2d",
    "loop_depth","ai",
    "N","M","K","O","logN","logM","logK","logO","ratio_O_over_N",
    "tile_i","tile_j","tile_k",
    "ratio_i","ratio_j","ratio_k",
    "mis_i","mis_j","mis_k",
    "conv_ratio_K_over_N",
]

