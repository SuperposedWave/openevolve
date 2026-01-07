# EVOLVE-BLOCK-START
"""
Constructor-based binary linear code for a fixed (n, k).
Goal: maximize minimum Hamming distance d.
"""
import numpy as np


def construct_generator():
    """
    Construct a binary generator matrix G for a fixed (n, k).
    """
    # -------------------------------------------------
    # FIXED TARGET (edit once, evolve only for this pair)
    # -------------------------------------------------
    n = 35
    k = 10

    assert k <= n

    # Systematic form: G = [I_k | P]
    G = np.zeros((k, n), dtype=np.uint8)
    G[:, :k] = np.eye(k, dtype=np.uint8)

    # Parity part P
    m = n - k
    P = np.zeros((k, m), dtype=np.uint8)

    # Simple structured baseline (evolution will improve this)
    for j in range(m):
        for t in [
            (j * 3 + 1) % k,
            (j * 5 + 2) % k,
            (j * 7 + 3) % k,
        ]:
            P[t, j] ^= 1

        # add a short window
        w = 3 + (j % 3)
        start = (j * 2) % k
        for r in range(w):
            P[(start + r) % k, j] ^= 1

    G[:, k:] = P

    # Light row-mixing (preserves code, helps exploration)
    for i in range(1, k, 2):
        G[i] ^= G[i - 1]

    return G


# EVOLVE-BLOCK-END


def run_code():
    """
    Fixed interface called by evaluator.
    """
    return construct_generator()


if __name__ == "__main__":
    G = run_code()
    print("G shape:", G.shape)
