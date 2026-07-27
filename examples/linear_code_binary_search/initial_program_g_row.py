"""Baseline priority for G-row incremental legality search."""


# EVOLVE-BLOCK-START
def _mix01(value):
    value = int(value) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    return value / float(1 << 64)


def priority(
    row_mask,
    n,
    k,
    d,
    step=0,
    row_weight=0,
    min_margin=0,
    near_margin_count=0,
    checked_constraints=0,
    legal_seen=0,
    attempts=0,
    selected_count=0,
    total_layer_count=0,
    max_layer_count=0,
    restart=0,
):
    """
    Score one legal r-bit row x for P in G=[I_k|P].

    The fixed evaluator has already checked exact incremental legality:
    every previous subset xor a of size s satisfies (s+1)+wt(a^x) >= d.
    This function only ranks legal candidates in the sampled pool.
    """
    r = n - k
    weight = int(row_weight) if row_weight else int(row_mask).bit_count()
    target_weight = max(float(d - 1), float(r) * 0.5)
    weight_balance = -abs(float(weight) - target_weight)
    near_rate = float(near_margin_count) / max(1.0, float(checked_constraints))
    fill_phase = float(step) / max(1.0, float(k))
    scarcity = float(attempts) / max(1.0, float(attempts + 256))
    novelty = _mix01(int(row_mask) + 1009 * int(step) + 9176 * int(restart))

    return (
        200.0 * float(min_margin)
        - 80.0 * near_rate
        + 1.2 * weight_balance
        - 0.15 * fill_phase * abs(float(weight) - target_weight)
        + 0.3 * scarcity
        + 0.01 * novelty
        - 0.000000001 * float(total_layer_count)
        - 0.000000001 * float(max_layer_count)
    )


# EVOLVE-BLOCK-END


def get_priority_function():
    """Stable entry point for evaluator and tests."""
    return priority


if __name__ == "__main__":
    from evaluator_g_row import evaluate

    result = evaluate(__file__)
    print("Metrics:", result.metrics)
    print("Artifacts:", result.artifacts)
