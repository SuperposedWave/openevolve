/*
 * Baseline static priority heuristic for C linear-code construction.
 *
 * The fixed search skeleton lives in c_search_skeleton.c. OpenEvolve-generated
 * variants should only change the EVOLVE-BLOCK inside oe_linear_code_priority().
 */

#include <stdint.h>

double oe_linear_code_priority(
    uint64_t column_mask,
    int n,
    int k,
    int d,
    int step,
    int column_weight,
    uint64_t forbidden_count,
    uint64_t new_forbidden_count,
    uint64_t overlap_forbidden_count
) {
    /* # EVOLVE-BLOCK-START */
    int r = n - k;
    int weight = column_weight;
    if (weight <= 0) {
        uint64_t value = column_mask;
        while (value) {
            value &= value - 1ULL;
            weight++;
        }
    }

    int target_weight = d - 1;
    int half_plus = r / 2 + 1;
    if (half_plus > target_weight) {
        target_weight = half_plus;
    }

    int left = 0;
    int right = 0;
    for (int i = 0; i < r / 2; i++) {
        if ((column_mask >> i) & 1ULL) {
            left++;
        }
        if ((column_mask >> (r - 1 - i)) & 1ULL) {
            right++;
        }
    }

    int runs = 1;
    int previous = (int)(column_mask & 1ULL);
    for (int i = 1; i < r; i++) {
        int bit = (int)((column_mask >> i) & 1ULL);
        if (bit != previous) {
            runs++;
        }
        previous = bit;
    }

    double endpoint_bonus = 0.0;
    if (column_mask & 1ULL) {
        endpoint_bonus += 0.5;
    }
    if ((column_mask >> (r - 1)) & 1ULL) {
        endpoint_bonus += 0.5;
    }

    int weight_delta = weight - target_weight;
    if (weight_delta < 0) {
        weight_delta = -weight_delta;
    }
    int symmetry_delta = left - right;
    if (symmetry_delta < 0) {
        symmetry_delta = -symmetry_delta;
    }

    double weight_score = -(double)weight_delta;
    double symmetry_score = -(double)symmetry_delta;
    uint64_t damage_total = new_forbidden_count + overlap_forbidden_count;
    double damage_score = 0.0;
    double overlap_score = 0.0;
    if (damage_total > 0ULL) {
        damage_score = -((double)new_forbidden_count / (double)damage_total);
        overlap_score = (double)overlap_forbidden_count / (double)damage_total;
    }
    double fill_phase = (double)step / (double)(k > 0 ? k : 1);
    double space_pressure = 0.0;
    if (r > 0 && r < 63) {
        double universe = (double)(1ULL << r);
        space_pressure = (double)forbidden_count / universe;
    }

    return 1.2 * damage_score
        + 0.4 * overlap_score
        + 2.5 * weight_score
        + 0.8 * symmetry_score
        + 0.35 * (double)runs
        + endpoint_bonus
        - 0.1 * fill_phase * space_pressure;
    /* # EVOLVE-BLOCK-END */
}
