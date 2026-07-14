/*
 * Baseline dynamic priority heuristic for C maximum-code construction.
 *
 * The fixed search skeleton lives in c_search_skeleton.c. OpenEvolve-generated
 * variants should only change the EVOLVE-BLOCK inside oe_max_code_priority().
 */

#include <stdint.h>

double oe_max_code_priority(
    uint64_t word_mask,
    int n,
    int d,
    int step,
    int word_weight,
    uint64_t forbidden_count,
    uint64_t new_forbidden_count,
    uint64_t overlap_forbidden_count,
    uint64_t local_available_count
) {
    /* # EVOLVE-BLOCK-START */
    int weight = word_weight;
    if (weight <= 0) {
        uint64_t value = word_mask;
        while (value) {
            value &= value - 1ULL;
            weight++;
        }
    }

    int target_weight = d > n / 2 ? d : n / 2;
    int weight_delta = weight - target_weight;
    if (weight_delta < 0) {
        weight_delta = -weight_delta;
    }

    int left = 0;
    int right = 0;
    for (int bit_index = 0; bit_index < n / 2; bit_index++) {
        if ((word_mask >> bit_index) & 1ULL) {
            left++;
        }
        if ((word_mask >> (n - 1 - bit_index)) & 1ULL) {
            right++;
        }
    }
    int balance_delta = left - right;
    if (balance_delta < 0) {
        balance_delta = -balance_delta;
    }

    int runs = 1;
    int previous = (int)(word_mask & 1ULL);
    for (int bit_index = 1; bit_index < n; bit_index++) {
        int bit = (int)((word_mask >> bit_index) & 1ULL);
        if (bit != previous) {
            runs++;
        }
        previous = bit;
    }

    double endpoint_bonus = 0.0;
    if (word_mask & 1ULL) {
        endpoint_bonus += 0.25;
    }
    if ((word_mask >> (n - 1)) & 1ULL) {
        endpoint_bonus += 0.25;
    }

    uint64_t damage_total = new_forbidden_count + overlap_forbidden_count;
    double damage_score = 0.0;
    double overlap_score = 0.0;
    if (damage_total > 0ULL) {
        damage_score = -((double)new_forbidden_count / (double)damage_total);
        overlap_score = (double)overlap_forbidden_count / (double)damage_total;
    }
    double local_score = (double)local_available_count / (double)(n > 0 ? n * n : 1);
    double phase = (double)step / (double)(1U << (n < 12 ? n : 12));
    double pressure = 0.0;
    if (n > 0 && n < 63) {
        pressure = (double)forbidden_count / (double)(1ULL << n);
    }

    return 7.0 * damage_score
        + 1.5 * overlap_score
        + 0.8 * local_score
        - 2.0 * (double)weight_delta
        + 0.25 * (double)runs
        - 0.5 * (double)balance_delta
        + endpoint_bonus
        - 0.05 * phase * pressure * (double)weight_delta;
    /* # EVOLVE-BLOCK-END */
}
