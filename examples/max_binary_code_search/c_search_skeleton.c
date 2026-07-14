#define _POSIX_C_SOURCE 200809L

/*
 * Fixed C search skeleton for binary maximum-code search.
 *
 * Evolved C variants provide only oe_max_code_priority(). This file owns
 * candidate enumeration, sorting, exact forbidden-ball legality, dynamic
 * damage features, bounded MCTS repair, metrics, and selected-word output.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define METRIC_CODE_SIZE 0
#define METRIC_VALID 1
#define METRIC_CANDIDATE_COUNT 2
#define METRIC_SCORED_CANDIDATES 3
#define METRIC_REPAIR_ROLLOUT_EVALUATIONS 4
#define METRIC_REPAIR_EVENTS 5
#define METRIC_RESTART_INDEX 6
#define METRIC_BLOCKED_CANDIDATES 7
#define METRIC_FORBIDDEN_COUNT 8
#define METRIC_CANDIDATE_GENERATION_SECONDS 9
#define METRIC_CANDIDATE_SCORING_SECONDS 10
#define METRIC_RESTART_SORT_SECONDS 11
#define METRIC_STATE_INIT_SECONDS 12
#define METRIC_GREEDY_SCAN_SECONDS 13
#define METRIC_FORBIDDEN_COUNT_SECONDS 14
#define METRIC_C_RUN_SECONDS 15
#define METRIC_MINIMUM_DISTANCE 16
#define METRIC_COUNT 17

#define MAX_N_LIMIT 24
#define BASELINE_MAX_CANDIDATES 1000000000ULL

typedef struct {
    uint64_t mask;
    double score;
    uint32_t tie;
} Candidate;

typedef struct {
    uint64_t *bits;
    uint64_t universe_size;
    uint64_t word_count;
    uint64_t forbidden_count;
    const uint64_t *offsets;
    uint64_t offset_count;
    uint64_t *codewords;
    int codeword_count;
    int codeword_cap;
} BallState;

typedef struct {
    int constructed_count;
    int dropped_count;
    int steps;
    uint64_t forbidden_count;
} RepairReward;

typedef struct {
    int visits;
    int has_best;
    RepairReward best_reward;
    uint64_t success_count;
    uint64_t total_constructed;
    uint64_t total_dropped;
    uint64_t total_steps;
    uint64_t total_forbidden;
} RepairStats;

typedef struct {
    Candidate *candidates;
    uint64_t candidate_count;
    const uint64_t *codewords;
    int codeword_count;
    int n;
    int d;
    int restart;
    int repair_event_index;
    uint64_t seed;
    uint64_t dynamic_window;
    uint64_t rollout_depth;
    uint64_t drop_topk;
    const uint64_t *tabu;
    int tabu_count;
    const uint64_t *offsets;
    uint64_t offset_count;
    const uint64_t *local_offsets;
    uint64_t local_offset_count;
    uint64_t simulation_start;
    uint64_t simulation_end;
    RepairStats *root_stats;
    uint64_t dynamic_evaluations;
} MctsJob;

extern double oe_max_code_priority(
    uint64_t word_mask,
    int n,
    int d,
    int step,
    int word_weight,
    uint64_t forbidden_count,
    uint64_t new_forbidden_count,
    uint64_t overlap_forbidden_count,
    uint64_t local_available_count
);

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static int popcount64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value) {
        value &= value - 1ULL;
        count++;
    }
    return count;
#endif
}

static void write_error(char *error_out, int error_cap, const char *message) {
    if (!error_out || error_cap <= 0) {
        return;
    }
    int index = 0;
    while (message[index] && index + 1 < error_cap) {
        error_out[index] = message[index];
        index++;
    }
    error_out[index] = '\0';
}

static uint64_t env_u64(const char *name, uint64_t default_value) {
    const char *raw_value = getenv(name);
    if (!raw_value || !raw_value[0]) {
        return default_value;
    }
    char *end_ptr = NULL;
    unsigned long long parsed = strtoull(raw_value, &end_ptr, 10);
    if (end_ptr == raw_value) {
        return default_value;
    }
    return (uint64_t)parsed;
}

static int env_equals(const char *name, const char *expected_value) {
    const char *raw_value = getenv(name);
    return raw_value && strcmp(raw_value, expected_value) == 0;
}

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

static uint64_t rng_next(uint64_t *state) {
    *state = mix64(*state + 0x9e3779b97f4a7c15ULL);
    return *state;
}

static uint32_t deterministic_tiebreak(uint64_t mask, int restart_index) {
    return (uint32_t)(
        mask * 1103515245ULL + (uint64_t)restart_index * 2654435761ULL + 12345ULL
    );
}

static int compare_candidates_desc(const void *left_ptr, const void *right_ptr) {
    const Candidate *left = (const Candidate *)left_ptr;
    const Candidate *right = (const Candidate *)right_ptr;
    if (left->score < right->score) {
        return 1;
    }
    if (left->score > right->score) {
        return -1;
    }
    if (left->tie < right->tie) {
        return 1;
    }
    if (left->tie > right->tie) {
        return -1;
    }
    if (left->mask < right->mask) {
        return 1;
    }
    if (left->mask > right->mask) {
        return -1;
    }
    return 0;
}

static int get_bit(const uint64_t *bits, uint64_t value) {
    return (bits[value >> 6] >> (value & 63U)) & 1ULL;
}

static int set_bit_if_new(uint64_t *bits, uint64_t value) {
    uint64_t word_index = value >> 6;
    uint64_t bit = 1ULL << (value & 63U);
    if (bits[word_index] & bit) {
        return 0;
    }
    bits[word_index] |= bit;
    return 1;
}

static uint64_t binomial_u64(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }
    __uint128_t result = 1;
    for (int i = 1; i <= k; i++) {
        result = (result * (uint64_t)(n - k + i)) / (uint64_t)i;
        if (result > UINT64_MAX) {
            return UINT64_MAX;
        }
    }
    return (uint64_t)result;
}

static uint64_t next_combination(uint64_t mask) {
    uint64_t smallest = mask & (~mask + 1ULL);
    uint64_t ripple = mask + smallest;
    if (ripple == 0) {
        return 0;
    }
    return ripple | (((mask ^ ripple) >> 2) / smallest);
}

static uint64_t count_weight_range(int n, int min_weight, int max_weight) {
    uint64_t count = 0;
    for (int weight = min_weight; weight <= max_weight; weight++) {
        count += binomial_u64(n, weight);
    }
    return count;
}

static uint64_t fill_weight_range(uint64_t *values, int n, int min_weight, int max_weight) {
    uint64_t index = 0;
    uint64_t limit = 1ULL << n;
    for (int weight = min_weight; weight <= max_weight; weight++) {
        if (weight == 0) {
            values[index++] = 0ULL;
            continue;
        }
        uint64_t mask = (1ULL << weight) - 1ULL;
        while (mask && mask < limit) {
            values[index++] = mask;
            mask = next_combination(mask);
        }
    }
    return index;
}

static uint64_t make_offsets(uint64_t **offsets_out, int n, int d) {
    int max_weight = d - 1;
    if (max_weight > n) {
        max_weight = n;
    }
    uint64_t count = count_weight_range(n, 0, max_weight);
    uint64_t *offsets = (uint64_t *)calloc((size_t)count, sizeof(uint64_t));
    if (!offsets) {
        *offsets_out = NULL;
        return 0;
    }
    fill_weight_range(offsets, n, 0, max_weight);
    *offsets_out = offsets;
    return count;
}

static uint64_t make_local_offsets(uint64_t **offsets_out, int n, int d, uint64_t sample_size) {
    uint64_t exact_count = binomial_u64(n, d);
    if (sample_size == 0 || exact_count == 0) {
        *offsets_out = NULL;
        return 0;
    }
    uint64_t *all = (uint64_t *)calloc((size_t)exact_count, sizeof(uint64_t));
    if (!all) {
        *offsets_out = NULL;
        return 0;
    }
    fill_weight_range(all, n, d, d);
    uint64_t count = exact_count < sample_size ? exact_count : sample_size;
    uint64_t stride = exact_count / count;
    if (stride == 0) {
        stride = 1;
    }
    uint64_t *sampled = (uint64_t *)calloc((size_t)count, sizeof(uint64_t));
    if (!sampled) {
        free(all);
        *offsets_out = NULL;
        return 0;
    }
    for (uint64_t i = 0; i < count; i++) {
        uint64_t source_index = i * stride;
        if (source_index >= exact_count) {
            source_index = exact_count - 1;
        }
        sampled[i] = all[source_index];
    }
    free(all);
    *offsets_out = sampled;
    return count;
}

static uint64_t make_candidates(Candidate **candidates_out, int n, int d) {
    uint64_t count = count_weight_range(n, d, n);
    Candidate *candidates = (Candidate *)calloc((size_t)count, sizeof(Candidate));
    if (!candidates) {
        *candidates_out = NULL;
        return 0;
    }
    uint64_t index = 0;
    uint64_t limit = 1ULL << n;
    for (int weight = d; weight <= n; weight++) {
        uint64_t mask = (1ULL << weight) - 1ULL;
        while (mask && mask < limit) {
            candidates[index++].mask = mask;
            mask = next_combination(mask);
        }
    }
    *candidates_out = candidates;
    return index;
}

static int init_state(
    BallState *state,
    int n,
    const uint64_t *offsets,
    uint64_t offset_count,
    int codeword_cap
) {
    memset(state, 0, sizeof(*state));
    state->universe_size = 1ULL << n;
    state->word_count = (state->universe_size + 63ULL) >> 6;
    state->offsets = offsets;
    state->offset_count = offset_count;
    state->codeword_cap = codeword_cap;
    state->bits = (uint64_t *)calloc((size_t)state->word_count, sizeof(uint64_t));
    state->codewords = (uint64_t *)calloc((size_t)codeword_cap, sizeof(uint64_t));
    if (!state->bits || !state->codewords) {
        free(state->bits);
        free(state->codewords);
        memset(state, 0, sizeof(*state));
        return 0;
    }
    return 1;
}

static void free_state(BallState *state) {
    if (!state) {
        return;
    }
    free(state->bits);
    free(state->codewords);
    memset(state, 0, sizeof(*state));
}

static int can_add(BallState *state, uint64_t mask) {
    return !get_bit(state->bits, mask);
}

static int add_word(BallState *state, uint64_t mask) {
    if (!can_add(state, mask) || state->codeword_count >= state->codeword_cap) {
        return 0;
    }
    state->codewords[state->codeword_count++] = mask;
    for (uint64_t i = 0; i < state->offset_count; i++) {
        uint64_t value = mask ^ state->offsets[i];
        if (set_bit_if_new(state->bits, value)) {
            state->forbidden_count++;
        }
    }
    return 1;
}

static int rebuild_state(
    BallState *state,
    int n,
    const uint64_t *offsets,
    uint64_t offset_count,
    const uint64_t *codewords,
    int codeword_count,
    int codeword_cap
) {
    if (!init_state(state, n, offsets, offset_count, codeword_cap)) {
        return 0;
    }
    for (int i = 0; i < codeword_count; i++) {
        if (!add_word(state, codewords[i])) {
            free_state(state);
            return 0;
        }
    }
    return 1;
}

static int mask_in_list(const uint64_t *values, int count, uint64_t mask) {
    for (int i = 0; i < count; i++) {
        if (values[i] == mask) {
            return 1;
        }
    }
    return 0;
}

static void damage_features(
    BallState *state,
    uint64_t mask,
    const uint64_t *local_offsets,
    uint64_t local_offset_count,
    uint64_t *new_out,
    uint64_t *overlap_out,
    uint64_t *local_out
) {
    uint64_t new_count = 0;
    for (uint64_t i = 0; i < state->offset_count; i++) {
        if (!get_bit(state->bits, mask ^ state->offsets[i])) {
            new_count++;
        }
    }
    uint64_t local_count = 0;
    for (uint64_t i = 0; i < local_offset_count; i++) {
        if (!get_bit(state->bits, mask ^ local_offsets[i])) {
            local_count++;
        }
    }
    *new_out = new_count;
    *overlap_out = state->offset_count > new_count ? state->offset_count - new_count : 0;
    *local_out = local_count;
}

static int choose_dynamic_candidate(
    Candidate *candidates,
    uint64_t candidate_count,
    BallState *state,
    int n,
    int d,
    int restart,
    uint64_t window_size,
    const uint64_t *tabu,
    int tabu_count,
    const uint64_t *local_offsets,
    uint64_t local_offset_count,
    uint64_t *selected_mask,
    double *selected_score,
    uint64_t *blocked_out,
    uint64_t *evaluations_out
) {
    double best_score = 0.0;
    uint32_t best_tie = 0;
    uint64_t best_mask = 0;
    int found = 0;
    uint64_t legal_seen = 0;
    uint64_t limit = window_size == 0 ? candidate_count : window_size;
    int step = state->codeword_count;

    for (uint64_t i = 0; i < candidate_count; i++) {
        uint64_t mask = candidates[i].mask;
        if (mask_in_list(tabu, tabu_count, mask)) {
            continue;
        }
        if (!can_add(state, mask)) {
            (*blocked_out)++;
            continue;
        }
        uint64_t new_forbidden = 0;
        uint64_t overlap = 0;
        uint64_t local_available = 0;
        damage_features(
            state,
            mask,
            local_offsets,
            local_offset_count,
            &new_forbidden,
            &overlap,
            &local_available
        );
        double score = oe_max_code_priority(
            mask,
            n,
            d,
            step,
            popcount64(mask),
            state->forbidden_count,
            new_forbidden,
            overlap,
            local_available
        );
        uint32_t tie = deterministic_tiebreak(mask, restart + step + 1);
        (*evaluations_out)++;
        if (!found || score > best_score || (score == best_score && tie > best_tie)) {
            found = 1;
            best_score = score;
            best_tie = tie;
            best_mask = mask;
        }
        legal_seen++;
        if (legal_seen >= limit) {
            break;
        }
    }
    if (!found) {
        return 0;
    }
    *selected_mask = best_mask;
    *selected_score = best_score;
    return 1;
}

static uint64_t count_legal_prefix(
    Candidate *candidates,
    uint64_t candidate_count,
    BallState *state,
    uint64_t prefix_size,
    const uint64_t *tabu,
    int tabu_count
) {
    uint64_t limit = prefix_size == 0 || prefix_size > candidate_count ? candidate_count : prefix_size;
    uint64_t count = 0;
    for (uint64_t i = 0; i < limit; i++) {
        uint64_t mask = candidates[i].mask;
        if (!mask_in_list(tabu, tabu_count, mask) && can_add(state, mask)) {
            count++;
        }
    }
    return count;
}

static int drop_choice_is_better(
    uint64_t legal_count,
    uint64_t release,
    uint32_t tie,
    uint64_t current_legal,
    uint64_t current_release,
    uint32_t current_tie
) {
    return legal_count > current_legal
        || (legal_count == current_legal && release > current_release)
        || (legal_count == current_legal && release == current_release && tie > current_tie);
}

static int choose_rollout_drop_index(
    Candidate *candidates,
    uint64_t candidate_count,
    const uint64_t *codewords,
    int codeword_count,
    int n,
    int d,
    const uint64_t *offsets,
    uint64_t offset_count,
    int restart,
    int repair_event_index,
    int step,
    uint64_t before_forbidden_count,
    uint64_t candidate_window,
    const uint64_t *tabu,
    int tabu_count,
    uint64_t drop_topk,
    uint64_t *rng_state
) {
    (void)d;
    int removable_count = 0;
    for (int i = 1; i < codeword_count; i++) {
        if (!mask_in_list(tabu, tabu_count, codewords[i])) {
            removable_count++;
        }
    }
    if (removable_count <= 0) {
        return -1;
    }
    if (drop_topk == 0) {
        int chosen_pos = (int)(rng_next(rng_state) % (uint64_t)removable_count);
        for (int i = 1; i < codeword_count; i++) {
            if (!mask_in_list(tabu, tabu_count, codewords[i]) && chosen_pos-- == 0) {
                return i;
            }
        }
    }

    uint64_t top_cap = drop_topk < (uint64_t)removable_count ? drop_topk : (uint64_t)removable_count;
    int *top_indices = (int *)calloc((size_t)top_cap, sizeof(int));
    uint64_t *top_legal = (uint64_t *)calloc((size_t)top_cap, sizeof(uint64_t));
    uint64_t *top_release = (uint64_t *)calloc((size_t)top_cap, sizeof(uint64_t));
    uint32_t *top_tie = (uint32_t *)calloc((size_t)top_cap, sizeof(uint32_t));
    if (!top_indices || !top_legal || !top_release || !top_tie) {
        free(top_indices);
        free(top_legal);
        free(top_release);
        free(top_tie);
        return -1;
    }
    uint64_t top_count = 0;
    int cap = codeword_count + 8;
    uint64_t *trial_words = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    if (!trial_words) {
        free(top_indices);
        free(top_legal);
        free(top_release);
        free(top_tie);
        return -1;
    }

    for (int drop_index = 1; drop_index < codeword_count; drop_index++) {
        if (mask_in_list(tabu, tabu_count, codewords[drop_index])) {
            continue;
        }
        int trial_count = 0;
        for (int i = 0; i < codeword_count; i++) {
            if (i != drop_index) {
                trial_words[trial_count++] = codewords[i];
            }
        }
        BallState trial_state;
        if (!rebuild_state(&trial_state, n, offsets, offset_count, trial_words, trial_count, cap)) {
            continue;
        }
        uint64_t release = before_forbidden_count > trial_state.forbidden_count
            ? before_forbidden_count - trial_state.forbidden_count
            : 0;
        uint64_t legal = count_legal_prefix(candidates, candidate_count, &trial_state, candidate_window, tabu, tabu_count);
        uint32_t tie = deterministic_tiebreak(codewords[drop_index], restart + repair_event_index + step + 4099);
        free_state(&trial_state);

        uint64_t insert_at = top_count;
        while (
            insert_at > 0
            && drop_choice_is_better(
                legal,
                release,
                tie,
                top_legal[insert_at - 1],
                top_release[insert_at - 1],
                top_tie[insert_at - 1]
            )
        ) {
            if (insert_at < top_cap) {
                top_indices[insert_at] = top_indices[insert_at - 1];
                top_legal[insert_at] = top_legal[insert_at - 1];
                top_release[insert_at] = top_release[insert_at - 1];
                top_tie[insert_at] = top_tie[insert_at - 1];
            }
            insert_at--;
        }
        if (insert_at < top_cap) {
            top_indices[insert_at] = drop_index;
            top_legal[insert_at] = legal;
            top_release[insert_at] = release;
            top_tie[insert_at] = tie;
            if (top_count < top_cap) {
                top_count++;
            }
        }
    }

    int chosen = -1;
    if (top_count > 0) {
        chosen = top_indices[rng_next(rng_state) % top_count];
    }
    free(trial_words);
    free(top_indices);
    free(top_legal);
    free(top_release);
    free(top_tie);
    return chosen;
}

static int reward_is_better(const RepairReward *candidate, const RepairReward *current, int has_current) {
    if (!has_current) {
        return 1;
    }
    if (candidate->constructed_count != current->constructed_count) {
        return candidate->constructed_count > current->constructed_count;
    }
    if (candidate->dropped_count != current->dropped_count) {
        return candidate->dropped_count < current->dropped_count;
    }
    if (candidate->steps != current->steps) {
        return candidate->steps < current->steps;
    }
    return candidate->forbidden_count < current->forbidden_count;
}

static void update_stats(RepairStats *stats, RepairReward reward, int original_count) {
    stats->visits++;
    stats->total_constructed += (uint64_t)reward.constructed_count;
    stats->total_dropped += (uint64_t)reward.dropped_count;
    stats->total_steps += (uint64_t)reward.steps;
    stats->total_forbidden += reward.forbidden_count;
    if (reward.constructed_count >= original_count) {
        stats->success_count++;
    }
    if (reward_is_better(&reward, &stats->best_reward, stats->has_best)) {
        stats->best_reward = reward;
        stats->has_best = 1;
    }
}

static int stats_is_better(const RepairStats *candidate, const RepairStats *current, int has_current) {
    if (candidate->visits <= 0 || !candidate->has_best) {
        return 0;
    }
    if (!has_current || current->visits <= 0 || !current->has_best) {
        return 1;
    }
    if (candidate->success_count * (uint64_t)current->visits != current->success_count * (uint64_t)candidate->visits) {
        return candidate->success_count * (uint64_t)current->visits > current->success_count * (uint64_t)candidate->visits;
    }
    if (reward_is_better(&candidate->best_reward, &current->best_reward, 1)) {
        return 1;
    }
    if (reward_is_better(&current->best_reward, &candidate->best_reward, 1)) {
        return 0;
    }
    if (candidate->total_constructed * (uint64_t)current->visits != current->total_constructed * (uint64_t)candidate->visits) {
        return candidate->total_constructed * (uint64_t)current->visits > current->total_constructed * (uint64_t)candidate->visits;
    }
    if (candidate->total_dropped * (uint64_t)current->visits != current->total_dropped * (uint64_t)candidate->visits) {
        return candidate->total_dropped * (uint64_t)current->visits < current->total_dropped * (uint64_t)candidate->visits;
    }
    if (candidate->total_steps * (uint64_t)current->visits != current->total_steps * (uint64_t)candidate->visits) {
        return candidate->total_steps * (uint64_t)current->visits < current->total_steps * (uint64_t)candidate->visits;
    }
    return candidate->total_forbidden * (uint64_t)current->visits < current->total_forbidden * (uint64_t)candidate->visits;
}

static int rollout_after_first_drop(
    Candidate *candidates,
    uint64_t candidate_count,
    const uint64_t *codewords,
    int codeword_count,
    int first_drop_index,
    int n,
    int d,
    const uint64_t *offsets,
    uint64_t offset_count,
    const uint64_t *local_offsets,
    uint64_t local_offset_count,
    int restart,
    int repair_event_index,
    uint64_t seed,
    uint64_t dynamic_window,
    uint64_t rollout_depth,
    uint64_t drop_topk,
    uint64_t candidate_window,
    const uint64_t *tabu,
    int tabu_count,
    RepairReward *reward_out,
    uint64_t *evaluations_out
) {
    int cap = codeword_count + (int)rollout_depth + 8;
    uint64_t *trial_words = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    uint64_t *trial_tabu = (uint64_t *)calloc((size_t)(tabu_count + rollout_depth + 2), sizeof(uint64_t));
    if (!trial_words || !trial_tabu) {
        free(trial_words);
        free(trial_tabu);
        return 0;
    }
    for (int i = 0; i < tabu_count; i++) {
        trial_tabu[i] = tabu[i];
    }
    int trial_tabu_count = tabu_count;
    int trial_tabu_cap = tabu_count + (int)rollout_depth + 2;

    int trial_count = 0;
    for (int i = 0; i < codeword_count; i++) {
        if (i != first_drop_index) {
            trial_words[trial_count++] = codewords[i];
        }
    }
    BallState trial_state;
    if (!rebuild_state(&trial_state, n, offsets, offset_count, trial_words, trial_count, cap)) {
        free(trial_words);
        free(trial_tabu);
        return 0;
    }
    trial_tabu[trial_tabu_count++] = codewords[first_drop_index];
    uint64_t rng_state = mix64(seed ^ ((uint64_t)restart << 32) ^ ((uint64_t)repair_event_index << 16) ^ (uint64_t)(first_drop_index + 1));
    int dropped_count = 1;
    int steps = 0;
    uint64_t evaluations = 0;

    while ((uint64_t)steps < rollout_depth) {
        uint64_t mask = 0;
        double score = 0.0;
        uint64_t blocked = 0;
        uint64_t evals = 0;
        if (choose_dynamic_candidate(
            candidates,
            candidate_count,
            &trial_state,
            n,
            d,
            restart + repair_event_index + steps + 1,
            dynamic_window,
            trial_tabu,
            trial_tabu_count,
            local_offsets,
            local_offset_count,
            &mask,
            &score,
            &blocked,
            &evals
        )) {
            (void)score;
            evaluations += evals;
            if (!add_word(&trial_state, mask)) {
                break;
            }
            trial_words[trial_count++] = mask;
            steps++;
            continue;
        }
        evaluations += evals;
        if (trial_count <= 1 || trial_tabu_count >= trial_tabu_cap) {
            break;
        }
        int drop_index = choose_rollout_drop_index(
            candidates,
            candidate_count,
            trial_words,
            trial_count,
            n,
            d,
            offsets,
            offset_count,
            restart,
            repair_event_index,
            steps,
            trial_state.forbidden_count,
            candidate_window,
            trial_tabu,
            trial_tabu_count,
            drop_topk,
            &rng_state
        );
        if (drop_index <= 0) {
            break;
        }
        uint64_t dropped = trial_words[drop_index];
        trial_tabu[trial_tabu_count++] = dropped;
        for (int i = drop_index + 1; i < trial_count; i++) {
            trial_words[i - 1] = trial_words[i];
        }
        trial_count--;
        BallState rebuilt;
        if (!rebuild_state(&rebuilt, n, offsets, offset_count, trial_words, trial_count, cap)) {
            break;
        }
        free_state(&trial_state);
        trial_state = rebuilt;
        dropped_count++;
        steps++;
    }

    reward_out->constructed_count = trial_count;
    reward_out->dropped_count = dropped_count;
    reward_out->steps = steps;
    reward_out->forbidden_count = trial_state.forbidden_count;
    *evaluations_out = evaluations;
    free_state(&trial_state);
    free(trial_words);
    free(trial_tabu);
    return 1;
}

static void merge_stats(RepairStats *target, const RepairStats *source) {
    if (source->visits <= 0) {
        return;
    }
    target->visits += source->visits;
    target->success_count += source->success_count;
    target->total_constructed += source->total_constructed;
    target->total_dropped += source->total_dropped;
    target->total_steps += source->total_steps;
    target->total_forbidden += source->total_forbidden;
    if (source->has_best && reward_is_better(&source->best_reward, &target->best_reward, target->has_best)) {
        target->best_reward = source->best_reward;
        target->has_best = 1;
    }
}

static void run_mcts_range(MctsJob *job) {
    uint64_t evaluations = 0;
    int removable_count = job->codeword_count - 1;
    for (uint64_t simulation = job->simulation_start; simulation < job->simulation_end; simulation++) {
        int drop_index = 1 + (int)(simulation % (uint64_t)removable_count);
        if (mask_in_list(job->tabu, job->tabu_count, job->codewords[drop_index])) {
            continue;
        }
        RepairReward reward;
        uint64_t evals = 0;
        if (!rollout_after_first_drop(
            job->candidates,
            job->candidate_count,
            job->codewords,
            job->codeword_count,
            drop_index,
            job->n,
            job->d,
            job->offsets,
            job->offset_count,
            job->local_offsets,
            job->local_offset_count,
            job->restart,
            job->repair_event_index,
            job->seed + simulation * 0x9e3779b97f4a7c15ULL,
            job->dynamic_window,
            job->rollout_depth,
            job->drop_topk,
            env_u64("MAX_CODE_REPAIR_CANDIDATE_WINDOW", 65536ULL),
            job->tabu,
            job->tabu_count,
            &reward,
            &evals
        )) {
            continue;
        }
        evaluations += evals;
        update_stats(&job->root_stats[drop_index], reward, job->codeword_count);
    }
    job->dynamic_evaluations = evaluations;
}

static void *mcts_worker(void *arg) {
    run_mcts_range((MctsJob *)arg);
    return NULL;
}

static int choose_mcts_drop_index(
    Candidate *candidates,
    uint64_t candidate_count,
    BallState *state,
    int n,
    int d,
    int restart,
    int repair_event_index,
    uint64_t seed,
    uint64_t dynamic_window,
    uint64_t rollout_depth,
    uint64_t simulations,
    uint64_t drop_topk,
    const uint64_t *tabu,
    int tabu_count,
    const uint64_t *local_offsets,
    uint64_t local_offset_count,
    const uint64_t *offsets,
    uint64_t offset_count,
    int *drop_index_out,
    uint64_t *evaluations_out
) {
    if (state->codeword_count <= 1 || simulations == 0 || rollout_depth == 0) {
        return 0;
    }
    RepairStats *root_stats = (RepairStats *)calloc((size_t)state->codeword_cap, sizeof(RepairStats));
    if (!root_stats) {
        return 0;
    }
    uint64_t worker_count = env_u64("MAX_CODE_REPAIR_MCTS_WORKERS", 1ULL);
    if (worker_count == 0ULL) {
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        worker_count = cpu_count > 1 ? (uint64_t)cpu_count : 1ULL;
    }
    if (worker_count > simulations) {
        worker_count = simulations;
    }
    if (worker_count > 64ULL) {
        worker_count = 64ULL;
    }

    uint64_t total_evaluations = 0;
    if (worker_count <= 1ULL) {
        MctsJob job;
        memset(&job, 0, sizeof(job));
        job.candidates = candidates;
        job.candidate_count = candidate_count;
        job.codewords = state->codewords;
        job.codeword_count = state->codeword_count;
        job.n = n;
        job.d = d;
        job.restart = restart;
        job.repair_event_index = repair_event_index;
        job.seed = seed;
        job.dynamic_window = dynamic_window;
        job.rollout_depth = rollout_depth;
        job.drop_topk = drop_topk;
        job.tabu = tabu;
        job.tabu_count = tabu_count;
        job.offsets = offsets;
        job.offset_count = offset_count;
        job.local_offsets = local_offsets;
        job.local_offset_count = local_offset_count;
        job.simulation_start = 0;
        job.simulation_end = simulations;
        job.root_stats = root_stats;
        run_mcts_range(&job);
        total_evaluations = job.dynamic_evaluations;
    } else {
        pthread_t *threads = (pthread_t *)calloc((size_t)worker_count, sizeof(pthread_t));
        MctsJob *jobs = (MctsJob *)calloc((size_t)worker_count, sizeof(MctsJob));
        RepairStats *worker_stats = (RepairStats *)calloc((size_t)(worker_count * (uint64_t)state->codeword_cap), sizeof(RepairStats));
        if (!threads || !jobs || !worker_stats) {
            free(threads);
            free(jobs);
            free(worker_stats);
            free(root_stats);
            return 0;
        }
        uint64_t chunk = (simulations + worker_count - 1ULL) / worker_count;
        uint64_t created = 0;
        for (uint64_t worker = 0; worker < worker_count; worker++) {
            uint64_t start = worker * chunk;
            uint64_t end = start + chunk;
            if (start >= simulations) {
                break;
            }
            if (end > simulations) {
                end = simulations;
            }
            MctsJob *job = &jobs[worker];
            job->candidates = candidates;
            job->candidate_count = candidate_count;
            job->codewords = state->codewords;
            job->codeword_count = state->codeword_count;
            job->n = n;
            job->d = d;
            job->restart = restart;
            job->repair_event_index = repair_event_index;
            job->seed = seed;
            job->dynamic_window = dynamic_window;
            job->rollout_depth = rollout_depth;
            job->drop_topk = drop_topk;
            job->tabu = tabu;
            job->tabu_count = tabu_count;
            job->offsets = offsets;
            job->offset_count = offset_count;
            job->local_offsets = local_offsets;
            job->local_offset_count = local_offset_count;
            job->simulation_start = start;
            job->simulation_end = end;
            job->root_stats = &worker_stats[worker * (uint64_t)state->codeword_cap];
            if (pthread_create(&threads[worker], NULL, mcts_worker, job) != 0) {
                break;
            }
            created++;
        }
        for (uint64_t worker = 0; worker < created; worker++) {
            pthread_join(threads[worker], NULL);
            total_evaluations += jobs[worker].dynamic_evaluations;
            RepairStats *local = &worker_stats[worker * (uint64_t)state->codeword_cap];
            for (int i = 1; i < state->codeword_count; i++) {
                merge_stats(&root_stats[i], &local[i]);
            }
        }
        free(threads);
        free(jobs);
        free(worker_stats);
    }

    int best_index = -1;
    int has_best = 0;
    for (int i = 1; i < state->codeword_count; i++) {
        if (mask_in_list(tabu, tabu_count, state->codewords[i])) {
            continue;
        }
        if (stats_is_better(&root_stats[i], best_index >= 0 ? &root_stats[best_index] : NULL, has_best)) {
            best_index = i;
            has_best = 1;
        }
    }
    if (!has_best || best_index <= 0 || root_stats[best_index].best_reward.constructed_count < state->codeword_count) {
        free(root_stats);
        *evaluations_out = total_evaluations;
        return 0;
    }
    *drop_index_out = best_index;
    *evaluations_out = total_evaluations;
    free(root_stats);
    return 1;
}

static int minimum_distance(const uint64_t *codewords, int count, int n) {
    if (count < 2) {
        return n + 1;
    }
    int best = n + 1;
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            int dist = popcount64(codewords[i] ^ codewords[j]);
            if (dist < best) {
                best = dist;
            }
        }
    }
    return best;
}

int oe_max_code_run(
    int n,
    int d,
    int restarts,
    uint64_t seed,
    uint64_t *selected_out,
    int selected_cap,
    double *metrics_out,
    int metrics_cap,
    char *error_out,
    int error_cap
) {
    double run_started = monotonic_seconds();
    if (metrics_cap < METRIC_COUNT) {
        write_error(error_out, error_cap, "metrics buffer too small");
        return -1;
    }
    for (int i = 0; i < metrics_cap; i++) {
        metrics_out[i] = 0.0;
    }
    if (n <= 0 || n > MAX_N_LIMIT || d <= 0 || d > n) {
        write_error(error_out, error_cap, "unsupported n or d");
        return -2;
    }
    if (restarts <= 0) {
        restarts = 1;
    }

    uint64_t total_candidate_count = (1ULL << n) - 1ULL;
    if (total_candidate_count > env_u64("MAX_CODE_MAX_CANDIDATES", BASELINE_MAX_CANDIDATES)) {
        write_error(error_out, error_cap, "candidate count exceeds MAX_CODE_MAX_CANDIDATES");
        return -3;
    }

    double stage_started = monotonic_seconds();
    uint64_t *offsets = NULL;
    uint64_t offset_count = make_offsets(&offsets, n, d);
    uint64_t *local_offsets = NULL;
    uint64_t local_offset_count = make_local_offsets(&local_offsets, n, d, env_u64("MAX_CODE_LOCAL_SAMPLE_SIZE", 64ULL));
    Candidate *candidates = NULL;
    uint64_t candidate_count = make_candidates(&candidates, n, d);
    if (!offsets || !candidates || offset_count == 0 || candidate_count == 0) {
        free(offsets);
        free(local_offsets);
        free(candidates);
        write_error(error_out, error_cap, "candidate or offset allocation failed");
        return -4;
    }
    metrics_out[METRIC_CANDIDATE_GENERATION_SECONDS] = monotonic_seconds() - stage_started;

    stage_started = monotonic_seconds();
    for (uint64_t i = 0; i < candidate_count; i++) {
        int weight = popcount64(candidates[i].mask);
        candidates[i].score = oe_max_code_priority(candidates[i].mask, n, d, 0, weight, 0, 0, 0, 0);
    }
    metrics_out[METRIC_CANDIDATE_SCORING_SECONDS] = monotonic_seconds() - stage_started;

    uint64_t *best_codewords = (uint64_t *)calloc((size_t)selected_cap, sizeof(uint64_t));
    if (!best_codewords) {
        free(offsets);
        free(local_offsets);
        free(candidates);
        write_error(error_out, error_cap, "best codeword allocation failed");
        return -5;
    }
    int best_count = 0;
    int best_min_distance = 0;
    int best_restart = 0;
    uint64_t best_forbidden_count = 0;
    uint64_t best_blocked = 0;
    uint64_t total_rollout_evaluations = 0;
    uint64_t total_repair_events = 0;

    uint64_t dynamic_window = env_u64("MAX_CODE_DYNAMIC_WINDOW", 4096ULL);
    uint64_t repair_events = env_u64("MAX_CODE_REPAIR_EVENTS", 4ULL);
    uint64_t repair_drop_count = env_u64("MAX_CODE_REPAIR_DROP_COUNT", 1ULL);
    uint64_t repair_tabu_tenure = env_u64("MAX_CODE_REPAIR_TABU_TENURE", repair_events * repair_drop_count);
    uint64_t repair_mcts_simulations = env_u64("MAX_CODE_REPAIR_MCTS_SIMULATIONS", 64ULL);
    uint64_t repair_mcts_depth = env_u64("MAX_CODE_REPAIR_MCTS_DEPTH", 4ULL);
    uint64_t repair_mcts_drop_topk = env_u64("MAX_CODE_REPAIR_MCTS_DROP_TOPK", 2ULL);
    int use_mcts = env_equals("MAX_CODE_REPAIR_MODE", "mcts");

    for (int restart = 0; restart < restarts; restart++) {
        for (uint64_t i = 0; i < candidate_count; i++) {
            candidates[i].tie = deterministic_tiebreak(candidates[i].mask, restart);
        }
        stage_started = monotonic_seconds();
        qsort(candidates, (size_t)candidate_count, sizeof(Candidate), compare_candidates_desc);
        metrics_out[METRIC_RESTART_SORT_SECONDS] += monotonic_seconds() - stage_started;

        stage_started = monotonic_seconds();
        BallState state;
        int cap = selected_cap;
        if (!init_state(&state, n, offsets, offset_count, cap)) {
            continue;
        }
        add_word(&state, 0ULL);
        metrics_out[METRIC_STATE_INIT_SECONDS] += monotonic_seconds() - stage_started;

        uint64_t *tabu = NULL;
        int tabu_count = 0;
        int tabu_next = 0;
        int tabu_capacity = (int)(repair_tabu_tenure > (uint64_t)cap ? (uint64_t)cap : repair_tabu_tenure);
        if (tabu_capacity > 0) {
            tabu = (uint64_t *)calloc((size_t)tabu_capacity, sizeof(uint64_t));
        }
        uint64_t blocked = 0;
        int current_repair_events = 0;
        stage_started = monotonic_seconds();
        while (state.forbidden_count < state.universe_size && state.codeword_count < cap) {
            uint64_t mask = 0;
            double score = 0.0;
            uint64_t evals = 0;
            if (choose_dynamic_candidate(
                candidates,
                candidate_count,
                &state,
                n,
                d,
                restart,
                dynamic_window,
                tabu,
                tabu_count,
                local_offsets,
                local_offset_count,
                &mask,
                &score,
                &blocked,
                &evals
            )) {
                (void)score;
                add_word(&state, mask);
                continue;
            }
            if (!use_mcts || (uint64_t)current_repair_events >= repair_events || state.codeword_count <= 1) {
                break;
            }
            int repaired = 0;
            for (uint64_t drop_event = 0; drop_event < repair_drop_count && (uint64_t)current_repair_events < repair_events; drop_event++) {
                int drop_index = -1;
                uint64_t rollout_evals = 0;
                if (!choose_mcts_drop_index(
                    candidates,
                    candidate_count,
                    &state,
                    n,
                    d,
                    restart,
                    current_repair_events,
                    seed,
                    dynamic_window,
                    repair_mcts_depth,
                    repair_mcts_simulations,
                    repair_mcts_drop_topk,
                    tabu,
                    tabu_count,
                    local_offsets,
                    local_offset_count,
                    offsets,
                    offset_count,
                    &drop_index,
                    &rollout_evals
                )) {
                    total_rollout_evaluations += rollout_evals;
                    break;
                }
                total_rollout_evaluations += rollout_evals;
                uint64_t dropped = state.codewords[drop_index];
                for (int i = drop_index + 1; i < state.codeword_count; i++) {
                    state.codewords[i - 1] = state.codewords[i];
                }
                int new_count = state.codeword_count - 1;
                BallState rebuilt;
                if (!rebuild_state(&rebuilt, n, offsets, offset_count, state.codewords, new_count, cap)) {
                    break;
                }
                free_state(&state);
                state = rebuilt;
                if (tabu && tabu_capacity > 0) {
                    if (tabu_count < tabu_capacity) {
                        tabu[tabu_count++] = dropped;
                    } else {
                        tabu[tabu_next] = dropped;
                        tabu_next = (tabu_next + 1) % tabu_capacity;
                    }
                }
                current_repair_events++;
                total_repair_events++;
                repaired = 1;
            }
            if (!repaired) {
                break;
            }
        }
        metrics_out[METRIC_GREEDY_SCAN_SECONDS] += monotonic_seconds() - stage_started;

        int min_dist = minimum_distance(state.codewords, state.codeword_count, n);
        if (
            state.codeword_count > best_count
            || (state.codeword_count == best_count && min_dist > best_min_distance)
        ) {
            best_count = state.codeword_count;
            best_min_distance = min_dist;
            best_restart = restart;
            best_forbidden_count = state.forbidden_count;
            best_blocked = blocked;
            int copy_count = best_count < selected_cap ? best_count : selected_cap;
            for (int i = 0; i < copy_count; i++) {
                best_codewords[i] = state.codewords[i];
            }
        }
        free(tabu);
        free_state(&state);
    }

    int output_count = best_count < selected_cap ? best_count : selected_cap;
    for (int i = 0; i < output_count; i++) {
        selected_out[i] = best_codewords[i];
    }

    metrics_out[METRIC_CODE_SIZE] = (double)best_count;
    metrics_out[METRIC_VALID] = (best_min_distance >= d) ? 1.0 : 0.0;
    metrics_out[METRIC_CANDIDATE_COUNT] = (double)total_candidate_count;
    metrics_out[METRIC_SCORED_CANDIDATES] = (double)candidate_count;
    metrics_out[METRIC_REPAIR_ROLLOUT_EVALUATIONS] = (double)total_rollout_evaluations;
    metrics_out[METRIC_REPAIR_EVENTS] = (double)total_repair_events;
    metrics_out[METRIC_RESTART_INDEX] = (double)best_restart;
    metrics_out[METRIC_BLOCKED_CANDIDATES] = (double)best_blocked;
    metrics_out[METRIC_FORBIDDEN_COUNT] = (double)best_forbidden_count;
    metrics_out[METRIC_MINIMUM_DISTANCE] = (double)best_min_distance;
    metrics_out[METRIC_C_RUN_SECONDS] = monotonic_seconds() - run_started;

    free(best_codewords);
    free(offsets);
    free(local_offsets);
    free(candidates);
    return best_min_distance >= d ? 0 : 1;
}
