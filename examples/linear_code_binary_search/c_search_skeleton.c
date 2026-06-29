#define _POSIX_C_SOURCE 200809L

/*
 * Fixed C search skeleton for the binary linear-code feasibility example.
 *
 * OpenEvolve-generated C variants provide only oe_linear_code_priority().
 * This file owns the fixed ABI entry point, candidate enumeration, sorting,
 * exact legality checks, metrics, and selected-column output.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define METRIC_SUCCESS 0
#define METRIC_CONSTRUCTED_COLUMNS 1
#define METRIC_CANDIDATE_COUNT 2
#define METRIC_SCORED_CANDIDATES 3
#define METRIC_SAMPLE_ATTEMPTS 4
#define METRIC_BACKTRACK_EVENTS 5
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

#define BASELINE_R_LIMIT 60
#define BASELINE_MAX_CANDIDATES 1000000000ULL

typedef struct {
    uint64_t mask;
    double score;
    uint32_t tie;
} Candidate;

typedef struct {
    uint64_t **layers;
    uint64_t *forbidden_union;
    uint64_t *layer_counts;
    int max_subset_size;
    uint64_t value_count;
    uint64_t word_count;
} DenseState;

typedef struct {
    Candidate *candidates;
    uint64_t start;
    uint64_t end;
    int n;
    int k;
    int d;
} ScoreJob;

typedef struct {
    int constructed_count;
    int dropped_count;
    int steps;
    uint64_t forbidden_count_value;
} RepairReward;

extern double oe_linear_code_priority(
    uint64_t column_mask,
    int n,
    int k,
    int d,
    int step,
    int column_weight,
    uint64_t forbidden_count,
    uint64_t new_forbidden_count,
    uint64_t overlap_forbidden_count
);

static uint64_t next_combination(uint64_t mask);

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
        value &= value - 1;
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

static int get_bit(uint64_t *bits, uint64_t value) {
    return (bits[value >> 6] >> (value & 63U)) & 1ULL;
}

static void set_bit(uint64_t *bits, uint64_t value) {
    bits[value >> 6] |= 1ULL << (value & 63U);
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

static void free_state(DenseState *state) {
    if (!state) {
        return;
    }
    if (state->layers) {
        for (int i = 0; i <= state->max_subset_size; i++) {
            free(state->layers[i]);
        }
        free(state->layers);
    }
    free(state->forbidden_union);
    free(state->layer_counts);
    state->layers = NULL;
    state->forbidden_union = NULL;
    state->layer_counts = NULL;
}

static int init_state(DenseState *state, int r, int d) {
    memset(state, 0, sizeof(*state));
    state->max_subset_size = d - 2;
    if (state->max_subset_size < 0) {
        state->max_subset_size = 0;
    }
    state->value_count = 1ULL << r;
    state->word_count = (state->value_count + 63ULL) >> 6;
    state->layers = (uint64_t **)calloc((size_t)state->max_subset_size + 1U, sizeof(uint64_t *));
    if (!state->layers) {
        return 0;
    }
    state->forbidden_union = (uint64_t *)calloc((size_t)state->word_count, sizeof(uint64_t));
    if (!state->forbidden_union) {
        free_state(state);
        return 0;
    }
    state->layer_counts = (uint64_t *)calloc((size_t)state->max_subset_size + 1U, sizeof(uint64_t));
    if (!state->layer_counts) {
        free_state(state);
        return 0;
    }
    for (int i = 0; i <= state->max_subset_size; i++) {
        state->layers[i] = (uint64_t *)calloc((size_t)state->word_count, sizeof(uint64_t));
        if (!state->layers[i]) {
            free_state(state);
            return 0;
        }
    }
    set_bit(state->layers[0], 0);
    set_bit(state->forbidden_union, 0);
    state->layer_counts[0] = 1;
    return 1;
}

static void add_column(DenseState *state, uint64_t column_mask) {
    for (int subset_size = state->max_subset_size; subset_size >= 1; subset_size--) {
        uint64_t *previous = state->layers[subset_size - 1];
        uint64_t *target = state->layers[subset_size];
        for (uint64_t word_index = 0; word_index < state->word_count; word_index++) {
            uint64_t bits = previous[word_index];
            while (bits) {
#if defined(__GNUC__) || defined(__clang__)
                int offset = __builtin_ctzll(bits);
#else
                int offset = 0;
                uint64_t probe = bits;
                while ((probe & 1ULL) == 0) {
                    probe >>= 1;
                    offset++;
                }
#endif
                uint64_t value = (word_index << 6) + (uint64_t)offset;
                uint64_t new_value = value ^ column_mask;
                if (set_bit_if_new(target, new_value)) {
                    state->layer_counts[subset_size]++;
                }
                set_bit(state->forbidden_union, new_value);
                bits &= bits - 1ULL;
            }
        }
    }
}

static int can_add(DenseState *state, uint64_t column_mask) {
    return !get_bit(state->forbidden_union, column_mask);
}

static int mask_in_list(const uint64_t *values, int count, uint64_t mask) {
    if (!values || count <= 0) {
        return 0;
    }
    for (int i = 0; i < count; i++) {
        if (values[i] == mask) {
            return 1;
        }
    }
    return 0;
}

static uint64_t forbidden_count(DenseState *state) {
    uint64_t count = 0;
    for (uint64_t word_index = 0; word_index < state->word_count; word_index++) {
        count += (uint64_t)popcount64(state->forbidden_union[word_index]);
    }
    return count;
}

static uint64_t generated_values_for_add(DenseState *state) {
    uint64_t total = 0;
    for (int subset_size = 1; subset_size <= state->max_subset_size; subset_size++) {
        total += state->layer_counts[subset_size - 1];
    }
    return total;
}

static uint64_t estimate_forbidden_growth(
    DenseState *state,
    uint64_t column_mask,
    uint32_t *scratch_marks,
    uint32_t *scratch_epoch
) {
    uint64_t new_count = 0;
    if (state->max_subset_size <= 0) {
        return 0;
    }
    uint32_t epoch = ++(*scratch_epoch);
    if (epoch == 0U) {
        memset(scratch_marks, 0, (size_t)state->value_count * sizeof(uint32_t));
        epoch = ++(*scratch_epoch);
    }
    for (int subset_size = state->max_subset_size; subset_size >= 1; subset_size--) {
        uint64_t *previous = state->layers[subset_size - 1];
        for (uint64_t word_index = 0; word_index < state->word_count; word_index++) {
            uint64_t bits = previous[word_index];
            while (bits) {
#if defined(__GNUC__) || defined(__clang__)
                int offset = __builtin_ctzll(bits);
#else
                int offset = 0;
                uint64_t probe = bits;
                while ((probe & 1ULL) == 0) {
                    probe >>= 1;
                    offset++;
                }
#endif
                uint64_t value = (word_index << 6) + (uint64_t)offset;
                uint64_t new_value = value ^ column_mask;
                if (
                    !get_bit(state->forbidden_union, new_value)
                    && scratch_marks[new_value] != epoch
                ) {
                    scratch_marks[new_value] = epoch;
                    new_count++;
                }
                bits &= bits - 1ULL;
            }
        }
    }
    return new_count;
}

static int initialize_systematic_columns(DenseState *state, int r) {
    int max_weight = state->max_subset_size;
    if (max_weight > r) {
        max_weight = r;
    }
    for (int weight = 1; weight <= max_weight; weight++) {
        uint64_t mask = (1ULL << weight) - 1ULL;
        uint64_t limit = 1ULL << r;
        while (mask && mask < limit) {
            if (set_bit_if_new(state->layers[weight], mask)) {
                state->layer_counts[weight]++;
            }
            set_bit(state->forbidden_union, mask);
            mask = next_combination(mask);
        }
    }
    return 1;
}

static int rebuild_state_from_selection(
    DenseState *state,
    int r,
    int d,
    const uint64_t *selected,
    int selected_count
) {
    if (!init_state(state, r, d)) {
        return 0;
    }
    initialize_systematic_columns(state, r);
    for (int i = 0; i < selected_count; i++) {
        add_column(state, selected[i]);
    }
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

static uint64_t candidate_count_for_instance(int r, int d) {
    uint64_t count = 0;
    int min_weight = d - 1;
    if (min_weight < 1) {
        min_weight = 1;
    }
    for (int weight = min_weight; weight <= r; weight++) {
        uint64_t layer_count = binomial_u64(r, weight);
        if (UINT64_MAX - count < layer_count) {
            return UINT64_MAX;
        }
        count += layer_count;
        if (count > BASELINE_MAX_CANDIDATES) {
            return count;
        }
    }
    return count;
}

static uint64_t next_combination(uint64_t mask) {
    uint64_t smallest = mask & (~mask + 1ULL);
    uint64_t ripple = mask + smallest;
    if (ripple == 0) {
        return 0;
    }
    return ripple | (((mask ^ ripple) >> 2) / smallest);
}

static uint64_t generate_candidate_masks(Candidate *candidates, int r, int d) {
    uint64_t index = 0;
    uint64_t limit = 1ULL << r;
    int min_weight = d - 1;
    if (min_weight < 1) {
        min_weight = 1;
    }
    for (int weight = min_weight; weight <= r; weight++) {
        uint64_t mask = (1ULL << weight) - 1ULL;
        while (mask && mask < limit) {
            candidates[index].mask = mask;
            index++;
            mask = next_combination(mask);
        }
    }
    return index;
}

static void score_candidate_range(Candidate *candidates, uint64_t start, uint64_t end, int n, int k, int d) {
    for (uint64_t index = start; index < end; index++) {
        int weight = popcount64(candidates[index].mask);
        candidates[index].score = oe_linear_code_priority(
            candidates[index].mask,
            n,
            k,
            d,
            0,
            weight,
            0,
            0,
            0
        );
    }
}

static void *score_candidate_worker(void *arg) {
    ScoreJob *job = (ScoreJob *)arg;
    score_candidate_range(job->candidates, job->start, job->end, job->n, job->k, job->d);
    return NULL;
}

static int score_candidates(Candidate *candidates, uint64_t candidate_count, int n, int k, int d) {
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    int thread_count = cpu_count > 1 ? (int)cpu_count : 1;
    if (thread_count > 16) {
        thread_count = 16;
    }
    if (candidate_count < 65536ULL || thread_count <= 1) {
        score_candidate_range(candidates, 0, candidate_count, n, k, d);
        return 1;
    }
    if ((uint64_t)thread_count > candidate_count) {
        thread_count = (int)candidate_count;
    }

    pthread_t *threads = (pthread_t *)calloc((size_t)thread_count, sizeof(pthread_t));
    ScoreJob *jobs = (ScoreJob *)calloc((size_t)thread_count, sizeof(ScoreJob));
    if (!threads || !jobs) {
        free(threads);
        free(jobs);
        return 0;
    }

    uint64_t chunk_size = (candidate_count + (uint64_t)thread_count - 1ULL) / (uint64_t)thread_count;
    int created = 0;
    for (int thread_index = 0; thread_index < thread_count; thread_index++) {
        uint64_t start = (uint64_t)thread_index * chunk_size;
        uint64_t end = start + chunk_size;
        if (start >= candidate_count) {
            break;
        }
        if (end > candidate_count) {
            end = candidate_count;
        }
        jobs[thread_index].candidates = candidates;
        jobs[thread_index].start = start;
        jobs[thread_index].end = end;
        jobs[thread_index].n = n;
        jobs[thread_index].k = k;
        jobs[thread_index].d = d;
        if (pthread_create(&threads[thread_index], NULL, score_candidate_worker, &jobs[thread_index]) != 0) {
            for (int join_index = 0; join_index < created; join_index++) {
                pthread_join(threads[join_index], NULL);
            }
            free(threads);
            free(jobs);
            return 0;
        }
        created++;
    }
    for (int thread_index = 0; thread_index < created; thread_index++) {
        pthread_join(threads[thread_index], NULL);
    }
    free(threads);
    free(jobs);
    return 1;
}

static int choose_dynamic_candidate(
    Candidate *candidates,
    uint64_t candidate_count,
    DenseState *state,
    int n,
    int k,
    int d,
    int step,
    int restart,
    uint64_t current_forbidden_count,
    uint64_t window_size,
    const uint64_t *tabu_masks,
    int tabu_count,
    uint32_t *scratch_marks,
    uint32_t *scratch_epoch,
    uint64_t *selected_mask,
    uint64_t *selected_growth,
    uint64_t *blocked_out,
    uint64_t *dynamic_evaluations_out
) {
    double best_score = 0.0;
    uint32_t best_tie = 0;
    uint64_t best_mask = 0;
    uint64_t best_growth = 0;
    int found = 0;
    uint64_t legal_seen = 0;
    uint64_t generated_count = generated_values_for_add(state);

    for (uint64_t i = 0; i < candidate_count; i++) {
        uint64_t mask = candidates[i].mask;
        if (mask_in_list(tabu_masks, tabu_count, mask)) {
            continue;
        }
        if (!can_add(state, mask)) {
            (*blocked_out)++;
            continue;
        }
        uint64_t growth = estimate_forbidden_growth(
            state,
            mask,
            scratch_marks,
            scratch_epoch
        );
        uint64_t overlap = generated_count > growth ? generated_count - growth : 0;
        double dynamic_score = oe_linear_code_priority(
            mask,
            n,
            k,
            d,
            step,
            popcount64(mask),
            current_forbidden_count,
            growth,
            overlap
        );
        uint32_t tie = deterministic_tiebreak(mask, restart + step + 1);
        (*dynamic_evaluations_out)++;
        if (
            !found
            || dynamic_score > best_score
            || (dynamic_score == best_score && tie > best_tie)
            || (dynamic_score == best_score && tie == best_tie && mask > best_mask)
        ) {
            found = 1;
            best_score = dynamic_score;
            best_tie = tie;
            best_mask = mask;
            best_growth = growth;
        }
        legal_seen++;
        if (legal_seen >= window_size) {
            break;
        }
    }

    if (!found) {
        return 0;
    }
    *selected_mask = best_mask;
    *selected_growth = best_growth;
    return 1;
}

static uint64_t count_legal_candidates_in_prefix(
    Candidate *candidates,
    uint64_t candidate_count,
    DenseState *state,
    uint64_t prefix_size,
    const uint64_t *selected,
    int selected_count,
    const uint64_t *tabu_masks,
    int tabu_count
) {
    uint64_t legal_count = 0;
    uint64_t limit = prefix_size;
    if (limit == 0 || limit > candidate_count) {
        limit = candidate_count;
    }
    for (uint64_t i = 0; i < limit; i++) {
        uint64_t mask = candidates[i].mask;
        if (
            mask_in_list(selected, selected_count, mask)
            || mask_in_list(tabu_masks, tabu_count, mask)
        ) {
            continue;
        }
        if (can_add(state, mask)) {
            legal_count++;
        }
    }
    return legal_count;
}

static int choose_repair_drop_index(
    Candidate *candidates,
    uint64_t candidate_count,
    const uint64_t *selected,
    int selected_count,
    int r,
    int d,
    int restart,
    int repair_event_index,
    uint64_t before_forbidden_count,
    uint64_t repair_candidate_window,
    const uint64_t *tabu_masks,
    int tabu_count,
    int *drop_index_out,
    uint64_t *after_forbidden_count_out
) {
    int found = 0;
    int best_index = -1;
    uint64_t best_legal_count = 0;
    uint64_t best_forbidden_release = 0;
    uint32_t best_tie = 0;
    uint64_t best_after_forbidden_count = 0;

    if (selected_count <= 0) {
        return 0;
    }

    for (int drop_index = 0; drop_index < selected_count; drop_index++) {
        DenseState trial_state;
        if (!init_state(&trial_state, r, d)) {
            return 0;
        }
        initialize_systematic_columns(&trial_state, r);
        for (int i = 0; i < selected_count; i++) {
            if (i != drop_index) {
                add_column(&trial_state, selected[i]);
            }
        }

        uint64_t after_forbidden_count = forbidden_count(&trial_state);
        uint64_t forbidden_release = (
            before_forbidden_count > after_forbidden_count
            ? before_forbidden_count - after_forbidden_count
            : 0
        );
        uint64_t legal_count = count_legal_candidates_in_prefix(
            candidates,
            candidate_count,
            &trial_state,
            repair_candidate_window,
            selected,
            selected_count,
            tabu_masks,
            tabu_count
        );
        uint64_t dropped_mask = selected[drop_index];
        uint32_t tie = deterministic_tiebreak(
            dropped_mask,
            restart + repair_event_index + 1009
        );
        free_state(&trial_state);

        if (
            !found
            || legal_count > best_legal_count
            || (
                legal_count == best_legal_count
                && forbidden_release > best_forbidden_release
            )
            || (
                legal_count == best_legal_count
                && forbidden_release == best_forbidden_release
                && tie > best_tie
            )
        ) {
            found = 1;
            best_index = drop_index;
            best_legal_count = legal_count;
            best_forbidden_release = forbidden_release;
            best_tie = tie;
            best_after_forbidden_count = after_forbidden_count;
        }
    }

    if (!found || (best_legal_count == 0 && best_forbidden_release == 0)) {
        return 0;
    }
    *drop_index_out = best_index;
    *after_forbidden_count_out = best_after_forbidden_count;
    return 1;
}

static int reward_is_better(RepairReward candidate, RepairReward current, int has_current) {
    if (!has_current) {
        return 1;
    }
    if (candidate.constructed_count != current.constructed_count) {
        return candidate.constructed_count > current.constructed_count;
    }
    if (candidate.dropped_count != current.dropped_count) {
        return candidate.dropped_count < current.dropped_count;
    }
    if (candidate.steps != current.steps) {
        return candidate.steps < current.steps;
    }
    return candidate.forbidden_count_value < current.forbidden_count_value;
}

static int rebuild_after_drop(
    DenseState *state,
    int r,
    int d,
    const uint64_t *selected,
    int selected_count,
    int drop_index,
    uint64_t *out_selected
) {
    int out_count = 0;
    for (int i = 0; i < selected_count; i++) {
        if (i != drop_index) {
            out_selected[out_count++] = selected[i];
        }
    }
    if (!rebuild_state_from_selection(state, r, d, out_selected, out_count)) {
        return -1;
    }
    return out_count;
}

static int rollout_after_first_drop(
    Candidate *candidates,
    uint64_t candidate_count,
    const uint64_t *selected,
    int selected_count,
    int first_drop_index,
    int n,
    int k,
    int d,
    int restart,
    int repair_event_index,
    uint64_t seed,
    uint64_t dynamic_window,
    uint64_t rollout_depth,
    const uint64_t *tabu_masks,
    int tabu_count,
    RepairReward *reward_out,
    uint64_t *after_first_drop_forbidden_out,
    uint64_t *dynamic_evaluations_out
) {
    int r = n - k;
    uint64_t *trial_selected = (uint64_t *)calloc((size_t)k, sizeof(uint64_t));
    uint64_t *trial_tabu = (uint64_t *)calloc((size_t)(tabu_count + rollout_depth + 1ULL), sizeof(uint64_t));
    if (!trial_selected || !trial_tabu) {
        free(trial_selected);
        free(trial_tabu);
        return 0;
    }
    for (int i = 0; i < tabu_count; i++) {
        trial_tabu[i] = tabu_masks[i];
    }
    int trial_tabu_count = tabu_count;
    int trial_tabu_capacity = tabu_count + (int)rollout_depth + 1;

    DenseState trial_state;
    int trial_count = rebuild_after_drop(
        &trial_state,
        r,
        d,
        selected,
        selected_count,
        first_drop_index,
        trial_selected
    );
    if (trial_count < 0) {
        free(trial_selected);
        free(trial_tabu);
        return 0;
    }
    uint64_t first_drop_forbidden_count = forbidden_count(&trial_state);
    uint64_t current_forbidden_count = first_drop_forbidden_count;
    trial_tabu[trial_tabu_count++] = selected[first_drop_index];

    uint32_t *scratch_marks = (uint32_t *)calloc((size_t)trial_state.value_count, sizeof(uint32_t));
    if (!scratch_marks) {
        free_state(&trial_state);
        free(trial_selected);
        free(trial_tabu);
        return 0;
    }
    uint32_t scratch_epoch = 0;
    uint64_t rng_state = mix64(
        seed
        ^ ((uint64_t)restart << 32)
        ^ ((uint64_t)repair_event_index << 16)
        ^ (uint64_t)(first_drop_index + 1)
    );
    int dropped_count = 1;
    int steps = 0;
    uint64_t dynamic_evaluations = 0;

    while (trial_count < k && (uint64_t)steps < rollout_depth) {
        uint64_t mask = 0;
        uint64_t growth = 0;
        uint64_t evals = 0;
        uint64_t blocked = 0;
        if (choose_dynamic_candidate(
            candidates,
            candidate_count,
            &trial_state,
            n,
            k,
            d,
            trial_count,
            restart + repair_event_index + steps + 1,
            current_forbidden_count,
            dynamic_window,
            trial_tabu,
            trial_tabu_count,
            scratch_marks,
            &scratch_epoch,
            &mask,
            &growth,
            &blocked,
            &evals
        )) {
            add_column(&trial_state, mask);
            current_forbidden_count += growth;
            trial_selected[trial_count++] = mask;
            dynamic_evaluations += evals;
            steps++;
            continue;
        }

        dynamic_evaluations += evals;
        if (trial_count <= 0 || trial_tabu_count >= trial_tabu_capacity) {
            break;
        }
        int drop_index = (int)(rng_next(&rng_state) % (uint64_t)trial_count);
        uint64_t dropped_mask = trial_selected[drop_index];
        for (int shift_index = drop_index + 1; shift_index < trial_count; shift_index++) {
            trial_selected[shift_index - 1] = trial_selected[shift_index];
        }
        trial_count--;
        trial_tabu[trial_tabu_count++] = dropped_mask;

        DenseState rebuilt_state;
        if (!rebuild_state_from_selection(&rebuilt_state, r, d, trial_selected, trial_count)) {
            break;
        }
        free_state(&trial_state);
        trial_state = rebuilt_state;
        current_forbidden_count = forbidden_count(&trial_state);
        dropped_count++;
        steps++;
    }

    reward_out->constructed_count = trial_count;
    reward_out->dropped_count = dropped_count;
    reward_out->steps = steps;
    reward_out->forbidden_count_value = current_forbidden_count;
    *after_first_drop_forbidden_out = first_drop_forbidden_count;
    *dynamic_evaluations_out = dynamic_evaluations;

    free(scratch_marks);
    free_state(&trial_state);
    free(trial_selected);
    free(trial_tabu);
    return 1;
}

static int choose_repair_drop_index_mcts(
    Candidate *candidates,
    uint64_t candidate_count,
    const uint64_t *selected,
    int selected_count,
    int n,
    int k,
    int d,
    int restart,
    int repair_event_index,
    uint64_t seed,
    uint64_t dynamic_window,
    uint64_t rollout_depth,
    uint64_t simulations,
    const uint64_t *tabu_masks,
    int tabu_count,
    int *drop_index_out,
    uint64_t *after_forbidden_count_out,
    uint64_t *dynamic_evaluations_out
) {
    if (selected_count <= 0 || rollout_depth == 0 || simulations == 0) {
        return 0;
    }
    RepairReward best_reward = {0, 0, 0, 0};
    int has_best = 0;
    int best_drop_index = -1;
    uint64_t best_after_forbidden = 0;
    uint64_t total_dynamic_evaluations = 0;

    for (uint64_t simulation = 0; simulation < simulations; simulation++) {
        int drop_index = (int)(simulation % (uint64_t)selected_count);
        RepairReward reward;
        uint64_t after_forbidden = 0;
        uint64_t evals = 0;
        if (!rollout_after_first_drop(
            candidates,
            candidate_count,
            selected,
            selected_count,
            drop_index,
            n,
            k,
            d,
            restart,
            repair_event_index,
            seed + simulation * 0x9e3779b97f4a7c15ULL,
            dynamic_window,
            rollout_depth,
            tabu_masks,
            tabu_count,
            &reward,
            &after_forbidden,
            &evals
        )) {
            continue;
        }
        total_dynamic_evaluations += evals;
        if (reward_is_better(reward, best_reward, has_best)) {
            best_reward = reward;
            best_drop_index = drop_index;
            best_after_forbidden = after_forbidden;
            has_best = 1;
        }
    }

    if (!has_best || best_reward.constructed_count < selected_count) {
        return 0;
    }
    *drop_index_out = best_drop_index;
    *after_forbidden_count_out = best_after_forbidden;
    *dynamic_evaluations_out = total_dynamic_evaluations;
    return 1;
}

int oe_linear_code_run(
    int n,
    int k,
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
    double c_run_started_at = monotonic_seconds();
    if (metrics_out && metrics_cap > 0) {
        for (int i = 0; i < metrics_cap; i++) {
            metrics_out[i] = 0.0;
        }
    }
    if (!selected_out || selected_cap < k || !metrics_out || metrics_cap <= METRIC_FORBIDDEN_COUNT) {
        write_error(error_out, error_cap, "invalid output buffers");
        return -1;
    }
    if (n <= 0 || k <= 0 || d <= 0 || k >= n) {
        write_error(error_out, error_cap, "invalid instance parameters");
        return -2;
    }
    int r = n - k;
    if (r <= 0 || r > BASELINE_R_LIMIT) {
        write_error(error_out, error_cap, "baseline C kernel requires 0 < r <= 60");
        return -3;
    }
    if (d > n || d - 1 > r) {
        write_error(error_out, error_cap, "distance is too large for baseline C search");
        return -4;
    }
    if (restarts <= 0) {
        restarts = 1;
    }

    uint64_t candidate_count = candidate_count_for_instance(r, d);
    if (candidate_count > BASELINE_MAX_CANDIDATES) {
        write_error(error_out, error_cap, "too many candidates for baseline C kernel");
        return -5;
    }
    Candidate *candidates = (Candidate *)calloc((size_t)candidate_count, sizeof(Candidate));
    if (!candidates) {
        write_error(error_out, error_cap, "candidate allocation failed");
        return -6;
    }

    double stage_started_at = monotonic_seconds();
    uint64_t generated_count = generate_candidate_masks(candidates, r, d);
    metrics_out[METRIC_CANDIDATE_GENERATION_SECONDS] = monotonic_seconds() - stage_started_at;
    if (generated_count != candidate_count) {
        free(candidates);
        write_error(error_out, error_cap, "candidate generation count mismatch");
        return -7;
    }
    stage_started_at = monotonic_seconds();
    if (!score_candidates(candidates, candidate_count, n, k, d)) {
        free(candidates);
        write_error(error_out, error_cap, "candidate scoring failed");
        return -8;
    }
    metrics_out[METRIC_CANDIDATE_SCORING_SECONDS] = monotonic_seconds() - stage_started_at;

    uint64_t *best_selected = (uint64_t *)calloc((size_t)k, sizeof(uint64_t));
    if (!best_selected) {
        free(candidates);
        write_error(error_out, error_cap, "best selection allocation failed");
        return -9;
    }

    int best_count = 0;
    int best_restart = 0;
    uint64_t best_blocked = 0;
    uint64_t best_forbidden_count = 0;
    uint64_t dynamic_window = env_u64("LINEAR_CODE_DYNAMIC_WINDOW", 4096ULL);
    uint64_t repair_max_events = env_u64("LINEAR_CODE_REPAIR_EVENTS", 4ULL);
    uint64_t repair_drop_count = env_u64("LINEAR_CODE_REPAIR_DROP_COUNT", 1ULL);
    if (repair_drop_count == 0) {
        repair_drop_count = 1;
    }
    uint64_t default_repair_candidate_window = dynamic_window > 0 ? dynamic_window : 4096ULL;
    if (default_repair_candidate_window < UINT64_MAX / 16ULL) {
        default_repair_candidate_window *= 16ULL;
    }
    uint64_t repair_candidate_window = env_u64(
        "LINEAR_CODE_REPAIR_CANDIDATE_WINDOW",
        default_repair_candidate_window
    );
    uint64_t repair_tabu_tenure = env_u64(
        "LINEAR_CODE_REPAIR_TABU_TENURE",
        repair_max_events * repair_drop_count
    );
    int repair_mcts_enabled = env_equals("LINEAR_CODE_REPAIR_MODE", "mcts");
    uint64_t repair_mcts_simulations = env_u64("LINEAR_CODE_REPAIR_MCTS_SIMULATIONS", 64ULL);
    uint64_t repair_mcts_depth = env_u64("LINEAR_CODE_REPAIR_MCTS_DEPTH", 4ULL);
    uint64_t repair_tabu_cap_u64 = repair_tabu_tenure;
    uint64_t max_reasonable_tabu = (uint64_t)k + repair_max_events * repair_drop_count + 1ULL;
    if (repair_tabu_cap_u64 > max_reasonable_tabu) {
        repair_tabu_cap_u64 = max_reasonable_tabu;
    }
    uint64_t total_dynamic_evaluations = 0;
    uint64_t total_repair_events = 0;
    for (int restart = 0; restart < restarts; restart++) {
        for (uint64_t i = 0; i < candidate_count; i++) {
            candidates[i].tie = deterministic_tiebreak(candidates[i].mask, restart);
        }
        stage_started_at = monotonic_seconds();
        qsort(candidates, (size_t)candidate_count, sizeof(Candidate), compare_candidates_desc);
        metrics_out[METRIC_RESTART_SORT_SECONDS] += monotonic_seconds() - stage_started_at;

        stage_started_at = monotonic_seconds();
        DenseState state;
        if (!init_state(&state, r, d)) {
            free(candidates);
            free(best_selected);
            write_error(error_out, error_cap, "state allocation failed");
            return -10;
        }
        initialize_systematic_columns(&state, r);
        metrics_out[METRIC_STATE_INIT_SECONDS] += monotonic_seconds() - stage_started_at;
        uint64_t current_forbidden_count = forbidden_count(&state);
        uint64_t *current_selected = (uint64_t *)calloc((size_t)k, sizeof(uint64_t));
        if (!current_selected) {
            free_state(&state);
            free(candidates);
            free(best_selected);
            write_error(error_out, error_cap, "selection allocation failed");
            return -11;
        }
        uint32_t *scratch_marks = NULL;
        uint32_t scratch_epoch = 0;
        if (dynamic_window > 0) {
            scratch_marks = (uint32_t *)calloc((size_t)state.value_count, sizeof(uint32_t));
            if (!scratch_marks) {
                free(current_selected);
                free_state(&state);
                free(candidates);
                free(best_selected);
                write_error(error_out, error_cap, "dynamic scratch allocation failed");
                return -12;
            }
        }
        uint64_t *tabu_masks = NULL;
        int tabu_count = 0;
        int tabu_next = 0;
        int tabu_capacity = (int)repair_tabu_cap_u64;
        if (dynamic_window > 0 && repair_max_events > 0 && tabu_capacity > 0) {
            tabu_masks = (uint64_t *)calloc((size_t)tabu_capacity, sizeof(uint64_t));
            if (!tabu_masks) {
                free(scratch_marks);
                free(current_selected);
                free_state(&state);
                free(candidates);
                free(best_selected);
                write_error(error_out, error_cap, "repair tabu allocation failed");
                return -13;
            }
        }

        int current_count = 0;
        int current_repair_events = 0;
        uint64_t blocked = 0;
        stage_started_at = monotonic_seconds();
        if (dynamic_window > 0) {
            while (current_count < k) {
                uint64_t mask = 0;
                uint64_t growth = 0;
                uint64_t dynamic_evaluations = 0;
                if (!choose_dynamic_candidate(
                    candidates,
                    candidate_count,
                    &state,
                    n,
                    k,
                    d,
                    current_count,
                    restart,
                    current_forbidden_count,
                    dynamic_window,
                    tabu_masks,
                    tabu_count,
                    scratch_marks,
                    &scratch_epoch,
                    &mask,
                    &growth,
                    &blocked,
                    &dynamic_evaluations
                )) {
                    int repaired = 0;
                    if (
                        repair_max_events == 0
                        || (uint64_t)current_repair_events >= repair_max_events
                        || current_count <= 0
                    ) {
                        break;
                    }
                    for (
                        uint64_t drop_event = 0;
                        drop_event < repair_drop_count
                            && (uint64_t)current_repair_events < repair_max_events
                            && current_count > 0;
                        drop_event++
                    ) {
                        int drop_index = -1;
                        uint64_t repaired_forbidden_count = 0;
                        uint64_t repair_dynamic_evaluations = 0;
                        int chose_repair = 0;
                        if (repair_mcts_enabled) {
                            chose_repair = choose_repair_drop_index_mcts(
                                candidates,
                                candidate_count,
                                current_selected,
                                current_count,
                                n,
                                k,
                                d,
                                restart,
                                current_repair_events,
                                seed,
                                dynamic_window,
                                repair_mcts_depth,
                                repair_mcts_simulations,
                                tabu_masks,
                                tabu_count,
                                &drop_index,
                                &repaired_forbidden_count,
                                &repair_dynamic_evaluations
                            );
                            total_dynamic_evaluations += repair_dynamic_evaluations;
                        }
                        if (!chose_repair) {
                            chose_repair = choose_repair_drop_index(
                                candidates,
                                candidate_count,
                                current_selected,
                                current_count,
                                r,
                                d,
                                restart,
                                current_repair_events,
                                current_forbidden_count,
                                repair_candidate_window,
                                tabu_masks,
                                tabu_count,
                                &drop_index,
                                &repaired_forbidden_count
                            );
                        }
                        if (!chose_repair) {
                            break;
                        }
                        uint64_t dropped_mask = current_selected[drop_index];
                        for (int shift_index = drop_index + 1; shift_index < current_count; shift_index++) {
                            current_selected[shift_index - 1] = current_selected[shift_index];
                        }
                        current_count--;
                        if (tabu_capacity > 0) {
                            tabu_masks[tabu_next] = dropped_mask;
                            tabu_next = (tabu_next + 1) % tabu_capacity;
                            if (tabu_count < tabu_capacity) {
                                tabu_count++;
                            }
                        }
                        DenseState rebuilt_state;
                        if (!rebuild_state_from_selection(
                            &rebuilt_state,
                            r,
                            d,
                            current_selected,
                            current_count
                        )) {
                            free(tabu_masks);
                            free(scratch_marks);
                            free(current_selected);
                            free_state(&state);
                            free(candidates);
                            free(best_selected);
                            write_error(error_out, error_cap, "repair rebuild failed");
                            return -14;
                        }
                        free_state(&state);
                        state = rebuilt_state;
                        current_forbidden_count = repaired_forbidden_count;
                        current_repair_events++;
                        total_repair_events++;
                        repaired = 1;
                    }
                    if (!repaired) {
                        break;
                    }
                    continue;
                }
                add_column(&state, mask);
                current_forbidden_count += growth;
                current_selected[current_count++] = mask;
                total_dynamic_evaluations += dynamic_evaluations;
                if (current_count > best_count) {
                    best_count = current_count;
                    best_restart = restart;
                    best_blocked = blocked;
                    best_forbidden_count = current_forbidden_count;
                    memcpy(best_selected, current_selected, (size_t)current_count * sizeof(uint64_t));
                }
            }
        } else {
            for (uint64_t i = 0; i < candidate_count && current_count < k; i++) {
                uint64_t mask = candidates[i].mask;
                if (can_add(&state, mask)) {
                    add_column(&state, mask);
                    current_selected[current_count++] = mask;
                } else {
                    blocked++;
                }
            }
        }
        metrics_out[METRIC_GREEDY_SCAN_SECONDS] += monotonic_seconds() - stage_started_at;

        if (current_count > best_count) {
            best_count = current_count;
            best_restart = restart;
            best_blocked = blocked;
            stage_started_at = monotonic_seconds();
            best_forbidden_count = forbidden_count(&state);
            metrics_out[METRIC_FORBIDDEN_COUNT_SECONDS] += monotonic_seconds() - stage_started_at;
            memcpy(best_selected, current_selected, (size_t)current_count * sizeof(uint64_t));
        }
        free(tabu_masks);
        free(scratch_marks);
        free(current_selected);
        free_state(&state);
        if (best_count == k) {
            break;
        }
    }

    for (int i = 0; i < best_count; i++) {
        selected_out[i] = best_selected[i];
    }
    metrics_out[METRIC_SUCCESS] = best_count == k ? 1.0 : 0.0;
    metrics_out[METRIC_CONSTRUCTED_COLUMNS] = (double)best_count;
    metrics_out[METRIC_CANDIDATE_COUNT] = (double)candidate_count;
    metrics_out[METRIC_SCORED_CANDIDATES] = (double)candidate_count;
    metrics_out[METRIC_SAMPLE_ATTEMPTS] = (double)total_dynamic_evaluations;
    metrics_out[METRIC_BACKTRACK_EVENTS] = (double)total_repair_events;
    metrics_out[METRIC_RESTART_INDEX] = (double)best_restart;
    metrics_out[METRIC_BLOCKED_CANDIDATES] = (double)best_blocked;
    metrics_out[METRIC_FORBIDDEN_COUNT] = (double)best_forbidden_count;
    metrics_out[METRIC_C_RUN_SECONDS] = monotonic_seconds() - c_run_started_at;

    free(best_selected);
    free(candidates);
    write_error(error_out, error_cap, "");
    return best_count == k ? 0 : 1;
}
