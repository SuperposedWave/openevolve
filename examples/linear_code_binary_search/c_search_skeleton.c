#define _POSIX_C_SOURCE 200809L

/*
 * Fixed C search skeleton for the binary linear-code feasibility example.
 *
 * OpenEvolve-generated C variants provide only oe_linear_code_priority().
 * This file owns the fixed ABI entry point, candidate enumeration, sorting,
 * exact legality checks, metrics, and selected-column output.
 */

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
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

static double g_verbose_started_at = 0.0;
static pthread_mutex_t g_verbose_log_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    uint64_t mask;
    double score;
    uint32_t tie;
} Candidate;

typedef struct {
    uint64_t **layers;
    uint64_t **layer_values;
    uint64_t *forbidden_union;
    uint64_t *layer_counts;
    uint64_t *layer_value_caps;
    int max_subset_size;
    int r;
    int d;
    int g_layer_legality;
    int sparse_layers_valid;
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
    Candidate *candidates;
    const uint64_t *legal_indices;
    uint64_t start;
    uint64_t end;
    DenseState *state;
    int n;
    int k;
    int d;
    int step;
    int restart;
    uint64_t current_forbidden_count;
    uint64_t generated_count;
    int estimate_growth;
    uint64_t scratch_touched_cap;
    int found;
    double best_score;
    uint32_t best_tie;
    uint64_t best_mask;
    uint64_t best_growth;
    uint64_t evaluations;
    int status;
} DynamicJob;

typedef struct {
    int constructed_count;
    int dropped_count;
    int steps;
    uint64_t forbidden_count_value;
} RepairReward;

typedef struct {
    int visits;
    int has_best;
    RepairReward best_reward;
    uint64_t best_after_forbidden;
    uint64_t success_count;
    uint64_t total_constructed;
    uint64_t total_dropped;
    uint64_t total_steps;
    uint64_t total_forbidden;
} RepairRootStats;

typedef struct {
    Candidate *candidates;
    uint64_t candidate_count;
    const uint64_t *selected;
    int selected_count;
    int n;
    int k;
    int d;
    int restart;
    int repair_event_index;
    uint64_t seed;
    uint64_t dynamic_window;
    uint64_t rollout_depth;
    uint64_t drop_topk;
    const uint64_t *tabu_masks;
    int tabu_count;
    int estimate_growth;
    uint64_t simulation_start;
    uint64_t simulation_end;
    RepairRootStats *root_stats;
    uint64_t dynamic_evaluations;
    double elapsed_seconds;
} MctsRolloutJob;

typedef struct {
    uint64_t *bits;
    uint64_t *touched_values;
    uint64_t touched_count;
    uint64_t touched_cap;
    uint64_t word_count;
    int overflowed;
} ScratchSet;

typedef struct {
    int restart;
    int count;
    uint64_t blocked;
    uint64_t forbidden_count;
    uint64_t dynamic_evaluations;
    uint64_t repair_events;
    double restart_sort_seconds;
    double state_init_seconds;
    double greedy_scan_seconds;
    double forbidden_count_seconds;
    uint64_t *selected;
    int status;
} RestartResult;

typedef struct {
    Candidate *base_candidates;
    uint64_t candidate_count;
    int n;
    int k;
    int d;
    int restart;
    uint64_t seed;
    uint64_t dynamic_window;
    uint64_t repair_max_events;
    uint64_t repair_drop_count;
    uint64_t repair_candidate_window;
    uint64_t repair_tabu_tenure;
    int repair_mcts_enabled;
    uint64_t repair_mcts_simulations;
    uint64_t repair_mcts_depth;
    uint64_t repair_mcts_drop_topk;
    int estimate_dynamic_growth;
    RestartResult *result;
} RestartJob;

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

static int verbose_progress_enabled(void) {
    const char *raw_value = getenv("LINEAR_CODE_VERBOSE_PROGRESS");
    if (!raw_value || !raw_value[0]) {
        return 0;
    }
    return strcmp(raw_value, "0") != 0
        && strcmp(raw_value, "false") != 0
        && strcmp(raw_value, "False") != 0
        && strcmp(raw_value, "no") != 0
        && strcmp(raw_value, "off") != 0;
}

static void verbose_progress_log(const char *format, ...) {
    if (!verbose_progress_enabled()) {
        return;
    }
    double started_at = g_verbose_started_at > 0.0 ? g_verbose_started_at : monotonic_seconds();
    double elapsed = monotonic_seconds() - started_at;
    pthread_mutex_lock(&g_verbose_log_mutex);
    fprintf(stderr, "[linear-code-c +%.3fs] ", elapsed);
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    fflush(stderr);
    pthread_mutex_unlock(&g_verbose_log_mutex);
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

static int init_scratch_set(ScratchSet *scratch, uint64_t word_count, uint64_t touched_cap) {
    memset(scratch, 0, sizeof(*scratch));
    scratch->word_count = word_count;
    scratch->touched_cap = touched_cap;
    scratch->bits = (uint64_t *)calloc((size_t)word_count, sizeof(uint64_t));
    if (!scratch->bits) {
        return 0;
    }
    if (touched_cap > 0ULL) {
        scratch->touched_values = (uint64_t *)calloc((size_t)touched_cap, sizeof(uint64_t));
        if (!scratch->touched_values) {
            free(scratch->bits);
            memset(scratch, 0, sizeof(*scratch));
            return 0;
        }
    }
    return 1;
}

static void clear_scratch_set(ScratchSet *scratch) {
    if (!scratch || !scratch->bits) {
        return;
    }
    if (scratch->overflowed) {
        memset(scratch->bits, 0, (size_t)scratch->word_count * sizeof(uint64_t));
    } else {
        for (uint64_t i = 0; i < scratch->touched_count; i++) {
            uint64_t value = scratch->touched_values[i];
            scratch->bits[value >> 6] &= ~(1ULL << (value & 63U));
        }
    }
    scratch->touched_count = 0;
    scratch->overflowed = 0;
}

static void free_scratch_set(ScratchSet *scratch) {
    if (!scratch) {
        return;
    }
    free(scratch->bits);
    free(scratch->touched_values);
    memset(scratch, 0, sizeof(*scratch));
}

static int scratch_set_if_new(ScratchSet *scratch, uint64_t value) {
    uint64_t word_index = value >> 6;
    uint64_t bit = 1ULL << (value & 63U);
    if (scratch->bits[word_index] & bit) {
        return 0;
    }
    scratch->bits[word_index] |= bit;
    if (!scratch->overflowed && scratch->touched_count < scratch->touched_cap) {
        scratch->touched_values[scratch->touched_count++] = value;
    } else {
        scratch->overflowed = 1;
    }
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

static int use_g_layer_legality(void) {
    return env_equals("LINEAR_CODE_LEGALITY_ENGINE", "g_layers")
        || env_equals("LINEAR_CODE_LEGALITY_ENGINE", "g");
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
    if (state->layer_values) {
        for (int i = 0; i <= state->max_subset_size; i++) {
            free(state->layer_values[i]);
        }
        free(state->layer_values);
    }
    free(state->forbidden_union);
    free(state->layer_counts);
    free(state->layer_value_caps);
    state->layers = NULL;
    state->layer_values = NULL;
    state->forbidden_union = NULL;
    state->layer_counts = NULL;
    state->layer_value_caps = NULL;
}

static void append_sparse_layer_value(DenseState *state, int subset_size, uint64_t value) {
    if (!state->g_layer_legality || !state->sparse_layers_valid) {
        return;
    }
    uint64_t index = state->layer_counts[subset_size];
    if (index >= state->layer_value_caps[subset_size]) {
        uint64_t old_cap = state->layer_value_caps[subset_size];
        uint64_t new_cap = old_cap > 0ULL ? old_cap * 2ULL : 16ULL;
        while (new_cap <= index) {
            new_cap *= 2ULL;
        }
        uint64_t *new_values = (uint64_t *)realloc(
            state->layer_values[subset_size],
            (size_t)new_cap * sizeof(uint64_t)
        );
        if (!new_values) {
            state->sparse_layers_valid = 0;
            return;
        }
        state->layer_values[subset_size] = new_values;
        state->layer_value_caps[subset_size] = new_cap;
    }
    state->layer_values[subset_size][index] = value;
}

static int init_state(DenseState *state, int r, int d) {
    memset(state, 0, sizeof(*state));
    state->r = r;
    state->d = d;
    state->g_layer_legality = use_g_layer_legality();
    state->sparse_layers_valid = state->g_layer_legality;
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
    if (state->g_layer_legality) {
        state->layer_values = (uint64_t **)calloc((size_t)state->max_subset_size + 1U, sizeof(uint64_t *));
        state->layer_value_caps = (uint64_t *)calloc((size_t)state->max_subset_size + 1U, sizeof(uint64_t));
        if (!state->layer_values || !state->layer_value_caps) {
            free_state(state);
            return 0;
        }
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
    append_sparse_layer_value(state, 0, 0);
    state->layer_counts[0] = 1;
    return 1;
}

static void add_column(DenseState *state, uint64_t column_mask) {
    if (state->g_layer_legality && state->sparse_layers_valid) {
        for (int subset_size = state->max_subset_size; subset_size >= 1; subset_size--) {
            uint64_t source_count = state->layer_counts[subset_size - 1];
            uint64_t *previous_values = state->layer_values[subset_size - 1];
            uint64_t *target = state->layers[subset_size];
            for (uint64_t index = 0; index < source_count; index++) {
                uint64_t new_value = previous_values[index] ^ column_mask;
                if (set_bit_if_new(target, new_value)) {
                    append_sparse_layer_value(state, subset_size, new_value);
                    state->layer_counts[subset_size]++;
                }
                set_bit(state->forbidden_union, new_value);
            }
        }
        return;
    }
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
    if (state->g_layer_legality) {
        if (state->sparse_layers_valid) {
            for (int subset_size = 0; subset_size <= state->max_subset_size; subset_size++) {
                int threshold = state->d - (subset_size + 1);
                if (threshold <= 0) {
                    continue;
                }
                uint64_t count = state->layer_counts[subset_size];
                uint64_t *values = state->layer_values[subset_size];
                for (uint64_t index = 0; index < count; index++) {
                    if (popcount64(values[index] ^ column_mask) < threshold) {
                        return 0;
                    }
                }
            }
            return 1;
        }
        for (int subset_size = 0; subset_size <= state->max_subset_size; subset_size++) {
            int threshold = state->d - (subset_size + 1);
            if (threshold <= 0) {
                continue;
            }
            uint64_t *layer = state->layers[subset_size];
            for (uint64_t word_index = 0; word_index < state->word_count; word_index++) {
                uint64_t bits = layer[word_index];
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
                    if (popcount64(value ^ column_mask) < threshold) {
                        return 0;
                    }
                    bits &= bits - 1ULL;
                }
            }
        }
        return 1;
    }
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
    ScratchSet *scratch
) {
    uint64_t new_count = 0;
    if (state->max_subset_size <= 0) {
        return 0;
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
                    && scratch_set_if_new(scratch, new_value)
                ) {
                    new_count++;
                }
                bits &= bits - 1ULL;
            }
        }
    }
    clear_scratch_set(scratch);
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
    if (!state->g_layer_legality) {
        initialize_systematic_columns(state, r);
    }
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
    }
    return count;
}

static uint64_t next_power_of_two_u64(uint64_t value) {
    if (value <= 1ULL) {
        return 1ULL;
    }
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return value + 1ULL;
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

static int insert_seen_mask(uint64_t *seen, uint64_t seen_cap, uint64_t mask) {
    uint64_t index = mix64(mask) & (seen_cap - 1ULL);
    while (seen[index] != 0ULL) {
        if (seen[index] == mask) {
            return 0;
        }
        index = (index + 1ULL) & (seen_cap - 1ULL);
    }
    seen[index] = mask;
    return 1;
}

static int choose_sample_weight(int r, int d, uint64_t total_count, uint64_t *rng_state) {
    int min_weight = d - 1;
    if (min_weight < 1) {
        min_weight = 1;
    }
    uint64_t rank = rng_next(rng_state) % total_count;
    for (int weight = min_weight; weight <= r; weight++) {
        uint64_t layer_count = binomial_u64(r, weight);
        if (rank < layer_count) {
            return weight;
        }
        rank -= layer_count;
    }
    return r;
}

static uint64_t random_mask_with_weight(int r, int weight, uint64_t *rng_state) {
    uint64_t mask = 0ULL;
    int chosen = 0;
    while (chosen < weight) {
        int bit_index = (int)(rng_next(rng_state) % (uint64_t)r);
        uint64_t bit = 1ULL << bit_index;
        if ((mask & bit) == 0ULL) {
            mask |= bit;
            chosen++;
        }
    }
    return mask;
}

static uint64_t generate_sampled_candidate_masks(
    Candidate *candidates,
    uint64_t sample_count,
    uint64_t total_count,
    int r,
    int d,
    uint64_t seed
) {
    uint64_t seen_cap = next_power_of_two_u64(sample_count * 4ULL);
    if (seen_cap < 16ULL) {
        seen_cap = 16ULL;
    }
    if (seen_cap < sample_count) {
        return 0ULL;
    }
    uint64_t *seen = (uint64_t *)calloc((size_t)seen_cap, sizeof(uint64_t));
    if (!seen) {
        return 0ULL;
    }

    uint64_t rng_state = mix64(seed ^ ((uint64_t)r << 32) ^ ((uint64_t)d << 16) ^ sample_count);
    uint64_t index = 0;
    uint64_t attempts = 0;
    uint64_t max_attempts = sample_count * 64ULL + 1024ULL;
    while (index < sample_count && attempts < max_attempts) {
        attempts++;
        int weight = choose_sample_weight(r, d, total_count, &rng_state);
        uint64_t mask = random_mask_with_weight(r, weight, &rng_state);
        if (mask != 0ULL && insert_seen_mask(seen, seen_cap, mask)) {
            candidates[index].mask = mask;
            index++;
        }
    }

    if (index < sample_count) {
        uint64_t limit = 1ULL << r;
        int min_weight = d - 1;
        if (min_weight < 1) {
            min_weight = 1;
        }
        for (int weight = min_weight; weight <= r && index < sample_count; weight++) {
            uint64_t mask = (1ULL << weight) - 1ULL;
            while (mask && mask < limit && index < sample_count) {
                if (insert_seen_mask(seen, seen_cap, mask)) {
                    candidates[index].mask = mask;
                    index++;
                }
                mask = next_combination(mask);
            }
        }
    }

    free(seen);
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
    uint64_t requested_workers = env_u64("LINEAR_CODE_CANDIDATE_WORKERS", UINT64_MAX);
    int thread_count = cpu_count > 1 ? (int)cpu_count : 1;
    if (requested_workers != UINT64_MAX) {
        if (requested_workers == 0ULL) {
            thread_count = cpu_count > 1 ? (int)cpu_count : 1;
        } else {
            thread_count = (int)requested_workers;
        }
    }
    if (thread_count < 1) {
        thread_count = 1;
    }
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

static int dynamic_choice_is_better(
    double candidate_score,
    uint32_t candidate_tie,
    uint64_t candidate_mask,
    double current_score,
    uint32_t current_tie,
    uint64_t current_mask,
    int has_current
) {
    return !has_current
        || candidate_score > current_score
        || (candidate_score == current_score && candidate_tie > current_tie)
        || (
            candidate_score == current_score
            && candidate_tie == current_tie
            && candidate_mask > current_mask
        );
}

static void run_dynamic_candidate_range(DynamicJob *job) {
    if (!job) {
        return;
    }
    ScratchSet scratch;
    memset(&scratch, 0, sizeof(scratch));
    ScratchSet *scratch_ptr = NULL;
    if (job->estimate_growth) {
        if (!init_scratch_set(&scratch, job->state->word_count, job->scratch_touched_cap)) {
            job->status = -1;
            return;
        }
        scratch_ptr = &scratch;
    }

    for (uint64_t pos = job->start; pos < job->end; pos++) {
        Candidate *candidate = &job->candidates[job->legal_indices[pos]];
        uint64_t mask = candidate->mask;
        uint64_t growth = job->estimate_growth
            ? estimate_forbidden_growth(job->state, mask, scratch_ptr)
            : 0ULL;
        uint64_t overlap = job->estimate_growth && job->generated_count > growth
            ? job->generated_count - growth
            : 0ULL;
        double dynamic_score = oe_linear_code_priority(
            mask,
            job->n,
            job->k,
            job->d,
            job->step,
            popcount64(mask),
            job->current_forbidden_count,
            growth,
            overlap
        );
        uint32_t tie = deterministic_tiebreak(mask, job->restart + job->step + 1);
        job->evaluations++;
        if (dynamic_choice_is_better(
            dynamic_score,
            tie,
            mask,
            job->best_score,
            job->best_tie,
            job->best_mask,
            job->found
        )) {
            job->found = 1;
            job->best_score = dynamic_score;
            job->best_tie = tie;
            job->best_mask = mask;
            job->best_growth = growth;
        }
    }

    free_scratch_set(&scratch);
}

static void *dynamic_candidate_worker(void *arg) {
    run_dynamic_candidate_range((DynamicJob *)arg);
    return NULL;
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
    ScratchSet *scratch,
    int estimate_growth,
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
    uint64_t generated_count = estimate_growth ? generated_values_for_add(state) : 0ULL;
    uint64_t legal_target = window_size == 0ULL ? 1ULL : window_size;
    if (legal_target > candidate_count) {
        legal_target = candidate_count;
    }
    uint64_t *legal_indices = (uint64_t *)calloc((size_t)legal_target, sizeof(uint64_t));
    if (!legal_indices) {
        return 0;
    }

    for (uint64_t i = 0; i < candidate_count; i++) {
        uint64_t mask = candidates[i].mask;
        if (mask_in_list(tabu_masks, tabu_count, mask)) {
            continue;
        }
        if (!can_add(state, mask)) {
            (*blocked_out)++;
            continue;
        }
        legal_indices[legal_seen++] = i;
        if (legal_seen >= legal_target) {
            break;
        }
    }

    if (legal_seen == 0) {
        free(legal_indices);
        return 0;
    }

    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    uint64_t requested_workers = env_u64("LINEAR_CODE_DYNAMIC_WORKERS", 1ULL);
    if (requested_workers == 0ULL) {
        requested_workers = cpu_count > 1 ? (uint64_t)cpu_count : 1ULL;
    }
    if (requested_workers > legal_seen) {
        requested_workers = legal_seen;
    }
    if (requested_workers > 64ULL) {
        requested_workers = 64ULL;
    }

    if (requested_workers <= 1ULL) {
        DynamicJob job;
        memset(&job, 0, sizeof(job));
        job.candidates = candidates;
        job.legal_indices = legal_indices;
        job.start = 0;
        job.end = legal_seen;
        job.state = state;
        job.n = n;
        job.k = k;
        job.d = d;
        job.step = step;
        job.restart = restart;
        job.current_forbidden_count = current_forbidden_count;
        job.generated_count = generated_count;
        job.estimate_growth = estimate_growth;
        job.scratch_touched_cap = scratch ? scratch->touched_cap : env_u64("LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP", 1048576ULL);
        run_dynamic_candidate_range(&job);
        if (job.status != 0 || !job.found) {
            free(legal_indices);
            return 0;
        }
        found = job.found;
        best_score = job.best_score;
        best_tie = job.best_tie;
        best_mask = job.best_mask;
        best_growth = job.best_growth;
        *dynamic_evaluations_out += job.evaluations;
    } else {
        pthread_t *threads = (pthread_t *)calloc((size_t)requested_workers, sizeof(pthread_t));
        DynamicJob *jobs = (DynamicJob *)calloc((size_t)requested_workers, sizeof(DynamicJob));
        if (!threads || !jobs) {
            free(threads);
            free(jobs);
            free(legal_indices);
            return 0;
        }
        uint64_t chunk_size = (legal_seen + requested_workers - 1ULL) / requested_workers;
        uint64_t created = 0;
        for (uint64_t worker = 0; worker < requested_workers; worker++) {
            uint64_t start = worker * chunk_size;
            uint64_t end = start + chunk_size;
            if (start >= legal_seen) {
                break;
            }
            if (end > legal_seen) {
                end = legal_seen;
            }
            DynamicJob *job = &jobs[worker];
            job->candidates = candidates;
            job->legal_indices = legal_indices;
            job->start = start;
            job->end = end;
            job->state = state;
            job->n = n;
            job->k = k;
            job->d = d;
            job->step = step;
            job->restart = restart;
            job->current_forbidden_count = current_forbidden_count;
            job->generated_count = generated_count;
            job->estimate_growth = estimate_growth;
            job->scratch_touched_cap = scratch ? scratch->touched_cap : env_u64("LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP", 1048576ULL);
            if (pthread_create(&threads[worker], NULL, dynamic_candidate_worker, job) != 0) {
                for (uint64_t join_index = 0; join_index < created; join_index++) {
                    pthread_join(threads[join_index], NULL);
                }
                free(threads);
                free(jobs);
                free(legal_indices);
                return 0;
            }
            created++;
        }

        for (uint64_t worker = 0; worker < created; worker++) {
            pthread_join(threads[worker], NULL);
            DynamicJob *job = &jobs[worker];
            if (job->status != 0) {
                free(threads);
                free(jobs);
                free(legal_indices);
                return 0;
            }
            *dynamic_evaluations_out += job->evaluations;
            if (
                job->found
                && dynamic_choice_is_better(
                    job->best_score,
                    job->best_tie,
                    job->best_mask,
                    best_score,
                    best_tie,
                    best_mask,
                    found
                )
            ) {
                found = 1;
                best_score = job->best_score;
                best_tie = job->best_tie;
                best_mask = job->best_mask;
                best_growth = job->best_growth;
            }
        }
        free(threads);
        free(jobs);
    }

    free(legal_indices);
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
        if (!trial_state.g_layer_legality) {
            initialize_systematic_columns(&trial_state, r);
        }
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

static int ratio_greater(uint64_t left_num, uint64_t left_den, uint64_t right_num, uint64_t right_den) {
    return (__uint128_t)left_num * (__uint128_t)right_den
        > (__uint128_t)right_num * (__uint128_t)left_den;
}

static int ratio_less(uint64_t left_num, uint64_t left_den, uint64_t right_num, uint64_t right_den) {
    return (__uint128_t)left_num * (__uint128_t)right_den
        < (__uint128_t)right_num * (__uint128_t)left_den;
}

static int root_stats_is_better(
    const RepairRootStats *candidate,
    const RepairRootStats *current,
    int has_current
) {
    if (!candidate || candidate->visits <= 0 || !candidate->has_best) {
        return 0;
    }
    if (!has_current || !current || current->visits <= 0 || !current->has_best) {
        return 1;
    }
    uint64_t candidate_visits = (uint64_t)candidate->visits;
    uint64_t current_visits = (uint64_t)current->visits;
    if (ratio_greater(candidate->success_count, candidate_visits, current->success_count, current_visits)) {
        return 1;
    }
    if (ratio_greater(current->success_count, current_visits, candidate->success_count, candidate_visits)) {
        return 0;
    }
    if (reward_is_better(candidate->best_reward, current->best_reward, current->has_best)) {
        return 1;
    }
    if (reward_is_better(current->best_reward, candidate->best_reward, candidate->has_best)) {
        return 0;
    }
    if (ratio_greater(candidate->total_constructed, candidate_visits, current->total_constructed, current_visits)) {
        return 1;
    }
    if (ratio_greater(current->total_constructed, current_visits, candidate->total_constructed, candidate_visits)) {
        return 0;
    }
    if (ratio_less(candidate->total_dropped, candidate_visits, current->total_dropped, current_visits)) {
        return 1;
    }
    if (ratio_less(current->total_dropped, current_visits, candidate->total_dropped, candidate_visits)) {
        return 0;
    }
    if (ratio_less(candidate->total_steps, candidate_visits, current->total_steps, current_visits)) {
        return 1;
    }
    if (ratio_less(current->total_steps, current_visits, candidate->total_steps, candidate_visits)) {
        return 0;
    }
    return ratio_less(candidate->total_forbidden, candidate_visits, current->total_forbidden, current_visits);
}

static void update_root_stats(
    RepairRootStats *stats,
    RepairReward reward,
    uint64_t after_forbidden,
    int original_selected_count
) {
    stats->visits++;
    stats->total_constructed += (uint64_t)reward.constructed_count;
    stats->total_dropped += (uint64_t)reward.dropped_count;
    stats->total_steps += (uint64_t)reward.steps;
    stats->total_forbidden += reward.forbidden_count_value;
    if (reward.constructed_count >= original_selected_count) {
        stats->success_count++;
    }
    if (reward_is_better(reward, stats->best_reward, stats->has_best)) {
        stats->best_reward = reward;
        stats->best_after_forbidden = after_forbidden;
        stats->has_best = 1;
    }
}

static void merge_root_stats(RepairRootStats *target, const RepairRootStats *source) {
    if (!target || !source || source->visits <= 0) {
        return;
    }
    target->visits += source->visits;
    target->success_count += source->success_count;
    target->total_constructed += source->total_constructed;
    target->total_dropped += source->total_dropped;
    target->total_steps += source->total_steps;
    target->total_forbidden += source->total_forbidden;
    if (
        source->has_best
        && reward_is_better(source->best_reward, target->best_reward, target->has_best)
    ) {
        target->best_reward = source->best_reward;
        target->best_after_forbidden = source->best_after_forbidden;
        target->has_best = 1;
    }
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

static int drop_choice_is_better(
    uint64_t candidate_legal_count,
    uint64_t candidate_release,
    uint32_t candidate_tie,
    uint64_t current_legal_count,
    uint64_t current_release,
    uint32_t current_tie
) {
    return candidate_legal_count > current_legal_count
        || (
            candidate_legal_count == current_legal_count
            && candidate_release > current_release
        )
        || (
            candidate_legal_count == current_legal_count
            && candidate_release == current_release
            && candidate_tie > current_tie
        );
}

static int choose_rollout_drop_index(
    Candidate *candidates,
    uint64_t candidate_count,
    const uint64_t *selected,
    int selected_count,
    int r,
    int d,
    int restart,
    int repair_event_index,
    int step,
    uint64_t before_forbidden_count,
    uint64_t repair_candidate_window,
    const uint64_t *tabu_masks,
    int tabu_count,
    uint64_t drop_topk,
    uint64_t *rng_state
) {
    if (selected_count <= 0) {
        return -1;
    }
    if (drop_topk == 0ULL) {
        return (int)(rng_next(rng_state) % (uint64_t)selected_count);
    }
    if (drop_topk > (uint64_t)selected_count) {
        drop_topk = (uint64_t)selected_count;
    }

    int *top_indices = (int *)calloc((size_t)drop_topk, sizeof(int));
    uint64_t *top_legal_counts = (uint64_t *)calloc((size_t)drop_topk, sizeof(uint64_t));
    uint64_t *top_releases = (uint64_t *)calloc((size_t)drop_topk, sizeof(uint64_t));
    uint32_t *top_ties = (uint32_t *)calloc((size_t)drop_topk, sizeof(uint32_t));
    if (!top_indices || !top_legal_counts || !top_releases || !top_ties) {
        free(top_indices);
        free(top_legal_counts);
        free(top_releases);
        free(top_ties);
        return (int)(rng_next(rng_state) % (uint64_t)selected_count);
    }

    int top_count = 0;
    for (int drop_index = 0; drop_index < selected_count; drop_index++) {
        DenseState trial_state;
        if (!init_state(&trial_state, r, d)) {
            continue;
        }
        if (!trial_state.g_layer_legality) {
            initialize_systematic_columns(&trial_state, r);
        }
        for (int i = 0; i < selected_count; i++) {
            if (i != drop_index) {
                add_column(&trial_state, selected[i]);
            }
        }

        uint64_t after_forbidden_count = forbidden_count(&trial_state);
        uint64_t forbidden_release = before_forbidden_count > after_forbidden_count
            ? before_forbidden_count - after_forbidden_count
            : 0ULL;
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
        uint32_t tie = deterministic_tiebreak(
            selected[drop_index],
            restart + repair_event_index + step + 4099
        );
        free_state(&trial_state);

        int insert_at = top_count;
        while (
            insert_at > 0
            && drop_choice_is_better(
                legal_count,
                forbidden_release,
                tie,
                top_legal_counts[insert_at - 1],
                top_releases[insert_at - 1],
                top_ties[insert_at - 1]
            )
        ) {
            if ((uint64_t)insert_at < drop_topk) {
                top_indices[insert_at] = top_indices[insert_at - 1];
                top_legal_counts[insert_at] = top_legal_counts[insert_at - 1];
                top_releases[insert_at] = top_releases[insert_at - 1];
                top_ties[insert_at] = top_ties[insert_at - 1];
            }
            insert_at--;
        }
        if ((uint64_t)insert_at < drop_topk) {
            top_indices[insert_at] = drop_index;
            top_legal_counts[insert_at] = legal_count;
            top_releases[insert_at] = forbidden_release;
            top_ties[insert_at] = tie;
            if ((uint64_t)top_count < drop_topk) {
                top_count++;
            }
        }
    }

    int chosen_index = -1;
    if (top_count > 0) {
        chosen_index = top_indices[rng_next(rng_state) % (uint64_t)top_count];
    }
    free(top_indices);
    free(top_legal_counts);
    free(top_releases);
    free(top_ties);
    if (chosen_index < 0) {
        chosen_index = (int)(rng_next(rng_state) % (uint64_t)selected_count);
    }
    return chosen_index;
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
    uint64_t drop_topk,
    const uint64_t *tabu_masks,
    int tabu_count,
    int estimate_growth,
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

    ScratchSet scratch;
    memset(&scratch, 0, sizeof(scratch));
    uint64_t scratch_touched_cap = env_u64("LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP", 1048576ULL);
    if (estimate_growth && !init_scratch_set(&scratch, trial_state.word_count, scratch_touched_cap)) {
        free_state(&trial_state);
        free(trial_selected);
        free(trial_tabu);
        return 0;
    }
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
            &scratch,
            estimate_growth,
            &mask,
            &growth,
            &blocked,
            &evals
        )) {
            add_column(&trial_state, mask);
            if (estimate_growth) {
                current_forbidden_count += growth;
            } else {
                current_forbidden_count = forbidden_count(&trial_state);
            }
            trial_selected[trial_count++] = mask;
            dynamic_evaluations += evals;
            steps++;
            continue;
        }

        dynamic_evaluations += evals;
        if (trial_count <= 0 || trial_tabu_count >= trial_tabu_capacity) {
            break;
        }
        int drop_index = choose_rollout_drop_index(
            candidates,
            candidate_count,
            trial_selected,
            trial_count,
            r,
            d,
            restart,
            repair_event_index,
            steps,
            current_forbidden_count,
            dynamic_window,
            trial_tabu,
            trial_tabu_count,
            drop_topk,
            &rng_state
        );
        if (drop_index < 0) {
            break;
        }
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

    free_scratch_set(&scratch);
    free_state(&trial_state);
    free(trial_selected);
    free(trial_tabu);
    return 1;
}

static void run_mcts_rollout_range(MctsRolloutJob *job) {
    if (!job || !job->root_stats) {
        return;
    }
    double worker_started_at = monotonic_seconds();
    uint64_t dynamic_evaluations = 0;
    for (uint64_t simulation = job->simulation_start; simulation < job->simulation_end; simulation++) {
        int drop_index = (int)(simulation % (uint64_t)job->selected_count);
        RepairReward reward;
        uint64_t after_forbidden = 0;
        uint64_t evals = 0;
        if (!rollout_after_first_drop(
            job->candidates,
            job->candidate_count,
            job->selected,
            job->selected_count,
            drop_index,
            job->n,
            job->k,
            job->d,
            job->restart,
            job->repair_event_index,
            job->seed + simulation * 0x9e3779b97f4a7c15ULL,
            job->dynamic_window,
            job->rollout_depth,
            job->drop_topk,
            job->tabu_masks,
            job->tabu_count,
            job->estimate_growth,
            &reward,
            &after_forbidden,
            &evals
        )) {
            continue;
        }
        dynamic_evaluations += evals;
        update_root_stats(
            &job->root_stats[drop_index],
            reward,
            after_forbidden,
            job->selected_count
        );
    }
    job->dynamic_evaluations = dynamic_evaluations;
    job->elapsed_seconds = monotonic_seconds() - worker_started_at;
}

static void *mcts_rollout_worker(void *arg) {
    run_mcts_rollout_range((MctsRolloutJob *)arg);
    return NULL;
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
    uint64_t drop_topk,
    const uint64_t *tabu_masks,
    int tabu_count,
    int estimate_growth,
    int *drop_index_out,
    uint64_t *after_forbidden_count_out,
    uint64_t *dynamic_evaluations_out
) {
    if (selected_count <= 0 || rollout_depth == 0 || simulations == 0) {
        return 0;
    }
    double mcts_started_at = monotonic_seconds();
    RepairRootStats *root_stats = (RepairRootStats *)calloc(
        (size_t)selected_count,
        sizeof(RepairRootStats)
    );
    if (!root_stats) {
        return 0;
    }
    uint64_t total_dynamic_evaluations = 0;
    uint64_t requested_workers = env_u64("LINEAR_CODE_REPAIR_MCTS_WORKERS", 1ULL);
    if (requested_workers == 0ULL) {
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        requested_workers = cpu_count > 1 ? (uint64_t)cpu_count : 1ULL;
    }
    if (requested_workers > simulations) {
        requested_workers = simulations;
    }
    if (requested_workers > 64ULL) {
        requested_workers = 64ULL;
    }
    verbose_progress_log(
        "mcts repair start: selected=%d simulations=%llu depth=%llu workers=%llu drop_topk=%llu dynamic_window=%llu tabu=%d estimate_growth=%d",
        selected_count,
        (unsigned long long)simulations,
        (unsigned long long)rollout_depth,
        (unsigned long long)requested_workers,
        (unsigned long long)drop_topk,
        (unsigned long long)dynamic_window,
        tabu_count,
        estimate_growth
    );

    if (requested_workers <= 1ULL) {
        MctsRolloutJob job;
        memset(&job, 0, sizeof(job));
        job.candidates = candidates;
        job.candidate_count = candidate_count;
        job.selected = selected;
        job.selected_count = selected_count;
        job.n = n;
        job.k = k;
        job.d = d;
        job.restart = restart;
        job.repair_event_index = repair_event_index;
        job.seed = seed;
        job.dynamic_window = dynamic_window;
        job.rollout_depth = rollout_depth;
        job.drop_topk = drop_topk;
        job.tabu_masks = tabu_masks;
        job.tabu_count = tabu_count;
        job.estimate_growth = estimate_growth;
        job.simulation_start = 0;
        job.simulation_end = simulations;
        job.root_stats = root_stats;
        run_mcts_rollout_range(&job);
        total_dynamic_evaluations = job.dynamic_evaluations;
        verbose_progress_log(
            "mcts worker done: simulations=[%llu,%llu) dynamic_evals=%llu elapsed=%.3fs",
            (unsigned long long)job.simulation_start,
            (unsigned long long)job.simulation_end,
            (unsigned long long)job.dynamic_evaluations,
            job.elapsed_seconds
        );
    } else {
        uint64_t worker_count = requested_workers;
        pthread_t *threads = (pthread_t *)calloc((size_t)worker_count, sizeof(pthread_t));
        MctsRolloutJob *jobs = (MctsRolloutJob *)calloc((size_t)worker_count, sizeof(MctsRolloutJob));
        RepairRootStats *worker_stats = (RepairRootStats *)calloc(
            (size_t)(worker_count * (uint64_t)selected_count),
            sizeof(RepairRootStats)
        );
        if (!threads || !jobs || !worker_stats) {
            free(threads);
            free(jobs);
            free(worker_stats);
            free(root_stats);
            return 0;
        }

        uint64_t chunk_size = (simulations + worker_count - 1ULL) / worker_count;
        uint64_t created = 0;
        for (uint64_t worker_index = 0; worker_index < worker_count; worker_index++) {
            uint64_t start = worker_index * chunk_size;
            uint64_t end = start + chunk_size;
            if (start >= simulations) {
                break;
            }
            if (end > simulations) {
                end = simulations;
            }
            MctsRolloutJob *job = &jobs[worker_index];
            job->candidates = candidates;
            job->candidate_count = candidate_count;
            job->selected = selected;
            job->selected_count = selected_count;
            job->n = n;
            job->k = k;
            job->d = d;
            job->restart = restart;
            job->repair_event_index = repair_event_index;
            job->seed = seed;
            job->dynamic_window = dynamic_window;
            job->rollout_depth = rollout_depth;
            job->drop_topk = drop_topk;
            job->tabu_masks = tabu_masks;
            job->tabu_count = tabu_count;
            job->estimate_growth = estimate_growth;
            job->simulation_start = start;
            job->simulation_end = end;
            job->root_stats = &worker_stats[worker_index * (uint64_t)selected_count];
            if (pthread_create(&threads[worker_index], NULL, mcts_rollout_worker, job) != 0) {
                for (uint64_t join_index = 0; join_index < created; join_index++) {
                    pthread_join(threads[join_index], NULL);
                }
                free(threads);
                free(jobs);
                free(worker_stats);
                free(root_stats);
                return 0;
            }
            created++;
        }

        for (uint64_t worker_index = 0; worker_index < created; worker_index++) {
            pthread_join(threads[worker_index], NULL);
            total_dynamic_evaluations += jobs[worker_index].dynamic_evaluations;
            verbose_progress_log(
                "mcts worker %llu done: simulations=[%llu,%llu) dynamic_evals=%llu elapsed=%.3fs",
                (unsigned long long)worker_index,
                (unsigned long long)jobs[worker_index].simulation_start,
                (unsigned long long)jobs[worker_index].simulation_end,
                (unsigned long long)jobs[worker_index].dynamic_evaluations,
                jobs[worker_index].elapsed_seconds
            );
            RepairRootStats *local_stats = &worker_stats[worker_index * (uint64_t)selected_count];
            for (int drop_index = 0; drop_index < selected_count; drop_index++) {
                merge_root_stats(&root_stats[drop_index], &local_stats[drop_index]);
            }
        }

        free(threads);
        free(jobs);
        free(worker_stats);
    }

    int has_best_root = 0;
    int best_drop_index = -1;
    for (int drop_index = 0; drop_index < selected_count; drop_index++) {
        if (root_stats_is_better(
            &root_stats[drop_index],
            best_drop_index >= 0 ? &root_stats[best_drop_index] : NULL,
            has_best_root
        )) {
            best_drop_index = drop_index;
            has_best_root = 1;
        }
    }

    if (
        !has_best_root
        || best_drop_index < 0
        || root_stats[best_drop_index].best_reward.constructed_count < selected_count
    ) {
        verbose_progress_log(
            "mcts repair failed: best_drop=%d has_best=%d dynamic_evals=%llu elapsed=%.3fs",
            best_drop_index,
            has_best_root,
            (unsigned long long)total_dynamic_evaluations,
            monotonic_seconds() - mcts_started_at
        );
        free(root_stats);
        return 0;
    }
    *drop_index_out = best_drop_index;
    *after_forbidden_count_out = root_stats[best_drop_index].best_after_forbidden;
    *dynamic_evaluations_out = total_dynamic_evaluations;
    verbose_progress_log(
        "mcts repair done: best_drop=%d visits=%d best_constructed=%d best_dropped=%d best_steps=%d after_forbidden=%llu dynamic_evals=%llu elapsed=%.3fs",
        best_drop_index,
        root_stats[best_drop_index].visits,
        root_stats[best_drop_index].best_reward.constructed_count,
        root_stats[best_drop_index].best_reward.dropped_count,
        root_stats[best_drop_index].best_reward.steps,
        (unsigned long long)root_stats[best_drop_index].best_after_forbidden,
        (unsigned long long)total_dynamic_evaluations,
        monotonic_seconds() - mcts_started_at
    );
    free(root_stats);
    return 1;
}

static void run_restart_job(RestartJob *job) {
    RestartResult *result = job->result;
    uint64_t *selected_storage = result->selected;
    memset(result, 0, sizeof(*result));
    result->selected = selected_storage;
    result->restart = job->restart;
    result->status = 0;
    verbose_progress_log(
        "restart %d start: candidates=%llu dynamic_window=%llu repair_events=%llu",
        job->restart,
        (unsigned long long)job->candidate_count,
        (unsigned long long)job->dynamic_window,
        (unsigned long long)job->repair_max_events
    );

    Candidate *candidates = (Candidate *)calloc((size_t)job->candidate_count, sizeof(Candidate));
    if (!candidates) {
        result->status = -20;
        verbose_progress_log("restart %d failed: candidate copy allocation", job->restart);
        return;
    }
    memcpy(candidates, job->base_candidates, (size_t)job->candidate_count * sizeof(Candidate));

    double stage_started_at = monotonic_seconds();
    for (uint64_t i = 0; i < job->candidate_count; i++) {
        candidates[i].tie = deterministic_tiebreak(candidates[i].mask, job->restart);
    }
    qsort(candidates, (size_t)job->candidate_count, sizeof(Candidate), compare_candidates_desc);
    result->restart_sort_seconds += monotonic_seconds() - stage_started_at;
    verbose_progress_log(
        "restart %d sorted candidates in %.3fs",
        job->restart,
        result->restart_sort_seconds
    );

    stage_started_at = monotonic_seconds();
    DenseState state;
    int r = job->n - job->k;
    if (!init_state(&state, r, job->d)) {
        free(candidates);
        result->status = -10;
        verbose_progress_log("restart %d failed: state allocation", job->restart);
        return;
    }
    if (!state.g_layer_legality) {
        initialize_systematic_columns(&state, r);
    }
    result->state_init_seconds += monotonic_seconds() - stage_started_at;
    uint64_t current_forbidden_count = forbidden_count(&state);
    verbose_progress_log(
        "restart %d initialized DenseState in %.3fs",
        job->restart,
        result->state_init_seconds
    );

    uint64_t *current_selected = (uint64_t *)calloc((size_t)job->k, sizeof(uint64_t));
    if (!current_selected) {
        free_state(&state);
        free(candidates);
        result->status = -11;
        verbose_progress_log("restart %d failed: selection allocation", job->restart);
        return;
    }

    ScratchSet scratch;
    memset(&scratch, 0, sizeof(scratch));
    if (job->dynamic_window > 0 && job->estimate_dynamic_growth) {
        uint64_t scratch_touched_cap = env_u64("LINEAR_CODE_GROWTH_SCRATCH_TOUCHED_CAP", 1048576ULL);
        if (!init_scratch_set(&scratch, state.word_count, scratch_touched_cap)) {
            free(current_selected);
            free_state(&state);
            free(candidates);
            result->status = -12;
            verbose_progress_log("restart %d failed: scratch allocation", job->restart);
            return;
        }
    }

    uint64_t max_reasonable_tabu = (uint64_t)job->k + job->repair_max_events * job->repair_drop_count + 1ULL;
    uint64_t repair_tabu_cap_u64 = job->repair_tabu_tenure;
    if (repair_tabu_cap_u64 > max_reasonable_tabu) {
        repair_tabu_cap_u64 = max_reasonable_tabu;
    }
    uint64_t *tabu_masks = NULL;
    int tabu_count = 0;
    int tabu_next = 0;
    int tabu_capacity = (int)repair_tabu_cap_u64;
    if (job->dynamic_window > 0 && job->repair_max_events > 0 && tabu_capacity > 0) {
        tabu_masks = (uint64_t *)calloc((size_t)tabu_capacity, sizeof(uint64_t));
        if (!tabu_masks) {
            free_scratch_set(&scratch);
            free(current_selected);
            free_state(&state);
            free(candidates);
            result->status = -13;
            verbose_progress_log("restart %d failed: tabu allocation", job->restart);
            return;
        }
    }

    int current_count = 0;
    int current_repair_events = 0;
    uint64_t blocked = 0;
    int best_count = 0;
    uint64_t best_blocked = 0;
    uint64_t best_forbidden_count = current_forbidden_count;

    stage_started_at = monotonic_seconds();
    if (job->dynamic_window > 0) {
        while (current_count < job->k) {
            uint64_t mask = 0;
            uint64_t growth = 0;
            uint64_t dynamic_evaluations = 0;
            uint64_t blocked_before_step = blocked;
            double step_started_at = monotonic_seconds();
            verbose_progress_log(
                "restart %d step %d start: forbidden=%llu blocked_total=%llu",
                job->restart,
                current_count,
                (unsigned long long)current_forbidden_count,
                (unsigned long long)blocked
            );
            if (!choose_dynamic_candidate(
                candidates,
                job->candidate_count,
                &state,
                job->n,
                job->k,
                job->d,
                current_count,
                job->restart,
                current_forbidden_count,
                job->dynamic_window,
                tabu_masks,
                tabu_count,
                &scratch,
                job->estimate_dynamic_growth,
                &mask,
                &growth,
                &blocked,
                &dynamic_evaluations
            )) {
                int repaired = 0;
                verbose_progress_log(
                    "restart %d step %d no candidate: scanned_blocked_delta=%llu dynamic_evals=%llu elapsed=%.3fs",
                    job->restart,
                    current_count,
                    (unsigned long long)(blocked - blocked_before_step),
                    (unsigned long long)dynamic_evaluations,
                    monotonic_seconds() - step_started_at
                );
                if (
                    job->repair_max_events == 0
                    || (uint64_t)current_repair_events >= job->repair_max_events
                    || current_count <= 0
                ) {
                    verbose_progress_log(
                        "restart %d step %d stop: repair unavailable repair_events=%d/%llu",
                        job->restart,
                        current_count,
                        current_repair_events,
                        (unsigned long long)job->repair_max_events
                    );
                    break;
                }
                for (
                    uint64_t drop_event = 0;
                    drop_event < job->repair_drop_count
                        && (uint64_t)current_repair_events < job->repair_max_events
                        && current_count > 0;
                    drop_event++
                ) {
                    int drop_index = -1;
                    uint64_t repaired_forbidden_count = 0;
                    uint64_t repair_dynamic_evaluations = 0;
                    int chose_repair = 0;
                    double repair_started_at = monotonic_seconds();
                    verbose_progress_log(
                        "restart %d repair event %d start: current_columns=%d mode=%s",
                        job->restart,
                        current_repair_events,
                        current_count,
                        job->repair_mcts_enabled ? "mcts" : "greedy"
                    );
                    if (job->repair_mcts_enabled) {
                        chose_repair = choose_repair_drop_index_mcts(
                            candidates,
                            job->candidate_count,
                            current_selected,
                            current_count,
                            job->n,
                            job->k,
                            job->d,
                            job->restart,
                            current_repair_events,
                            job->seed,
                            job->dynamic_window,
                            job->repair_mcts_depth,
                            job->repair_mcts_simulations,
                            job->repair_mcts_drop_topk,
                            tabu_masks,
                            tabu_count,
                            job->estimate_dynamic_growth,
                            &drop_index,
                            &repaired_forbidden_count,
                            &repair_dynamic_evaluations
                        );
                        result->dynamic_evaluations += repair_dynamic_evaluations;
                    }
                    if (!chose_repair) {
                        double fallback_started_at = monotonic_seconds();
                        verbose_progress_log(
                            "restart %d repair event %d fallback greedy drop start",
                            job->restart,
                            current_repair_events
                        );
                        chose_repair = choose_repair_drop_index(
                            candidates,
                            job->candidate_count,
                            current_selected,
                            current_count,
                            r,
                            job->d,
                            job->restart,
                            current_repair_events,
                            current_forbidden_count,
                            job->repair_candidate_window,
                            tabu_masks,
                            tabu_count,
                            &drop_index,
                            &repaired_forbidden_count
                        );
                        verbose_progress_log(
                            "restart %d repair event %d fallback greedy drop %s: drop_index=%d after_forbidden=%llu elapsed=%.3fs",
                            job->restart,
                            current_repair_events,
                            chose_repair ? "done" : "failed",
                            drop_index,
                            (unsigned long long)repaired_forbidden_count,
                            monotonic_seconds() - fallback_started_at
                        );
                    }
                    if (!chose_repair) {
                        verbose_progress_log(
                            "restart %d repair event %d failed: elapsed=%.3fs",
                            job->restart,
                            current_repair_events,
                            monotonic_seconds() - repair_started_at
                        );
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
                        job->d,
                        current_selected,
                        current_count
                    )) {
                        free(tabu_masks);
                        free_scratch_set(&scratch);
                        free(current_selected);
                        free_state(&state);
                        free(candidates);
                        result->status = -14;
                        verbose_progress_log("restart %d failed: repair rebuild", job->restart);
                        return;
                    }
                    free_state(&state);
                    state = rebuilt_state;
                    current_forbidden_count = repaired_forbidden_count;
                    current_repair_events++;
                    result->repair_events++;
                    repaired = 1;
                    verbose_progress_log(
                        "restart %d repair event %d done: dropped_index=%d dropped_mask=%llu columns=%d forbidden=%llu repair_dynamic_evals=%llu elapsed=%.3fs",
                        job->restart,
                        current_repair_events - 1,
                        drop_index,
                        (unsigned long long)dropped_mask,
                        current_count,
                        (unsigned long long)current_forbidden_count,
                        (unsigned long long)repair_dynamic_evaluations,
                        monotonic_seconds() - repair_started_at
                    );
                }
                if (!repaired) {
                    verbose_progress_log(
                        "restart %d step %d stop: repair did not recover",
                        job->restart,
                        current_count
                    );
                    break;
                }
                continue;
            }

            verbose_progress_log(
                "restart %d step %d candidate chosen: mask=%llu weight=%d dynamic_evals=%llu blocked_delta=%llu scan_and_score=%.3fs",
                job->restart,
                current_count,
                (unsigned long long)mask,
                popcount64(mask),
                (unsigned long long)dynamic_evaluations,
                (unsigned long long)(blocked - blocked_before_step),
                monotonic_seconds() - step_started_at
            );
            double add_started_at = monotonic_seconds();
            add_column(&state, mask);
            if (job->estimate_dynamic_growth) {
                current_forbidden_count += growth;
            } else {
                current_forbidden_count = forbidden_count(&state);
            }
            current_selected[current_count++] = mask;
            result->dynamic_evaluations += dynamic_evaluations;
            verbose_progress_log(
                "restart %d step %d add done: columns=%d/%d forbidden=%llu growth=%llu add_and_count=%.3fs total_step=%.3fs",
                job->restart,
                current_count - 1,
                current_count,
                job->k,
                (unsigned long long)current_forbidden_count,
                (unsigned long long)growth,
                monotonic_seconds() - add_started_at,
                monotonic_seconds() - step_started_at
            );
            if (current_count > best_count) {
                best_count = current_count;
                best_blocked = blocked;
                best_forbidden_count = current_forbidden_count;
                memcpy(result->selected, current_selected, (size_t)current_count * sizeof(uint64_t));
            }
        }
    } else {
        for (uint64_t i = 0; i < job->candidate_count && current_count < job->k; i++) {
            uint64_t mask = candidates[i].mask;
            if (can_add(&state, mask)) {
                add_column(&state, mask);
                current_selected[current_count++] = mask;
            } else {
                blocked++;
            }
        }
    }
    result->greedy_scan_seconds += monotonic_seconds() - stage_started_at;

    if (current_count > best_count) {
        best_count = current_count;
        best_blocked = blocked;
        stage_started_at = monotonic_seconds();
        best_forbidden_count = forbidden_count(&state);
        result->forbidden_count_seconds += monotonic_seconds() - stage_started_at;
        memcpy(result->selected, current_selected, (size_t)current_count * sizeof(uint64_t));
    }

    result->count = best_count;
    result->blocked = best_blocked;
    result->forbidden_count = best_forbidden_count;
    verbose_progress_log(
        "restart %d done: columns=%d/%d blocked=%llu repair_events=%llu dynamic_evals=%llu greedy_scan=%.3fs forbidden=%llu",
        job->restart,
        result->count,
        job->k,
        (unsigned long long)result->blocked,
        (unsigned long long)result->repair_events,
        (unsigned long long)result->dynamic_evaluations,
        result->greedy_scan_seconds,
        (unsigned long long)result->forbidden_count
    );

    free(tabu_masks);
    free_scratch_set(&scratch);
    free(current_selected);
    free_state(&state);
    free(candidates);
}

static void *restart_worker(void *arg) {
    run_restart_job((RestartJob *)arg);
    return NULL;
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
    g_verbose_started_at = c_run_started_at;
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
    verbose_progress_log(
        "run start: n=%d k=%d d=%d r=%d restarts=%d seed=%llu",
        n,
        k,
        d,
        r,
        restarts,
        (unsigned long long)seed
    );

    uint64_t total_candidate_count = candidate_count_for_instance(r, d);
    uint64_t max_candidates = env_u64("LINEAR_CODE_MAX_CANDIDATES", BASELINE_MAX_CANDIDATES);
    uint64_t candidate_count = total_candidate_count;
    int use_sampled_candidates = 0;
    if (max_candidates > 0ULL && candidate_count > max_candidates) {
        candidate_count = max_candidates;
        use_sampled_candidates = 1;
    }
    if (!use_sampled_candidates && candidate_count > BASELINE_MAX_CANDIDATES) {
        write_error(error_out, error_cap, "too many candidates for baseline C kernel");
        return -5;
    }
    if (candidate_count == 0ULL) {
        write_error(error_out, error_cap, "empty candidate set");
        return -5;
    }
    verbose_progress_log(
        "candidate sizing: total=%llu selected=%llu mode=%s max_candidates=%llu",
        (unsigned long long)total_candidate_count,
        (unsigned long long)candidate_count,
        use_sampled_candidates ? "sampled" : "full",
        (unsigned long long)max_candidates
    );
    Candidate *candidates = (Candidate *)calloc((size_t)candidate_count, sizeof(Candidate));
    if (!candidates) {
        write_error(error_out, error_cap, "candidate allocation failed");
        return -6;
    }

    double stage_started_at = monotonic_seconds();
    verbose_progress_log("candidate generation start");
    uint64_t generated_count = use_sampled_candidates
        ? generate_sampled_candidate_masks(candidates, candidate_count, total_candidate_count, r, d, seed)
        : generate_candidate_masks(candidates, r, d);
    metrics_out[METRIC_CANDIDATE_GENERATION_SECONDS] = monotonic_seconds() - stage_started_at;
    verbose_progress_log(
        "candidate generation done: generated=%llu seconds=%.3f",
        (unsigned long long)generated_count,
        metrics_out[METRIC_CANDIDATE_GENERATION_SECONDS]
    );
    if (generated_count != candidate_count) {
        free(candidates);
        write_error(error_out, error_cap, "candidate generation count mismatch");
        return -7;
    }
    stage_started_at = monotonic_seconds();
    verbose_progress_log(
        "candidate scoring start: candidates=%llu workers=%llu",
        (unsigned long long)candidate_count,
        (unsigned long long)env_u64("LINEAR_CODE_CANDIDATE_WORKERS", 0ULL)
    );
    if (!score_candidates(candidates, candidate_count, n, k, d)) {
        free(candidates);
        write_error(error_out, error_cap, "candidate scoring failed");
        return -8;
    }
    metrics_out[METRIC_CANDIDATE_SCORING_SECONDS] = monotonic_seconds() - stage_started_at;
    verbose_progress_log(
        "candidate scoring done: seconds=%.3f",
        metrics_out[METRIC_CANDIDATE_SCORING_SECONDS]
    );

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
    uint64_t repair_mcts_drop_topk = env_u64("LINEAR_CODE_REPAIR_MCTS_DROP_TOPK", 2ULL);
    int estimate_dynamic_growth = !env_equals("LINEAR_CODE_DYNAMIC_GROWTH_ESTIMATE", "0");
    uint64_t repair_tabu_cap_u64 = repair_tabu_tenure;
    uint64_t max_reasonable_tabu = (uint64_t)k + repair_max_events * repair_drop_count + 1ULL;
    if (repair_tabu_cap_u64 > max_reasonable_tabu) {
        repair_tabu_cap_u64 = max_reasonable_tabu;
    }
    uint64_t total_dynamic_evaluations = 0;
    uint64_t total_repair_events = 0;
    uint64_t requested_restart_workers = env_u64("LINEAR_CODE_RESTART_WORKERS", 1ULL);
    if (requested_restart_workers == 0ULL) {
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        requested_restart_workers = cpu_count > 1 ? (uint64_t)cpu_count : 1ULL;
    }
    if (requested_restart_workers > (uint64_t)restarts) {
        requested_restart_workers = (uint64_t)restarts;
    }
    if (requested_restart_workers > 64ULL) {
        requested_restart_workers = 64ULL;
    }
    uint64_t restart_parallel_candidate_limit = env_u64(
        "LINEAR_CODE_RESTART_PARALLEL_MAX_CANDIDATES",
        2000000ULL
    );
    if (candidate_count > restart_parallel_candidate_limit) {
        requested_restart_workers = 1ULL;
    }
    verbose_progress_log(
        "restart scheduling: restarts=%d workers=%llu parallel_candidate_limit=%llu dynamic_workers=%llu mcts_workers=%llu",
        restarts,
        (unsigned long long)requested_restart_workers,
        (unsigned long long)restart_parallel_candidate_limit,
        (unsigned long long)env_u64("LINEAR_CODE_DYNAMIC_WORKERS", 1ULL),
        (unsigned long long)env_u64("LINEAR_CODE_REPAIR_MCTS_WORKERS", 1ULL)
    );

    RestartResult *restart_results = (RestartResult *)calloc((size_t)restarts, sizeof(RestartResult));
    uint64_t *restart_selected = (uint64_t *)calloc((size_t)restarts * (size_t)k, sizeof(uint64_t));
    if (!restart_results || !restart_selected) {
        free(restart_results);
        free(restart_selected);
        free(best_selected);
        free(candidates);
        write_error(error_out, error_cap, "restart result allocation failed");
        return -15;
    }
    for (int restart = 0; restart < restarts; restart++) {
        restart_results[restart].selected = &restart_selected[(size_t)restart * (size_t)k];
    }

    int next_restart = 0;
    while (next_restart < restarts && best_count < k) {
        uint64_t batch_workers = requested_restart_workers;
        if (batch_workers < 1ULL) {
            batch_workers = 1ULL;
        }
        if (batch_workers > (uint64_t)(restarts - next_restart)) {
            batch_workers = (uint64_t)(restarts - next_restart);
        }
        verbose_progress_log(
            "restart batch start: first_restart=%d workers=%llu current_best=%d/%d",
            next_restart,
            (unsigned long long)batch_workers,
            best_count,
            k
        );

        RestartJob *jobs = (RestartJob *)calloc((size_t)batch_workers, sizeof(RestartJob));
        pthread_t *threads = NULL;
        if (batch_workers > 1ULL) {
            threads = (pthread_t *)calloc((size_t)batch_workers, sizeof(pthread_t));
        }
        if (!jobs || (batch_workers > 1ULL && !threads)) {
            free(jobs);
            free(threads);
            free(restart_selected);
            free(restart_results);
            free(best_selected);
            free(candidates);
            write_error(error_out, error_cap, "restart worker allocation failed");
            return -16;
        }

        uint64_t created = 0;
        for (uint64_t worker = 0; worker < batch_workers; worker++) {
            int restart = next_restart + (int)worker;
            RestartJob *job = &jobs[worker];
            job->base_candidates = candidates;
            job->candidate_count = candidate_count;
            job->n = n;
            job->k = k;
            job->d = d;
            job->restart = restart;
            job->seed = seed;
            job->dynamic_window = dynamic_window;
            job->repair_max_events = repair_max_events;
            job->repair_drop_count = repair_drop_count;
            job->repair_candidate_window = repair_candidate_window;
            job->repair_tabu_tenure = repair_tabu_tenure;
            job->repair_mcts_enabled = repair_mcts_enabled;
            job->repair_mcts_simulations = repair_mcts_simulations;
            job->repair_mcts_depth = repair_mcts_depth;
            job->repair_mcts_drop_topk = repair_mcts_drop_topk;
            job->estimate_dynamic_growth = estimate_dynamic_growth;
            job->result = &restart_results[restart];
            if (batch_workers <= 1ULL) {
                run_restart_job(job);
                created++;
            } else if (pthread_create(&threads[worker], NULL, restart_worker, job) == 0) {
                created++;
            } else {
                for (uint64_t join_index = 0; join_index < created; join_index++) {
                    pthread_join(threads[join_index], NULL);
                }
                free(jobs);
                free(threads);
                free(restart_selected);
                free(restart_results);
                free(best_selected);
                free(candidates);
                write_error(error_out, error_cap, "restart worker creation failed");
                return -17;
            }
        }

        if (batch_workers > 1ULL) {
            for (uint64_t worker = 0; worker < created; worker++) {
                pthread_join(threads[worker], NULL);
            }
        }

        for (uint64_t worker = 0; worker < created; worker++) {
            int restart = next_restart + (int)worker;
            RestartResult *result = &restart_results[restart];
            if (result->status != 0) {
                free(jobs);
                free(threads);
                free(restart_selected);
                free(restart_results);
                free(best_selected);
                free(candidates);
                write_error(error_out, error_cap, "restart worker failed");
                return result->status;
            }
            metrics_out[METRIC_RESTART_SORT_SECONDS] += result->restart_sort_seconds;
            metrics_out[METRIC_STATE_INIT_SECONDS] += result->state_init_seconds;
            metrics_out[METRIC_GREEDY_SCAN_SECONDS] += result->greedy_scan_seconds;
            metrics_out[METRIC_FORBIDDEN_COUNT_SECONDS] += result->forbidden_count_seconds;
            total_dynamic_evaluations += result->dynamic_evaluations;
            total_repair_events += result->repair_events;
            if (result->count > best_count) {
                best_count = result->count;
                best_restart = result->restart;
                best_blocked = result->blocked;
                best_forbidden_count = result->forbidden_count;
                memcpy(best_selected, result->selected, (size_t)result->count * sizeof(uint64_t));
            }
        }
        verbose_progress_log(
            "restart batch done: first_restart=%d completed=%llu best=%d/%d best_restart=%d",
            next_restart,
            (unsigned long long)created,
            best_count,
            k,
            best_restart
        );

        next_restart += (int)created;
        free(jobs);
        free(threads);
        if (created == 0) {
            break;
        }
    }

    free(restart_selected);
    free(restart_results);

    for (int i = 0; i < best_count; i++) {
        selected_out[i] = best_selected[i];
    }
    metrics_out[METRIC_SUCCESS] = best_count == k ? 1.0 : 0.0;
    metrics_out[METRIC_CONSTRUCTED_COLUMNS] = (double)best_count;
    metrics_out[METRIC_CANDIDATE_COUNT] = (double)total_candidate_count;
    metrics_out[METRIC_SCORED_CANDIDATES] = (double)candidate_count;
    metrics_out[METRIC_SAMPLE_ATTEMPTS] = (double)total_dynamic_evaluations;
    metrics_out[METRIC_BACKTRACK_EVENTS] = (double)total_repair_events;
    metrics_out[METRIC_RESTART_INDEX] = (double)best_restart;
    metrics_out[METRIC_BLOCKED_CANDIDATES] = (double)best_blocked;
    metrics_out[METRIC_FORBIDDEN_COUNT] = (double)best_forbidden_count;
    metrics_out[METRIC_C_RUN_SECONDS] = monotonic_seconds() - c_run_started_at;
    verbose_progress_log(
        "run done: success=%d best=%d/%d best_restart=%d candidate_gen=%.3f candidate_score=%.3f restart_sort=%.3f state_init=%.3f greedy_scan=%.3f forbidden_count=%.3f c_total=%.3f",
        best_count == k ? 1 : 0,
        best_count,
        k,
        best_restart,
        metrics_out[METRIC_CANDIDATE_GENERATION_SECONDS],
        metrics_out[METRIC_CANDIDATE_SCORING_SECONDS],
        metrics_out[METRIC_RESTART_SORT_SECONDS],
        metrics_out[METRIC_STATE_INIT_SECONDS],
        metrics_out[METRIC_GREEDY_SCAN_SECONDS],
        metrics_out[METRIC_FORBIDDEN_COUNT_SECONDS],
        metrics_out[METRIC_C_RUN_SECONDS]
    );

    free(best_selected);
    free(candidates);
    write_error(error_out, error_cap, "");
    return best_count == k ? 0 : 1;
}
