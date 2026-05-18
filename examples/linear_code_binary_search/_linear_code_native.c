#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NATIVE_R_LIMIT 60
#define DEFAULT_MAX_INITIAL_VALUES 200000000ULL

typedef struct {
    uint64_t *keys;
    unsigned char *states;
    Py_ssize_t capacity;
    Py_ssize_t count;
    Py_ssize_t used;
} SparseSet;

typedef struct {
    uint64_t *items;
    Py_ssize_t size;
    Py_ssize_t cap;
} UIntVec;

typedef struct {
    uint64_t *items;
    Py_ssize_t size;
    Py_ssize_t cap;
} DeltaVec;

typedef struct {
    DeltaVec *layer_deltas;
    DeltaVec forbidden_delta;
    uint64_t selected_column;
} HistoryEntry;

typedef struct {
    SparseSet *set;
    UIntVec values;
} Layer;

typedef struct {
    PyObject_HEAD
    int r;
    int distance;
    int max_subset_size;
    uint64_t mask_limit;
    Layer *layers;
    SparseSet *forbidden;
    UIntVec selected;
    HistoryEntry *history;
    Py_ssize_t history_size;
    Py_ssize_t history_cap;
} NativeState;

static uint64_t hash_u64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static Py_ssize_t next_power_of_two(Py_ssize_t value) {
    Py_ssize_t capacity = 16;
    while (capacity < value) {
        capacity <<= 1;
    }
    return capacity;
}

static SparseSet *sparse_set_new(Py_ssize_t min_capacity) {
    SparseSet *set = PyMem_Calloc(1, sizeof(SparseSet));
    if (!set) {
        PyErr_NoMemory();
        return NULL;
    }
    set->capacity = next_power_of_two(min_capacity);
    set->keys = PyMem_Calloc((size_t)set->capacity, sizeof(uint64_t));
    set->states = PyMem_Calloc((size_t)set->capacity, sizeof(unsigned char));
    if (!set->keys || !set->states) {
        PyMem_Free(set->keys);
        PyMem_Free(set->states);
        PyMem_Free(set);
        PyErr_NoMemory();
        return NULL;
    }
    return set;
}

static void sparse_set_free(SparseSet *set) {
    if (!set) {
        return;
    }
    PyMem_Free(set->keys);
    PyMem_Free(set->states);
    PyMem_Free(set);
}

static int sparse_set_contains(SparseSet *set, uint64_t key) {
    Py_ssize_t mask = set->capacity - 1;
    Py_ssize_t index = (Py_ssize_t)(hash_u64(key) & (uint64_t)mask);
    for (;;) {
        unsigned char state = set->states[index];
        if (state == 0) {
            return 0;
        }
        if (state == 1 && set->keys[index] == key) {
            return 1;
        }
        index = (index + 1) & mask;
    }
}

static int sparse_set_rehash(SparseSet *set, Py_ssize_t min_capacity) {
    SparseSet replacement;
    memset(&replacement, 0, sizeof(replacement));
    replacement.capacity = next_power_of_two(min_capacity);
    replacement.keys = PyMem_Calloc((size_t)replacement.capacity, sizeof(uint64_t));
    replacement.states = PyMem_Calloc((size_t)replacement.capacity, sizeof(unsigned char));
    if (!replacement.keys || !replacement.states) {
        PyMem_Free(replacement.keys);
        PyMem_Free(replacement.states);
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < set->capacity; i++) {
        if (set->states[i] != 1) {
            continue;
        }
        uint64_t key = set->keys[i];
        Py_ssize_t mask = replacement.capacity - 1;
        Py_ssize_t index = (Py_ssize_t)(hash_u64(key) & (uint64_t)mask);
        while (replacement.states[index] == 1) {
            index = (index + 1) & mask;
        }
        replacement.keys[index] = key;
        replacement.states[index] = 1;
        replacement.count++;
        replacement.used++;
    }

    PyMem_Free(set->keys);
    PyMem_Free(set->states);
    set->keys = replacement.keys;
    set->states = replacement.states;
    set->capacity = replacement.capacity;
    set->count = replacement.count;
    set->used = replacement.used;
    return 0;
}

static int sparse_set_add(SparseSet *set, uint64_t key) {
    if ((set->used + 1) * 10 >= set->capacity * 7) {
        if (sparse_set_rehash(set, set->capacity * 2) < 0) {
            return -1;
        }
    }

    Py_ssize_t mask = set->capacity - 1;
    Py_ssize_t index = (Py_ssize_t)(hash_u64(key) & (uint64_t)mask);
    Py_ssize_t first_deleted = -1;
    for (;;) {
        unsigned char state = set->states[index];
        if (state == 0) {
            Py_ssize_t insert_at = first_deleted >= 0 ? first_deleted : index;
            set->keys[insert_at] = key;
            if (set->states[insert_at] == 0) {
                set->used++;
            }
            set->states[insert_at] = 1;
            set->count++;
            return 1;
        }
        if (state == 2) {
            if (first_deleted < 0) {
                first_deleted = index;
            }
        } else if (set->keys[index] == key) {
            return 0;
        }
        index = (index + 1) & mask;
    }
}

static int sparse_set_remove(SparseSet *set, uint64_t key) {
    Py_ssize_t mask = set->capacity - 1;
    Py_ssize_t index = (Py_ssize_t)(hash_u64(key) & (uint64_t)mask);
    for (;;) {
        unsigned char state = set->states[index];
        if (state == 0) {
            return 0;
        }
        if (state == 1 && set->keys[index] == key) {
            set->states[index] = 2;
            set->count--;
            return 1;
        }
        index = (index + 1) & mask;
    }
}

static SparseSet *sparse_set_clone(SparseSet *src) {
    SparseSet *dst = sparse_set_new(src->capacity);
    if (!dst) {
        return NULL;
    }
    memcpy(dst->keys, src->keys, (size_t)src->capacity * sizeof(uint64_t));
    memcpy(dst->states, src->states, (size_t)src->capacity * sizeof(unsigned char));
    dst->capacity = src->capacity;
    dst->count = src->count;
    dst->used = src->used;
    return dst;
}

static int vec_push(UIntVec *vec, uint64_t value) {
    if (vec->size == vec->cap) {
        Py_ssize_t new_cap = vec->cap ? vec->cap * 2 : 16;
        uint64_t *items = PyMem_Realloc(vec->items, (size_t)new_cap * sizeof(uint64_t));
        if (!items) {
            PyErr_NoMemory();
            return -1;
        }
        vec->items = items;
        vec->cap = new_cap;
    }
    vec->items[vec->size++] = value;
    return 0;
}

static int delta_push(DeltaVec *vec, uint64_t value) {
    if (vec->size == vec->cap) {
        Py_ssize_t new_cap = vec->cap ? vec->cap * 2 : 16;
        uint64_t *items = PyMem_Realloc(vec->items, (size_t)new_cap * sizeof(uint64_t));
        if (!items) {
            PyErr_NoMemory();
            return -1;
        }
        vec->items = items;
        vec->cap = new_cap;
    }
    vec->items[vec->size++] = value;
    return 0;
}

static void vec_free(UIntVec *vec) {
    PyMem_Free(vec->items);
    vec->items = NULL;
    vec->size = 0;
    vec->cap = 0;
}

static void delta_free(DeltaVec *vec) {
    PyMem_Free(vec->items);
    vec->items = NULL;
    vec->size = 0;
    vec->cap = 0;
}

static int layer_add(Layer *layer, uint64_t value) {
    int changed = sparse_set_add(layer->set, value);
    if (changed < 0) {
        return -1;
    }
    if (changed && vec_push(&layer->values, value) < 0) {
        return -1;
    }
    return changed;
}

static int history_push(NativeState *self, HistoryEntry *entry) {
    if (self->history_size == self->history_cap) {
        Py_ssize_t new_cap = self->history_cap ? self->history_cap * 2 : 16;
        HistoryEntry *items = PyMem_Realloc(
            self->history, (size_t)new_cap * sizeof(HistoryEntry)
        );
        if (!items) {
            PyErr_NoMemory();
            return -1;
        }
        self->history = items;
        self->history_cap = new_cap;
    }
    self->history[self->history_size++] = *entry;
    return 0;
}

static void history_entry_free(HistoryEntry *entry, int layer_count) {
    if (entry->layer_deltas) {
        for (int i = 0; i < layer_count; i++) {
            delta_free(&entry->layer_deltas[i]);
        }
        PyMem_Free(entry->layer_deltas);
    }
    delta_free(&entry->forbidden_delta);
    entry->layer_deltas = NULL;
}

static uint64_t initial_limit_from_env(void) {
    const char *raw = getenv("LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES");
    if (!raw || !*raw) {
        return DEFAULT_MAX_INITIAL_VALUES;
    }
    char *end = NULL;
    unsigned long long parsed = strtoull(raw, &end, 10);
    if (end == raw || parsed == 0) {
        return DEFAULT_MAX_INITIAL_VALUES;
    }
    return (uint64_t)parsed;
}

static int initial_layer_value_count_exceeds(int r, int max_subset_size, uint64_t limit) {
    long double total = 0.0L;
    long double comb = 1.0L;
    int capped_subset_size = max_subset_size < r ? max_subset_size : r;
    for (int w = 0; w <= capped_subset_size; w++) {
        if (w == 0) {
            comb = 1.0L;
        } else {
            comb *= (long double)(r - w + 1) / (long double)w;
        }
        total += comb;
        if (total > (long double)limit) {
            return 1;
        }
    }
    return 0;
}

static int state_update_layers(NativeState *self, uint64_t column, HistoryEntry *history) {
    int max_s = self->max_subset_size;
    for (int s = max_s; s >= 1; s--) {
        Py_ssize_t prev_size = self->layers[s - 1].values.size;
        uint64_t *prev_items = self->layers[s - 1].values.items;
        for (Py_ssize_t i = 0; i < prev_size; i++) {
            uint64_t value = prev_items[i] ^ column;
            if (!sparse_set_contains(self->layers[s].set, value)) {
                int was_forbidden = sparse_set_contains(self->forbidden, value);
                if (layer_add(&self->layers[s], value) < 0) {
                    return -1;
                }
                if (history && delta_push(&history->layer_deltas[s], value) < 0) {
                    return -1;
                }
                if (!was_forbidden) {
                    int changed = sparse_set_add(self->forbidden, value);
                    if (changed < 0) {
                        return -1;
                    }
                    if (changed && history && delta_push(&history->forbidden_delta, value) < 0) {
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

static int state_init_storage(NativeState *self, int r, int distance) {
    if (r < 0 || r > NATIVE_R_LIMIT) {
        PyErr_Format(PyExc_ValueError, "Native legality engine supports r <= %d", NATIVE_R_LIMIT);
        return -1;
    }
    if (distance < 1) {
        PyErr_SetString(PyExc_ValueError, "distance must be positive");
        return -1;
    }
    self->r = r;
    self->distance = distance;
    self->max_subset_size = distance - 2 > 0 ? distance - 2 : 0;
    self->mask_limit = 1ULL << r;

    uint64_t initial_limit = initial_limit_from_env();
    if (initial_layer_value_count_exceeds(r, self->max_subset_size, initial_limit)) {
        PyErr_Format(
            PyExc_MemoryError,
            "Native exact initialization for r=%d, d=%d exceeds "
            "LINEAR_CODE_NATIVE_MAX_INITIAL_VALUES=%llu; exact low-weight forbidden "
            "layers are too large for this run",
            r,
            distance,
            (unsigned long long)initial_limit
        );
        return -1;
    }

    self->layers = PyMem_Calloc((size_t)self->max_subset_size + 1, sizeof(Layer));
    if (!self->layers) {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i <= self->max_subset_size; i++) {
        self->layers[i].set = sparse_set_new(16);
        if (!self->layers[i].set) {
            return -1;
        }
    }
    self->forbidden = sparse_set_new(16);
    if (!self->forbidden) {
        return -1;
    }
    if (layer_add(&self->layers[0], 0) < 0 || sparse_set_add(self->forbidden, 0) < 0) {
        return -1;
    }
    for (int bit = 0; bit < r; bit++) {
        if (state_update_layers(self, 1ULL << bit, NULL) < 0) {
            return -1;
        }
    }
    return 0;
}

static void NativeState_dealloc(NativeState *self) {
    if (self->layers) {
        for (int i = 0; i <= self->max_subset_size; i++) {
            sparse_set_free(self->layers[i].set);
            vec_free(&self->layers[i].values);
        }
        PyMem_Free(self->layers);
    }
    sparse_set_free(self->forbidden);
    vec_free(&self->selected);
    if (self->history) {
        for (Py_ssize_t i = 0; i < self->history_size; i++) {
            history_entry_free(&self->history[i], self->max_subset_size + 1);
        }
        PyMem_Free(self->history);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int NativeState_init(NativeState *self, PyObject *args, PyObject *kwds) {
    int r, distance;
    static char *kwlist[] = {"r", "distance", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist, &r, &distance)) {
        return -1;
    }
    return state_init_storage(self, r, distance);
}

static PyObject *NativeState_can_add(NativeState *self, PyObject *args) {
    unsigned long long mask;
    if (!PyArg_ParseTuple(args, "K", &mask)) {
        return NULL;
    }
    if ((uint64_t)mask >= self->mask_limit) {
        Py_RETURN_FALSE;
    }
    if (sparse_set_contains(self->forbidden, (uint64_t)mask)) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *NativeState_add(NativeState *self, PyObject *args) {
    unsigned long long mask;
    if (!PyArg_ParseTuple(args, "K", &mask)) {
        return NULL;
    }
    uint64_t column = (uint64_t)mask;
    if (column >= self->mask_limit || sparse_set_contains(self->forbidden, column)) {
        PyErr_SetString(PyExc_ValueError, "Illegal free column");
        return NULL;
    }
    HistoryEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.selected_column = column;
    entry.layer_deltas = PyMem_Calloc((size_t)self->max_subset_size + 1, sizeof(DeltaVec));
    if (!entry.layer_deltas) {
        PyErr_NoMemory();
        return NULL;
    }
    uint64_t before = (uint64_t)self->forbidden->count;
    if (state_update_layers(self, column, &entry) < 0 || vec_push(&self->selected, column) < 0) {
        history_entry_free(&entry, self->max_subset_size + 1);
        return NULL;
    }
    if (history_push(self, &entry) < 0) {
        history_entry_free(&entry, self->max_subset_size + 1);
        return NULL;
    }
    return PyLong_FromUnsignedLongLong((uint64_t)self->forbidden->count - before);
}

static PyObject *NativeState_undo(NativeState *self, PyObject *args) {
    int count;
    if (!PyArg_ParseTuple(args, "i", &count)) {
        return NULL;
    }
    if (count < 0 || count > self->history_size) {
        PyErr_SetString(PyExc_ValueError, "Invalid undo count");
        return NULL;
    }
    for (int c = 0; c < count; c++) {
        HistoryEntry *entry = &self->history[self->history_size - 1];
        for (int s = self->max_subset_size; s >= 0; s--) {
            DeltaVec *delta = &entry->layer_deltas[s];
            for (Py_ssize_t i = delta->size - 1; i >= 0; i--) {
                sparse_set_remove(self->layers[s].set, delta->items[i]);
            }
            self->layers[s].values.size -= delta->size;
        }
        for (Py_ssize_t i = entry->forbidden_delta.size - 1; i >= 0; i--) {
            sparse_set_remove(self->forbidden, entry->forbidden_delta.items[i]);
        }
        if (self->selected.size > 0) {
            self->selected.size--;
        }
        history_entry_free(entry, self->max_subset_size + 1);
        self->history_size--;
    }
    Py_RETURN_NONE;
}

static PyObject *NativeState_clone(NativeState *self, PyObject *Py_UNUSED(ignored)) {
    NativeState *clone = PyObject_New(NativeState, Py_TYPE(self));
    if (!clone) {
        return NULL;
    }
    memset(((char *)clone) + sizeof(PyObject), 0, sizeof(NativeState) - sizeof(PyObject));
    clone->r = self->r;
    clone->distance = self->distance;
    clone->max_subset_size = self->max_subset_size;
    clone->mask_limit = self->mask_limit;
    clone->layers = PyMem_Calloc((size_t)self->max_subset_size + 1, sizeof(Layer));
    if (!clone->layers) {
        Py_DECREF(clone);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i <= self->max_subset_size; i++) {
        clone->layers[i].set = sparse_set_clone(self->layers[i].set);
        if (!clone->layers[i].set) {
            Py_DECREF(clone);
            return NULL;
        }
        clone->layers[i].values.cap = self->layers[i].values.size;
        clone->layers[i].values.size = self->layers[i].values.size;
        if (clone->layers[i].values.size) {
            clone->layers[i].values.items = PyMem_Malloc(
                (size_t)clone->layers[i].values.size * sizeof(uint64_t)
            );
            if (!clone->layers[i].values.items) {
                Py_DECREF(clone);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(
                clone->layers[i].values.items,
                self->layers[i].values.items,
                (size_t)clone->layers[i].values.size * sizeof(uint64_t)
            );
        }
    }
    clone->forbidden = sparse_set_clone(self->forbidden);
    if (!clone->forbidden) {
        Py_DECREF(clone);
        return NULL;
    }
    clone->selected.cap = self->selected.size;
    clone->selected.size = self->selected.size;
    if (clone->selected.size) {
        clone->selected.items = PyMem_Malloc((size_t)clone->selected.size * sizeof(uint64_t));
        if (!clone->selected.items) {
            Py_DECREF(clone);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(
            clone->selected.items,
            self->selected.items,
            (size_t)clone->selected.size * sizeof(uint64_t)
        );
    }
    return (PyObject *)clone;
}

static PyObject *NativeState_selected_columns(NativeState *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *tuple = PyTuple_New(self->selected.size);
    if (!tuple) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->selected.size; i++) {
        PyTuple_SET_ITEM(tuple, i, PyLong_FromUnsignedLongLong(self->selected.items[i]));
    }
    return tuple;
}

static PyObject *NativeState_forbidden_count(NativeState *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromSsize_t(self->forbidden->count);
}

static PyObject *NativeState_layer_counts(NativeState *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *tuple = PyTuple_New(self->max_subset_size + 1);
    if (!tuple) {
        return NULL;
    }
    for (int i = 0; i <= self->max_subset_size; i++) {
        PyTuple_SET_ITEM(tuple, i, PyLong_FromSsize_t(self->layers[i].values.size));
    }
    return tuple;
}

static PyMethodDef NativeState_methods[] = {
    {"can_add", (PyCFunction)NativeState_can_add, METH_VARARGS, "Return whether a mask can be added."},
    {"add", (PyCFunction)NativeState_add, METH_VARARGS, "Add a legal mask and return forbidden growth."},
    {"undo", (PyCFunction)NativeState_undo, METH_VARARGS, "Undo recent LIFO additions."},
    {"clone", (PyCFunction)NativeState_clone, METH_NOARGS, "Clone the native state."},
    {"selected_columns", (PyCFunction)NativeState_selected_columns, METH_NOARGS, "Return selected columns."},
    {"forbidden_count", (PyCFunction)NativeState_forbidden_count, METH_NOARGS, "Return forbidden count."},
    {"layer_counts", (PyCFunction)NativeState_layer_counts, METH_NOARGS, "Return reachable layer counts."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject NativeStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_linear_code_native.NativeForbiddenState",
    .tp_basicsize = sizeof(NativeState),
    .tp_dealloc = (destructor)NativeState_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Native exact sparse forbidden-state engine.",
    .tp_methods = NativeState_methods,
    .tp_init = (initproc)NativeState_init,
    .tp_new = PyType_GenericNew,
};

static PyObject *native_validate_columns(PyObject *self, PyObject *args) {
    int r, distance;
    PyObject *columns_obj;
    if (!PyArg_ParseTuple(args, "iiO", &r, &distance, &columns_obj)) {
        return NULL;
    }
    PyObject *state_args = Py_BuildValue("(ii)", r, distance);
    if (!state_args) {
        return NULL;
    }
    NativeState *state = (NativeState *)PyObject_CallObject((PyObject *)&NativeStateType, state_args);
    Py_DECREF(state_args);
    if (!state) {
        return NULL;
    }
    PyObject *seq = PySequence_Fast(columns_obj, "columns must be a sequence");
    if (!seq) {
        Py_DECREF(state);
        return NULL;
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < size; i++) {
        uint64_t mask = PyLong_AsUnsignedLongLong(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            Py_DECREF(state);
            return NULL;
        }
        if (mask >= state->mask_limit || sparse_set_contains(state->forbidden, mask)) {
            Py_DECREF(seq);
            Py_DECREF(state);
            Py_RETURN_FALSE;
        }
        PyObject *add_args = Py_BuildValue("(K)", (unsigned long long)mask);
        if (!add_args) {
            Py_DECREF(seq);
            Py_DECREF(state);
            return NULL;
        }
        PyObject *growth = NativeState_add(state, add_args);
        Py_DECREF(add_args);
        Py_XDECREF(growth);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            Py_DECREF(state);
            return NULL;
        }
    }
    Py_DECREF(seq);
    Py_DECREF(state);
    Py_RETURN_TRUE;
}

static PyObject *native_r_limit(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromLong(NATIVE_R_LIMIT);
}

static PyMethodDef module_methods[] = {
    {"validate_columns", native_validate_columns, METH_VARARGS, "Validate free columns exactly."},
    {"native_r_limit", native_r_limit, METH_NOARGS, "Return native r limit."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_linear_code_native",
    "Native binary linear-code legality engine.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__linear_code_native(void) {
    if (PyType_Ready(&NativeStateType) < 0) {
        return NULL;
    }
    PyObject *module = PyModule_Create(&module_def);
    if (!module) {
        return NULL;
    }
    Py_INCREF(&NativeStateType);
    if (PyModule_AddObject(module, "NativeForbiddenState", (PyObject *)&NativeStateType) < 0) {
        Py_DECREF(&NativeStateType);
        Py_DECREF(module);
        return NULL;
    }
    PyModule_AddIntConstant(module, "NATIVE_R_LIMIT", NATIVE_R_LIMIT);
    return module;
}
