#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NATIVE_R_LIMIT 30
#define PAGE_BITS 4096u
#define PAGE_WORDS (PAGE_BITS / 64u)
#define PAGE_SHIFT 12u
#define PAGE_MASK (PAGE_BITS - 1u)

typedef struct {
    size_t refcount;
    uint64_t words[PAGE_WORDS];
} BitPage;

typedef struct {
    uint32_t page_count;
    BitPage **pages;
    uint64_t count;
} Bitset;

typedef struct {
    uint32_t *items;
    Py_ssize_t size;
    Py_ssize_t cap;
} UIntVec;

typedef struct {
    uint32_t *items;
    Py_ssize_t size;
    Py_ssize_t cap;
} DeltaVec;

typedef struct {
    DeltaVec *layer_deltas;
    DeltaVec forbidden_delta;
    uint32_t selected_column;
} HistoryEntry;

typedef struct {
    Bitset *bits;
    UIntVec values;
} Layer;

typedef struct {
    PyObject_HEAD
    int r;
    int distance;
    int max_subset_size;
    uint32_t mask_limit;
    Layer *layers;
    Bitset *forbidden;
    UIntVec selected;
    HistoryEntry *history;
    Py_ssize_t history_size;
    Py_ssize_t history_cap;
} NativeState;

static int vec_push(UIntVec *vec, uint32_t value) {
    if (vec->size == vec->cap) {
        Py_ssize_t new_cap = vec->cap ? vec->cap * 2 : 16;
        uint32_t *items = PyMem_Realloc(vec->items, (size_t)new_cap * sizeof(uint32_t));
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

static int delta_push(DeltaVec *vec, uint32_t value) {
    if (vec->size == vec->cap) {
        Py_ssize_t new_cap = vec->cap ? vec->cap * 2 : 16;
        uint32_t *items = PyMem_Realloc(vec->items, (size_t)new_cap * sizeof(uint32_t));
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

static BitPage *page_new(void) {
    BitPage *page = PyMem_Calloc(1, sizeof(BitPage));
    if (!page) {
        PyErr_NoMemory();
        return NULL;
    }
    page->refcount = 1;
    return page;
}

static void page_decref(BitPage *page) {
    if (!page) {
        return;
    }
    if (--page->refcount == 0) {
        PyMem_Free(page);
    }
}

static Bitset *bitset_new(int r) {
    uint64_t bit_count = 1ULL << r;
    uint32_t page_count = (uint32_t)((bit_count + PAGE_BITS - 1u) >> PAGE_SHIFT);
    Bitset *bitset = PyMem_Calloc(1, sizeof(Bitset));
    if (!bitset) {
        PyErr_NoMemory();
        return NULL;
    }
    bitset->pages = PyMem_Calloc(page_count, sizeof(BitPage *));
    if (!bitset->pages) {
        PyMem_Free(bitset);
        PyErr_NoMemory();
        return NULL;
    }
    bitset->page_count = page_count;
    bitset->count = 0;
    return bitset;
}

static Bitset *bitset_clone(Bitset *src) {
    Bitset *dst = PyMem_Calloc(1, sizeof(Bitset));
    if (!dst) {
        PyErr_NoMemory();
        return NULL;
    }
    dst->pages = PyMem_Malloc((size_t)src->page_count * sizeof(BitPage *));
    if (!dst->pages) {
        PyMem_Free(dst);
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(dst->pages, src->pages, (size_t)src->page_count * sizeof(BitPage *));
    dst->page_count = src->page_count;
    dst->count = src->count;
    for (uint32_t i = 0; i < src->page_count; i++) {
        if (dst->pages[i]) {
            dst->pages[i]->refcount++;
        }
    }
    return dst;
}

static void bitset_free(Bitset *bitset) {
    if (!bitset) {
        return;
    }
    for (uint32_t i = 0; i < bitset->page_count; i++) {
        page_decref(bitset->pages[i]);
    }
    PyMem_Free(bitset->pages);
    PyMem_Free(bitset);
}

static int bitset_ensure_unique_page(Bitset *bitset, uint32_t page_index) {
    BitPage *page = bitset->pages[page_index];
    if (!page) {
        page = page_new();
        if (!page) {
            return -1;
        }
        bitset->pages[page_index] = page;
        return 0;
    }
    if (page->refcount == 1) {
        return 0;
    }
    BitPage *copy = PyMem_Malloc(sizeof(BitPage));
    if (!copy) {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(copy, page, sizeof(BitPage));
    copy->refcount = 1;
    page->refcount--;
    bitset->pages[page_index] = copy;
    return 0;
}

static int bitset_get(Bitset *bitset, uint32_t value) {
    uint32_t page_index = value >> PAGE_SHIFT;
    BitPage *page = bitset->pages[page_index];
    if (!page) {
        return 0;
    }
    uint32_t bit = value & PAGE_MASK;
    return (page->words[bit >> 6] >> (bit & 63u)) & 1u;
}

static int bitset_set(Bitset *bitset, uint32_t value) {
    uint32_t page_index = value >> PAGE_SHIFT;
    uint32_t bit = value & PAGE_MASK;
    if (bitset_get(bitset, value)) {
        return 0;
    }
    if (bitset_ensure_unique_page(bitset, page_index) < 0) {
        return -1;
    }
    bitset->pages[page_index]->words[bit >> 6] |= 1ULL << (bit & 63u);
    bitset->count++;
    return 1;
}

static int bitset_clear(Bitset *bitset, uint32_t value) {
    uint32_t page_index = value >> PAGE_SHIFT;
    uint32_t bit = value & PAGE_MASK;
    if (!bitset_get(bitset, value)) {
        return 0;
    }
    if (bitset_ensure_unique_page(bitset, page_index) < 0) {
        return -1;
    }
    bitset->pages[page_index]->words[bit >> 6] &= ~(1ULL << (bit & 63u));
    bitset->count--;
    return 1;
}

static int layer_add(Layer *layer, uint32_t value) {
    int changed = bitset_set(layer->bits, value);
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

static int state_update_layers(NativeState *self, uint32_t column, HistoryEntry *history) {
    int max_s = self->max_subset_size;
    for (int s = max_s; s >= 1; s--) {
        Py_ssize_t prev_size = self->layers[s - 1].values.size;
        uint32_t *prev_items = self->layers[s - 1].values.items;
        for (Py_ssize_t i = 0; i < prev_size; i++) {
            uint32_t value = prev_items[i] ^ column;
            if (!bitset_get(self->layers[s].bits, value)) {
                int was_forbidden = bitset_get(self->forbidden, value);
                if (layer_add(&self->layers[s], value) < 0) {
                    return -1;
                }
                if (history && delta_push(&history->layer_deltas[s], value) < 0) {
                    return -1;
                }
                if (!was_forbidden) {
                    int changed = bitset_set(self->forbidden, value);
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
    self->mask_limit = r == 32 ? 0xFFFFFFFFu : (uint32_t)(1ULL << r);
    self->layers = PyMem_Calloc((size_t)self->max_subset_size + 1, sizeof(Layer));
    if (!self->layers) {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i <= self->max_subset_size; i++) {
        self->layers[i].bits = bitset_new(r);
        if (!self->layers[i].bits) {
            return -1;
        }
    }
    self->forbidden = bitset_new(r);
    if (!self->forbidden) {
        return -1;
    }
    if (layer_add(&self->layers[0], 0) < 0 || bitset_set(self->forbidden, 0) < 0) {
        return -1;
    }
    for (int bit = 0; bit < r; bit++) {
        if (state_update_layers(self, (uint32_t)(1u << bit), NULL) < 0) {
            return -1;
        }
    }
    return 0;
}

static void NativeState_dealloc(NativeState *self) {
    if (self->layers) {
        for (int i = 0; i <= self->max_subset_size; i++) {
            bitset_free(self->layers[i].bits);
            vec_free(&self->layers[i].values);
        }
        PyMem_Free(self->layers);
    }
    bitset_free(self->forbidden);
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
    uint32_t mask;
    if (!PyArg_ParseTuple(args, "I", &mask)) {
        return NULL;
    }
    if (mask >= self->mask_limit) {
        Py_RETURN_FALSE;
    }
    if (bitset_get(self->forbidden, mask)) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *NativeState_add(NativeState *self, PyObject *args) {
    uint32_t mask;
    if (!PyArg_ParseTuple(args, "I", &mask)) {
        return NULL;
    }
    if (mask >= self->mask_limit || bitset_get(self->forbidden, mask)) {
        PyErr_SetString(PyExc_ValueError, "Illegal free column");
        return NULL;
    }
    HistoryEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.selected_column = mask;
    entry.layer_deltas = PyMem_Calloc((size_t)self->max_subset_size + 1, sizeof(DeltaVec));
    if (!entry.layer_deltas) {
        PyErr_NoMemory();
        return NULL;
    }
    uint64_t before = self->forbidden->count;
    if (state_update_layers(self, mask, &entry) < 0 || vec_push(&self->selected, mask) < 0) {
        history_entry_free(&entry, self->max_subset_size + 1);
        return NULL;
    }
    if (history_push(self, &entry) < 0) {
        history_entry_free(&entry, self->max_subset_size + 1);
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(self->forbidden->count - before);
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
                if (bitset_clear(self->layers[s].bits, delta->items[i]) < 0) {
                    return NULL;
                }
            }
            self->layers[s].values.size -= delta->size;
        }
        for (Py_ssize_t i = entry->forbidden_delta.size - 1; i >= 0; i--) {
            if (bitset_clear(self->forbidden, entry->forbidden_delta.items[i]) < 0) {
                return NULL;
            }
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
        clone->layers[i].bits = bitset_clone(self->layers[i].bits);
        if (!clone->layers[i].bits) {
            Py_DECREF(clone);
            return NULL;
        }
        clone->layers[i].values.cap = self->layers[i].values.size;
        clone->layers[i].values.size = self->layers[i].values.size;
        if (clone->layers[i].values.size) {
            clone->layers[i].values.items = PyMem_Malloc(
                (size_t)clone->layers[i].values.size * sizeof(uint32_t)
            );
            if (!clone->layers[i].values.items) {
                Py_DECREF(clone);
                PyErr_NoMemory();
                return NULL;
            }
            memcpy(
                clone->layers[i].values.items,
                self->layers[i].values.items,
                (size_t)clone->layers[i].values.size * sizeof(uint32_t)
            );
        }
    }
    clone->forbidden = bitset_clone(self->forbidden);
    if (!clone->forbidden) {
        Py_DECREF(clone);
        return NULL;
    }
    clone->selected.cap = self->selected.size;
    clone->selected.size = self->selected.size;
    if (clone->selected.size) {
        clone->selected.items = PyMem_Malloc((size_t)clone->selected.size * sizeof(uint32_t));
        if (!clone->selected.items) {
            Py_DECREF(clone);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(
            clone->selected.items,
            self->selected.items,
            (size_t)clone->selected.size * sizeof(uint32_t)
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
        PyTuple_SET_ITEM(tuple, i, PyLong_FromUnsignedLong(self->selected.items[i]));
    }
    return tuple;
}

static PyObject *NativeState_forbidden_count(NativeState *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromUnsignedLongLong(self->forbidden->count);
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
    .tp_doc = "Native exact forbidden-state engine.",
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
        uint32_t mask = (uint32_t)PyLong_AsUnsignedLong(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            Py_DECREF(state);
            return NULL;
        }
        if (mask >= state->mask_limit || bitset_get(state->forbidden, mask)) {
            Py_DECREF(seq);
            Py_DECREF(state);
            Py_RETURN_FALSE;
        }
        PyObject *add_args = Py_BuildValue("(I)", mask);
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
