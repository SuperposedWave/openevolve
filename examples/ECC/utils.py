from itertools import combinations

def _rows_to_bitmasks(G):
    """G: k x n list of 0/1 -> list[int] row bitmasks (LSB = col0)."""
    k = len(G)
    n = len(G[0]) if k else 0
    rows = []
    for r in range(k):
        m = 0
        row = G[r]
        for c in range(n):
            if row[c] & 1:
                m |= (1 << c)
        rows.append(m)
    return rows, k, n

def _extract_kxk_cols(rows, cols):
    """Extract kxk matrix (as list[int] row bitmasks of length k) from full-row bitmasks."""
    # Map selected columns to positions 0..k-1
    pos = {col: j for j, col in enumerate(cols)}
    k = len(rows)
    mat_rows = []
    for i in range(k):
        m = 0
        r = rows[i]
        for col, j in pos.items():
            if (r >> col) & 1:
                m |= (1 << j)
        mat_rows.append(m)
    return mat_rows

def _invert_gf2(mat_rows, k):
    """
    Invert kxk GF(2) matrix given as list[int] row bitmasks (length k).
    Return inv_rows (list[int] row bitmasks length k) or None if singular.
    """
    A = mat_rows[:]  # left
    I = [(1 << i) for i in range(k)]  # right
    # Gaussian elimination to RREF, simultaneously on I
    for col in range(k):
        pivot = None
        for r in range(col, k):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            I[col], I[pivot] = I[pivot], I[col]
        # eliminate other rows
        for r in range(k):
            if r != col and ((A[r] >> col) & 1):
                A[r] ^= A[col]
                I[r] ^= I[col]
    return I

def _apply_left_transform(inv_rows, rows):
    """
    rows: list[int] original rows (k rows, n-bit)
    inv_rows: list[int] row bitmasks (k-bit) representing kxk matrix A (rows of A)
    Return new_rows = A * G over GF(2), still as n-bit row masks.
    """
    k = len(rows)
    new_rows = []
    for i in range(k):
        comb = inv_rows[i]  # which original rows to xor
        acc = 0
        x = comb
        while x:
            lsb = x & -x
            j = (lsb.bit_length() - 1)
            acc ^= rows[j]
            x ^= lsb
        new_rows.append(acc)
    return new_rows

def _permute_columns_rows(rows, perm, n):
    """
    Apply column permutation 'perm' where perm[new_index] = old_index.
    rows are n-bit masks; return masks in new column order.
    """
    new_rows = []
    for r in rows:
        nr = 0
        for new_c, old_c in enumerate(perm):
            if (r >> old_c) & 1:
                nr |= (1 << new_c)
        new_rows.append(nr)
    return new_rows

def _columns_of_B_from_systematic(rows_sys, k, n):
    """
    rows_sys are in order of columns [info-set first k | rest n-k]
    Assume first k columns are I. Extract B columns as list[int] column bitmasks (k-bit),
    for columns k..n-1.
    """
    m = n - k
    cols = []
    for j in range(m):
        colmask = 0
        col_idx = k + j
        for i in range(k):
            if (rows_sys[i] >> col_idx) & 1:
                colmask |= (1 << i)
        cols.append(colmask)
    return cols

def _build_rows_from_I_and_Bcols(k, Bcols):
    """
    Return rows of [I | B] as list[int] bitmasks length k + len(Bcols).
    Columns: first k are I, then Bcols in given order.
    """
    m = len(Bcols)
    n = k + m
    rows = [0] * k
    # put I
    for i in range(k):
        rows[i] |= (1 << i)
    # put B
    for j, colmask in enumerate(Bcols):
        col_idx = k + j
        x = colmask
        while x:
            lsb = x & -x
            i = (lsb.bit_length() - 1)
            rows[i] |= (1 << col_idx)
            x ^= lsb
    return rows, n

def _rows_bitmasks_to_list(rows, k, n):
    G = [[0]*n for _ in range(k)]
    for i in range(k):
        r = rows[i]
        for c in range(n):
            G[i][c] = (r >> c) & 1
    return G

def canonical_form_binary_linear_code(G):
    """
    Compute a deterministic canonical systematic generator [I | B*] for a binary linear code
    under column-permutation equivalence.

    Returns:
      G_can: k x n list[list[int]] in systematic form
      signature: bytes/string usable for hashing/dedup
    """
    rows, k, n = _rows_to_bitmasks(G)
    if k == 0:
        return [], "k=0"
    if n == 0:
        return [[] for _ in range(k)], "n=0"

    best_key = None
    best_Bcols = None

    all_cols = list(range(n))
    for S in combinations(all_cols, k):
        # check if G_S invertible
        sub = _extract_kxk_cols(rows, S)
        inv = _invert_gf2(sub, k)
        if inv is None:
            continue

        # Left-multiply to make selected columns become I (in original column order)
        rows_left = _apply_left_transform(inv, rows)

        # Permute columns so S comes first (in increasing order), rest after (in increasing order)
        rest = [c for c in all_cols if c not in set(S)]
        perm = list(S) + rest  # perm[new]=old
        rows_sys = _permute_columns_rows(rows_left, perm, n)

        # Extract B and sort its columns canonically (lex by integer)
        Bcols = _columns_of_B_from_systematic(rows_sys, k, n)
        Bcols_sorted = sorted(Bcols)

        # Key: (sorted B columns) gives canonical tie-break for this info set
        key = tuple(Bcols_sorted)
        if best_key is None or key < best_key:
            best_key = key
            best_Bcols = Bcols_sorted

    if best_key is None:
        raise ValueError("No full-rank information set found; is G rank < k?")

    rows_can, n_can = _build_rows_from_I_and_Bcols(k, best_Bcols)
    G_can = _rows_bitmasks_to_list(rows_can, k, n_can)

    # A compact signature for hashing / DB lookup
    signature = f"n={n_can},k={k},Bcols={best_key}"
    return G_can, signature


# ------------------ example ------------------
if __name__ == "__main__":
    G = [
        [1,0,1,1,0,0],
        [0,1,1,0,1,0],
        [0,0,1,1,1,1],
    ]
    G_can, sig = canonical_form_binary_linear_code(G)
    print("signature:", sig)
    print("G_can:")
    for r in G_can:
        print(r)
