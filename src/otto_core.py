"""otto_core.py - fonctions essentielles du projet OTTO (testées contre un calcul brut ou la version notebook)."""
import time
import numpy as np

WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
TYPE2ID = {"clicks": 0, "carts": 1, "orders": 2}


# ---------------------------------------------------------------- co-visitation
def covisit_block(aid_b, ts_b, sizes_b, K, window, tmax=None):
    """Paires ordonnées (i, j) des K derniers événements (ts < tmax) de chaque session.
    Retourne (clés triées, compteurs). clé = i * 2**21 + j."""
    S = len(sizes_b)
    sess = np.repeat(np.arange(S), sizes_b)
    if tmax is not None:                                   # anti-fuite : on retire les événements >= tmax
        keep = ts_b < tmax
        sess, aid_b, ts_b = sess[keep], aid_b[keep], ts_b[keep]
    sz = np.bincount(sess, minlength=S)
    starts = np.cumsum(sz) - sz
    pos_end = sz[sess] - 1 - (np.arange(len(sess)) - starts[sess])      # 0 = dernier événement
    sel = pos_end < K
    A = np.full((S, K), -1, dtype=np.int64)
    Tm = np.zeros((S, K), dtype=np.int64)
    A[sess[sel], pos_end[sel]] = aid_b[sel]
    Tm[sess[sel], pos_end[sel]] = ts_b[sel]
    ai, aj = A[:, :, None], A[:, None, :]
    ok = (ai >= 0) & (aj >= 0) & (ai != aj) & (np.abs(Tm[:, :, None] - Tm[:, None, :]) <= window)
    keys = (np.broadcast_to(ai, ok.shape)[ok] << 21) | np.broadcast_to(aj, ok.shape)[ok]
    return np.unique(keys, return_counts=True)


def merge(parts):
    k = np.concatenate([p[0] for p in parts]); c = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, weights=c).astype(np.int64)


def build_topn(aid_arr, ts_arr, sizes_arr, cum, n_items, K=10, window=86400, tmax=None,
               N=50, P=4, BLOCK=50_000):
    """Table des N meilleurs voisins par article : nbr_j[i] = voisins, nbr_c[i] = compteurs."""
    nbr_j = np.full((n_items, N), -1, dtype=np.int32)
    nbr_c = np.zeros((n_items, N), dtype=np.float32)
    n_sess, MASK = len(sizes_arr), (1 << 21) - 1
    for p in range(P):
        t0, parts = time.time(), []
        for s in range(0, n_sess, BLOCK):
            e = min(s + BLOCK, n_sess); a, z = cum[s], cum[e]
            k, c = covisit_block(aid_arr[a:z], ts_arr[a:z], sizes_arr[s:e], K, window, tmax)
            m = ((k >> 21) % P) == p
            parts.append((k[m], c[m]))
            if len(parts) >= 20:
                parts = [merge(parts)]
        keys, counts = merge(parts)
        i = (keys >> 21).astype(np.int64); j = (keys & MASK).astype(np.int32)
        order = np.lexsort((-counts, i))
        i, j, counts = i[order], j[order], counts[order]
        rank = np.arange(len(i)) - np.searchsorted(i, i, side="left")
        keep = rank < N
        nbr_j[i[keep], rank[keep]] = j[keep]
        nbr_c[i[keep], rank[keep]] = counts[keep]
        print(f"partition {p+1}/{P} : {len(keys):,} paires distinctes | {time.time()-t0:.0f}s")
        del keys, counts, i, j, order, parts
    return nbr_j, nbr_c


# ---------------------------------------------------------------- popularité et listes
def top_lists(aid_arr, type_arr, n_items, mask=None, k=40):
    """Top-k articles par type (clicks, carts, orders), sur les événements où mask est vrai."""
    m = np.ones(len(aid_arr), bool) if mask is None else mask
    return {t: np.argsort(-np.bincount(aid_arr[m & (type_arr == i)], minlength=n_items),
                          kind="stable")[:k].tolist() for t, i in TYPE2ID.items()}


def topup(base, top, n=20):
    lst = list(base); s = set(lst)
    for x in top:
        if len(lst) == n: break
        if x not in s: lst.append(x)
    return lst


def lists_b4star(seen, tops, nj, nc, n_seed=10, gamma=0.5, row_norm=True, n=20):
    """B4* : articles de la session (récents d'abord), puis candidats de co-visitation, puis popularité par type.
    seen : articles distincts de la session, du plus récent au plus ancien."""
    base = seen[:n]; sset = set(base)
    if len(base) < n:
        sd = np.asarray(seen[:n_seed])
        cs = nc[sd].astype(np.float64)
        if row_norm:
            cs = cs / np.maximum(cs.sum(axis=1, keepdims=True), 1)
        cs = cs * (gamma ** np.arange(len(sd)))[:, None]
        js = nj[sd].ravel(); cs = cs.ravel(); m = js >= 0
        if m.any():
            u, inv = np.unique(js[m], return_inverse=True)
            sc = np.bincount(inv, weights=cs[m])
            order = np.argsort(-sc, kind="stable")
            base = base + [int(x) for x in u[order] if int(x) not in sset][:n - len(base)]
    return {t: topup(base, tops[t], n) for t in WEIGHTS}


def write_submission(path, ids, t_aid, t_size, tops, nj, nc, **kw):
    """Écrit le fichier de soumission en flux : 3 lignes par session (clicks, carts, orders)."""
    off = np.concatenate([[0], np.cumsum(t_size, dtype=np.int64)])
    t0 = time.time()
    with open(path, "w") as f:
        f.write("session_type,labels\n")
        for i in range(len(ids)):
            seen = list(dict.fromkeys(map(int, t_aid[off[i]:off[i + 1]][::-1])))
            r = lists_b4star(seen, tops, nj, nc, **kw)
            for t in WEIGHTS:
                f.write(f"{ids[i]}_{t},{' '.join(map(str, r[t]))}\n")
            if (i + 1) % 500_000 == 0:
                print(f"{i+1:,} sessions ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------- validation et métrique
def split_labels(suf_aid, suf_type):
    clicks = suf_aid[suf_type == 0]
    return {"clicks": {int(clicks[0])} if len(clicks) else set(),
            "carts": set(map(int, suf_aid[suf_type == 1])),
            "orders": set(map(int, suf_aid[suf_type == 2]))}


def per_session(make, GT, k=20):
    sids = list(GT)
    H = {t: np.zeros(len(sids)) for t in WEIGHTS}; D = {t: np.zeros(len(sids)) for t in WEIGHTS}
    for i, sid in enumerate(sids):
        lists = make(sid)
        for t in WEIGHTS:
            gt = GT[sid][t]
            if gt:
                H[t][i] = len(set(lists[t][:k]) & gt); D[t][i] = min(k, len(gt))
    return H, D


def score_from(H, D, idx=None):
    out = {}
    for t in WEIGHTS:
        h = H[t] if idx is None else H[t][idx]; d = D[t] if idx is None else D[t][idx]
        out[t] = h.sum() / d.sum()
    out["score"] = sum(WEIGHTS[t] * out[t] for t in WEIGHTS)
    return out


def boot_diff_all(HA, HB, D, B=100, seed=0):
    r = np.random.default_rng(seed); n = len(next(iter(D.values())))
    diffs = {k: [] for k in list(WEIGHTS) + ["score"]}
    for _ in range(B):
        idx = r.integers(0, n, n); a = score_from(HA, D, idx); b = score_from(HB, D, idx)
        for k in diffs: diffs[k].append(b[k] - a[k])
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in diffs.items()}
