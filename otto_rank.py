"""otto_rank.py - classeur appris (reranker) : variables par couple (session, candidat), jeux d'entraînement, évaluation."""
import numpy as np
from otto_core import WEIGHTS, TYPE2ID, split_labels

FEATS = ["gen_score", "raw_sum", "n_seeds", "best_seed", "in_session", "n_events", "last_pos",
         "n_click", "n_cart", "n_order", "age_min", "logpop", "logpop_t", "top_rank",
         "prefix_len", "n_distinct", "last_type", "span_min"]


def make_ctx(aid_arr, ts_arr, type_arr, cum, n_items, nj, nc, tops, pop_mask=None):
    """Regroupe tout ce dont cand_features a besoin. pop_mask : événements utilisés pour la popularité (ts < T)."""
    m = np.ones(len(aid_arr), bool) if pop_mask is None else pop_mask
    return {"aid": aid_arr, "ts": ts_arr, "type": type_arr, "cum": cum, "nj": nj, "nc": nc, "tops": tops,
            "logpop": np.log1p(np.bincount(aid_arr[m], minlength=n_items)),
            "logpop_t": {t: np.log1p(np.bincount(aid_arr[m & (type_arr == i)], minlength=n_items))
                         for t, i in TYPE2ID.items()}}


def cand_features(sid, k, t, ctx, n_seed=10, gamma=0.5):
    """Pool et matrice de variables (une ligne par candidat) pour la session sid vue jusqu'à son k-ième événement."""
    a = ctx["cum"][sid]
    ev_aid = ctx["aid"][a:a + k].astype(np.int64); ev_type = ctx["type"][a:a + k].astype(np.int64)
    ev_ts = ctx["ts"][a:a + k].astype(np.int64)
    seen = np.asarray(list(dict.fromkeys(map(int, ev_aid[::-1]))), dtype=np.int64)    # récents d'abord
    sd = seen[:n_seed]; S = len(sd)
    nbr = ctx["nj"][sd]; cnt = ctx["nc"][sd].astype(np.float64)
    rnc = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1)                         # normalisation par graine
    decay = gamma ** np.arange(S)
    top = np.asarray(ctx["tops"][t], dtype=np.int64)
    flat = nbr.ravel(); m = flat >= 0
    pool = np.unique(np.concatenate([seen, flat[m].astype(np.int64), top])); P = len(pool)
    idx = np.searchsorted(pool, flat[m]); seed_r = np.repeat(np.arange(S), nbr.shape[1])[m]
    gen = np.bincount(idx, weights=(rnc * decay[:, None]).ravel()[m], minlength=P)
    raw = np.bincount(idx, weights=cnt.ravel()[m], minlength=P)
    nse = np.bincount(idx, minlength=P)
    best = np.full(P, 99.0); np.minimum.at(best, idx, seed_r)
    pidx = np.searchsorted(pool, ev_aid)
    n_ev = np.bincount(pidx, minlength=P)
    n_cl, n_ca, n_or = (np.bincount(pidx[ev_type == q], minlength=P) for q in (0, 1, 2))
    last_pos = np.full(P, 99.0); last_pos[np.searchsorted(pool, seen)] = np.arange(len(seen))
    last_ts = np.full(P, -1, dtype=np.int64); np.maximum.at(last_ts, pidx, ev_ts)
    age = np.where(last_ts >= 0, (ev_ts[-1] - last_ts) / 60.0, -1.0)
    toprank = np.full(P, 99.0); toprank[np.searchsorted(pool, top)] = np.arange(len(top))
    full = lambda v: np.full(P, v, dtype=np.float64)
    X = np.column_stack([gen, raw, nse, best, (n_ev > 0).astype(float), n_ev, last_pos,
                         n_cl, n_ca, n_or, age, ctx["logpop"][pool], ctx["logpop_t"][t][pool], toprank,
                         full(k), full(len(seen)), full(ev_type[-1]),
                         full((ev_ts[-1] - ev_ts[0]) / 60.0)]).astype(np.float32)
    return pool, X


def make_examples(ctx, sids, seed):
    """Même procédé que le dev : coupure uniforme dans [1, n-1]."""
    cum = ctx["cum"]; r = np.random.default_rng(seed); GT_, PL_ = {}, {}
    for sid in sids:
        a, z = cum[sid], cum[sid + 1]; k = int(r.integers(1, z - a))
        GT_[int(sid)] = split_labels(ctx["aid"][a + k:z], ctx["type"][a + k:z]); PL_[int(sid)] = k
    return GT_, PL_


def sample_labeled(ctx, pool_ids, t, n_target, seed, chunk=2000):
    """Tire des sessions au hasard jusqu'à en avoir n_target avec au moins un label du type t."""
    order = np.random.default_rng(seed).permutation(pool_ids); GT_, PL_ = {}, {}
    for c, s in enumerate(range(0, len(order), chunk)):
        g, p = make_examples(ctx, order[s:s + chunk], seed * 1000 + c)
        for sid in g:
            if g[sid][t] and len(GT_) < n_target:
                GT_[sid] = g[sid]; PL_[sid] = p[sid]
        if len(GT_) >= n_target: break
    return GT_, PL_


def build_dataset(ctx, GT_, PL_, t):
    Xs, ys = [], []
    for sid, g in GT_.items():
        pool, X = cand_features(sid, PL_[sid], t, ctx)
        Xs.append(X); ys.append(np.isin(pool, list(g[t])).astype(np.int8))
    return np.vstack(Xs), np.concatenate(ys)


def rerank_hits(model, ctx, GT_, PL_, t, oracle=False, chunk=500):
    """Hits par session du top-20 du classeur (oracle=True : classe par l'étiquette vraie)."""
    sids = list(GT_); H = np.zeros(len(sids))
    for s in range(0, len(sids), chunk):
        part, pools, Xs = sids[s:s + chunk], [], []
        for sid in part:
            pool, X = cand_features(sid, PL_[sid], t, ctx); pools.append(pool); Xs.append(X)
        if oracle:
            sc_all = np.concatenate([np.isin(p, list(GT_[sid][t])).astype(float) for p, sid in zip(pools, part)])
        else:
            sc_all = model.predict_proba(np.vstack(Xs))[:, 1]
        off = 0
        for j, (sid, pool) in enumerate(zip(part, pools)):
            sc = sc_all[off:off + len(pool)]; off += len(pool)
            top20 = pool[np.argsort(-sc, kind="stable")[:20]]
            H[s + j] = len(set(map(int, top20)) & GT_[sid][t])
    return H
