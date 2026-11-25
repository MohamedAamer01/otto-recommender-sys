"""otto_infer.py - inférence du classeur sur le test et écriture des fichiers de soumission."""
import time
import numpy as np
from otto_core import lists_b4star
from otto_rank import cand_features


def ctx_for_sessions(ctx_pop, aid, ts, typ, cum):
    """Contexte dont la popularité vient du train (ctx_pop) mais dont les sessions viennent d'autres tableaux (test)."""
    c = dict(ctx_pop); c.update({"aid": aid, "ts": ts, "type": typ, "cum": cum})
    return c


def infer_orders(indices, ks, models, ctx_s, t="orders", chunk=500):
    """Top-20 de chaque modèle pour chaque session (indices) vue jusqu'à son k-ième événement.
    Retourne {nom_du_modèle: [liste de 20 aid par session]}."""
    out = {name: [] for name in models}
    for s in range(0, len(indices), chunk):
        idx, kk = indices[s:s + chunk], ks[s:s + chunk]
        pools, Xs = [], []
        for sid, k in zip(idx, kk):
            pool, X = cand_features(int(sid), int(k), t, ctx_s); pools.append(pool); Xs.append(X)
        Xall = np.vstack(Xs)
        for name, m in models.items():
            sc = m.predict_proba(Xall)[:, 1]; off = 0
            for pool in pools:
                p = sc[off:off + len(pool)]; off += len(pool)
                out[name].append([int(x) for x in pool[np.argsort(-p, kind="stable")[:20]]])
    return out


def write_submissions(paths, models, ids, t_aid, t_size, ctx_t, tops, nj, nc, chunk=500):
    """Un fichier par modèle : clicks et carts par B4*, orders par le classeur. Les variables sont calculées une seule fois."""
    off = np.concatenate([[0], np.cumsum(t_size, dtype=np.int64)])
    files = {k: open(p, "w") for k, p in paths.items()}
    for f in files.values():
        f.write("session_type,labels\n")
    t0, n = time.time(), len(ids)
    try:
        for s in range(0, n, chunk):
            idx = np.arange(s, min(s + chunk, n))
            top = infer_orders(idx, t_size[idx], models, ctx_t, chunk=chunk)
            for j, i in enumerate(idx):
                seen = list(dict.fromkeys(map(int, t_aid[off[i]:off[i + 1]][::-1])))
                r = lists_b4star(seen, tops, nj, nc)
                for name, f in files.items():
                    f.write(f"{ids[i]}_clicks,{' '.join(map(str, r['clicks']))}\n")
                    f.write(f"{ids[i]}_carts,{' '.join(map(str, r['carts']))}\n")
                    f.write(f"{ids[i]}_orders,{' '.join(map(str, top[name][j]))}\n")
            if s > 0 and s % 100_000 == 0:
                print(f"{s:,} sessions ({time.time()-t0:.0f}s)")
    finally:
        for f in files.values():
            f.close()
