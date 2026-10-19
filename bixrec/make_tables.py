"""Collect runs/ into LaTeX tables + macros for the paper.

    python -m bixrec.make_tables --runs runs --data_root data --out paper/generated

Writes <out>/tables.tex (\\input in the paper) and <out>/summary.json.
Missing runs appear as "--" so the paper always compiles.

Protocol: full-catalogue ranking, history items kept in the ranking (primary). This is the protocol of the
published side-information results quoted for reference: RecBole's full-sort evaluation does not mask history for
sequential models, and DIFF states that it does not penalise repeated items. A secondary table removes the history
items in the input window (BSARec / DLFSRec-style evaluation). Ties are broken by a fixed random item order.
The selected ("proposed") model is not hard-coded: it is the main-table model with the best validation NDCG@10,
averaged over datasets after normalising by the best model on each dataset; it is only chosen once every
main-table model has runs on every dataset.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import torch

from .stats import holm, mcnemar_exact, ndcg_per_user, paired_seed_test, seed_summary, wilcoxon

ALL_DATASETS = [("ml100k", "ML-100K"), ("ml1m", "ML-1M"), ("beauty", "Beauty"), ("sports", "Sports"),
                ("toys", "Toys"), ("beauty_text", "Beauty (text)"), ("sports_text", "Sports (text)")]
DATASETS = [d for d in ALL_DATASETS if d[0] != "ml100k"]  # replaced in main() by the datasets found
DAG = "$^\\dagger$"  # trained with an additional per-stream loss: a different objective
DAG_NOTE = DAG + r"Trained with an additional per-stream loss (a different objective)."
NAMES = {
    "sasrec": "SASRec (ID)",
    "sasrec_bce": "SASRec, BCE + 1 negative",
    "sasrec_lastpos": "SASRec, last position only",
    "text": "Content expert",
    "poe50": "Static PoE ($\\alpha{=}0.5$)",
    "poe": "Tuned PoE ($\\alpha$ on val.)",
    "hybrid": "Early fusion (ID + text)",
    "dual_late": "Joint late fusion",
    "dual_late_pen": "Joint late fusion + penalty",
    "bix_nogate": "+ cross-attention",
    "bix_gate": "+ cross-attention + gate",
    "bix": "BiXRec (full)",
    "bix_lastpos": "BiXRec, last position only",
    "dual_late_aux": "Joint late fusion + stream loss" + DAG,
    "bix_aux": "BiXRec + stream loss" + DAG,
    "dual_late_ef_aux": "Joint late fusion, EF stream + stream loss" + DAG,
    "bix_ef_aux": "BiXRec-EF + stream loss" + DAG,
}
MAIN_ROWS = ["sasrec", "text", "poe50", "poe", "hybrid", "dual_late", "bix", "dual_late_aux", "bix_aux",
             "dual_late_ef_aux", "bix_ef_aux"]
ABL_ROWS = ["poe50", "dual_late", "dual_late_pen", "bix_nogate", "bix_gate", "bix"]
AUDIT_ROWS = ["sasrec_lastpos", "sasrec_bce", "sasrec", "bix_lastpos", "bix"]
SPLIT = "test"         # primary evaluation split (see module docstring)
PROPOSED = None        # set in main() by choose_proposed(); None until every main model has run everywhere
N_SEEDS = 3            # set in main(): the largest seed count of any main-table cell

# Published results on the same Amazon Beauty / Sports / Toys data (5-core; identical user/item/interaction
# counts), leave-one-out, full ranking, history items NOT removed. Copied from Kim et al., "DIFF: Dual
# Side-Information Filtering and Fusion for Sequential Recommendation", SIGIR 2025, Table 2 (R@k = HR@k for a
# single target). Not re-run by us; shown for orientation, in the unmasked tables only.
PUBLISHED_SRC = "Kim et al., SIGIR 2025, Table 2"
PUBLISHED = {
    "beauty": {"SASRec": (0.0842, 0.0424, 0.1191, 0.0511), "DIF-SR": (0.0891, 0.0444, 0.1281, 0.0542),
               "DIFF": (0.0935, 0.0526, 0.1347, 0.0632)},
    "sports": {"SASRec": (0.0487, 0.0231, 0.0709, 0.0287), "DIF-SR": (0.0534, 0.0251, 0.0784, 0.0314),
               "DIFF": (0.0574, 0.0310, 0.0853, 0.0381)},
    "toys": {"SASRec": (0.0889, 0.0436, 0.1225, 0.0521), "DIF-SR": (0.1011, 0.0504, 0.1379, 0.0597),
             "DIFF": (0.1023, 0.0553, 0.1425, 0.0656)},
}
PUB_KEYS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")


def load_runs(root):
    runs = {}
    for f in glob(os.path.join(root, "*", "*", "seed*", "metrics.json")):
        seed_dir = os.path.dirname(f)
        var_dir = os.path.dirname(seed_dir)
        ds = os.path.basename(os.path.dirname(var_dir))
        var = os.path.basename(var_dir)
        if "@" in var:  # tuning trials live in their own root; never mix them into result tables
            continue
        seed = int(os.path.basename(seed_dir)[4:])
        with open(f) as fh:
            m = json.load(fh)
        m["_dir"] = seed_dir
        runs.setdefault(ds, {}).setdefault(var, {})[seed] = m
    return runs


def ranks(run, split=None):
    return np.load(os.path.join(run["_dir"], f"ranks_{split or SPLIT}.npy"))


def fmt(s, scale=100, bold=False, digits=2):
    if s is None:
        return "--"
    txt = f"{s['mean'] * scale:.{digits}f}" + (f"\\,$\\pm$\\,{s['sd'] * scale:.{digits}f}" if s["n"] > 1 else "")
    if s["n"] < N_SEEDS:  # incomplete: mark the number of seeds
        txt += f"$^{{n={s['n']}}}$"
    return f"\\textbf{{{txt}}}" if bold else txt


SEED_NOTE = r"$^{n=k}$: only $k$ seeds so far."


def metric_summary(runs, ds, var, key, split=None):
    r = runs.get(ds, {}).get(var)
    if not r or any((split or SPLIT) not in m for m in r.values()):
        return None
    return seed_summary([m[split or SPLIT][key] for _, m in sorted(r.items())])


def choose_proposed(runs):
    """Main-table model with the best normalised validation NDCG@10, averaged over datasets. Reads validation
    numbers only, so the choice cannot depend on test results. Returns (None, info) until every main-table model
    has runs on every dataset, instead of silently choosing among the models that happen to be finished."""
    ds_list = [ds for ds, _ in DATASETS if ds in runs]
    missing = [f"{ds}:{v}" for ds in ds_list for v in MAIN_ROWS if not runs[ds].get(v)]
    if not ds_list or missing:
        return None, {"missing": missing}
    val = {(ds, v): float(np.mean([m["val"]["NDCG@10"] for m in runs[ds][v].values()]))
           for ds in ds_list for v in MAIN_ROWS}
    score = {v: float(np.mean([val[(ds, v)] / max(val[(ds, u)] for u in MAIN_ROWS) for ds in ds_list]))
             for v in MAIN_ROWS}
    return max(MAIN_ROWS, key=score.get), score


def env_report(runs):
    """Devices and code versions per dataset; more than one of either in a table is flagged."""
    out = {}
    for ds, vs in runs.items():
        envs = [m.get("env", {}) for r in vs.values() for m in r.values()]
        out[ds] = {"devices": sorted({e.get("device", "unknown") for e in envs}),
                   "git_shas": sorted({e.get("git_sha", "unknown") for e in envs})}
        for k in ("devices", "git_shas"):
            if len(out[ds][k]) > 1:
                print(f"[warn] {ds}: runs come from several {k}: {out[ds][k]}")
    return out


def hparam_report(runs, data_root):
    """Every trained run must use its dataset's tuned d and lr (equal tuning budget)."""
    bad = []
    for ds, vs in runs.items():
        p = os.path.join(data_root, ds, "hparams.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            hp = json.load(f)
        for v, r in vs.items():
            for s, m in r.items():
                c = m.get("config", {})
                if c.get("d") != hp["d"] or (v in hp["lr"] and c.get("lr") != hp["lr"][v]):
                    bad.append(f"{ds}:{v}:seed{s} d={c.get('d')} lr={c.get('lr')}")
    if bad:
        print(f"[warn] runs not using the tuned hyperparameters: {bad}")
    return bad


def seed_avg_ndcg(runs, ds, var, seeds, split):
    return np.mean([ndcg_per_user(ranks(runs[ds][var][s], split)) for s in seeds], 0)


def compare(runs, ds, a, b, split=None):
    """a (selected) vs b on the seeds both have."""
    split = split or SPLIT
    ra, rb = runs.get(ds, {}).get(a, {}), runs.get(ds, {}).get(b, {})
    seeds = sorted(s for s in set(ra) & set(rb) if split in ra[s] and split in rb[s])
    if not seeds:
        return None
    na = [ra[s][split]["NDCG@10"] for s in seeds]
    nb = [rb[s][split]["NDCG@10"] for s in seeds]
    per_seed_mc = [mcnemar_exact(ranks(ra[s], split) <= 10, ranks(rb[s], split) <= 10) for s in seeds]
    ua, ub = seed_avg_ndcg(runs, ds, a, seeds, split), seed_avg_ndcg(runs, ds, b, seeds, split)
    return {
        "seeds": seeds,
        "delta_ndcg": float(np.mean(na) - np.mean(nb)),
        "rel_ndcg": float((np.mean(na) - np.mean(nb)) / max(np.mean(nb), 1e-12)),
        "delta_hits10": float(np.mean([ra[s][split]["hits@10"] - rb[s][split]["hits@10"] for s in seeds])),
        "seed_t": paired_seed_test(na, nb),
        "wilcoxon_p": wilcoxon(ua, ub),
        "mcnemar_p_per_seed": [m["p"] for m in per_seed_mc],
        "wins": int(sum(x > y for x, y in zip(na, nb))),
    }


def table_main(runs, split, keys, label, caption, published=False):
    ncol = 1 + len(keys) * len(DATASETS)
    lines = [r"\begin{table*}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small",
             r"\begin{tabular}{l" + "c" * (ncol - 1) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(keys)}}}{{c}}{{{d}}}" for _, d in DATASETS) + r" \\",
             " ".join(f"\\cmidrule(lr){{{2 + len(keys) * i}-{1 + len(keys) * (i + 1)}}}" for i in range(len(DATASETS))),
             "Model & " + " & ".join([" & ".join(keys)] * len(DATASETS)) + r" \\", r"\midrule"]
    best = {}
    for ds, _ in DATASETS:
        for key in keys:
            vals = {v: metric_summary(runs, ds, v, key, split) for v in MAIN_ROWS}
            vals = {k: x["mean"] for k, x in vals.items() if x}
            best[(ds, key)] = max(vals, key=vals.get) if vals else None
    for i, v in enumerate(MAIN_ROWS):
        if v == PROPOSED and i > 0:
            lines.append(r"\midrule")
        cells = [fmt(metric_summary(runs, ds, v, key, split), bold=best[(ds, key)] == v)
                 for ds, _ in DATASETS for key in keys]
        lines.append(f"{NAMES[v]} & " + " & ".join(cells) + r" \\")
        if v == PROPOSED and i < len(MAIN_ROWS) - 1:
            lines.append(r"\midrule")
    if published and any(ds in PUBLISHED for ds, _ in DATASETS):
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Published, same data and protocol (history kept), "
                     f"not re-run by us ({PUBLISHED_SRC})}}}} \\\\")
        for model in ("SASRec", "DIF-SR", "DIFF"):
            cells = []
            for ds, _ in DATASETS:
                pub = PUBLISHED.get(ds, {}).get(model)
                cells += [f"\\textit{{{100 * pub[PUB_KEYS.index(k)]:.2f}}}" if pub else "--" for k in keys]
            lines.append(f"\\textit{{{model}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_significance(runs):
    comps, keys = [], []
    if PROPOSED:
        for ds, _ in DATASETS:
            for b in MAIN_ROWS:
                if b == PROPOSED:
                    continue
                c = compare(runs, ds, PROPOSED, b)
                if c:
                    comps.append(c)
                    keys.append((ds, b))
    adj = holm([c["wilcoxon_p"] for c in comps]) if comps else []
    who = NAMES[PROPOSED] if PROPOSED else "The selected model"
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{{who} (selected on validation) against each other model, on the seeds both have: mean "
             r"relative change of NDCG@10 over seeds, seeds won, two-sided Wilcoxon signed-rank test on per-user "
             r"NDCG@10 averaged over seeds (Holm-adjusted over all rows), and the largest per-seed exact McNemar "
             f"$p$ on HR@10. {DAG_NOTE}}}",
             r"\label{tab:sig}", r"\small", r"\begin{tabular}{llrcrr}", r"\toprule",
             r"Data & Other model & $\Delta$NDCG & Wins & $p_\mathrm{Holm}$ & $\max p_\mathrm{McN}$ \\", r"\midrule"]
    disp = dict(DATASETS)
    out = {}
    for (ds, b), c, p in zip(keys, comps, adj):
        c["wilcoxon_p_holm"] = p
        out[f"{ds}:{b}"] = c
        lines.append(f"{disp[ds]} & {NAMES[b]} & {100 * c['rel_ndcg']:+.1f}\\% & {c['wins']}/{len(c['seeds'])} & "
                     f"{p:.1e} & {max(c['mcnemar_p_per_seed']):.1e} \\\\")
    if not comps:
        lines.append(r"\multicolumn{6}{c}{-- results pending --} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), out


def table_rows(runs, rows, label, caption, key="NDCG@10"):
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\small",
             r"\begin{tabular}{l" + "c" * len(DATASETS) + "}", r"\toprule",
             "Variant & " + " & ".join(d for _, d in DATASETS) + r" \\", r"\midrule"]
    for v in rows:
        lines.append(f"{NAMES[v]} & " + " & ".join(fmt(metric_summary(runs, ds, v, key)) for ds, _ in DATASETS)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_masking(runs):
    rows = [v for v in dict.fromkeys(["sasrec", "poe", "hybrid", "bix", PROPOSED]) if v]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Effect of removing the history items in the input window from the ranking (NDCG@10, \%; "
             r"the masked column corresponds to BSARec/DLFSRec-style evaluation). Because repeated user--item pairs "
             f"are removed, the target is never in the history and masking can only help. {DAG_NOTE}}}",
             r"\label{tab:mask}", r"\small", r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for _, d in DATASETS) + r" \\",
             "Model & " + " & ".join(["kept & removed"] * len(DATASETS)) + r" \\", r"\midrule"]
    for v in rows:
        cells = []
        for ds, _ in DATASETS:
            cells += [fmt(metric_summary(runs, ds, v, "NDCG@10", "test")),
                      fmt(metric_summary(runs, ds, v, "NDCG@10", "test_masked"))]
        lines.append(f"{NAMES[v]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_length(runs, data_root):
    """NDCG@10 by test-history-length tercile (seed-averaged)."""
    cols = [v for v in dict.fromkeys(["sasrec", "poe", "hybrid", PROPOSED]) if v]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{NDCG@10 (\%) by history-length tercile of the test users (short / medium / long), "
             f"averaged over seeds. {DAG_NOTE}}}", r"\label{tab:length}", r"\small",
             r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             r"Data & Tercile (len.) & " + " & ".join(NAMES[v] for v in cols) + r" \\", r"\midrule"]
    out = {}
    for ds, disp in DATASETS:
        p = os.path.join(data_root, ds, "dataset.pt")
        if not os.path.exists(p) or ds not in runs:
            continue
        lens = np.array([len(s) - 1 for s in torch.load(p, weights_only=False)["user_seqs"]])
        cuts = np.quantile(lens, [1 / 3, 2 / 3])
        groups = [lens <= cuts[0], (lens > cuts[0]) & (lens <= cuts[1]), lens > cuts[1]]
        ranges = [f"$\\le${int(cuts[0])}", f"{int(cuts[0]) + 1}--{int(cuts[1])}", f"$>${int(cuts[1])}"]
        for gname, g, rng in zip(["short", "medium", "long"], groups, ranges):
            cells = []
            for v in cols:
                r = {s: m for s, m in runs[ds].get(v, {}).items() if SPLIT in m}
                if not r:
                    cells.append("--")
                    continue
                val = np.mean([ndcg_per_user(ranks(m))[g].mean() for m in r.values()])
                out[f"{ds}:{gname}:{v}"] = float(val)
                cells.append(f"{100 * val:.2f}")
            lines.append(f"{disp} & {gname} ({rng}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines), out


def table_overlap(runs):
    """Hit overlap of the separately trained ID and text experts (HR@10), seed-averaged.
    Any per-user selector between the two experts can gain at most n01 hits over SASRec and
    risks n10; it needs precision > 50% on the users it moves, against a base rate n01/(n01+n10)."""
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Hit overlap at HR@10 of the separately trained ID expert (C) and content expert (S), mean "
             r"over seeds. $n_{11}$: both hit; $n_{10}$: only C; $n_{01}$: only S. A per-user selector between C "
             r"and S gains at most $n_{01}$ hits over C and puts $n_{10}$ at risk; base rate "
             r"$=n_{01}/(n_{01}+n_{10})$. Last column: HR@10 hits of the selected model minus those of C (the "
             r"selector bound does not apply to jointly trained models).}",
             r"\label{tab:overlap}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Data & Users & $n_{11}$ & $n_{10}$ & $n_{01}$ & Base rate & Selected $-$ C \\", r"\midrule"]
    out = {}
    for ds, disp in DATASETS:
        rc = {s: m for s, m in runs.get(ds, {}).get("sasrec", {}).items() if SPLIT in m}
        rs = {s: m for s, m in runs.get(ds, {}).get("text", {}).items() if SPLIT in m}
        seeds = sorted(set(rc) & set(rs))
        if not seeds:
            continue
        cells = []
        for s in seeds:
            c, t = ranks(rc[s]) <= 10, ranks(rs[s]) <= 10
            cells.append(((c & t).sum(), (c & ~t).sum(), (~c & t).sum()))
        n11, n10, n01 = np.mean(cells, 0)
        rb = runs[ds].get(PROPOSED, {}) if PROPOSED else {}
        shared = [s for s in seeds if s in rb and SPLIT in rb[s]]
        gain = np.mean([rb[s][SPLIT]["hits@10"] - rc[s][SPLIT]["hits@10"] for s in shared]) if shared else None
        out[ds] = {"n11": float(n11), "n10": float(n10), "n01": float(n01), "selected_minus_c": gain,
                   "users": len(ranks(rc[seeds[0]]))}
        lines.append(f"{disp} & {out[ds]['users']} & {n11:.1f} & {n10:.1f} & {n01:.1f} & "
                     f"{100 * n01 / max(n01 + n10, 1e-9):.1f}\\% & " + ("--" if gain is None else f"{gain:+.1f}")
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), out


def table_datasets(data_root):
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Dataset statistics after 5-core filtering and "
             r"removal of repeated user--item pairs.}", r"\label{tab:data}", r"\small",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Data & Users & Items & Inter. & Avg.\ len. & Density \\", r"\midrule"]
    found = False
    for ds, disp in DATASETS:
        p = os.path.join(data_root, ds, "dataset.pt")
        if not os.path.exists(p):
            continue
        found = True
        st = torch.load(p, weights_only=False)["stats"]
        lines.append(f"{disp} & {st['users']:,} & {st['items']:,} & {st['interactions']:,} & "
                     f"{st['seq_len_mean']:.1f} & {100 * st['density']:.3f}\\% \\\\")
    if not found:
        lines.append(r"\multicolumn{6}{c}{-- generated after data preparation --} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def macros(runs, sig):
    """Headline numbers as LaTeX macros.
    \\res<Data><Model><Ndcg|Hr>  : test, history kept (primary; same meaning as in v2/v3)
    \\res<Data><Model><NdcgM|HrM>: test, history removed (secondary)
    \\gain<Data><Baseline>       : BiXRec (bix) vs baseline, history kept (same meaning as in v2/v3)
    \\gainSel<Data><Baseline>    : selected model vs baseline (rows of tab:sig); \\selName = its name"""
    tag = {"ml100k": "MlHundred", "ml1m": "MlOne", "beauty": "Beauty", "sports": "Sports", "toys": "Toys",
           "beauty_text": "BeautyText", "sports_text": "SportsText"}
    vtag = {"sasrec": "Sasrec", "poe": "Poe", "poe50": "PoeFifty", "hybrid": "Hybrid", "dual_late": "Late",
            "bix": "Bix", "text": "Text", "dual_late_aux": "LateAux", "bix_aux": "BixAux",
            "dual_late_ef_aux": "LateEfAux", "bix_ef_aux": "BixEfAux"}
    out = ["\\newcommand{\\selName}{%s}" % (NAMES[PROPOSED] if PROPOSED else "--")]
    for ds, t in tag.items():
        for v, vt in vtag.items():
            for split, sfx in (("test", ""), ("test_masked", "M")):
                for key, kt in (("NDCG@10", "Ndcg"), ("HR@10", "Hr")):
                    s = metric_summary(runs, ds, v, key, split)
                    val = "--" if s is None else "%.2f" % (100 * s["mean"])
                    out.append("\\newcommand{\\res%s%s%s%s}{%s}" % (t, vt, kt, sfx, val))
        for b, bt in (("poe", "Poe"), ("hybrid", "Hybrid"), ("sasrec", "Sasrec"), ("dual_late", "Late")):
            c = compare(runs, ds, "bix", b, "test")
            out.append("\\newcommand{\\gain%s%s}{%s}" % (t, bt, "--" if c is None else "%+.1f" % (100 * c["rel_ndcg"])))
            c = sig.get(f"{ds}:{b}")
            out.append("\\newcommand{\\gainSel%s%s}{%s}" % (t, bt, "--" if c is None else "%+.1f" % (100 * c["rel_ndcg"])))
    return "\n".join(out) + "\n"


def fit_width(tex: str) -> str:
    """Scale every tabular down to the line width when it is wider (one- and two-column tables)."""
    out = []
    for line in tex.split("\n"):
        if line.startswith(r"\begin{tabular}"):
            line = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{" + line
        if line.startswith(r"\end{tabular}"):
            line = line + "}"
        out.append(line)
    return "\n".join(out)


MAIN_CAPTION = (r"Full-catalogue leave-one-out results without negative sampling, %s, mean\,$\pm$\,sd over seeds, "
                r"in \%%; exact score ties are broken in a fixed random item order. Best of our runs per column in "
                r"bold; the model selected on validation NDCG@10 sits between rules. " + DAG_NOTE + r" Compare "
                r"those BiXRec variants with the joint late-fusion control trained the same way. " + SEED_NOTE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--exclude", nargs="*", default=[], help="datasets to leave out (e.g. still running)")
    a = ap.parse_args()
    runs = {k: v for k, v in load_runs(a.runs).items() if k not in a.exclude}
    global DATASETS, PROPOSED, N_SEEDS
    found = [d for d in ALL_DATASETS if d[0] in runs]
    if found:
        DATASETS = found
    N_SEEDS = max([len(r) for vs in runs.values() for v, r in vs.items() if v in MAIN_ROWS] or [3])
    PROPOSED, sel_info = choose_proposed(runs)
    if PROPOSED is None:
        print(f"[info] no model selected yet; missing main-table runs: {sel_info.get('missing')}")
    env = env_report(runs)
    hp_bad = hparam_report(runs, a.data_root)
    os.makedirs(a.out, exist_ok=True)
    sig_tex, sig = table_significance(runs)
    len_tex, length = table_length(runs, a.data_root)
    ov_tex, overlap = table_overlap(runs)
    tex = "% generated by bixrec.make_tables -- do not edit by hand\n"
    tex += macros(runs, sig)
    tex += "\\newcommand{\\tabMain}{\n" + table_main(
        runs, "test", ("HR@10", "NDCG@10"), "tab:main",
        MAIN_CAPTION % "history items kept in the ranking", published=True) + "}\n"
    tex += "\\newcommand{\\tabMainTwenty}{\n" + table_main(
        runs, "test", ("HR@20", "NDCG@20"), "tab:main20",
        MAIN_CAPTION % "cutoff 20, history items kept", published=True) + "}\n"
    tex += "\\newcommand{\\tabMainMasked}{\n" + table_main(
        runs, "test_masked", ("HR@10", "NDCG@10"), "tab:main_masked",
        MAIN_CAPTION % "history items in the input window removed from the ranking (BSARec/DLFSRec-style; "
        "not comparable with the published rows of Table~\\ref{tab:main})") + "}\n"
    tex += "\\newcommand{\\tabSig}{\n" + sig_tex + "}\n"
    tex += "\\newcommand{\\tabAbl}{\n" + table_rows(
        runs, ABL_ROWS, "tab:ablation",
        r"Ablation (NDCG@10, \%, mean\,$\pm$\,sd). Static PoE fuses two separately trained experts; "
        r"all other rows are trained end to end with the same loss and budget. " + SEED_NOTE) + "}\n"
    tex += "\\newcommand{\\tabAudit}{\n" + table_rows(
        runs, AUDIT_ROWS, "tab:audit",
        r"Training-protocol audit (NDCG@10, \%). ``Last position only'' and ``BCE + 1 negative'' "
        r"reproduce the training choices of our earlier pipeline. " + SEED_NOTE) + "}\n"
    tex += "\\newcommand{\\tabMask}{\n" + table_masking(runs) + "}\n"
    tex += "\\newcommand{\\tabLength}{\n" + len_tex + "}\n"
    tex += "\\newcommand{\\tabOverlap}{\n" + ov_tex + "}\n"
    tex += "\\newcommand{\\tabData}{\n" + table_datasets(a.data_root) + "}\n"
    tex = fit_width(tex)
    with open(os.path.join(a.out, "tables.tex"), "w") as f:
        f.write(tex)
    summary = {"selected": {"model": PROPOSED, "info": sel_info,
                            "rule": "best mean normalised validation NDCG@10 over datasets (history kept)"},
               "excluded_datasets": a.exclude, "not_using_tuned_hparams": hp_bad,
               "significance": sig, "length": length, "overlap": overlap, "env": env,
               "runs": {ds: {v: sorted(r) for v, r in vs.items()} for ds, vs in runs.items()}}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"selected model: {PROPOSED}; wrote {a.out}/tables.tex; runs found: " + json.dumps(summary["runs"]))


if __name__ == "__main__":
    main()
