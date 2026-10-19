"""Collect runs/ into LaTeX tables + macros for the paper.

    python -m bixrec.make_tables --runs runs --data_root data --out paper/generated

Writes <out>/tables.tex (\\input in the paper) and <out>/summary.json.
Missing runs appear as "--" so the paper always compiles.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import torch

from .stats import holm, mcnemar_exact, ndcg_per_user, paired_seed_test, seed_summary, wilcoxon

ALL_DATASETS = [("ml100k", "ML-100K"), ("ml1m", "ML-1M"), ("beauty", "Beauty"), ("sports", "Sports")]
DATASETS = [d for d in ALL_DATASETS if d[0] != "ml100k"]  # replaced in main() by the datasets found
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
}
MAIN_ROWS = ["sasrec", "text", "poe50", "poe", "hybrid", "dual_late", "bix"]
BASELINES = ["sasrec", "text", "poe50", "poe", "hybrid", "dual_late"]
ABL_ROWS = ["poe50", "dual_late", "dual_late_pen", "bix_nogate", "bix_gate", "bix"]
AUDIT_ROWS = ["sasrec_lastpos", "sasrec_bce", "sasrec", "bix_lastpos", "bix"]
PROPOSED = "bix"


def load_runs(root):
    runs = {}
    for f in glob(os.path.join(root, "*", "*", "seed*", "metrics.json")):
        seed_dir = os.path.dirname(f)
        var_dir = os.path.dirname(seed_dir)
        ds = os.path.basename(os.path.dirname(var_dir))
        var = os.path.basename(var_dir)
        seed = int(os.path.basename(seed_dir)[4:])
        with open(f) as fh:
            m = json.load(fh)
        m["_dir"] = seed_dir
        runs.setdefault(ds, {}).setdefault(var, {})[seed] = m
    return runs


def ranks(run, split="test"):
    return np.load(os.path.join(run["_dir"], f"ranks_{split}.npy"))


def fmt(s, scale=100, bold=False, digits=2):
    if s is None:
        return "--"
    txt = f"{s['mean'] * scale:.{digits}f}" + (f"\\,$\\pm$\\,{s['sd'] * scale:.{digits}f}" if s["n"] > 1 else "")
    if s["n"] < 3:  # incomplete: mark the number of seeds
        txt += f"$^{{n={s['n']}}}$"
    return f"\\textbf{{{txt}}}" if bold else txt


def metric_summary(runs, ds, var, key, split="test"):
    r = runs.get(ds, {}).get(var)
    if not r:
        return None
    return seed_summary([m[split][key] for _, m in sorted(r.items())])


def seed_avg_ndcg(runs, ds, var, seeds):
    return np.mean([ndcg_per_user(ranks(runs[ds][var][s])) for s in seeds], 0)


def compare(runs, ds, a, b):
    """a (proposed) vs b on shared seeds."""
    ra, rb = runs.get(ds, {}).get(a, {}), runs.get(ds, {}).get(b, {})
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        return None
    na = [ra[s]["test"]["NDCG@10"] for s in seeds]
    nb = [rb[s]["test"]["NDCG@10"] for s in seeds]
    per_seed_mc = [mcnemar_exact(ranks(ra[s]) <= 10, ranks(rb[s]) <= 10) for s in seeds]
    ua, ub = seed_avg_ndcg(runs, ds, a, seeds), seed_avg_ndcg(runs, ds, b, seeds)
    return {
        "seeds": seeds,
        "delta_ndcg": float(np.mean(na) - np.mean(nb)),
        "rel_ndcg": float((np.mean(na) - np.mean(nb)) / max(np.mean(nb), 1e-12)),
        "delta_hits10": float(np.mean([ra[s]["test"]["hits@10"] - rb[s]["test"]["hits@10"] for s in seeds])),
        "seed_t": paired_seed_test(na, nb),
        "wilcoxon_p": wilcoxon(ua, ub),
        "mcnemar_p_per_seed": [m["p"] for m in per_seed_mc],
        "wins": int(sum(x > y for x, y in zip(na, nb))),
    }


def table_main(runs):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Full-catalog leave-one-out results (no negative sampling), mean\,$\pm$\,sd over seeds, in \%. "
             r"Best per column in bold.}", r"\label{tab:main}", r"\small",
             r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for _, d in DATASETS) + r" \\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS))),
             "Model & " + " & ".join(["HR@10 & NDCG@10"] * len(DATASETS)) + r" \\", r"\midrule"]
    best = {}
    for ds, _ in DATASETS:
        for key in ("HR@10", "NDCG@10"):
            vals = {v: metric_summary(runs, ds, v, key) for v in MAIN_ROWS}
            vals = {k: x["mean"] for k, x in vals.items() if x}
            best[(ds, key)] = max(vals, key=vals.get) if vals else None
    for v in MAIN_ROWS:
        if v == PROPOSED:
            lines.append(r"\midrule")
        cells = []
        for ds, _ in DATASETS:
            for key in ("HR@10", "NDCG@10"):
                cells.append(fmt(metric_summary(runs, ds, v, key), bold=best[(ds, key)] == v))
        lines.append(f"{NAMES[v]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_significance(runs):
    comps, keys = [], []
    for ds, _ in DATASETS:
        for b in BASELINES:
            c = compare(runs, ds, PROPOSED, b)
            if c:
                comps.append(c)
                keys.append((ds, b))
    adj = holm([c["wilcoxon_p"] for c in comps]) if comps else []
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{BiXRec against each baseline: mean relative change of NDCG@10 over seeds, "
             r"seeds won, two-sided Wilcoxon signed-rank test on per-user NDCG@10 averaged over seeds "
             r"(Holm-adjusted over all rows), and the largest per-seed exact McNemar $p$ on HR@10.}",
             r"\label{tab:sig}", r"\small", r"\begin{tabular}{llrcrr}", r"\toprule",
             r"Data & Baseline & $\Delta$NDCG & Wins & $p_\mathrm{Holm}$ & $\max p_\mathrm{McN}$ \\", r"\midrule"]
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
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Effect of removing history items from the ranking (NDCG@10, \%). Because repeated "
             r"user--item pairs are removed, the target is never in the history and masking can only help.}",
             r"\label{tab:mask}", r"\small", r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d}}}" for _, d in DATASETS) + r" \\",
             "Model & " + " & ".join(["unmasked & masked"] * len(DATASETS)) + r" \\", r"\midrule"]
    for v in ["sasrec", "poe", "hybrid", "bix"]:
        cells = []
        for ds, _ in DATASETS:
            cells += [fmt(metric_summary(runs, ds, v, "NDCG@10")),
                      fmt(metric_summary(runs, ds, v, "NDCG@10", "test_masked"))]
        lines.append(f"{NAMES[v]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_length(runs, data_root):
    """NDCG@10 by test-history-length tercile (seed-averaged)."""
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{NDCG@10 (\%) by history-length tercile of the test users (short / medium / long), "
             r"averaged over seeds.}", r"\label{tab:length}", r"\small",
             r"\begin{tabular}{ll" + "c" * 4 + "}", r"\toprule",
             r"Data & Tercile (len.) & SASRec & Tuned PoE & Early fusion & BiXRec \\", r"\midrule"]
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
            for v in ["sasrec", "poe", "hybrid", "bix"]:
                r = runs[ds].get(v)
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
             r"\caption{Hit overlap at HR@10 of the separately trained ID expert (C) and content expert (S), "
             r"mean over seeds. $n_{11}$: both hit; $n_{10}$: only C; $n_{01}$: only S. A per-user selector "
             r"between C and S gains at most $n_{01}$ hits over C and puts $n_{10}$ at risk; "
             r"base rate $=n_{01}/(n_{01}+n_{10})$.}",
             r"\label{tab:overlap}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Data & Users & $n_{11}$ & $n_{10}$ & $n_{01}$ & Base rate & BiXRec $-$ C \\", r"\midrule"]
    out = {}
    for ds, disp in DATASETS:
        rc, rs = runs.get(ds, {}).get("sasrec", {}), runs.get(ds, {}).get("text", {})
        seeds = sorted(set(rc) & set(rs))
        if not seeds:
            continue
        cells = []
        for s in seeds:
            c, t = ranks(rc[s]) <= 10, ranks(rs[s]) <= 10
            cells.append(((c & t).sum(), (c & ~t).sum(), (~c & t).sum()))
        n11, n10, n01 = np.mean(cells, 0)
        rb = runs[ds].get("bix", {})
        shared = [s for s in seeds if s in rb]
        gain = np.mean([rb[s]["test"]["hits@10"] - rc[s]["test"]["hits@10"] for s in shared]) if shared else None
        out[ds] = {"n11": float(n11), "n10": float(n10), "n01": float(n01), "bix_minus_c": gain,
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
    """Headline numbers as LaTeX macros, e.g. \\resMlOneBixNdcg."""
    tag = {"ml100k": "MlHundred", "ml1m": "MlOne", "beauty": "Beauty", "sports": "Sports"}
    vtag = {"sasrec": "Sasrec", "poe": "Poe", "poe50": "PoeFifty", "hybrid": "Hybrid", "dual_late": "Late",
            "bix": "Bix", "text": "Text"}
    out = []
    for ds, t in tag.items():
        for v, vt in vtag.items():
            for key, kt in (("NDCG@10", "Ndcg"), ("HR@10", "Hr")):
                s = metric_summary(runs, ds, v, key)
                val = "--" if s is None else "%.2f" % (100 * s["mean"])
                out.append("\\newcommand{\\res%s%s%s}{%s}" % (t, vt, kt, val))
        for b, bt in (("poe", "Poe"), ("hybrid", "Hybrid"), ("sasrec", "Sasrec"), ("dual_late", "Late")):
            c = sig.get(f"{ds}:{b}")
            val = "--" if c is None else "%+.1f" % (100 * c["rel_ndcg"])
            out.append("\\newcommand{\\gain%s%s}{%s}" % (t, bt, val))
    return "\n".join(out) + "\n"


def fit_single_column(tex: str) -> str:
    """Scale tabulars of single-column tables down to the column width when they are wider."""
    out, inside = [], False
    for line in tex.split("\n"):
        if line.startswith(r"\begin{table}["):
            inside = True
        elif line.startswith(r"\begin{table*}"):
            inside = False
        if inside and line.startswith(r"\begin{tabular}"):
            line = r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{" + line
        if inside and line.startswith(r"\end{tabular}"):
            line = line + "}"
        if line.startswith(r"\end{table}"):
            inside = False
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--exclude", nargs="*", default=[], help="datasets to leave out (e.g. still running)")
    a = ap.parse_args()
    runs = {k: v for k, v in load_runs(a.runs).items() if k not in a.exclude}
    global DATASETS
    found = [d for d in ALL_DATASETS if d[0] in runs]
    if found:
        DATASETS = found
    os.makedirs(a.out, exist_ok=True)
    sig_tex, sig = table_significance(runs)
    len_tex, length = table_length(runs, a.data_root)
    ov_tex, overlap = table_overlap(runs)
    tex = "% generated by bixrec.make_tables -- do not edit by hand\n"
    tex += macros(runs, sig)
    tex += "\\newcommand{\\tabMain}{\n" + table_main(runs) + "}\n"
    tex += "\\newcommand{\\tabSig}{\n" + sig_tex + "}\n"
    tex += "\\newcommand{\\tabAbl}{\n" + table_rows(
        runs, ABL_ROWS, "tab:ablation",
        r"Ablation (NDCG@10, \%, mean\,$\pm$\,sd). Static PoE fuses two separately trained experts; all other "
        r"rows are trained end to end with the same loss and budget.") + "}\n"
    tex += "\\newcommand{\\tabAudit}{\n" + table_rows(
        runs, AUDIT_ROWS, "tab:audit",
        r"Training-protocol audit (NDCG@10, \%). ``Last position only'' and ``BCE + 1 negative'' reproduce "
        r"the training choices of our earlier pipeline.") + "}\n"
    tex += "\\newcommand{\\tabMask}{\n" + table_masking(runs) + "}\n"
    tex += "\\newcommand{\\tabLength}{\n" + len_tex + "}\n"
    tex += "\\newcommand{\\tabOverlap}{\n" + ov_tex + "}\n"
    tex += "\\newcommand{\\tabData}{\n" + table_datasets(a.data_root) + "}\n"
    tex = fit_single_column(tex)
    with open(os.path.join(a.out, "tables.tex"), "w") as f:
        f.write(tex)
    summary = {"significance": sig, "length": length, "overlap": overlap,
               "runs": {ds: {v: sorted(r) for v, r in vs.items()} for ds, vs in runs.items()}}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {a.out}/tables.tex; runs found: " + json.dumps(summary["runs"]))


if __name__ == "__main__":
    main()
