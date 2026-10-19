"""Quick router ablation table from existing JSON results."""
import sys, io, os, json, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.makedirs('results', exist_ok=True)

ORACLE_HR10 = 0.0618

def og(hr):
    return f"{(ORACLE_HR10 - hr) / ORACLE_HR10 * 100:.1f}%"

with open('ablation_router_formulations.json', encoding='utf-8') as f:
    abl = json.load(f)

rows = [
    ("Static Fusion (50/50)",           0.0394, 0.0183,  50.0,   50.0,  0.0,  og(0.0394), ""),
    ("Length Gate (heuristic)",          0.0237, 0.0121, "var",  "var",  0.0,  og(0.0237), ""),
    ("2-Action Hard {C vs S}",    abl[4]['HR@10'], abl[4]['NDCG@10'], "?", "?", 0.0, og(abl[4]['HR@10']), ""),
    ("3-Action Hard (argmax)",     abl[3]['HR@10'], abl[3]['NDCG@10'], "?", "?", "?", og(abl[3]['HR@10']), ""),
    ("2-Action Soft {C,S}",        abl[2]['HR@10'], abl[2]['NDCG@10'], "var","var",0.0, og(abl[2]['HR@10']), ""),
    ("Bandit TS (offline replay*)",  "TBD",  "TBD",  "?", "?", "?",    "?",  "needs cache"),
    ("MMoE Router K=4",              "TBD",  "TBD",  "?", "?", "?",    "?",  "needs cache"),
    ("Attention Gate (top-50)",      "TBD",  "TBD",  "?", "?", "?",    "?",  "needs cache"),
    ("MLP Router v1 8D  [Proposed]", abl[0]['HR@10'], abl[0]['NDCG@10'], "~35","~45","~20", og(abl[0]['HR@10']), "BEST"),
    ("MLP Router v2 12D [Proposed]", "TBD",  "TBD",  "?", "?", "?",    "?",  "needs cache"),
    ("Oracle Router (upper bound)",  0.0618, 0.0304, 34.2,  33.6, 32.2,  "0.0%", ""),
]

W = 38
print("=" * 82)
print("  ROUTER MECHANISM ABLATION TABLE  --  MovieLens-1M, seed=42, full catalog")
print("=" * 82)
hdr = f"  {'Mechanism':<{W}} {'HR@10':>7} {'NDCG@10':>9} {'C%':>5} {'S%':>5} {'H%':>5} {'Gap':>7}"
print(hdr)
print("  " + "-"*W + " " + "-"*7 + " " + "-"*9 + " " + "-"*5 + " " + "-"*5 + " " + "-"*5 + " " + "-"*7)
for name, hr, ndcg, c, s, h, gap, note in rows:
    hr_s   = f"{hr:.4f}" if isinstance(hr, float) else str(hr)
    ndcg_s = f"{ndcg:.4f}" if isinstance(ndcg, float) else str(ndcg)
    flag   = f"  <- {note}" if note else ""
    print(f"  {name:<{W}} {hr_s:>7} {ndcg_s:>9} {str(c):>5} {str(s):>5} {str(h):>5} {gap:>7}{flag}")

print()
print("  * Bandit (TS offline): OFFLINE REPLAY SIMULATION -- not a real-time online bandit.")
print("    Reported HR@K is cumulative hit rate across shuffled test users.")
print("    See models/competing_routers.py BanditRouter docstring for full disclaimer.")
print()
print("  TBD entries require: python train_and_evaluate_evidence_routing.py")
print("  then: python run_router_mechanism_ablation.py  (will auto-load cache)")
print()

results_out = {}
for name, hr, ndcg, c, s, h, gap, note in rows:
    results_out[name] = {
        "HR@10": hr, "NDCG@10": ndcg,
        "routing": {"C%": c, "S%": s, "H%": h},
        "oracle_gap": gap,
        "note": note,
    }

with open('results/router_mechanism_ablation.json', 'w', encoding='utf-8') as f:
    json.dump(results_out, f, indent=2, default=str)
print("Saved -> results/router_mechanism_ablation.json")
