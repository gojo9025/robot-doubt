"""Multi-seed evaluation with confidence intervals.
  python evaluate.py oracle          # perfect judge (upper bound, no Ollama needed, fast)
  python evaluate.py vlm             # real VLM judge via Ollama
Optional: python evaluate.py vlm 50 0.015 3 1,2,3   (episodes/seed, noise, max_tries, seeds)"""
import json, sys
import numpy as np
import pickplace as pp
import vlm_retry as vr


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def main(arm, n_per_seed=50, noise=0.015, max_tries=3, seeds=(1, 2, 3)):
    if arm == "oracle":
        vr.vlm_says_success = lambda env: env.succeeded()[0]
    env = pp.PickPlaceEnv(1.0, render=(arm == "vlm"))
    rows = []
    for seed in seeds:
        for i in range(n_per_seed):
            rng = np.random.default_rng([seed, i])     # same episode in every arm
            r = vr.episode(env, rng, noise, max_tries, ep_id=seed * 1000 + i)
            rows.append(r); print(".", end="", flush=True)
        sub = rows[-n_per_seed:]
        print(f"seed {seed}: first try {100*np.mean([r['first_ok'] for r in sub]):.0f}%  ->  "
              f"final {100*np.mean([r['final_ok'] for r in sub]):.0f}%", flush=True)
    n = len(rows)
    first = np.array([r["first_ok"] for r in rows], float)
    final = np.array([r["final_ok"] for r in rows], float)
    b = wilson(first.sum(), n); f = wilson(final.sum(), n)
    rec = int(((final == 1) & (first == 0)).sum()); brk = int(((final == 0) & (first == 1)).sum())
    boot = np.random.default_rng(0)
    lifts = [100 * (final[idx] - first[idx]).mean() for idx in (boot.integers(0, n, n) for _ in range(5000))]
    lo, hi = np.percentile(lifts, [2.5, 97.5])
    ver = [v for r in rows for v in r["verdicts"]]
    failed = sum(not t for s, t in ver); missed = sum((not t) and s for s, t in ver)
    succ = sum(t for s, t in ver); alarm = sum(t and (not s) for s, t in ver)
    print(f"\n=== {arm.upper()} judge | noise {noise*100:.1f} cm | up to {max_tries} tries | {n} episodes ({len(seeds)} seeds) ===")
    print(f"Without self-correction: {b[0]:.1f}%   (95% CI {b[1]:.1f}-{b[2]:.1f})")
    print(f"With self-correction:    {f[0]:.1f}%   (95% CI {f[1]:.1f}-{f[2]:.1f})")
    print(f"Lift: {100*(final-first).mean():+.1f} points   (95% CI {lo:+.1f} to {hi:+.1f})   recovered {rec}, broken {brk}")
    print(f"Average tries: {np.mean([r['tries'] for r in rows]):.2f}")
    print(f"Judge: missed {missed} of {failed} failed attempts; false alarm on {alarm} of {succ} successes")
    json.dump(rows, open(f"eval_{arm}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    arm = a[0] if a else "oracle"
    main(arm,
         int(a[1]) if len(a) > 1 else 50,
         float(a[2]) if len(a) > 2 else 0.015,
         int(a[3]) if len(a) > 3 else 3,
         tuple(int(x) for x in a[4].split(",")) if len(a) > 4 else (1, 2, 3))
