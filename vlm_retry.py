"""Self-doubting robot: after each attempt the VLM judges the result; on NO, the robot retries.
Paired comparison: success after the FIRST attempt (no self-correction) vs. final success
(with self-correction), measured on the same episodes against simulator ground truth."""
import json, os, sys
import numpy as np
from PIL import Image
import pickplace as pp
from vlm_judge import ask_vlm

TARGET = (0.5, -0.25)


def vlm_says_success(env):
    text, yes = ask_vlm(env.frame())
    return yes


def episode(env, rng, noise, max_tries, ep_id=0):
    cube_xy = (rng.uniform(0.40, 0.60), rng.uniform(0.10, 0.30))
    env.reset(cube_xy, TARGET)
    first_truth = None
    verdicts = []
    for k in range(1, max_tries + 1):
        est = env.cube_pos()[:2] + rng.normal(0, noise, 2)   # re-look at the cube
        env.attempt(est, TARGET)
        truth = env.succeeded()[0]
        if first_truth is None:
            first_truth = truth
        said = vlm_says_success(env)
        if said != truth:   # keep a picture of every disagreement so we can look at it
            os.makedirs("mismatch_frames", exist_ok=True)
            dist_cm = env.succeeded()[1] * 100
            Image.fromarray(env.frame()).save(
                f"mismatch_frames/ep{ep_id}_try{k}_truth-{'ok' if truth else 'fail'}_vlm-{'yes' if said else 'no'}_dist{dist_cm:.1f}cm.png")
        verdicts.append((said, truth))
        if said:
            break
    return dict(first_ok=first_truth, final_ok=env.succeeded()[0], tries=k, verdicts=verdicts)


def main(n, noise, max_tries, seed=123):
    rng = np.random.default_rng(seed)
    env = pp.PickPlaceEnv(1.0, render=True)
    rows = []
    for i in range(n):
        r = episode(env, rng, noise, max_tries, i)
        rows.append(r)
        print(f"episode {i:2d}: first try {'OK  ' if r['first_ok'] else 'FAIL'} -> "
              f"final {'OK  ' if r['final_ok'] else 'FAIL'}  (tries={r['tries']})", flush=True)
    json.dump(rows, open("retry_results.json", "w"), indent=2)
    base = 100 * np.mean([r["first_ok"] for r in rows])
    fin = 100 * np.mean([r["final_ok"] for r in rows])
    avg = np.mean([r["tries"] for r in rows])
    wrong = sum(s != t for r in rows for s, t in r["verdicts"])
    total = sum(len(r["verdicts"]) for r in rows)
    print(f"\n=== Self-correction (perception noise {noise*100:.1f} cm, up to {max_tries} tries) ===")
    print(f"Without self-correction: {base:.1f}%")
    print(f"With VLM self-correction: {fin:.1f}%   (avg {avg:.2f} tries)")
    print(f"VLM judge mistakes: {wrong} of {total} judgements")
    failed = sum(not t for r in rows for s, t in r["verdicts"])
    missed = sum((not t) and s for r in rows for s, t in r["verdicts"])
    print(f"Failed attempts the VLM called a success: {missed} of {failed}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 20, float(a[1]) if len(a) > 1 else 0.03, int(a[2]) if len(a) > 2 else 3, int(a[3]) if len(a) > 3 else 123)
