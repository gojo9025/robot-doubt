"""Ask a VLM whether each pick-and-place attempt worked; compare with ground truth."""
import base64, io, json, os, sys, urllib.request
import numpy as np
from PIL import Image
import pickplace as pp

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5vl:3b"
PROMPT = ("This image shows a robot arm, a small red cube, and a green circular target "
          "on a table. Is the red cube completely inside the green circle, with no part of it touching or hanging over the edge? "
          "Answer with exactly one word: YES or NO.")


def ask_vlm(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "images": [b64],
                       "stream": False, "keep_alive": "10m",
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.loads(r.read())["response"].strip()
    return text, text.upper().startswith("YES")


def main(n):
    os.makedirs("vlm_frames", exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    conditions = [("normal", 1.0, 0.0),
                  ("noisy perception", 1.0, 0.015),
                  ("slippery cube", 0.2, 0.0)]
    for name, friction, noise in conditions:
        env = pp.PickPlaceEnv(friction, render=True)
        for i in range(n):
            cube_xy = (rng.uniform(0.40, 0.60), rng.uniform(0.10, 0.30))
            target_xy = (0.5, -0.25)
            env.reset(cube_xy, target_xy)
            est = np.array(cube_xy) + rng.normal(0, noise, 2)
            env.attempt(est, target_xy)
            truth, dist = env.succeeded()
            img = env.frame()
            text, said_yes = ask_vlm(img)
            tag = f"{name.split()[0]}_{i}_truth-{'ok' if truth else 'fail'}_vlm-{'yes' if said_yes else 'no'}"
            Image.fromarray(img).save(f"vlm_frames/{tag}.png")
            rows.append(dict(condition=name, truth=truth, vlm_yes=said_yes, raw=text))
            print(f"{name:18s} #{i}  truth={'SUCCESS' if truth else 'FAIL   '}  "
                  f"VLM={'YES' if said_yes else 'NO '}  raw='{text[:30]}'", flush=True)

    json.dump(rows, open("results.json", "w"), indent=2)
    tp = sum(r["truth"] and r["vlm_yes"] for r in rows)
    tn = sum((not r["truth"]) and (not r["vlm_yes"]) for r in rows)
    fp = sum((not r["truth"]) and r["vlm_yes"] for r in rows)
    fn = sum(r["truth"] and (not r["vlm_yes"]) for r in rows)
    total = len(rows)
    print("\n=== VLM judge vs ground truth ===")
    print(f"Attempts: {total}   VLM accuracy: {100*(tp+tn)/total:.1f}%")
    print(f"Correctly said SUCCESS: {tp}   Correctly said FAIL: {tn}")
    print(f"Missed a failure (said SUCCESS but robot failed): {fp}")
    print(f"False alarm (said FAIL but robot succeeded):      {fn}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
