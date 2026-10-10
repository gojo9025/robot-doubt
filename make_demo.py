"""Record a demo video: the robot fails, the VLM notices, the robot retries.
  python make_demo.py            # picks a recovered 2-try episode from eval_vlm.json
  python make_demo.py 2 17       # or choose seed and episode index yourself
Output: demo.mp4 (needs Ollama running, plus ffmpeg)."""
import json, os, shutil, subprocess, sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import pickplace as pp

TARGET = (0.5, -0.25)
NOISE = 0.015
EVERY = 25          # one video frame every 25 physics steps (0.05 s) -> 20 fps, real time
FPS = 20
OUT = "demo_frames"
state = {"env": None, "n": 0, "frames": 0, "record": False, "title": "", "sub": ""}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def stamp(img, title, sub):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    d.rectangle([0, h - 74, w, h], fill=(0, 0, 0))
    d.text((12, h - 68), title, fill=(255, 255, 255), font=font(24))
    d.text((12, h - 36), sub, fill=(170, 215, 255), font=font(19))
    return np.asarray(im)


def save(img):
    Image.fromarray(img).save(f"{OUT}/{state['frames']:05d}.png")
    state["frames"] += 1


_real_step = mujoco.mj_step
def _step(m, d):
    _real_step(m, d)
    if state["record"]:
        state["n"] += 1
        if state["n"] % EVERY == 0:
            save(stamp(state["env"].frame(), state["title"], state["sub"]))
mujoco.mj_step = _step


def hold(env, title, sub, seconds):
    img = stamp(env.frame(), title, sub)
    for _ in range(int(seconds * FPS)):
        save(img)


def first_noise_draws(rng):
    cube_xy = (rng.uniform(0.40, 0.60), rng.uniform(0.10, 0.30))
    return cube_xy


def dry_run_ok(seed, i):
    """Physics-only check: attempt 1 fails and attempt 2 succeeds (ground truth)."""
    env = pp.PickPlaceEnv(1.0, render=False)
    rng = np.random.default_rng([seed, i])
    cube_xy = first_noise_draws(rng)
    env.reset(cube_xy, TARGET)
    outcomes = []
    for k in range(2):
        est = env.cube_pos()[:2] + rng.normal(0, NOISE, 2)
        env.attempt(est, TARGET)
        outcomes.append(env.succeeded()[0])
        if outcomes[-1]:
            break
    return outcomes == [False, True]


def pick_episode():
    rows = json.load(open("eval_vlm.json"))
    seeds = (1, 2, 3); per = len(rows) // len(seeds)
    for j, r in enumerate(rows):
        if r["tries"] == 2 and r["final_ok"] and not r["first_ok"]:
            seed, i = seeds[j // per], j % per
            if dry_run_ok(seed, i):
                return seed, i
    raise SystemExit("No suitable recovered episode found in eval_vlm.json")


def main(seed, i):
    from vlm_judge import ask_vlm
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    print(f"Recording seed {seed}, episode {i}")
    env = pp.PickPlaceEnv(1.0, render=True); state["env"] = env
    rng = np.random.default_rng([seed, i])
    cube_xy = first_noise_draws(rng)
    env.reset(cube_xy, TARGET)
    hold(env, "Self-doubting robot", "Task: put the red cube on the green circle", 1.5)
    for k in range(1, 4):
        est = env.cube_pos()[:2] + rng.normal(0, NOISE, 2)
        state["title"] = f"Attempt {k}: picking up the cube"
        state["sub"] = ("The robot's estimate of the cube position is slightly off" if k == 1
                        else "Looking at the cube again and trying again")
        state["record"] = True
        env.attempt(est, TARGET)
        state["record"] = False
        truth = env.succeeded()[0]
        text, said = ask_vlm(env.frame())
        gt = "Simulator ground truth: " + ("SUCCESS" if truth else "FAIL")
        if said:
            hold(env, "VLM judge: YES - the cube is on the target", gt, 3.0)
            break
        hold(env, "VLM judge: NO - that did not work, retrying", gt, 2.0)
    state["record"] = False
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", f"{OUT}/%05d.png", "-pix_fmt", "yuv420p", "demo.mp4"], check=True)
    print(f"Saved demo.mp4 ({state['frames']} frames, {state['frames']/FPS:.0f} s)")


if __name__ == "__main__":
    a = sys.argv[1:]
    seed, i = (int(a[0]), int(a[1])) if len(a) >= 2 else pick_episode()
    main(seed, i)
