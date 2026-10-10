"""
Scripted Franka Panda pick-and-place in MuJoCo, with:
  - failure injection (slippery cube, noisy cube-position estimate)
  - ground-truth success check straight from the simulator
  - a camera frame after each attempt (for the VLM later)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from pathlib import Path

PANDA_DIR = Path(os.environ.get("MENAGERIE", Path.home() / "mujoco_menagerie")) / "franka_emika_panda"
CUBE_HALF = 0.02          # 4 cm cube
TCP_OFFSET = 0.103        # fingertip-pad centre, measured along the hand's z axis
TARGET_TOL = 0.03         # success = cube fully inside the 5 cm disc (5 cm radius - 2 cm cube half-width)


def build_model(cube_friction=1.0):
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "scene.xml"))
    world = spec.worldbody

    # site at the fingertips, used as the end-effector for IK
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0])

    # light "mat" so the cube and target are easy to see
    world.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.35, 0.45, 0.001], pos=[0.5, 0, 0.001],
                   rgba=[0.85, 0.85, 0.82, 1], contype=0, conaffinity=0)

    # green target disc (visual only, no collision)
    world.add_geom(name="target", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.05, 0.001, 0], pos=[0.5, -0.25, 0.003],
                   rgba=[0.1, 0.8, 0.2, 1], contype=0, conaffinity=0)

    # red cube. priority=1 makes the cube's friction win over the gripper pads'
    cube = world.add_body(name="cube", pos=[0.5, 0.2, CUBE_HALF])
    cube.add_freejoint()
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[CUBE_HALF] * 3, mass=0.05, rgba=[0.9, 0.1, 0.1, 1],
                  friction=[cube_friction, 0.005, 0.0001], priority=1,
                  condim=4)

    # fixed camera looking at the workspace from the front-right
    p = np.array([1.05, 0.0, 0.75]); t = np.array([0.5, 0.0, 0.0])
    zc = (p - t) / np.linalg.norm(p - t)
    xc = np.cross([0, 0, 1], zc); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    world.add_camera(name="front", pos=p.tolist(),
                     xyaxes=np.concatenate([xc, yc]).tolist())
    return spec.compile()


class PickPlaceEnv:
    def __init__(self, cube_friction=1.0, render=True, img_size=(480, 640)):
        self.model = build_model(cube_friction)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.tcp = m.site("tcp").id
        self.cube_body = m.body("cube").id
        self.cube_qadr = m.jnt_qposadr[m.body_jntadr[self.cube_body]]
        self.target_pos = m.geom("target").pos.copy()
        self.renderer = mujoco.Renderer(m, *img_size) if render else None
        self.reset()

    # ---------- basics ----------
    def reset(self, cube_xy=(0.5, 0.2), target_xy=None):
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        a = self.cube_qadr
        d.qpos[a:a + 3] = [cube_xy[0], cube_xy[1], CUBE_HALF]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        if target_xy is not None:
            self.target_pos[:2] = target_xy
            m.geom("target").pos[:2] = target_xy
        mujoco.mj_forward(m, d)
        for _ in range(200):          # let everything settle
            mujoco.mj_step(m, d)

    def cube_pos(self):
        return self.data.xpos[self.cube_body].copy()

    def frame(self):
        self.renderer.update_scene(self.data, camera="front")
        return self.renderer.render().copy()

    # ---------- inverse kinematics (damped least squares) ----------
    def _ik(self, pos, iters=300, lam=0.05):
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        target_R = np.diag([1.0, -1.0, -1.0])     # gripper pointing straight down
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            err_p = pos - d.site_xpos[self.tcp]
            R = d.site_xmat[self.tcp].reshape(3, 3)
            Re = target_R @ R.T                     # rotation error in world frame
            q = np.zeros(4); mujoco.mju_mat2Quat(q, Re.flatten())
            err_r = np.zeros(3); mujoco.mju_quat2Vel(err_r, q, 1.0)
            err = np.concatenate([err_p, err_r])
            if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(err_r) < 1e-2:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
            d.qpos[:7] += 0.5 * dq
            d.qpos[:7] = np.clip(d.qpos[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])
        return d.qpos[:7].copy()

    # ---------- motion ----------
    def _move_to(self, pos, grip, seconds=1.2):
        q_goal = self._ik(np.asarray(pos, float))
        q_start = self.data.ctrl[:7].copy()
        g_start = self.data.ctrl[7]
        n = int(seconds / self.model.opt.timestep)
        for i in range(n):
            s = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / n)   # smooth start/stop
            self.data.ctrl[:7] = q_start + s * (q_goal - q_start)
            self.data.ctrl[7] = g_start + s * (grip - g_start)
            mujoco.mj_step(self.model, self.data)

    def _hold(self, grip, seconds=0.5):
        self.data.ctrl[7] = grip
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    # ---------- one full attempt ----------
    def attempt(self, cube_xy_est, target_xy_est):
        """Run pick-and-place using the robot's *estimates* of where things are."""
        OPEN, CLOSED = 255, 0
        cx, cy = cube_xy_est
        tx, ty = target_xy_est
        grasp_z = CUBE_HALF
        self.data.ctrl[:7] = self.data.qpos[:7]
        self.data.ctrl[7] = OPEN
        self._move_to([cx, cy, 0.20], OPEN)            # above cube
        self._move_to([cx, cy, grasp_z], OPEN, 1.0)    # down to cube
        self._hold(CLOSED, 0.6)                        # grasp
        self._move_to([cx, cy, 0.20], CLOSED)          # lift
        self._move_to([tx, ty, 0.20], CLOSED, 1.5)     # carry
        self._move_to([tx, ty, grasp_z + 0.005], CLOSED, 1.0)  # lower
        self._hold(OPEN, 0.5)                          # release
        self._move_to([tx, ty, 0.25], OPEN)            # retreat
        self._hold(OPEN, 0.5)

    # ---------- ground truth ----------
    def succeeded(self):
        p = self.cube_pos()
        dist = np.linalg.norm(p[:2] - self.target_pos[:2])
        return bool(dist < TARGET_TOL and p[2] < 2 * CUBE_HALF + 0.01), float(dist)


def run_trials(n, cube_friction=1.0, noise_std=0.0, seed=0, save_dir=None):
    rng = np.random.default_rng(seed)
    env = PickPlaceEnv(cube_friction, render=save_dir is not None)
    wins = 0
    for i in range(n):
        cube_xy = (rng.uniform(0.40, 0.60), rng.uniform(0.10, 0.30))
        target_xy = (0.5, -0.25)
        env.reset(cube_xy, target_xy)
        est = np.array(cube_xy) + rng.normal(0, noise_std, 2)   # imperfect "perception"
        env.attempt(est, target_xy)
        ok, dist = env.succeeded()
        wins += ok
        if save_dir is not None and i < 3:
            from PIL import Image
            Image.fromarray(env.frame()).save(f"{save_dir}/trial{i}_{'ok' if ok else 'fail'}.png")
    return wins / n


if __name__ == "__main__":
    import sys
    os.makedirs("frames", exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Running {n} trials per condition (ground-truth success rate)...")
    for name, fr, noise in [("normal", 1.0, 0.0),
                            ("noisy perception (1.5 cm)", 1.0, 0.015),
                            ("slippery cube", 0.2, 0.0)]:
        rate = run_trials(n, fr, noise, save_dir="frames")
        print(f"  {name:28s} success = {rate*100:5.1f}%")
