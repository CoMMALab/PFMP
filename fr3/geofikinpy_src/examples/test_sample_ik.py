#!/usr/bin/env python3
import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
import geofikinpy as g

# ---------- helpers ----------
POS_ATOL = 1e-5
ROT_ATOL = 1e-5
Q_ATOL   = 1e-4
SWIVEL_ATOL = 2e-2  # swivel closeness filter (numerical IK)

def rot_angle_err(Ra, Rb):
    M = Ra.T @ Rb
    tr = np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(tr))

def unwrap_close(q_ref, q, tol=Q_ATOL):
    q_ref = np.asarray(q_ref); q = np.asarray(q)
    dq = np.unwrap(np.vstack([q_ref, q]), axis=0)[1] - q_ref
    return np.linalg.norm(dq) < tol

def pose_from_fk(q):
    T = g.franka_fk(np.asarray(q, float))
    return T[:3, 3], T[:3, :3]

def valid_solutions(S):
    return [np.asarray(s, float) for s in S if np.all(np.isfinite(s))]

def dedupe_solutions(S, tol=Q_ATOL):
    uniq = []
    for s in S:
        if not any(unwrap_close(u, s, tol) for u in uniq):
            uniq.append(s)
    return uniq

def add_sphere(scene, pos, radius=0.01, rgba=(0,1,0,0.9)):
    if scene.ngeom >= scene.maxgeom: return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[radius,0,0], pos=np.asarray(pos),
                        mat=np.eye(3).flatten(), rgba=np.asarray(rgba))
    scene.ngeom += 1

def add_line(scene, p0, p1, radius=0.003, rgba=(0,0,1,0.6)):
    if scene.ngeom >= scene.maxgeom: return
    p0 = np.asarray(p0); p1 = np.asarray(p1)
    v = p1 - p0; L = np.linalg.norm(v)
    if L < 1e-9: return
    mid = 0.5*(p0+p1)
    z = v / L
    a = np.array([1.,0.,0.]) if abs(z[0]) < 0.9 else np.array([0.,1.,0.])
    x = a - np.dot(a,z)*z; x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(z,x)
    R = np.stack([x,y,z], axis=1)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[radius, 0.5*L, 0], pos=mid, mat=R.flatten(), rgba=np.asarray(rgba))
    scene.ngeom += 1

def find_arm_joint_ids(m):
    named = [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
             and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("panda_joint")]
    if len(named) >= 7: return named[:7]
    return [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE][:7]

def arm_qpos_indices_from_joint_ids(m, jids):
    return np.array([m.jnt_qposadr[j] for j in jids], dtype=int)

def _get_joint_range(m, j):
    """Return (lo, hi) for joint j, robust to Mujoco array layout."""
    rng = m.jnt_range
    if rng.ndim == 2:   # (njnt, 2)
        lo_raw, hi_raw = rng[j, 0], rng[j, 1]
    else:               # flat length 2*njnt
        lo_raw, hi_raw = rng[2*j], rng[2*j+1]
    return float(lo_raw), float(hi_raw)

def arm_joint_ranges(m, jids):
    # Typical Panda limits as fallback (radians)
    fallback = np.array([
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
    ], dtype=float)

    lohi = []
    for i, j in enumerate(jids):
        use_fallback = True
        # Only use model limits if the joint is marked limited and the numbers are sane
        limited = bool(int(m.jnt_limited[j]))
        if limited:
            lo, hi = _get_joint_range(m, j)
            if np.isfinite(lo) and np.isfinite(hi) and (lo < hi):
                lohi.append([lo, hi])
                use_fallback = False
        if use_fallback:
            lohi.append(fallback[i].tolist())
    return np.array(lohi, dtype=float)


# ---------- IK families ----------
FAMILY_ORDER = ("q7","q4","q6","swivel")

def compute_family_sets_and_profile(q_star):
    """Returns (fam_dict, timing_dict). fam[name]=(p_ref,R_ref,param,solutions)."""
    p_ref, R_ref = pose_from_fk(q_star)
    out, prof = {}, {}

    t0 = time.perf_counter()
    q7 = float(q_star[6])
    S = dedupe_solutions(valid_solutions(g.ik_q7(p_ref, R_ref, q7)))
    prof["q7"] = {"solve_ms": (time.perf_counter()-t0)*1000.0, "n_solutions": len(S)}
    if S: out["q7"] = (p_ref, R_ref, q7, S)

    t0 = time.perf_counter()
    q4 = float(q_star[3])
    S = dedupe_solutions(valid_solutions(g.ik_q4(p_ref, R_ref, q4)))
    prof["q4"] = {"solve_ms": (time.perf_counter()-t0)*1000.0, "n_solutions": len(S)}
    if S: out["q4"] = (p_ref, R_ref, q4, S)

    t0 = time.perf_counter()
    q6 = float(q_star[5])
    S = dedupe_solutions(valid_solutions(g.ik_q6(p_ref, R_ref, q6)))
    prof["q6"] = {"solve_ms": (time.perf_counter()-t0)*1000.0, "n_solutions": len(S)}
    if S: out["q6"] = (p_ref, R_ref, q6, S)

    t0 = time.perf_counter()
    theta = g.swivel_from_q(q_star)
    Ssw = []
    if np.isfinite(theta):
        S = dedupe_solutions(valid_solutions(g.ik_swivel(p_ref, R_ref, float(theta))))
        # keep swivel-close ones (numerical solver)
        for s in S:
            ths = g.swivel_from_q(s)
            if np.isfinite(ths):
                d = float((ths - theta + np.pi) % (2*np.pi) - np.pi)
                if abs(d) <= SWIVEL_ATOL:
                    Ssw.append(s)
    prof["swivel"] = {"solve_ms": (time.perf_counter()-t0)*1000.0, "n_solutions": len(Ssw)}
    if Ssw: out["swivel"] = (p_ref, R_ref, float(theta), Ssw)

    return out, prof

# ---------- seed stream (unique per session) ----------
class SeedStream:
    """
    Deterministic per-run stream: each call to next() increments an index
    and returns a new random q* within joint limits (with safety margins).
    """
    def __init__(self, m, arm_jids):
        self.m = m
        self.arm_jids = arm_jids
        self.q_ranges = arm_joint_ranges(m, arm_jids)
        self.idx = 0
        self.margin = 0.05  # rad

    def next(self):
        rng = np.random.default_rng(self.idx)
        self.idx += 1
        lo = self.q_ranges[:,0] + self.margin
        hi = self.q_ranges[:,1] - self.margin
        q = rng.uniform(lo, hi)
        # keep elbow bent reasonably
        lo4, hi4 = self.q_ranges[3]
        lo4, hi4 = max(lo4, -3.05), min(hi4, -0.10)
        q[3] = rng.uniform(lo4, hi4)
        # keep wrist pitch off extremes a bit
        q[5] = np.clip(q[5], lo[5]+0.2, hi[5]-0.2)
        return q

def print_help():
    print(
        "\nControls:\n"
        "  N / → : next solution     |  B / ← : previous solution\n"
        "  F / ↑ : next family       |  G / ↓ : previous family\n"
        "  Q     : new seed (never repeats this run)\n"
        "  A     : autoplay on/off   |  [ / ] : slower/faster autoplay\n"
        "  T     : print timing profile for current seed\n"
        "  H     : help\n"
    )

import os
def load_mj(name: str):
    os.chdir("tests/mjx_mod/assets")
    with open(f"{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    return (mj_model, mj_data)

def main():
    # Swap this for your loader. You said it's defined elsewhere.
    m, d = load_mj("mjx_single_cube")

    ee_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    if ee_sid < 0:
        raise RuntimeError("site 'gripper' not found in model")
    arm_jids = find_arm_joint_ids(m)
    arm_idx = arm_qpos_indices_from_joint_ids(m, arm_jids)

    stream = SeedStream(m, arm_jids)

    # current seed & data
    cur = {"q_star": None, "fam": {}, "prof": {}}

    def new_seed():
        # keep drawing until we get at least one family
        for _ in range(64):
            q_star = stream.next()
            fam, prof = compute_family_sets_and_profile(q_star)
            if fam:
                cur["q_star"] = q_star
                cur["fam"] = fam
                cur["prof"] = prof
                return True
        return False

    ok = new_seed()
    if not ok:
        raise RuntimeError("Failed to generate a usable seed after many tries")

    fam_names = [n for n in FAMILY_ORDER if n in cur["fam"]]
    fam_i, sol_i = 0, 0
    autoplay, dwell = True, 0.7

    def clamp_indices():
        nonlocal fam_names, fam_i, sol_i
        fam_names = [n for n in FAMILY_ORDER if n in cur["fam"]]
        fam_i = max(0, min(fam_i, len(fam_names)-1))
        S = cur["fam"][fam_names[fam_i]][3]
        sol_i = max(0, min(sol_i, len(S)-1))

    def next_sol():
        nonlocal sol_i
        S = cur["fam"][fam_names[fam_i]][3]
        sol_i = (sol_i + 1) % len(S)

    def prev_sol():
        nonlocal sol_i
        S = cur["fam"][fam_names[fam_i]][3]
        sol_i = (sol_i - 1) % len(S)

    def next_fam():
        nonlocal fam_i, sol_i
        fam_i = (fam_i + 1) % len(fam_names); sol_i = 0

    def prev_fam():
        nonlocal fam_i, sol_i
        fam_i = (fam_i - 1) % len(fam_names); sol_i = 0

    def print_profile():
        print(f"\n[Profile] q*={np.array2string(cur['q_star'], precision=3)}")
        for name in FAMILY_ORDER:
            dct = cur["prof"].get(name, {"solve_ms":0.0, "n_solutions":0})
            mark = " " if name in cur["fam"] else "*"
            print(f"  {name:7s}{mark} solve: {dct['solve_ms']:.3f} ms  sols: {dct['n_solutions']}")

    def key_callback(keycode: int):
        nonlocal autoplay, dwell, fam_i, sol_i
        ch = chr(keycode).lower() if 32 <= keycode <= 126 else None
        # GLFW arrow codes (what MuJoCo uses); constants inline if glfw not imported
        RIGHT, LEFT, UP, DOWN = 262, 263, 265, 264

        if ch == 'n' or keycode == RIGHT:
            autoplay = False; next_sol()
        elif ch == 'b' or keycode == LEFT:
            autoplay = False; prev_sol()
        elif ch == 'f' or keycode == UP:
            autoplay = False; next_fam()
        elif ch == 'g' or keycode == DOWN:
            autoplay = False; prev_fam()
        elif ch == 'q':  # NEW SEED (never repeats this run)
            autoplay = False
            if new_seed():
                fam_i, sol_i = 0, 0
                print_profile()
            else:
                print("\n[warn] couldn't find a usable seed; try again.")
        elif ch == 'a':
            autoplay = not autoplay
        elif ch == '[':
            dwell = max(0.05, dwell * 0.8); print(f"DWELL={dwell:.2f}s")
        elif ch == ']':
            dwell = min(3.0, dwell * 1.25); print(f"DWELL={dwell:.2f}s")
        elif ch == 't':
            print_profile()
        elif ch == 'h':
            print_help()

    print_help()
    last = time.perf_counter()

    with launch_passive(m, d, show_left_ui=False, show_right_ui=False, key_callback=key_callback) as viewer:
        while viewer.is_running():
            # ensure current seed has at least one family
            if not cur["fam"]:
                if not new_seed():
                    time.sleep(0.1); continue

            clamp_indices()
            name = fam_names[fam_i]
            p_ref, R_ref, param, S = cur["fam"][name]
            q = S[sol_i]

            d.qpos[arm_idx] = q
            d.qvel[:] = 0.0; d.qacc[:] = 0.0
            mujoco.mj_forward(m, d)

            viewer.user_scn.ngeom = 0
            p_cur = d.site_xpos[ee_sid].copy()
            add_sphere(viewer.user_scn, p_ref, radius=0.012, rgba=(0,1,0,0.85))   # target (green)
            add_sphere(viewer.user_scn, p_cur, radius=0.010, rgba=(0,0.4,1,0.85)) # current (blue)
            add_line(viewer.user_scn, p_cur, p_ref, radius=0.003, rgba=(0,0,1,0.5))

            perr = np.linalg.norm(p_cur - p_ref)
            Rerr = rot_angle_err(R_ref, d.site_xmat[ee_sid].reshape(3,3))
            print(f"\rSeedIdx~{(stream.idx-1):4d}  Family {name:<6}  " f"sol {sol_i+1}/{len(S)}  param={param:+.3f}  "
                  f"|dp|={perr:.2e}  drot={Rerr:.2e} rad   ",
                  end="", flush=True)

            viewer.sync()

            if autoplay:
                now = time.perf_counter()
                if now - last >= dwell:
                    next_sol(); last = now
            else:
                time.sleep(0.02)

        print("\nDone.")

if __name__ == "__main__":
    main()
