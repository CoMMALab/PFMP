#!/usr/bin/env python3
import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from geofikinpy import ik_q4, ik_q7, ik_q6, ik_swivel

# ===================== USER CONFIG =====================
EE_SITE = "gripper"

# Grid region (meters) at a fixed Z ~ table height
X_RANGE = (0.40, 0.80)     # left/right of base
Y_RANGE = (-0.2, 0.2)       # keep a safe buffer from 0!
Z_FIXED = 0.10             # absolute world Z for the targets (tweak for your model)

# Interpolation resolution
NX = 50                    # grid columns (x)
NY = 50                    # grid rows (y); total = NX*NY
SNAKE_SCAN = True          # zig-zag each row to preserve continuity

# Redundancy / continuity knobs
USE_Q7 = True              # call ik_q7 (we bound that earlier)
CONTINUITY_W = np.array([1,1,1,0.9,0.9,0.8,0.8], float)  # weight proximal joints
JUMP_WARN_THRESH = 0.5     # rad jump considered a branch switch
JUMP_FAIL_THRESH = 1.2     # rad jump considered a continuity failure

# Joint limits (tweak to your MJCF/URDF if needed)
PANDA_LIMITS = np.array([
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
], dtype=float)

# Preview options
PREVIEW = True           # show solved grid in MuJoCo (static FK only; no dynamics)
DWELL_SEC = 1e-6         # time per grid point during preview
# ======================================================

def finite_rows(qs):
    qs = np.asarray(qs, float).reshape(-1, 7)
    if qs.size == 0: return qs
    return qs[np.all(np.isfinite(qs), axis=1)]

def choose_closest(q_solutions, q_seed, limits=None, weights=None):
    """Pick solution closest to q_seed; drop any violating limits."""
    if q_solutions is None: return None
    qs = np.asarray(q_solutions, float).reshape(-1, 7)
    if qs.size == 0: return None
    if limits is not None:
        lo, hi = limits[:,0], limits[:,1]
        ok = np.all((qs >= lo) & (qs <= hi), axis=1)
        qs = qs[ok]
        if qs.size == 0: return None
    if weights is None: weights = np.ones(7)
    def wrapclose(q):
        # unwrap q around q_seed to avoid 2π jumps
        dq = np.unwrap(np.vstack([q_seed, q]), axis=0)[1] - q_seed
        return q_seed + dq
    costs = [np.linalg.norm(weights*(wrapclose(q) - q_seed)) for q in qs]
    return qs[int(np.argmin(costs))]

def add_cylinder_line_safe(scene, p0, p1, radius=0.004, color=(1,0,0,0.7)):
    if scene.ngeom >= scene.maxgeom: return
    p0 = np.asarray(p0); p1 = np.asarray(p1)
    v = p1 - p0; L = np.linalg.norm(v)
    if L < 1e-9: return
    mid = 0.5*(p0+p1)
    z = v / L
    a = np.array([1.0,0.0,0.0]) if abs(z[0]) < 0.9 else np.array([0.0,1.0,0.0])
    x = a - np.dot(a, z)*z; x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[radius, 0.5*L, 0.0], pos=mid, mat=R.flatten(), rgba=color)
    scene.ngeom += 1

def add_sphere_safe(scene, pos, radius=0.01, rgba=(0,1,0,0.9)):
    if scene.ngeom >= scene.maxgeom: return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[radius,0,0], pos=pos, mat=np.eye(3).flatten(), rgba=rgba)
    scene.ngeom += 1

def build_grid(x_range, y_range, nx, ny, z):
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    pts = []
    for j in range(ny):
        row = [(xs[i], ys[j], z) for i in range(nx)]
        if SNAKE_SCAN and (j % 2 == 1):
            row = row[::-1]
        pts.extend(row)
    return np.array(pts, float).reshape(-1, 3)

def run_grid_test(mj_model, mj_data, ee_sid, R_fixed, points):
    """Solve IK across grid sequentially; track continuity."""
    # indices into qpos for the 7 arm joints
    arm_dof = mj_model.actuator_trnid[:,0].astype(int)
    dof_jids = mj_model.dof_jntid[arm_dof]
    qpos_adr = mj_model.jnt_qposadr[dof_jids]

    mujoco.mj_forward(mj_model, mj_data)
    q_now = mj_data.qpos[qpos_adr].copy()
    q_prev = q_now.copy()         # for continuity tracking
    q7_pref = float(q_now[-1])    # redundancy bias
    weights = CONTINUITY_W

    solved_qpos = []
    success = []
    jumps = []

    print(f"[grid] {len(points)} targets @ Z={points[0,2]:.3f} m "
          f"({NX}x{NY}, snake={SNAKE_SCAN})")
    for k, p_des in enumerate(points):
        # analytic IK (GeoFIK)
        sols = ik_q7(p_des, R_fixed, q7_pref) if USE_Q7 else []
        # sols = ik_swivel(p_des, R_fixed, q7_pref) if USE_Q7 else []
        sols = finite_rows(sols)
        q_best = choose_closest(sols, q_prev, limits=PANDA_LIMITS, weights=weights)

        if q_best is None:
            solved_qpos.append(None); success.append(False); jumps.append(np.nan)
            print(f"  [{k:03d}] target={np.round(p_des,3)} -> NO SOLUTION")
            continue

        # continuity metric (wrapped distance)
        dq = np.unwrap(np.vstack([q_prev, q_best]), axis=0)[1] - q_prev
        djump = float(np.linalg.norm(weights*dq))
        jumps.append(djump)
        solved = mj_data.qpos.copy()
        solved[qpos_adr] = q_best
        solved_qpos.append(solved)
        success.append(True)

        # update "previous" and redundancy bias
        q_prev = q_best
        q7_pref = float(q_best[-1])

        if djump > JUMP_FAIL_THRESH:
            flag = "FAIL CONTINUITY"
        elif djump > JUMP_WARN_THRESH:
            flag = "warn"
        else:
            flag = "ok"
        print(f"  [{k:03d}] target={np.round(p_des,3)} -> jump={djump:.3f} ({flag})")

    succ = np.count_nonzero(success)
    fail = len(success) - succ
    jumps_arr = np.array([j for j in jumps if np.isfinite(j)])
    if jumps_arr.size:
        print(f"\n[summary] solved {succ}/{len(points)} "
              f"(mean jump={jumps_arr.mean():.3f}, max={jumps_arr.max():.3f})")
        print(f"  warnings (> {JUMP_WARN_THRESH}): "
              f"{np.count_nonzero(jumps_arr > JUMP_WARN_THRESH)}")
        print(f"  failures (> {JUMP_FAIL_THRESH}): "
              f"{np.count_nonzero(jumps_arr > JUMP_FAIL_THRESH)}")
    else:
        print(f"\n[summary] solved {succ}/{len(points)} (no jumps computed)")

    return solved_qpos, success, np.array(jumps)

def preview_solutions(mj_model, mj_data, ee_sid, points, qpos_list):
    """Static FK preview (no mj_step): load each solved qpos + draw target."""
    with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        for p_des, qp in zip(points, qpos_list):
            viewer.user_scn.ngeom = 0
            if qp is not None:
                mj_data.qpos[:] = qp
                mj_data.qvel[:] = 0.0
                mj_data.qacc[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)
                p_cur = mj_data.site_xpos[ee_sid].copy()
                add_cylinder_line_safe(viewer.user_scn, p_cur, p_des, radius=0.004, color=(1,0,0,0.7))
                add_sphere_safe(viewer.user_scn, p_des, radius=0.01, rgba=(0,1,0,0.9))
            else:
                # show target in red if no solution
                add_sphere_safe(viewer.user_scn, p_des, radius=0.012, rgba=(1,0,0,0.9))

            t0 = time.time()
            while time.time() - t0 < DWELL_SEC and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

import os
def load_mj(name: str):
    os.chdir("tests/mjx_mod/assets")
    with open(f"{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    return (mj_model, mj_data)

if __name__ == "__main__":
    # You said you’ve defined this elsewhere:
    (mj_model, mj_data) = load_mj("mjx_single_cube")

    ee_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if ee_sid < 0:
        raise RuntimeError(f"MuJoCo site '{EE_SITE}' not found.")

    # Use the *current* TCP orientation as the fixed R for the grid
    mujoco.mj_forward(mj_model, mj_data)
    R_fixed = mj_data.site_xmat[ee_sid].reshape(3,3).copy()
    # print(R_fixed)
    # exit()
    R_fixed = np.array([[0., 1., 0.],
          [1., 0., 0.],
          [0., 0.,-1.]])

    Rz90 = np.array([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]])

    R_fixed = R_fixed @ Rz90
    print(R_fixed)
    exit()

    # R_fixed = np.array([
    #     [1.,  0.,  0.],
    #     [0., -1.,  0.],
    #     [0.,  0., -1.]
    # ])

    # Build grid at fixed Z near table height
    grid_pts = build_grid(X_RANGE, Y_RANGE, NX, NY, Z_FIXED)

    # Solve across the grid with continuity tracking
    solved_qpos, success, jumps = run_grid_test(mj_model, mj_data, ee_sid, R_fixed, grid_pts)

    # Optional preview (safe static FK; no physics stepping)
    if PREVIEW:
        preview_solutions(mj_model, mj_data, ee_sid, grid_pts, solved_qpos)
