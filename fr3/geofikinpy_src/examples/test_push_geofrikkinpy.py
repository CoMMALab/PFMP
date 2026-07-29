#!/usr/bin/env python3
import time
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
import geofrikkinpy as gf

# ======= config =======
EE_SITE = "gripper"
NUM_TARGETS = 20          # how many random targets to sample
HOLD_PER_TARGET = 0.8     # seconds each target is displayed (per solution)
FIX_ORIENTATION = True    # True: keep current TCP orientation; False: random SO(3)
USE_Q7 = True             # we bound ik_q7
FREE_VAL = None           # if None, bias to current wrist joint each query
SEED = 42

# joint limits (tweak to your model)
PANDA_LIMITS = np.array([
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
], dtype=float)

# workspace sampling box (relative to robot base/world)
# Adjust to your scene! (meters)
X_RANGE = (0.25, 0.70)
Y_RANGE = (-0.40, 0.40)
Z_RANGE = (0.10, 0.70)

# ======= utils =======
rng = np.random.default_rng(SEED)

def finite_rows(qs):
    qs = np.asarray(qs, float).reshape(-1, 7)
    if qs.size == 0: return qs
    return qs[np.all(np.isfinite(qs), axis=1)]

def random_rotation():
    # Uniform random rotation via axis-angle
    axis = rng.normal(size=3); axis /= (np.linalg.norm(axis)+1e-12)
    angle = rng.uniform(0, np.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    return R

def add_cylinder_line_safe(scene, p0, p1, radius=0.004, color=(1,0,0,0.7)):
    if scene.ngeom >= scene.maxgeom: return
    p0 = np.asarray(p0); p1 = np.asarray(p1)
    mid = 0.5*(p0+p1)
    v = p1 - p0; L = np.linalg.norm(v) + 1e-12
    if L < 1e-9: return
    z = v / L
    a = np.array([1.0,0.0,0.0]) if abs(z[0]) < 0.9 else np.array([0.0,1.0,0.0])
    x = a - np.dot(a, z)*z; x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(z, x)
    R = np.stack([x,y,z], axis=1)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[radius, 0.5*L, 0.0], pos=mid, mat=R.flatten(), rgba=color)
    scene.ngeom += 1

def add_sphere_safe(scene, pos, radius=0.02, rgba=(0,1,0,0.85)):
    if scene.ngeom >= scene.maxgeom: return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[radius,0,0], pos=pos, mat=np.eye(3).flatten(), rgba=rgba)
    scene.ngeom += 1

def pose_error(model, data, site_id, p_des, R_des):
    p = data.site_xpos[site_id]
    R = data.site_xmat[site_id].reshape(3,3)
    e_pos = np.linalg.norm(p_des - p)
    Rt = R_des @ R.T
    tr = np.clip(np.trace(Rt), -1.0, 3.0)
    th = np.arccos(np.clip(0.5*(tr - 1.0), -1.0, 1.0))
    return e_pos, th

def compute_ik_solutions(mj_model, mj_data, ee_sid, p_des, R_des, free_val=None):
    arm_dof = mj_model.actuator_trnid[:,0].astype(int)
    dof_jids = mj_model.dof_jntid[arm_dof]
    qpos_adr = mj_model.jnt_qposadr[dof_jids]
    if free_val is None:
        free_val = float(mj_data.qpos[qpos_adr[-1]])

    sols = gf.ik_q7(p_des, R_des, free_val) if USE_Q7 else []
    sols = finite_rows(sols)
    if sols.size:
        lo, hi = PANDA_LIMITS[:,0], PANDA_LIMITS[:,1]
        keep = np.all((sols >= lo) & (sols <= hi), axis=1)
        sols = sols[keep]
    return sols, qpos_adr

import os
def load_mj(name: str):
    os.chdir("tests/mjx_mod/assets")
    with open(f"{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    return (mj_model, mj_data)

# ======= main =======
if __name__ == "__main__":
    (mj_model, mj_data) = load_mj("mjx_single_cube")

    ee_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if ee_sid < 0:
        raise RuntimeError(f"Site '{EE_SITE}' not found.")

    mujoco.mj_forward(mj_model, mj_data)
    R_current = mj_data.site_xmat[ee_sid].reshape(3,3).copy()

    # Sample random targets and precompute IK solutions (as full qpos arrays)
    targets = []
    arm_dof = mj_model.actuator_trnid[:,0].astype(int)
    dof_jids = mj_model.dof_jntid[arm_dof]
    qpos_adr = mj_model.jnt_qposadr[dof_jids]

    for k in range(NUM_TARGETS):
        p_des = np.array([
            rng.uniform(*X_RANGE),
            rng.uniform(*Y_RANGE),
            rng.uniform(*Z_RANGE),
        ], dtype=float)
        R_des = R_current if FIX_ORIENTATION else random_rotation()

        sols, qadr = compute_ik_solutions(mj_model, mj_data, ee_sid, p_des, R_des, FREE_VAL)

        qpos_list = []
        for q7 in sols:
            qp = mj_data.qpos.copy()
            qp[qadr] = q7
            qpos_list.append(qp)

        targets.append((p_des, R_des, qpos_list))

    # View them safely (no physics stepping)
    with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        for (p_des, R_des, qpos_list) in targets:
            if not qpos_list:
                # show target anyway (no solution)
                viewer.user_scn.ngeom = 0
                add_sphere_safe(viewer.user_scn, p_des, radius=0.02, rgba=(1,0,0,0.9))  # red sphere = no IK
                t0 = time.time()
                while time.time() - t0 < HOLD_PER_TARGET and viewer.is_running():
                    viewer.sync(); time.sleep(0.01)
                continue

            # cycle through each branch found
            for qp in qpos_list:
                mj_data.qpos[:] = qp
                mj_data.qvel[:] = 0.0
                mj_data.qacc[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)

                # draw
                viewer.user_scn.ngeom = 0
                p_cur = mj_data.site_xpos[ee_sid].copy()
                add_cylinder_line_safe(viewer.user_scn, p_cur, p_des, radius=0.004, color=(1,0,0,0.7))
                add_sphere_safe(viewer.user_scn, p_des, radius=0.02, rgba=(0,1,0,0.85))

                ep, er = pose_error(mj_model, mj_data, ee_sid, p_des, R_des)
                # quick console note (optional)
                print(f"target @ {p_des.round(3)} | branch pose_err: {ep:.3f} m, {er:.3f} rad")

                t0 = time.time()
                while time.time() - t0 < HOLD_PER_TARGET and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)
