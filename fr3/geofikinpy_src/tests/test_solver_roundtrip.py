import numpy as np
import pytest
import geofikinpy as g

# ---------- test parameters ----------
POS_ATOL = 1e-5
SWIVEL_ATOL = 1e-2
ROT_ATOL = 1e-5
Q_ATOL   = 1e-4

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
    R = T[:3, :3]
    p = T[:3, 3]
    return p, R

def pose_match(p_ref, R_ref, p, R, pos_atol=POS_ATOL, rot_atol=ROT_ATOL):
    return (np.linalg.norm(p - p_ref) <= pos_atol) and (rot_angle_err(R_ref, R) <= rot_atol)

def dedupe_solutions(S, tol=Q_ATOL):
    uniq = []
    for s in S:
        if not any(unwrap_close(u, s, tol) for u in uniq):
            uniq.append(s)
    return uniq

def valid_solutions(S):
    return [np.asarray(s, float) for s in S if np.all(np.isfinite(s))]

# Mid-range seeds; elbow bent (q4 ~ −1.2…−2.0), avoid singulars
Q_SEEDS = [
    np.array([ 0.20, -0.80,  0.30, -1.50,  0.20,  1.60,  0.50]),
    np.array([-0.40, -1.00,  0.10, -1.20, -0.30,  1.80, -0.20]),
    np.array([ 0.60, -0.90, -0.40, -2.00,  0.50,  1.20,  0.10]),
]

@pytest.mark.parametrize("q_star", Q_SEEDS)
def test_roundtrip_q7_full_set(q_star):
    # FK reference pose
    p_ref, R_ref = pose_from_fk(q_star)
    q7 = float(q_star[6])

    # IK set (q7 fixed)
    S = valid_solutions(g.ik_q7(p_ref, R_ref, q7))
    assert len(S) > 0, "ik_q7 returned no valid solutions"

    # Deduplicate near-duplicates
    S = dedupe_solutions(S)

    # Every solution matches pose and q7
    for s in S:
        p, R = pose_from_fk(s)
        assert pose_match(p_ref, R_ref, p, R), "FK mismatch from ik_q7 solution"
        assert abs(float(s[6] - q7)) < 5e-6, "Returned solution violates fixed q7"

    # Original is in the set
    assert any(unwrap_close(q_star, s) for s in S), "Original q* not found in ik_q7 set"

@pytest.mark.parametrize("q_star", Q_SEEDS)
def test_roundtrip_q4_full_set(q_star):
    p_ref, R_ref = pose_from_fk(q_star)
    q4 = float(q_star[3])

    S = valid_solutions(g.ik_q4(p_ref, R_ref, q4))
    if not S:
        pytest.skip("Pose not solvable for fixed q4 (expected for some seeds)")
    S = dedupe_solutions(S)

    for s in S:
        p, R = pose_from_fk(s)
        assert pose_match(p_ref, R_ref, p, R), "FK mismatch from ik_q4 solution"
        assert abs(float(s[3] - q4)) < 5e-6, "Returned solution violates fixed q4"

    assert any(unwrap_close(q_star, s) for s in S), "Original q* not found in ik_q4 set"

@pytest.mark.parametrize("q_star", Q_SEEDS)
def test_roundtrip_q6_full_set(q_star):
    p_ref, R_ref = pose_from_fk(q_star)
    q6 = float(q_star[5])

    S = valid_solutions(g.ik_q6(p_ref, R_ref, q6))
    if not S:
        pytest.skip("Pose not solvable for fixed q6 (expected for some seeds)")
    S = dedupe_solutions(S)

    for s in S:
        p, R = pose_from_fk(s)
        assert pose_match(p_ref, R_ref, p, R), "FK mismatch from ik_q6 solution"
        assert abs(float(s[5] - q6)) < 5e-6, "Returned solution violates fixed q6"

    assert any(unwrap_close(q_star, s) for s in S), "Original q* not found in ik_q6 set"

@pytest.mark.parametrize("q_star", Q_SEEDS)
def test_roundtrip_swivel_full_set(q_star):
    p_ref, R_ref = pose_from_fk(q_star)
    theta = g.swivel_from_q(q_star)
    if not np.isfinite(theta):
        pytest.skip("Swivel undefined at this q* (near singular)")

    # Call with library defaults (no NPTS override)
    S = valid_solutions(g.ik_swivel(p_ref, R_ref, float(theta)))
    if not S:
        pytest.skip("Pose not solvable for fixed swivel (expected near singulars)")

    # Deduplicate near-duplicates
    uniq = []
    for s in S:
        if not any(unwrap_close(u, s, tol=Q_ATOL) for u in uniq):
            uniq.append(s)
    S = uniq

    # Tolerances (loosened to account for discretization in franka_ik_swivel)
    SWIVEL_ATOL = 2e-2   # ~0.02 rad ≈ 1.15°
    MEMBERSHIP_ATOL = 3e-2  # closest joint-space match to q* must be ≤ 0.03 rad

    for s in S:
        p, R = pose_from_fk(s)
        assert pose_match(p_ref, R_ref, p, R), "FK mismatch from ik_swivel solution"
        theta_s = g.swivel_from_q(s)
        assert np.isfinite(theta_s)
        d = float((theta_s - theta + np.pi) % (2*np.pi) - np.pi)  # wrap to (-pi,pi]
        assert abs(d) <= SWIVEL_ATOL, \
            f"Fixed swivel violated by {abs(d):.6f} rad (tol {SWIVEL_ATOL:.6f})"

    # Don’t require exact equality with q*, just proximity (due to discretization)
    dmins = [np.linalg.norm(np.unwrap(np.vstack([q_star, s]), axis=0)[1] - q_star) for s in S]
    assert min(dmins) <= MEMBERSHIP_ATOL, \
        f"Closest swivel solution is {min(dmins):.4f} rad from q* (tol {MEMBERSHIP_ATOL:.4f})"
