import numpy as np
import geofikinpy as g

# ---------------- test parameters ----------------
POS_STD   = 1e-3    # meters, per axis (default perturb std)
ROT_STD   = 1e-3    # radians, per-axis so(3) (default perturb std)
GATE_Q    = 5e-2    # radians; max per-step joint distance to keep a track
SWIVEL_ATOL = 2e-2  # radians; keep swivel solutions close to base swivel

def _hat(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=float)

def _exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    W = _hat(w/th)
    return np.eye(3) + np.sin(th)*W + (1-np.cos(th))*(W@W)

def _unwrap_dist(qref, q):
    # small-step joint distance robust to wrap
    qref = np.asarray(qref); q = np.asarray(q)
    return np.linalg.norm(np.unwrap(np.vstack([qref, q]), axis=0)[1] - qref)

def _dedupe(sols, tol=1e-4):
    uniq = []
    for s in sols:
        if all(_unwrap_dist(u, s) > tol for u in uniq):
            uniq.append(s)
    return uniq

def _valid(S):
    return [np.asarray(s, float) for s in S if np.all(np.isfinite(s))]

def jitter_pose(p, R, pos_std=POS_STD, rot_std=ROT_STD, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    dp = rng.normal(0.0, pos_std, size=3)
    w  = rng.normal(0.0, rot_std, size=3)
    return p + dp, R @ _exp_so3(w)

def solve_family(family, p, R, base_q=None, base_theta=None):
    if family == "q7":
        assert base_q is not None
        sols = _valid(g.ik_q7(p, R, float(base_q[6])))
        return _dedupe(sols)
    if family == "q4":
        assert base_q is not None
        sols = _valid(g.ik_q4(p, R, float(base_q[3])))
        return _dedupe(sols)
    if family == "q6":
        assert base_q is not None
        sols = _valid(g.ik_q6(p, R, float(base_q[5])))
        return _dedupe(sols)
    if family == "swivel":
        theta = float(base_theta if base_theta is not None else g.swivel_from_q(base_q))
        sols = _valid(g.ik_swivel(p, R, theta))
        # keep swivel-close (numerical discretization)
        keep = []
        for s in sols:
            ths = g.swivel_from_q(s)
            if np.isfinite(ths):
                d = float((ths - theta + np.pi) % (2*np.pi) - np.pi)
                if abs(d) <= SWIVEL_ATOL:
                    keep.append(s)
        return _dedupe(keep)
    raise ValueError(f"unknown family: {family}")

# ---------------- main stability sweep ----------------
def stability_sweep(family, q_base, n_trials=50,
                    pos_std=POS_STD, rot_std=ROT_STD,
                    gate_q=GATE_Q, verbose=True, rng=None):
    """
    Given a base joint q_base, perturb its FK pose and check branch stability
    for the chosen IK family ('q7'|'q4'|'q6'|'swivel').
    Returns a dict of metrics and per-branch traces.
    """
    rng = np.random.default_rng() if rng is None else rng
    # reference pose
    T0 = g.franka_fk(q_base); p0, R0 = T0[:3,3].copy(), T0[:3,:3].copy()
    theta0 = g.swivel_from_q(q_base) if family == "swivel" else None

    # base solution set (and order = tracked branches)
    S0 = solve_family(family, p0, R0, base_q=q_base, base_theta=theta0)
    if len(S0) == 0:
        if verbose: print(f"[{family}] no base solutions; abort.")
        return {"ok": False, "reason": "no_base_solutions"}

    tracks = []
    for s in S0:
        tracks.append({
            "q_last": s,
            "alive": 0,
            "drops": 0,
            "step_dists": []
        })

    # sweep
    miss_trials = 0
    for t in range(n_trials):
        p_t, R_t = jitter_pose(p0, R0, pos_std, rot_std, rng)
        S_t = solve_family(family, p_t, R_t, base_q=q_base, base_theta=theta0)
        used = [False]*len(S_t)

        # greedy nearest-neighbor assign for each track
        for tr in tracks:
            if len(S_t) == 0:
                tr["drops"] += 1
                continue
            dists = [(_unwrap_dist(tr["q_last"], s), j) for j, s in enumerate(S_t) if not used[j]]
            if not dists:
                tr["drops"] += 1
                continue
            dmin, jbest = min(dists, key=lambda x: x[0])
            if dmin <= gate_q:
                tr["alive"] += 1
                tr["step_dists"].append(dmin)
                tr["q_last"] = S_t[jbest]
                used[jbest] = True
            else:
                tr["drops"] += 1

        if all(not u for u in used):
            miss_trials += 1

    # summarize
    surv = np.array([tr["alive"]/max(1,n_trials) for tr in tracks])
    step_mean = np.array([np.mean(tr["step_dists"]) if tr["step_dists"] else np.nan for tr in tracks])
    step_max  = np.array([np.max(tr["step_dists"])  if tr["step_dists"] else np.nan for tr in tracks])

    summary = {
        "ok": True,
        "family": family,
        "n_base": len(S0),
        "n_trials": n_trials,
        "miss_trials": miss_trials,
        "survival": surv,
        "step_mean": step_mean,
        "step_max": step_max,
        "tracks": tracks,
        "p0": p0, "R0": R0, "q_base": np.array(q_base)
    }

    if verbose:
        print(f"[{family}] base_solutions={len(S0)}  trials={n_trials}  miss_trials={miss_trials}")
        for i, tr in enumerate(tracks):
            m = step_mean[i]; mx = step_max[i]
            print(f"  branch {i:02d}: survival={surv[i]*100:5.1f}%  "
                  f"step_mean={m if np.isfinite(m) else np.nan:.3e}  "
                  f"step_max={mx if np.isfinite(mx) else np.nan:.3e} rad")

    return summary

# ---------------- example usage ----------------
if __name__ == "__main__":
    # pick a mid-range base joint (avoid singulars)
    q_base = np.array([ 0.20, -0.80,  0.30, -1.50,  0.20,  1.60,  0.50])
    for fam in ("q7","q4","q6","swivel"):
        stability_sweep(fam, q_base,
                        n_trials=100,
                        pos_std=1e-3,   # 1mm
                        rot_std=1e-3,   # ~0.057°
                        gate_q=5e-2,    # 0.05 rad per-step
                        verbose=True)
