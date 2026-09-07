# Franka FR3 push-T trial data

Per-trial records for the hardware experiment (Section V-C). All 140 records
come from 7 methods x 20 trials.

## fr3_trials.csv (20 rows)

The problem set: one row per trial, giving the specified initial and goal pose
of the T-block as (x, y, yaw) in metres and radians.

These are reproducible rather than arbitrary -- `sample_problem(seed, trial)` in
`fr3/tee_problem_gen.py` maps (seed=99, trial=0..19) to exactly these poses, so
the file is a convenience, not the source of truth. Poses are drawn on the arm's
reachable annulus at a controlled start-goal separation, so difficulty is
comparable across trials.

## fr3_results.csv (140 rows)

One row per (trial, method). Unlike the specs above, these are measurements and
cannot be regenerated.

| column | meaning |
|--------|---------|
| `trial`, `method` | which problem, which algorithm |
| `proposal`, `update` | the method's position on two of the three axes |
| `success` | 1 if the block reached the goal tolerance before the time limit |
| `start_*` | block pose as measured at trial start, after hand placement |
| `placement_err_m` | distance from the specified initial pose to the measured one |
| `final_*` | block pose at the end of the trial |
| `final_pos_err_m`, `final_angle_err_rad` | distance from the goal pose |
| `n_replans`, `duration_s` | replanning steps taken, wall-clock length |

Every method ran the same 20 problems, so comparisons are paired rather than
over independent draws.

Hand placement matched the specified pose to 3.5 mm on average (max 11 mm)
against a 5 cm success tolerance, so placement noise is small relative to the
criterion.

The raw motion-capture stream is not included; these are the per-trial
summaries derived from it.
