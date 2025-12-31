import jax
import jax.numpy as jnp
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from pfmp import PFMP
from pfmp.interpolate import hermite_spline


class Environment(NamedTuple):
    centers: jax.Array  # (N, 2)
    radii: jax.Array    # (N,)


class Params(NamedTuple):
    goal: jax.Array
    goal_ee: jax.Array
    env: Environment
    link_lengths: jax.Array
    q_min: jax.Array
    q_max: jax.Array
    u_stdev: float


def forward_kinematics(q: jax.Array, link_lengths: jax.Array) -> jax.Array:
    n = len(link_lengths)
    angles = jnp.cumsum(q)
    
    positions = [jnp.zeros(2)]
    pos = jnp.zeros(2)
    
    for i in range(n):
        pos = pos + link_lengths[i] * jnp.array([jnp.cos(angles[i]), jnp.sin(angles[i])])
        positions.append(pos)
    
    return jnp.stack(positions)


def arm_sdf_vmapped(q: jax.Array, link_lengths: jax.Array, env: Environment) -> float:
    positions = forward_kinematics(q, link_lengths)
    n_links = len(link_lengths)
    
    def segment_point_distance(p1, p2, center):
        v = p2 - p1
        w = center - p1
        t = jnp.clip(jnp.dot(w, v) / (jnp.dot(v, v) + 1e-8), 0.0, 1.0)
        closest = p1 + t * v
        return jnp.linalg.norm(center - closest)
    
    def link_sdf(i):
        p1 = positions[i]
        p2 = positions[i + 1]
        dists = jax.vmap(lambda c, r: segment_point_distance(p1, p2, c) - r)(
            env.centers, env.radii
        )
        return jnp.min(dists)
    
    link_sdfs = jax.vmap(link_sdf)(jnp.arange(n_links))
    return jnp.min(link_sdfs)


def collision_cost(sdf: float, margin: float = 0.1) -> float:
    return jnp.maximum(margin - sdf, 0.0) ** 2


def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    sdf = arm_sdf_vmapped(x, th.link_lengths, th.env)
    col_cost = 1000.0 * collision_cost(sdf, margin=0.1)
    
    positions = forward_kinematics(x, th.link_lengths)
    ee = positions[-1]
    ee_cost = 0.1 * jnp.linalg.norm(ee - th.goal_ee)
    
    return col_cost + ee_cost


def terminal_cost(x: jax.Array, th: Params) -> float:
    sdf = arm_sdf_vmapped(x, th.link_lengths, th.env)
    col_cost = 1000.0 * collision_cost(sdf, margin=0.1)
    
    positions = forward_kinematics(x, th.link_lengths)
    ee = positions[-1]
    ee_cost = 50.0 * jnp.linalg.norm(ee - th.goal_ee) ** 2
    
    return col_cost + ee_cost

def angle_diff(target, current):
    """Shortest angular difference, handles wrapping."""
    diff = target - current
    return jnp.arctan2(jnp.sin(diff), jnp.cos(diff))

def dynamics(x: jax.Array, u: jax.Array, theta: Params, t: float):
    # u is a "target" - take a small step toward it
    direction = jax.vmap(angle_diff)(u, x)
    dist = jnp.linalg.norm(direction)
    direction = direction / (dist + 1e-8)
    
    step = jnp.minimum(dist, 0.1)
    x_next = x + direction * step
    
    # Wrap to [-π, π]
    x_next = jnp.arctan2(jnp.sin(x_next), jnp.cos(x_next))
    
    return (x_next, x_next)


def init(key, x, theta: Params):
    k1, k2 = jax.random.split(key)
    r = jax.random.uniform(k1, ())
    target = jnp.where(
        r < 0.7,
        jax.random.uniform(k2, shape=x.shape, minval=theta.q_min, maxval=theta.q_max),
        theta.goal
    )
    return target

def proposal(key, x, u, theta: Params, t: float):
    k1, k2 = jax.random.split(key)
    r = jax.random.uniform(k1, ())
    target = jnp.where(
        r < 0.7,
        jax.random.uniform(k2, shape=x.shape, minval=theta.q_min, maxval=theta.q_max),
        theta.goal
    )
    return target

# === Visualization ===

def plot_arm(ax, q, link_lengths, color='blue', alpha=1.0, linewidth=4, zorder=10):
    positions = np.array(forward_kinematics(q, link_lengths))
    ax.plot(positions[:, 0], positions[:, 1], '-', 
            color=color, alpha=alpha, linewidth=linewidth, 
            solid_capstyle='round', zorder=zorder)
    ax.scatter(positions[:-1, 0], positions[:-1, 1], 
               s=linewidth*15, c=color, alpha=alpha, zorder=zorder+1)
    ax.scatter(positions[-1, 0], positions[-1, 1], 
               s=linewidth*20, c=color, alpha=alpha, marker='o', zorder=zorder+1)


def plot_environment(ax, env):
    for i in range(len(env.centers)):
        circle = Circle(env.centers[i], env.radii[i], 
                        facecolor='#404040', edgecolor='black', linewidth=2, zorder=5)
        ax.add_patch(circle)


def plot_ee_trace(ax, path, link_lengths, color='blue', alpha=0.6):
    ee_positions = []
    for q in path:
        positions = forward_kinematics(jnp.array(q), link_lengths)
        ee_positions.append(np.array(positions[-1]))
    ee_positions = np.array(ee_positions)
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], 
            '-', color=color, alpha=alpha, linewidth=2, zorder=8)


def plot_result(ax, path, link_lengths, env, title, q_goal, n_ghosts=5):
    ax.set_facecolor('#f5f5f5')
    plot_environment(ax, env)
    plot_ee_trace(ax, path, link_lengths, color='#2196F3', alpha=0.8)
    
    # Ghost arms
    if n_ghosts > 0:
        indices = np.linspace(0, len(path) - 1, n_ghosts + 2).astype(int)[1:-1]
        for idx in indices:
            plot_arm(ax, path[idx], link_lengths, 
                     color='#90CAF9', alpha=0.4, linewidth=3, zorder=6)
    
    # Start (red)
    plot_arm(ax, path[0], link_lengths, 
             color='#F44336', alpha=0.9, linewidth=5, zorder=11)
    
    # Goal config (gold, faded)
    plot_arm(ax, q_goal, link_lengths, 
             color='#FFC107', alpha=0.5, linewidth=4, zorder=9)
    
    # Result (green)
    plot_arm(ax, path[-1], link_lengths, 
             color='#4CAF50', alpha=0.9, linewidth=5, zorder=12)
    
    # Goal EE marker
    goal_ee = forward_kinematics(q_goal, link_lengths)[-1]
    ax.scatter([goal_ee[0]], [goal_ee[1]], s=200, c='gold', 
               edgecolors='black', linewidth=2, marker='*', zorder=15)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='#F44336', linewidth=4, label='Start'),
        Line2D([0], [0], color='#FFC107', linewidth=4, alpha=0.5, label='Goal'),
        Line2D([0], [0], color='#4CAF50', linewidth=4, label='Result'),
        Line2D([0], [0], color='#2196F3', linewidth=2, label='EE trace'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')


def test_setup(env):
    """Visualize configs to find valid start/goal."""
    n_dof = 3
    link_lengths = jnp.array([1.0, 1.0, 0.5])
    
    configs = [
        ("start candidate", jnp.array([-0.3, 0.5, 0.2])),
        ("goal candidate", jnp.array([1.2, -0.6, -0.2])),
        ("alt start", jnp.array([0.8, 0.8, 0.0])),
        ("alt goal", jnp.array([1.5, -0.8, 0.0])),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for ax, (name, q) in zip(axes, configs):
        sdf = arm_sdf_vmapped(q, link_lengths, env)
        plot_environment(ax, env)
        color = '#4CAF50' if sdf > 0 else '#F44336'
        plot_arm(ax, q, link_lengths, color=color, linewidth=5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        status = "CLEAR" if sdf > 0 else "COLLISION"
        ax.set_title(f"{name}\nSDF: {sdf:.2f} ({status})", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_setup.png', dpi=150)
    plt.show()

def main():
    n_dof = 3
    link_lengths = jnp.array([1.0, 1.0, 0.5])
    
    env = Environment(
        centers=jnp.array([
            [1.5, 0.8],
            [0.8, -0.4],
            [-0.5, 0.6],
            [1.2, -1.0],
        ]),
        radii=jnp.array([0.25, 0.25, 0.25, 0.25])
    )
    
    q_start = jnp.array([0., 0.5, 0.2])
    q_goal = jnp.array([1.2, -0.6, -0.2])
    
    start_sdf = arm_sdf_vmapped(q_start, link_lengths, env)
    goal_sdf = arm_sdf_vmapped(q_goal, link_lengths, env)
    print(f"Start SDF: {start_sdf:.3f} {'(CLEAR)' if start_sdf > 0 else '(COLLISION!)'}")
    print(f"Goal SDF: {goal_sdf:.3f} {'(CLEAR)' if goal_sdf > 0 else '(COLLISION!)'}")
    
    if start_sdf < 0 or goal_sdf < 0:
        print("\nRun test_setup() to find valid configs!")
        return
    
    goal_ee = forward_kinematics(q_goal, link_lengths)[-1]
    
    theta = Params(
        goal=q_goal,
        goal_ee=goal_ee,
        env=env,
        link_lengths=link_lengths,
        q_min=jnp.full(n_dof, -jnp.pi),
        q_max=jnp.full(n_dof, jnp.pi),
        u_stdev=0.05,
    )
    
    horizon = 1.0
    num_knots = 5
    particles = 512
    
    pfmp = PFMP(
        dynamics, cost_fn, terminal_cost, init, proposal,
        horizon, num_knots, particles,
        ess_resample_thresh=0.3
    )
    
    key = jax.random.key(42)
    
    # Run SIS
    print("\nRunning SIS...")
    import time
    for _ in range(5):
        t0 = time.time()
        traj_sis, scores_sis = pfmp.run_sis(key, q_start, theta)
        best_sis = jnp.argmin(jnp.sum(scores_sis, axis=-1))
        print("SIS elapsed time: %.3f sec" % (time.time() - t0))
    
    # Run SIR
    print("\nRunning SIR...")
    for _ in range(5):
        t0 = time.time()
        traj_sir, scores_sir = pfmp.run_sir(key, q_start, theta)
        best_sir = jnp.argmin(jnp.sum(scores_sir, axis=-1))
        print("SIR elapsed time: %.3f sec" % (time.time() - t0))
    
    # Extract paths
    tq = jnp.linspace(0, horizon, int(horizon / pfmp.dt))
    
    us_sis = hermite_spline(tq, pfmp.tk(), traj_sis[best_sis])
    _, path_sis = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us_sis)
    path_sis = np.array(path_sis)
    
    us_sir = hermite_spline(tq, pfmp.tk(), traj_sir[best_sir])
    _, path_sir = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us_sir)
    path_sir = np.array(path_sir)
    
    # Evaluation
    def check_collision(path):
        collisions = 0
        for q in path:
            sdf = arm_sdf_vmapped(jnp.array(q), link_lengths, env)
            if sdf < 0:
                collisions += 1
        return collisions / len(path)
    
    def goal_distance(path):
        return np.linalg.norm(path[-1] - np.array(q_goal))
    
    sis_col = check_collision(path_sis)
    sir_col = check_collision(path_sir)
    sis_dist = goal_distance(path_sis)
    sir_dist = goal_distance(path_sir)
    
    print(f"\nResults:")
    print(f"  SIS: collision={sis_col:.1%}, goal_dist={sis_dist:.3f}")
    print(f"  SIR: collision={sir_col:.1%}, goal_dist={sir_dist:.3f}")
    
    # Side-by-side animation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ee_trace_sis = np.array([forward_kinematics(jnp.array(q), link_lengths)[-1] for q in path_sis])
    ee_trace_sir = np.array([forward_kinematics(jnp.array(q), link_lengths)[-1] for q in path_sir])
    
    collision_flags_sis = np.array([arm_sdf_vmapped(jnp.array(q), link_lengths, env) < 0 for q in path_sis])
    collision_flags_sir = np.array([arm_sdf_vmapped(jnp.array(q), link_lengths, env) < 0 for q in path_sir])
    
    def animate(frame):
        for ax in axes:
            ax.clear()
            ax.set_facecolor('#f5f5f5')
            
            # Environment
            for i in range(len(env.centers)):
                circle = Circle(env.centers[i], env.radii[i], 
                                facecolor='#404040', edgecolor='black', linewidth=2, zorder=5)
                ax.add_patch(circle)
            
            # Goal config (faded)
            plot_arm(ax, q_goal, link_lengths, color='#FFC107', alpha=0.3, linewidth=3, zorder=4)
            goal_ee_pos = forward_kinematics(q_goal, link_lengths)[-1]
            ax.scatter([goal_ee_pos[0]], [goal_ee_pos[1]], s=200, c='gold', 
                       edgecolors='black', linewidth=2, marker='*', zorder=15)
            
            # Start config (faded)
            plot_arm(ax, q_start, link_lengths, color='#F44336', alpha=0.3, linewidth=2, zorder=3)
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # SIS
        if frame > 0:
            axes[0].plot(ee_trace_sis[:frame+1, 0], ee_trace_sis[:frame+1, 1], 
                         '-', color='#2196F3', alpha=0.6, linewidth=2, zorder=7)
        color_sis = '#F44336' if collision_flags_sis[frame] else '#4CAF50'
        plot_arm(axes[0], path_sis[frame], link_lengths, 
                 color=color_sis, alpha=0.9, linewidth=4, zorder=10)
        status_sis = "COLLISION!" if collision_flags_sis[frame] else "clear"
        axes[0].set_title(f'SIS (no resampling) - {status_sis}\ncol={sis_col:.0%}, dist={sis_dist:.2f}', 
                          fontsize=12, fontweight='bold')
        
        # SIR
        if frame > 0:
            axes[1].plot(ee_trace_sir[:frame+1, 0], ee_trace_sir[:frame+1, 1], 
                         '-', color='#2196F3', alpha=0.6, linewidth=2, zorder=7)
        color_sir = '#F44336' if collision_flags_sir[frame] else '#4CAF50'
        plot_arm(axes[1], path_sir[frame], link_lengths, 
                 color=color_sir, alpha=0.9, linewidth=4, zorder=10)
        status_sir = "COLLISION!" if collision_flags_sir[frame] else "clear"
        axes[1].set_title(f'SIR (with resampling) - {status_sir}\ncol={sir_col:.0%}, dist={sir_dist:.2f}', 
                          fontsize=12, fontweight='bold')
        
        return []
    
    n_frames = max(len(path_sis), len(path_sir))
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    
    print("\nSaving animation...")
    anim.save('sis_vs_sir.mp4', writer='ffmpeg', fps=20, dpi=150)
    print("Saved to sis_vs_sir.mp4")
    
    plt.close()
    
    # Static comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_result(axes[0], path_sis, link_lengths, env, 
                f'SIS (no resampling)\ncollision: {sis_col:.0%}, goal dist: {sis_dist:.2f}', 
                q_goal)
    plot_result(axes[1], path_sir, link_lengths, env, 
                f'SIR (with resampling)\ncollision: {sir_col:.0%}, goal dist: {sir_dist:.2f}', 
                q_goal)
    plt.tight_layout()
    plt.savefig('sis_vs_sir.png', dpi=150, bbox_inches='tight')
    print("Saved to sis_vs_sir.png")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_setup()
    else:
        main()
