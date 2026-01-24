# obstacle_avoidance.py

import jax
import jax.numpy as jnp
from typing import NamedTuple
from pfmp import PFMP
from pfmp.interpolate import Jet2, hermite_spline
import matplotlib.pyplot as plt


class Params(NamedTuple):
    goal: jax.Array           # Desired position
    obstacles: jax.Array      # ObstacleGrid
    L: float                  # wheelbase (m)
    v: float                  # forward speed (m/s)
    du_stdev: float           # tangent noise
    u_stdev: float            # point noise (local jitter)

def generate_obstacle_grid(num_x: int, num_y: int, box_size: float, gap: float):
    """ Generate a regular grid of AABB obstacles.

    Args:
        num_x: number of boxes along the x-axis
        num_y: number of boxes along the y-axis
        box_size: side length of each box (square)
        gap: gap between boxes (same in x and y)

    Returns:
        A (num_x * num_y, 4) array of AABB boxes as (xmin, xmax, ymin, ymax)
    """
    obstacles = []
    for i in range(num_x):
        for j in range(num_y):
            xmin = i * (box_size + gap)
            ymin = j * (box_size + gap)
            xmax = xmin + box_size
            ymax = ymin + box_size
            obstacles.append(jnp.array([xmin, xmax, ymin, ymax]))
    return jnp.stack(obstacles)
    
@jax.jit
def sdf_aabb(pos: jax.Array, bounds: jax.Array) -> float:
    xmin, xmax, ymin, ymax = bounds
    center = jnp.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    half_extents = jnp.array([(xmax - xmin) / 2, (ymax - ymin) / 2])
    
    d = jnp.abs(pos - center) - half_extents
    outside_dist = jnp.linalg.norm(jnp.maximum(d, 0.0))
    inside_dist = jnp.minimum(jnp.maximum(d[0], d[1]), 0.0)
    return outside_dist + inside_dist

@jax.jit
def hinge_loss(sdf: float, epsilon: float) -> float:
    return jnp.where(sdf < epsilon, 0.5 * (epsilon - sdf) ** 2, 0.0)

def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    loc = x[:2]
    sdf_vals = jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles)
    sdf_val = jnp.min(sdf_vals)
    obstacle_loss = 100. * hinge_loss(sdf_val, 0.25)
    # this is in-case `u.collision_multiplier == jnp.inf`
    obstacle_loss = jnp.nan_to_num(obstacle_loss, nan=0.0, posinf=jnp.inf, neginf=0.0)
    goal_loss = jnp.linalg.norm(loc - th.goal) / 10.
    return obstacle_loss**2 + goal_loss**2

def terminal_cost(x, th) -> float:
    loc = x[:2]
    return 10 * jnp.linalg.norm(loc - th.goal) ** 2

def dynamics(x: jax.Array, u: jax.Array, theta: Params, t):
    """
    x: [px, py, theta]
    j: Jet2(u, du) where u is steering angle δ
    """
    px, py, th = x
    L, v = theta.L, theta.v
    
    px_next = px + v * jnp.cos(th) * 0.01
    py_next = py + v * jnp.sin(th) * 0.01
    th_next = th + (v / L) * jnp.tan(u) * 0.01

    th_next = jnp.mod(th_next + jnp.pi, 2 * jnp.pi) - jnp.pi

    x_next = jnp.array([px_next, py_next, th_next])
    return (x_next, x_next)

def init(key, x, theta):
    u_next = jax.random.normal(key) * theta.u_stdev
    u_bound = 0.8
    return jnp.clip(u_next, -u_bound, u_bound)

def proposal(key, _x, _u: float, theta: Params, t):
    """
    key: PRNGKey
    x: state (ignored)
    u = current steering angle and rate
    theta: Params
    """

    u_next = jax.random.normal(key) * theta.u_stdev
    u_bound = 0.8
    u_next = jnp.clip(u_next, -u_bound, u_bound)
    return u_next

def plot_obstacles(ax, obstacles, color='black'):
    """Plot a list of AABB obstacles."""
    for obs in obstacles:
        xmin, xmax, ymin, ymax = obs
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height,
                             edgecolor=color, facecolor='black', linewidth=1)
        ax.add_patch(rect)

# if __name__ == "__main__":
#     horizon = 10
#     num_knots = int(2*horizon)
#     particles = 128
#     pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
#                 horizon, num_knots, particles,
#                 ess_resample_thresh=0.5)
#     x0 = jnp.array([0,0,jnp.pi/4])
#     # u0 = Jet2(0., 0.)
#     u0 = 0.0
#     theta = Params(
#         goal = jnp.array([12.0, 12.0]),
#         obstacles = generate_obstacle_grid(5, 5, 1.0, 1.0) + 1.0,
#         L=0.3,          # typical wheelbase for normalized bicycle model
#         v=2.0,          # moderate forward speed (m/s)
#         du_stdev=0.1,  # tangent noise — gives gentle arc variation
#         u_stdev=0.2    # tiny point jitter — stabilizes proposal
#     )

#     key = jax.random.key(1)
#     traj, scores = pfmp.run_sir(key, x0, theta)
#     tq = jnp.arange(0, horizon, pfmp.dt)

#     import matplotlib.patches as patches
#     import matplotlib.transforms as transforms

#     car_length = theta.L
#     car_width  = theta.L/2

#     ax = plt.gca()

#     xs = jnp.linspace(0, 12, 200)
#     ys = jnp.linspace(0, 12, 200)
#     X, Y = jnp.meshgrid(xs, ys)

#     # Flatten for vectorization
#     points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

#     # Wrap cost_fn so it only depends on position
#     def cost_at_loc(loc):
#         # Build a dummy x and u consistent with your system
#         x = loc  # fill in based on your state dimension
#         u = jnp.zeros(1)                          # fill in based on your control dimension
#         return cost_fn(x, u, theta, t=0.0)

#     cost_vmap = jax.vmap(cost_at_loc)
#     Z = cost_vmap(points).reshape(X.shape)

#     ax.contourf(X, Y, Z, levels=50)

#     plot_obstacles(ax, theta.obstacles)

#     interval_len = horizon / (num_knots - 1)
#     steps_per_knot = round(interval_len / pfmp.dt)
#     best_idx = jnp.argmin(jnp.sum(scores, axis=-1))
#     tq = jnp.linspace(0,pfmp.horizon,int(pfmp.horizon/pfmp.dt))
#     for i in range(20):
#         us = hermite_spline(tq, pfmp.tk(), traj[i])
#         _, xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
#         plt.plot(xy[:,0], xy[:,1], c="magenta", alpha=0.5)

#         for t in range(0, xy.shape[0], steps_per_knot):  # every 5 steps
#             x, y, th = xy[t]

#             # Create a rectangle centered at (0,0)
#             rect = patches.Rectangle(
#                 (-car_length/2, -car_width/2),  # center the box
#                 car_length,
#                 car_width,
#                 facecolor="cyan",
#                 edgecolor="black",
#                 alpha=1.0,
#                 zorder=5
#             )

#             # Build the transform: rotate, then translate
#             trans = transforms.Affine2D()
#             trans.rotate(th)
#             trans.translate(x, y)

#             rect.set_transform(trans + ax.transData)
#             ax.add_patch(rect)
#     us = hermite_spline(tq, pfmp.tk(), traj[best_idx])
#     _, best_xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
#     plt.plot(best_xy[:,0], best_xy[:,1], c="gold", lw=3, alpha=1)

#     plt.axis("equal")
#     plt.show()

if __name__ == "__main__":
    horizon = 10
    num_knots = int(2*horizon)
    particles = 256
    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon, num_knots, particles,
                ess_resample_thresh=0.5)
    x0 = jnp.array([0,0,jnp.pi/4])
    u0 = 0.0
    theta = Params(
        goal = jnp.array([12.0, 12.0]),
        obstacles = generate_obstacle_grid(5, 5, 1.0, 1.0) + 1.0,
        L=0.3,
        v=2.0,
        du_stdev=0.1,
        u_stdev=0.2
    )
    
    # Setup visualization with white background for publication
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms
    from matplotlib.animation import FuncAnimation
    
    # Use default white background style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot cost field with subtle colormap
    xs = jnp.linspace(0, 12, 200)
    ys = jnp.linspace(0, 12, 200)
    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    
    def cost_at_loc(loc):
        x = loc
        u = jnp.zeros(1)
        return cost_fn(x, u, theta, t=0.0)
    
    cost_vmap = jax.vmap(cost_at_loc)
    Z = cost_vmap(points).reshape(X.shape)
    
    # Use a subtle, printer-friendly colormap
    contour = ax.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.3)
    
    # Add colorbar with larger text
    cbar = plt.colorbar(contour, ax=ax, label='Cost')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Cost', fontsize=16, fontweight='bold')
    
    # Plot obstacles with high contrast
    plot_obstacles(ax, theta.obstacles)
    
    # Mark start and goal with clear, publication-quality markers
    ax.plot(x0[0], x0[1], 'o', color='darkgreen', markersize=14, 
            markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=10)
    ax.plot(theta.goal[0], theta.goal[1], '*', color='gold', markersize=20,
            markeredgecolor='black', markeredgewidth=1.5, label='Goal', zorder=10)
    
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel('X Position (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=14)
    
    # Add legend with clean styling and larger text
    ax.legend(loc='upper left', fontsize=14, framealpha=0.95, 
             edgecolor='black', fancybox=False, shadow=False)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # Initialize storage
    key = jax.random.key(1)
    interval_len = horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp.dt)
    
    # Run full SIR to get complete data
    states, ancestors, costs, final_xs = pfmp._sir(key, x0, theta)
    
    # Animation data
    trajectory_lines = []
    car_patches_list = []
    knot_markers = []
    progress_text = None
    
    # Define consistent colors
    particle_color = 'steelblue'  # Consistent color for all particles
    best_color = 'darkblue'       # Color for best trajectory
    
    def init_animation():
        return trajectory_lines + car_patches_list + knot_markers
    
    def animate(frame):
        global progress_text
        
        # Clear previous elements
        for line in trajectory_lines:
            line.remove()
        for patch in car_patches_list:
            patch.remove()
        for marker in knot_markers:
            marker.remove()
        if progress_text:
            progress_text.remove()
            
        trajectory_lines.clear()
        car_patches_list.clear()
        knot_markers.clear()
        
        # Calculate how many knots to use
        current_num_knots = min(frame + 2, num_knots)
        
        # Slice states and ancestors up to current knot
        current_states = jax.tree.map(lambda x: x[:current_num_knots], states)
        current_ancestors = ancestors[:current_num_knots-1]
        
        # Choose paths with current data
        current_trajs = pfmp.choose_paths(current_states, current_ancestors, 
                                         jnp.arange(pfmp.particles))
        
        # Calculate time points for current trajectory
        current_horizon = (current_num_knots - 1) * interval_len
        tq = jnp.linspace(0, current_horizon, int(current_horizon/pfmp.dt))
        tk = jnp.linspace(0, current_horizon, current_num_knots)
        
        # Calculate costs
        current_costs = costs[:, :current_num_knots-1]
        total_costs = jnp.sum(current_costs, axis=-1)
        top_indices = jnp.argsort(total_costs)[:20]
        best_idx = jnp.argmin(total_costs)
        
        # Plot particle trajectories with consistent color
        for i in top_indices:
            if i == best_idx:
                continue  # Skip best, will plot it separately
                
            us = hermite_spline(tq, tk, current_trajs[i])
            _, xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), 
                                x0, (us, tq))
            
            line, = ax.plot(xy[:,0], xy[:,1], color=particle_color, 
                           alpha=0.3, linewidth=1.5, zorder=2)
            trajectory_lines.append(line)
            
            # Draw cars at knot points with consistent color
            for t in range(0, xy.shape[0], steps_per_knot):
                x, y, th = xy[t]
                rect = patches.Rectangle(
                    (-theta.L/2, -theta.L/4),
                    theta.L,
                    theta.L/2,
                    facecolor=particle_color,
                    edgecolor='black',
                    linewidth=0.8,
                    alpha=0.4,
                    zorder=3
                )
                trans = transforms.Affine2D()
                trans.rotate(th)
                trans.translate(x, y)
                rect.set_transform(trans + ax.transData)
                ax.add_patch(rect)
                car_patches_list.append(rect)
        
        # Plot best trajectory with emphasis
        us = hermite_spline(tq, tk, current_trajs[best_idx])
        _, best_xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), 
                                 x0, (us, tq))
        
        # Best trajectory with shadow for depth
        line, = ax.plot(best_xy[:,0], best_xy[:,1], c='lightgray', 
                       lw=5, alpha=0.5, zorder=4, solid_capstyle='round')
        trajectory_lines.append(line)
        
        # Main best trajectory line
        line, = ax.plot(best_xy[:,0], best_xy[:,1], c=best_color, 
                       lw=3.5, alpha=1, zorder=5, label='Best Path',
                       solid_capstyle='round')
        trajectory_lines.append(line)
        
        # Draw best car with consistent color
        for t in range(0, best_xy.shape[0], steps_per_knot):
            x, y, th = best_xy[t]
            
            # Car body
            rect = patches.Rectangle(
                (-theta.L/2, -theta.L/4),
                theta.L,
                theta.L/2,
                facecolor=best_color,
                edgecolor='black',
                linewidth=2,
                alpha=0.9,
                zorder=6
            )
            trans = transforms.Affine2D()
            trans.rotate(th)
            trans.translate(x, y)
            rect.set_transform(trans + ax.transData)
            ax.add_patch(rect)
            car_patches_list.append(rect)
        
        # Mark knot points on best trajectory
        for t in range(0, best_xy.shape[0], steps_per_knot):
            marker, = ax.plot(best_xy[t, 0], best_xy[t, 1], 'o', 
                            color=best_color, markersize=6,
                            markeredgecolor='black', markeredgewidth=1,
                            zorder=8, alpha=0.9)
            knot_markers.append(marker)
        
        # Progress indicator - clean text box with larger text
        progress_pct = (current_num_knots / num_knots) * 100
        progress_text = ax.text(
            0.02, 0.98, 
            f'Knots: {current_num_knots}/{num_knots} ({progress_pct:.0f}%)\n'
            f'Best Cost: {float(jnp.min(total_costs)):.2f}',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                     alpha=0.9, edgecolor='black', linewidth=1.5),
            color='black',
            fontfamily='monospace',
            fontweight='bold'
        )
        
        return trajectory_lines + car_patches_list + knot_markers + [progress_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init_animation,
                        frames=num_knots-1, interval=1, blit=False, 
                        repeat=True, repeat_delay=1)
    
    plt.tight_layout()
    plt.show()
    
    # Save with publication quality
    # anim.save('pfmp_planning.mp4', writer='ffmpeg', fps=5, dpi=300, 
    #           bitrate=5000, extra_args=['-vcodec', 'libx264'])
    # 
    # # Or save as GIF for easier inclusion in documents
    # anim.save('pfmp_planning.gif', writer='pillow', fps=5, dpi=150)