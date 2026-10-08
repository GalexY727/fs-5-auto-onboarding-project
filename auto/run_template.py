import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
from pid_template import (
    make_car,
    update,
    calculate_desired_acceleration,
    acceleration_to_throttle_percentage,
    calculate_air_drag,
)

# Simulation parameters and initial gains
K_P = 2.5
K_I = 0.005
K_D = 0.025
DESIRED_V = 20.0

DESIRED_TIME_S = 75
DT = 0.02
STEPS = int(DESIRED_TIME_S / DT)
# Samples are recorded before each update, starting at t=0.
dt_axis = np.arange(STEPS) * DT
SETTLING_TOLERANCE = 0.01
SETTLING_LABEL = f"Settling Time (±{SETTLING_TOLERANCE:.0%})"


def run_simulation(kp: float, ki: float, kd: float):
    """
    Simulates the car's motion over time using the given PID gains.

    Inputs:
    kp: float, proportional gain (K_P)
    ki: float, integral gain (K_I)
    kd: float, derivative gain (K_D)

    Outputs:
    tuple[np.ndarray, np.ndarray, np.ndarray]: velocity, air drag force, and error arrays over each time step
    """
    if not np.all(np.isfinite([kp, ki, kd])):
        raise ValueError("PID gains must be finite numbers.")
    car = make_car(desired_v=DESIRED_V, dt=DT)
    velocity = np.zeros(STEPS)
    drag_force = np.zeros(STEPS)
    error = np.zeros(STEPS)

    for i in range(STEPS):
        velocity[i] = car["v"]
        drag_force[i] = calculate_air_drag(car["v"])
        desired_a, error[i] = calculate_desired_acceleration(car, kp, ki, kd)
        if not np.isfinite(desired_a):
            raise ValueError("Simulation produced a nonfinite command.")
        throttle_p = acceleration_to_throttle_percentage(desired_a)
        update(car, throttle_p)

    return velocity, drag_force, error


def calculate_metrics(
    time: np.ndarray,
    velocity: np.ndarray,
    target_v: float = DESIRED_V,
    error: np.ndarray = None,
    drag: np.ndarray = None,
):
    """
    Calculates step response metrics, key characteristic points, and graph display bounds.

    Inputs:
    time: np.ndarray, array of simulation time points
    velocity: np.ndarray, array of vehicle velocity values over time
    target_v: float, target steady-state velocity in m/s (default 20.0)
    error: np.ndarray (optional), array of velocity error values over time
    drag: np.ndarray (optional), array of aerodynamic drag force values over time

    Outputs:
    tuple[dict, dict, dict]: (metrics, points, bounds) containing metric display strings,
                             markup coordinates, and axis display bounds
    """
    if (
        time.ndim != 1 or velocity.ndim != 1 or time.size != velocity.size
        or time.size < 2 or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(velocity)) or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Provide matching finite arrays and increasing timestamps.")
    if not np.isfinite(target_v) or target_v <= 0 or velocity[0] != 0:
        raise ValueError("Metrics require a positive target and a start from rest.")

    # Rise time (10% to 90% of target_v)
    idx_10 = np.where(velocity >= 0.1 * target_v)[0]
    idx_90 = np.where(velocity >= 0.9 * target_v)[0]
    pt_10 = (time[idx_10[0]], velocity[idx_10[0]]) if len(idx_10) else None
    pt_90 = (time[idx_90[0]], velocity[idx_90[0]]) if len(idx_90) else None
    rise_str = f"{pt_90[0] - pt_10[0]:.2f} s" if pt_10 and pt_90 else "N/A"

    # Max overshoot
    peak_idx = int(np.argmax(velocity))
    peak_v = float(velocity[peak_idx])
    if peak_v > target_v:
        overshoot_pct = ((peak_v - target_v) / target_v) * 100.0
        pt_peak = (time[peak_idx], peak_v)
        overshoot_str = f"{overshoot_pct:.2f}%"
    else:
        overshoot_pct = 0.0
        pt_peak = None
        overshoot_str = "0.00%"

    # Settling time (within +/- 1% error band of target_v)
    tolerance = SETTLING_TOLERANCE * target_v
    out_of_band = np.where(np.abs(velocity - target_v) > tolerance)[0]
    if len(out_of_band) == 0:
        pt_settle = (time[0], velocity[0])
        settle_str = f"{time[0]:.2f} s"
    elif out_of_band[-1] < len(velocity) - 1:
        s_idx = out_of_band[-1] + 1
        pt_settle = (time[s_idx], velocity[s_idx])
        settle_str = f"{time[s_idx]:.2f} s"
    else:
        pt_settle = None
        settle_str = "N/A"

    # Ensure error and drag arrays for bounds calculation
    if error is None:
        error = target_v - velocity
    if drag is None:
        drag = np.array([calculate_air_drag(v) for v in velocity])

    # Calculate graph bounds so curves never exceed window limits
    vel_min = float(np.min(velocity))
    vel_max = float(np.max(velocity))
    err_min = float(np.min(error))
    err_max = float(np.max(error))

    y1_data_min = min(vel_min, err_min)
    y1_data_max = max(vel_max, err_max, float(target_v))
    y1_span = max(y1_data_max - y1_data_min, 1.0)

    # Primary y-axis (Velocity & Error) bounds
    if y1_data_min >= 0.0:
        y1_min = 0.0
    else:
        y1_min = y1_data_min - 0.08 * y1_span
    y1_max = y1_data_max + 0.12 * y1_span

    # Secondary y-axis (Aerodynamic Drag Force) bounds
    drag_data_min = float(np.min(drag))
    drag_data_max = float(np.max(drag))
    drag_span = max(drag_data_max - drag_data_min, 1.0)

    if drag_data_min >= 0.0:
        y2_min = 0.0
    else:
        y2_min = drag_data_min - 0.08 * drag_span
    y2_max = max(10.0, drag_data_max + 0.15 * drag_span)

    # Time (x-axis) bounds
    x_min = float(np.min(time))
    x_max = float(np.max(time))

    bounds = {
        "y1_min": y1_min,
        "y1_max": y1_max,
        "y2_min": y2_min,
        "y2_max": y2_max,
        "x_min": x_min,
        "x_max": x_max,
    }

    metrics = {
        "rise": rise_str,
        "settle": settle_str,
        "overshoot": overshoot_str,
    }
    points = {
        "pt_10": pt_10,
        "pt_90": pt_90,
        "pt_peak": pt_peak,
        "pt_settle": pt_settle,
    }
    return metrics, points, bounds


def create_response_marker(axis, color: str) -> tuple:
    """Create a guide line, point, and label for one response event."""
    guide, = axis.plot([], [], ":", color=color, linewidth=1.5, alpha=0.85, zorder=3)
    point, = axis.plot([], [], "o", color=color, markersize=5, zorder=4)
    label = axis.annotate(
        "", xy=(0, 0), xycoords=("data", "axes fraction"),
        xytext=(0, -8), textcoords="offset points", ha="left", va="top",
        rotation=-45, fontsize=8, fontweight="bold", color=color, visible=False,
    )
    return guide, point, label


def update_response_marker(axis, artists: tuple, coordinate, title: str) -> None:
    """Move an event marker, or hide it if that event was not reached."""
    guide, point, label = artists
    for artist in artists:
        artist.set_visible(coordinate is not None)
    if coordinate is None:
        return
    time, velocity = coordinate
    guide.set_data([time, time], [axis.get_ylim()[0], velocity])
    point.set_data([time], [velocity])
    label.xy = (time, 0)
    label.set_text(f"{title} ({time:.2f} s)")


# Initial simulation run and metrics calculation
velocity_over_time, drag_over_time, error_over_time = run_simulation(K_P, K_I, K_D)
metrics, points, bounds = calculate_metrics(
    dt_axis, velocity_over_time, DESIRED_V, error=error_over_time, drag=drag_over_time
)

# Single graph figure setup with space for right-side controls and bottom fields
fig, ax1 = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(left=0.08, right=0.73, top=0.92, bottom=0.25)

# Primary y-axis: Velocity (m/s) and Error (m/s)
line_vel, = ax1.plot(dt_axis, velocity_over_time, label="Velocity (m/s)", color="tab:blue", linewidth=2)
line_err, = ax1.plot(dt_axis, error_over_time, label="Error (m/s)", color="tab:orange", linestyle="--", linewidth=1.8)
line_target = ax1.axhline(DESIRED_V, color="gray", linestyle=":", alpha=0.6, label=f"Target ({DESIRED_V:g} m/s)")
ax1.set_xlabel("Time (s)", labelpad=42, fontsize=10)
ax1.set_ylabel("Velocity / Error (m/s)")
ax1.set_title("Vehicle Velocity, Error, and Aerodynamic Drag over Time")
ax1.grid(True)
ax1.set_xlim(bounds["x_min"], bounds["x_max"])
ax1.set_ylim(bounds["y1_min"], bounds["y1_max"])

# Secondary y-axis: Aerodynamic Drag Force (N)
ax2 = ax1.twinx()
line_drag, = ax2.plot(dt_axis, drag_over_time, label="Drag Force (N)", color="tab:green", linewidth=2)
ax2.set_ylabel("Drag Force (N)", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax2.set_ylim(bounds["y2_min"], bounds["y2_max"])

# Combined legend placed in the right center
lines = [line_vel, line_err, line_drag, line_target]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="center right")

# Metric display fields placed under the bottom of the graph
bbox_props = dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc", linewidth=1.2)
txt_rise = fig.text(0.19, 0.04, f"Rise Time (10-90%)\n{metrics['rise']}", ha="center", va="center", fontsize=10, bbox=bbox_props)
txt_settle = fig.text(0.41, 0.04, f"{SETTLING_LABEL}\n{metrics['settle']}", ha="center", va="center", fontsize=10, bbox=bbox_props)
txt_overshoot = fig.text(0.63, 0.04, f"Max Overshoot\n{metrics['overshoot']}", ha="center", va="center", fontsize=10, bbox=bbox_props)

# The same artists and update logic serve all four response events.
marker_styles = {
    "pt_10": ("Rise Start (10%)", "purple"),
    "pt_90": ("Rise End (90%)", "purple"),
    "pt_peak": ("Max Overshoot", "crimson"),
    "pt_settle": (SETTLING_LABEL, "darkcyan"),
}
markers = {
    name: create_response_marker(ax1, color)
    for name, (_, color) in marker_styles.items()
}


def update_markups(points: dict) -> None:
    """Refresh the response markers using the current graph limits."""
    for name, (title, _) in marker_styles.items():
        update_response_marker(ax1, markers[name], points[name], title)


# Initialize markups
update_markups(points)

# PID gain text fields on the right
input_status = fig.text(0.90, 0.30, "", fontsize=8, color="crimson", ha="center")
fig.text(0.91, 0.68, "PID Gains", fontsize=12, fontweight="bold", ha="center")

ax_kp = fig.add_axes([0.87, 0.59, 0.08, 0.045])
ax_ki = fig.add_axes([0.87, 0.51, 0.08, 0.045])
ax_kd = fig.add_axes([0.87, 0.43, 0.08, 0.045])

tb_kp = TextBox(ax_kp, "K_P ", initial=str(K_P))
tb_ki = TextBox(ax_ki, "K_I ", initial=str(K_I))
tb_kd = TextBox(ax_kd, "K_D ", initial=str(K_D))

fig.text(0.91, 0.37, "(Press Enter to update)", fontsize=8.5, color="gray", ha="center")


def update_plot(_=None):
    """
    Reads PID gains from the text boxes, re-runs the simulation, and updates the plot, markups, and metrics.
    Triggered when a user enters a new gain value into any PID text box.

    Inputs:
    _: text string passed by TextBox on_submit event (optional, unused)

    Outputs:
    None, but updates the plot lines, markup indicators, metric displays, and redraws the canvas
    """
    try:
        kp = float(tb_kp.text)
        ki = float(tb_ki.text)
        kd = float(tb_kd.text)
        vel, drag, err = run_simulation(kp, ki, kd)
    except (ValueError, OverflowError):
        input_status.set_text("Enter finite gains.\nRun must remain finite.")
        fig.canvas.draw_idle()
        return

    input_status.set_text("")

    line_vel.set_ydata(vel)
    line_err.set_ydata(err)
    line_drag.set_ydata(drag)

    # Recalculate metrics, markups, and bounds
    metrics, points, bounds = calculate_metrics(dt_axis, vel, DESIRED_V, error=err, drag=drag)

    # Apply calculated bounds so curves never exceed window limits
    ax1.set_ylim(bounds["y1_min"], bounds["y1_max"])
    ax2.set_ylim(bounds["y2_min"], bounds["y2_max"])

    update_markups(points)

    # Update bottom cards
    txt_rise.set_text(f"Rise Time (10-90%)\n{metrics['rise']}")
    txt_settle.set_text(f"{SETTLING_LABEL}\n{metrics['settle']}")
    txt_overshoot.set_text(f"Max Overshoot\n{metrics['overshoot']}")

    fig.canvas.draw_idle()


tb_kp.on_submit(update_plot)
tb_ki.on_submit(update_plot)
tb_kd.on_submit(update_plot)

plt.show()