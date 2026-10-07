import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
from pid_template import (
    make_car,
    update,
    calculate_desired_acceleration,
    acceleration_to_throttle_percentage,
)

# Simulation parameters and initial gains
K_P = 1.0
K_I = 0.0001
K_D = 0.025

DESIRED_TIME_S = 55
DT = 0.1
STEPS = int(DESIRED_TIME_S / DT)
dt_axis = np.linspace(0, DESIRED_TIME_S, STEPS)


def run_simulation(kp: float, ki: float, kd: float):
    car = make_car(desired_v=20.0, dt=DT)
    position = np.zeros(STEPS)
    velocity = np.zeros(STEPS)
    error = np.zeros(STEPS)

    for i in range(STEPS):
        position[i] = car["x"]
        velocity[i] = car["v"]
        desired_a, error[i] = calculate_desired_acceleration(car, kp, ki, kd)
        throttle_p = acceleration_to_throttle_percentage(desired_a)
        update(car, throttle_p)

    return position, velocity, error


# Initial run
position_over_dt, velocity_over_dt, error_over_dt = run_simulation(K_P, K_I, K_D)

# Figure setup with space on the right for controls
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
plt.subplots_adjust(left=0.08, right=0.73, hspace=0.25)

# Subplot 1: Velocity over time with Error layered
line_vel, = ax1.plot(dt_axis, velocity_over_dt, label="Velocity (m/s)",     
color="tab:blue")
line_err1, = ax1.plot(dt_axis, error_over_dt, label="Error (m/s)",
color="tab:orange", linestyle="--")
ax1.set_ylabel("Velocity / Error (m/s)")
ax1.set_title("Velocity & Error over Time")
ax1.legend(loc="upper right")
ax1.grid(True)

# Subplot 2: Position over time with Error layered
line_pos, = ax2.plot(dt_axis, position_over_dt, label="Position (m)", color="tab:green")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (m)", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax2.grid(True)

# Secondary y-axis for error on position plot
ax2_err = ax2.twinx()
line_err2, = ax2_err.plot(dt_axis, error_over_dt, label="Error (m/s)", color="tab:orange", linestyle="--")
ax2_err.set_ylabel("Error (m/s)", color="tab:orange")
ax2_err.tick_params(axis="y", labelcolor="tab:orange")

# Combined legend for the bottom subplot
lines_2 = [line_pos, line_err2]
labels_2 = [line.get_label() for line in lines_2]
ax2.legend(lines_2, labels_2, loc="upper left")
ax2.set_title("Position & Error over Time")

# PID gain text fields on the right
fig.text(0.91, 0.68, "PID Gains", fontsize=12, fontweight="bold", ha="center")

ax_kp = fig.add_axes([0.87, 0.59, 0.08, 0.045])
ax_ki = fig.add_axes([0.87, 0.51, 0.08, 0.045])
ax_kd = fig.add_axes([0.87, 0.43, 0.08, 0.045])

tb_kp = TextBox(ax_kp, "K_P ", initial=str(K_P))
tb_ki = TextBox(ax_ki, "K_I ", initial=str(K_I))
tb_kd = TextBox(ax_kd, "K_D ", initial=str(K_D))


def update_plot(_=None):
    try:
        kp = float(tb_kp.text)
        ki = float(tb_ki.text)
        kd = float(tb_kd.text)
    except ValueError:
        return

    pos, vel, err = run_simulation(kp, ki, kd)

    line_vel.set_ydata(vel)
    line_err1.set_ydata(err)
    line_pos.set_ydata(pos)
    line_err2.set_ydata(err)

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax2_err.relim()
    ax2_err.autoscale_view()

    fig.canvas.draw_idle()


tb_kp.on_submit(update_plot)
tb_ki.on_submit(update_plot)
tb_kd.on_submit(update_plot)

plt.show()