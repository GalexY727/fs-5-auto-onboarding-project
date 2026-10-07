import matplotlib.pyplot as plt
import numpy as np
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 1
K_I = 0.0001
K_D = 0.025
 
DESIRED_TIME_S = 55
DT = 0.1
STEPS = int(DESIRED_TIME_S/DT)

car = make_car(desired_v=20.0, dt=DT)

#WRITE CODE HERE
position_over_dt = np.zeros(STEPS)
velocity_over_dt = np.zeros(STEPS)
error_over_dt = np.zeros(STEPS)
dt_axis = np.linspace(0, DESIRED_TIME_S, STEPS)

for i in range(STEPS):
    position_over_dt[i] = car["x"]
    velocity_over_dt[i] = car["v"]
    (desired_a, error_over_dt[i]) = calculate_desired_acceleration(car, K_P, K_I, K_D)
    throttle_p = acceleration_to_throttle_percentage(desired_a)
    update(car, throttle_p);

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Velocity over time with Error layered
ax1.plot(dt_axis, velocity_over_dt, label="Velocity (m/s)", color="tab:blue")
ax1.plot(dt_axis, error_over_dt, label="Error (m/s)", color="tab:orange", linestyle="--")
ax1.set_ylabel("Velocity / Error (m/s)")
ax1.set_title("Velocity & Error over Time")
ax1.legend()
ax1.grid(True)

# Subplot 2: Position over time with Error layered
ax2.plot(dt_axis, position_over_dt, label="Position (m)", color="tab:green")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (m)", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax2.grid(True)

# Secondary y-axis for error on position plot due to scale differences (meters vs m/s)
ax2_err = ax2.twinx()
ax2_err.plot(dt_axis, error_over_dt, label="Error (m/s)", color="tab:orange", linestyle="--")
ax2_err.set_ylabel("Error (m/s)", color="tab:orange")
ax2_err.tick_params(axis="y", labelcolor="tab:orange")

# Combine legend entries for the position subplot
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_2_err, labels_2_err = ax2_err.get_legend_handles_labels()
ax2.legend(lines_2 + lines_2_err, labels_2 + labels_2_err, loc="center right")
ax2.set_title("Position & Error over Time")

plt.tight_layout()
plt.show()