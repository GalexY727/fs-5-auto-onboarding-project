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

plt.plot(dt_axis, velocity_over_dt)
plt.show()