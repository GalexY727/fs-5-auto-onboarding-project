import numpy as np


def make_car(desired_v: float = 20.0, dt: float = 0.1) -> dict:
    """
    Generates a dictionary that holds all the car's values and tracks state variables.

    Inputs:
    desired_v: float, desired velocity of your car to maintain (default 20.0)
    dt: float, time step increment for each update step (default 0.1)

    Outputs:
    dict: dictionary containing initialized vehicle state variables
    """
    car_state_dictionary : dict[str, float] = {
        "v" : 0, #velocity of your car 
        "a" : 0, #acceleration of your car
        "t" : 0, #time of your car
        "x" : 0, #position of your car
        "dt" : dt, #time step of your car, how much the time changes every time you update/step
        "desired_v" : desired_v, #desired velocity of your car, the velocity you want to maintain
        "step" : 0,
    
        #hint: use these variables in the integral and derivative portion of your PID control (steps 5 and 6 )
        "error_prev" : 0,
        "net_integral" : 0.0
    }
    return car_state_dictionary

def update(car: dict, throttle_perc: float, mass: float = 1000, max_throttle_force: float = 5000, friction: float = 2.0) -> None:
    """
    Updates the car's state variables based on the throttle percentage.
    Use this function after finding throttle percentage to update the car's state variables.

    Inputs:
    car: dictionary containing the car's state variables
    throttle_perc: float, throttle percentage (-1 to 1)
    mass: float, mass of the vehicle in kg (default 1000)
    max_throttle_force: float, maximum force exerted by throttle in N (default 5000)
    friction: float, resistive friction deceleration in m/s^2 (default 2.0)

    Outputs:
    None, but updates the car's state variables
    """
    force = throttle_perc * max_throttle_force
    car["a"] = (force / mass) - friction
    car["v"] += car["a"] * car["dt"]
    car["x"] += car["v"] * car["dt"]
    car["t"] += car["dt"]
    car["step"] += 1


def calculate_desired_acceleration(car: dict, K_P: float, K_I: float = 0.0, K_D: float = 0.0) -> tuple[float, float]:
    """
    Calculates desired acceleration using a PID controller based on velocity error.

    Inputs:
    car: dictionary containing the car's state variables
    K_P: float, proportional gain
    K_I: float, integral gain (default 0.0)
    K_D: float, derivative gain (default 0.0)

    Outputs:
    tuple[float, float]: (desired_a, error) desired acceleration and current velocity error
    """
    v_error = car["desired_v"] - car["v"]
    gain = K_P * v_error
    car["net_integral"] += K_I * v_error * car["dt"]
    anticipation = K_D * ((v_error - car["error_prev"]) / car["dt"])
    desired_a = gain + anticipation + car["net_integral"] # my goat PDI
    car["error_prev"] = v_error
    return (desired_a, v_error)



def acceleration_to_throttle_percentage(acceleration_desired: float, mass: float = 1000, max_throttle_force: float = 5000) -> float:
    """
    Converts desired acceleration to throttle percentage clipped between -1 and 1.

    Inputs:
    acceleration_desired: float, acceleration command to achieve
    mass: float, mass of the vehicle in kg (default 1000)
    max_throttle_force: float, maximum force exerted by throttle in N (default 5000)

    Outputs:
    float: throttle percentage clipped between -1 and 1 (-100% to 100%)
    """
    max_a = max_throttle_force / mass
    percentage = acceleration_desired / max_a
    return np.clip(percentage)
    