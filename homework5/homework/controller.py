import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    # Initialize constants
    goal_velocity = 29 # Target velocity
    steering_scaling_factor = 2.75 # Helps determine the steering direction
    drift_threshold = 0.77 # Helps determine if the car should drift

    # Calculate steering direction
    steering_direction = aim_point[0] * steering_scaling_factor # x-coordinate * scaling factor
    action.steer = max(-1, min(1, steering_direction)) # Makes the car turn in correct direction

    # Determine if needs to accelerate
    if current_vel < goal_velocity:
        action.acceleration = 0.75 # Accelerate
    else:
        action.acceleration = 0.0 # Do not accelerate

    # Determine if should brake
    if current_vel > goal_velocity + 3:
        action.brake = True
    else:
        action.brake = False
    
    # Determine if drift
    if abs(steering_direction) > drift_threshold:
        action.acceleration = 0.01
        action.brake = True
        action.drift = True
        action.brake = True
    else:
        action.drift = False

    # Determine if boost/nitro
    if aim_point[0] > -0.02 and aim_point[0] < 0.02:
        action.nitro = True
    else:
        action.nitro = False

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
