from matplotlib import pyplot as plt
import numpy as np 

from toy_f110.Simulator import TrackSim, ForestSim
import toy_f110.LibFunctions as lib




def follow_the_finder(obs):
    """
    Simple follow the longest range finder planner
    """
    ranges = obs[5:]
    max_range = np.argmax(ranges)

    w_range = np.argmax(ranges)

    max_range = int(round(w_range))

    dth = (np.pi * 2/ 3) / 9
    theta_dot = dth * max_range - np.pi/3

    ld = 0.5 # lookahead distance
    delta_ref = np.arctan(2*0.33*np.sin(theta_dot)/ld)
    delta_ref = np.clip(delta_ref, -0.4, 0.4)

    v_ref = 5

    return [delta_ref, v_ref]



def track_sim_test():
    """
    Test for the track simulator
    """
    sim_conf = lib.load_conf("toy_f110", "std_config")
    map_name = "porto"

    env = TrackSim(map_name, sim_conf)

    done, state, score = False, env.reset(True), 0.0
    while not done:
        action = follow_the_finder(state)
        s_p, r, done, _ = env.step_plan(action)
        score += r
        state = s_p

    print(f"Score: {score}")
    # env.history.show_history()
    env.render(True)


def forest_sim_test():
    """
    Test for the Forest simulator
    """
    sim_conf = lib.load_conf("toy_f110", "std_config")
    map_name = "forest"

    env = ForestSim(map_name)

    done, state, score = False, env.reset(), 0.0
    while not done:
        action = follow_the_finder(state)
        # s_p, r, done, _ = env.step(action)
        s_p, r, done, _ = env.step_plan(action)
        # s_p, r, done, _ = env.step_control(action)
        # env.render(False)
        score += r
        state = s_p

    print(f"Score: {score}")
    # env.history.show_history()
    env.render(wait=True)




if __name__ == "__main__":
    # track_sim_test()
    forest_sim_test()
