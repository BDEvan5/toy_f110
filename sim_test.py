from matplotlib import pyplot as plt
import numpy as np 

from src.Simulator import TrackSim, ForestSim
import src.LibFunctions as lib


def CorridorCS(obs):
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

    return [v_ref, delta_ref]



def track_sim_test():
    sim_conf = lib.load_conf("std_config")
    map_name = "porto"

    env = TrackSim(sim_conf, map_name)

    done, state, score = False, env.reset(None), 0.0
    while not done:
        action = CorridorCS(state)
        s_p, r, done, _ = env.step(action)
        score += r
        state = s_p

    print(f"Score: {score}")
    env.history.show_history()
    env.render(True)


def forest_sim_test():
    sim_conf = lib.load_conf("std_config")
    map_name = "forest"

    env = ForestSim(sim_conf, map_name)

    done, state, score = False, env.reset(), 0.0
    while not done:
        action = CorridorCS(state)
        s_p, r, done, _ = env.step(action)
        # env.render(False)
        score += r
        state = s_p

    print(f"Score: {score}")
    env.history.show_history()
    env.render(wait=True)




if __name__ == "__main__":
    # track_sim_test()
    forest_sim_test()
