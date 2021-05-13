import numpy as np
from numba import njit
from matplotlib import pyplot as plt

import toy_f110.LibFunctions as lib

from toy_f110.ScanSimulator import ScanSimulator

class BaseSim:
    """
    Base simulator class

    Important parameters:
        timestep: how long the simulation steps for
        max_steps: the maximum amount of steps the sim can take

    Data members:
        car: a model of a car with the ability to update the dynamics
        scan_sim: a simulator for a laser scanner
        action: the current action which has been given
        history: a data logger for the history
    """
    def __init__(self, env_map, done_fcn, sim_conf):
        """
        Init function

        Args:
            env_map: an env_map object which holds a map and has mapping functions
            done_fcn: a function which checks the state of the simulation for episode completeness
        """
        self.done_fcn = done_fcn
        self.env_map = env_map

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.wheelbase = sim_conf.l_r + sim_conf.l_f

        self.timestep = sim_conf.time_step
        self.max_steps = sim_conf.max_steps
        self.plan_steps = sim_conf.plan_steps

        # state = [x, y, theta, velocity, steering, th_dot]
        self.state = np.zeros(5)
        self.scan_sim = ScanSimulator(sim_conf.n_beams)
        self.scan_sim.init_sim_map(env_map)

        self.done = False
        self.colission = False
        self.reward = 0
        self.action = np.zeros((2))
        self.position_history = []
        self.steps = 0

        self.done_reason = ""

    def step(self, action):
        """
        Takes multiple control steps based on the number of control steps per planning step

        Args:
            action: [steering, speed]
            done_fcn: a no arg function which checks if the simulation is complete
        """
        action = np.array(action)
        self.action = action
        for _ in range(self.plan_steps):
            u = control_system(self.state, action, self.max_v, self.max_steer, 8, 3.2)
            self.state = update_kinematic_state(self.state, u, self.timestep, self.wheelbase, self.max_steer, self.max_v)
            self.steps += 1

            if self.done_fcn():
                break

        self.position_history.append(self.state[0:2])

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def reset(self, add_obs=True):
        """
        Resets the simulation

        Args:
            add_obs: a boolean flag if obstacles should be added to the map

        Returns:
            state observation
        """
        self.done = False
        self.done_reason = "Null"
        self.position_history.clear()
        self.steps = 0
        self.reward = 0

        self.state[0:3] = self.env_map.start_pose
        self.state[3:5] = np.zeros(2)

        if add_obs:
            self.env_map.add_obstacles()

        self.scan_sim.dt = self.env_map.set_dt()

        return self.get_observation()

    def render(self, wait=False, name="No vehicle name set"):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)
        plt.title(name)

        xs, ys = self.env_map.convert_positions(self.position_history)
        plt.plot(xs, ys, 'r', linewidth=3)
        plt.plot(xs, ys, '+', markersize=12)

        x, y = self.env_map.xy_to_row_column(self.state[0:2])
        plt.plot(x, y, 'x', markersize=20)

        text_x = self.env_map.map_width + 1
        text_y = self.env_map.map_height / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.state[0]:.2f}, {self.state[1]:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.state[3]:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.state[2] * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.state[4]*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()
     
    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        pose = self.state[0:3]
        scan = self.scan_sim.scan(pose)
        em = self.env_map
        s = calculate_progress(self.state[0:2], em.ref_pts, em.diffs, em.l2s, em.ss_normal)

        observation = {}
        observation['scan'] = scan
        observation['progress'] = s
        observation['state'] = self.state
        observation['reward'] = self.reward

        return observation


@njit(cache=True)
def calculate_progress(point, wpts, diffs, l2s, ss):
    dots = np.empty((wpts.shape[0]-1, ))
    dots_shape = dots.shape[0]
    for i in range(dots_shape):
        dots[i] = np.dot((point - wpts[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0  #np.clip, unsupported
    
    projections = wpts[:-1,:] + (t*diffs.T).T

    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))

    min_dist_segment = np.argmin(dists)
    dist_from_cur_pt = dists[min_dist_segment]

    s = ss[min_dist_segment] + dist_from_cur_pt
    # print(F"{min_dist_segment} --> SS: {ss[min_dist_segment]}, curr_pt: {dist_from_cur_pt}")

    s = s / ss[-1] # scales progress to range [0, 1]

    return s 

#Dynamics functions
@njit(cache=True)
def update_kinematic_state(x, u, dt, whlb, max_steer, max_v):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    dx = np.array([x[3]*np.sin(x[2]), # x
                x[3]*np.cos(x[2]), # y
                x[3]/whlb * np.tan(x[4]), # theta
                u[1], # velocity
                u[0]]) # steering

    new_state = x + dx * dt 

    # check limits
    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state

@njit(cache=True)
def control_system(state, action, max_v, max_steer, max_a, max_d_dot):
    """
    Generates acceleration and steering velocity commands to follow a reference
    Note: the controller gains are hand tuned in the fcn

    Args:
        v_ref: the reference velocity to be followed
        d_ref: reference steering to be followed

    Returns:
        a: acceleration
        d_dot: the change in delta = steering velocity
    """
    # clip action
    v_ref = min(action[1], max_v)
    d_ref = max(action[0], -max_steer)
    d_ref = min(action[0], max_steer)

    kp_a = 10
    a = (v_ref-state[3])*kp_a
    
    kp_delta = 40
    d_dot = (d_ref-state[4])*kp_delta

    # clip actions
    a = min(a, max_a)
    a = max(a, -max_a)
    d_dot = min(d_dot, max_d_dot)
    d_dot = max(d_dot, -max_d_dot)
    
    u = np.array([d_dot, a])

    return u

