import numpy as np 
from matplotlib import pyplot as plt
import os

import toy_f110.LibFunctions as lib
from toy_f110.SimMaps import TrackMap, ForestMap


class CarModel:
    """
    A simple class which holds the state of a car and can update the dynamics based on the bicycle model

    Data Members:
        x: x location of vehicle on map
        y: y location of vehicle on map
        theta: orientation of vehicle
        velocity: 
        steering: delta steering angle
        th_dot: the change in orientation due to steering

    """
    def __init__(self, sim_conf):
        """
        Init function

        Args:
            sim_conf: a config namespace with relevant car parameters
        """
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0
        self.th_dot = 0

        self.prev_loc = 0

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v
        self.max_friction_force = self.mass * self.mu * 9.81

    def update_kinematic_state(self, a, d_dot, dt):
        """
        Updates the internal state of the vehicle according to the kinematic equations for a bicycle model

        Args:
            a: acceleration
            d_dot: rate of change of steering angle
            dt: timestep in seconds

        """
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.th_dot = theta_dot
        dth = theta_dot * dt
        self.theta = lib.add_angles_complex(self.theta, dth)

        a = np.clip(a, -self.max_a, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

    def get_car_state(self):
        """
        Returns the state of the vehicle as an array

        Returns:
            state: [x, y, theta, velocity, steering]

        """
        state = []
        state.append(self.x) #0
        state.append(self.y)
        state.append(self.theta) # 2
        state.append(self.velocity) #3
        state.append(self.steering)  #4

        state = np.array(state)

        return state

    def reset_state(self, start_pose):
        """
        Resets the state of the vehicle

        Args:
            start_pose: the starting, [x, y, theta] to reset to
        """
        self.x = start_pose[0]
        self.y = start_pose[1]
        self.theta = start_pose[2]
        self.velocity = 0
        self.steering = 0
        self.prev_loc = [self.x, self.y]

class ScanSimulator:
    """
    A simulation class for a lidar scanner

    Parameters:
        number of beams: number of laser scans to return
        fov: field of view
        std_noise: the standard deviation of the noise which is added to the beams.

    Data members:
        scan_output: the last scan which was returned

    External Functions:
        set_check_fcn(fcn): give a function which can be called to check if a certain location falls in the driveable area
        get_scan(pose): returns a scan

    TODO: njit functions, precompute sines and cosines, improve the step searching

    """
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise
        self.rng = np.random.default_rng(seed=12345)

        self.dth = self.fov / (self.number_of_beams -1)
        self.scan_output = np.zeros(number_of_beams)

        self.step_size = 0.2
        self.n_searches = 20

        self.race_map = None
        self.x_bound = [1, 99]
        self.y_bound = [1, 99]

    def get_scan(self, pose):
        """
        A simple function to get a laser scan reading for a given pose.
        Adds noise with a std deviation as in the config file

        Args:
            pose: [x, y, theta] of the vehicle at present state
        
        Returns:
            scan: array of the output from the laser scan.
        """
        x = pose[0]
        y = pose[1]
        theta = pose[2]
        for i in range(self.number_of_beams):
            scan_theta = theta + self.dth * i - self.fov/2
            self.scan_output[i] = self._trace_ray(x, y, scan_theta)

        # noise = self.rng.normal(0., self.std_noise, size=self.number_of_beams)
        # self.scan_output = self.scan_output + noise

        return self.scan_output

    def _trace_ray(self, x, y, theta):
        """
        returns the % of the max range finder range which is in the driveable area for a single ray

        Args:
            x: x location
            y: y location
            theta: angle of orientation

        TODO: use pre computed sins and cosines
        """
        # obs_res = 10
        for j in range(self.n_searches): # number of search points
            fs = self.step_size * (j + 1)  # search from 1 step away from the point
            dx =  [np.sin(theta) * fs, np.cos(theta) * fs]
            search_val = lib.add_locations([x, y], dx)
            if self._check_location(search_val):
                break       

        ray = (j) / self.n_searches #* (1 + np.random.normal(0, self.std_noise))
        return ray

    def set_check_fcn(self, check_fcn):
        """
        Sets the function which is used interally to see if a location is driveable

        Args: 
            check_fcn: a function which can be called with a location as an argument
        """
        self._check_location = check_fcn


class SimHistory:
    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []


        self.ctr = 0

    def save_history(self):
        pos = np.array(self.positions)
        vel = np.array(self.velocities)
        steer = np.array(self.steering)
        obs = np.array(self.obs_locations)

        d = np.concatenate([pos, vel[:, None], steer[:, None]], axis=-1)

        d_name = 'Vehicles/TrainData/' + f'data{self.ctr}'
        o_name = 'Vehicles/TrainData/' + f"obs{self.ctr}"
        np.save(d_name, d)
        np.save(o_name, obs)

    def reset_history(self):
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.ctr += 1

    def show_history(self, vs=None):
        plt.figure(1)
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        if vs is not None:
            r = len(vs) / len(self.velocities)
            new_vs = []
            for i in range(len(self.velocities)):
                new_vs.append(vs[int(round(r*i))])
            plt.plot(new_vs)
            plt.legend(['Actual', 'Planned'])
        plt.pause(0.001)

    def show_forces(self):
        mu = self.sim_conf['car']['mu']
        m = self.sim_conf['car']['m']
        g = self.sim_conf['car']['g']
        l_f = self.sim_conf['car']['l_f']
        l_r = self.sim_conf['car']['l_r']
        f_max = mu * m * g
        f_long_max = l_f / (l_r + l_f) * f_max

        self.velocities = np.array(self.velocities)
        self.thetas = np.array(self.thetas)

        # divide by time taken for change to get per second
        t = self.sim_conf['sim']['timestep'] * self.sim_conf['sim']['update_f']
        v_dot = (self.velocities[1:] - self.velocities[:-1]) / t
        oms = (self.thetas[1:] - self.thetas[:-1]) / t

        f_lat = oms * self.velocities[:-1] * m
        f_long = v_dot * m
        f_total = (f_lat**2 + f_long**2)**0.5

        plt.figure(3)
        plt.clf()
        plt.title("Forces (lat, long)")
        plt.plot(f_lat)
        plt.plot(f_long)
        plt.plot(f_total, linewidth=2)
        plt.legend(['Lat', 'Long', 'total'])
        plt.plot(np.ones_like(f_lat) * f_max, '--')
        plt.plot(np.ones_like(f_lat) * f_long_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_long_max, '--')
        plt.pause(0.001)


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
    def __init__(self, env_map: TrackMap, done_fcn):
        """
        Init function

        Args:
            env_map: an env_map object which holds a map and has mapping functions
            done_fcn: a function which checks the state of the simulation for episode completeness
        """
        self.done_fcn = done_fcn
        self.env_map = env_map
        self.sim_conf = self.env_map.sim_conf #TODO: don't store the conf file, just use and throw away.
        self.n_obs = self.env_map.n_obs

        self.timestep = self.sim_conf.time_step
        self.max_steps = self.sim_conf.max_steps
        self.plan_steps = self.sim_conf.plan_steps

        self.car = CarModel(self.sim_conf)
        self.scan_sim = ScanSimulator(self.sim_conf.n_beams)
        self.scan_sim.set_check_fcn(self.env_map.check_scan_location)

        self.done = False
        self.colission = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.history = SimHistory(self.sim_conf)
        self.done_reason = ""

    def step_control(self, action):
        """
        Steps the simulator for a single step

        Args:
            action: [steer, speed]
        """
        d_ref = action[0]
        v_ref = action[1]
        acceleration, steer_dot = self.control_system(v_ref, d_ref)
        self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)

        return self.done_fcn()

    def step_plan(self, action):
        """
        Takes multiple control steps based on the number of control steps per planning step

        Args:
            action: [steering, speed]
            done_fcn: a no arg function which checks if the simulation is complete
        """

        for _ in range(self.plan_steps):
            if self.step_control(action):
                break

        self.record_history(action)

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def record_history(self, action):
        self.action = action
        self.history.velocities.append(self.car.velocity)
        self.history.steering.append(self.car.steering)
        self.history.positions.append([self.car.x, self.car.y])
        self.history.thetas.append(self.car.theta)

    def control_system(self, v_ref, d_ref):
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

        kp_a = 10
        a = (v_ref - self.car.velocity) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - self.car.steering) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

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
        self.action_memory = []
        self.steps = 0
        self.reward = 0

        #TODO: move this reset to inside car
        self.car.reset_state(self.env_map.start_pose)


        self.history.reset_history()

        if add_obs:
            self.env_map.add_obstacles()

        return self.get_observation()

    def render(self, wait=False):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)

        xs, ys = self.env_map.convert_positions(self.history.positions)
        plt.plot(xs, ys, 'r', linewidth=3)
        plt.plot(xs, ys, '+', markersize=12)

        x, y = self.env_map.xy_to_row_column([self.car.x, self.car.y])
        plt.plot(x, y, 'x', markersize=20)

        text_x = self.env_map.map_width + 1
        text_y = self.env_map.map_height / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def min_render(self, wait=False):
        """
        TODO: deprecate
        """
        fig = plt.figure(4)
        plt.clf()  

        ret_map = self.env_map.scan_map
        plt.imshow(ret_map)

        # plt.xlim([0, self.env_map.width])
        # plt.ylim([0, self.env_map.height])

        s_x, s_y = self.env_map.convert_to_plot(self.env_map.start)
        plt.plot(s_x, s_y, '*', markersize=12)

        c_x, c_y = self.env_map.convert_to_plot([self.car.x, self.car.y])
        plt.plot(c_x, c_y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            r_x, r_y = self.env_map.convert_to_plot(range_val)
            x = [c_x, r_x]
            y = [c_y, r_y]

            plt.plot(x, y)

        for pos in self.action_memory:
            p_x, p_y = self.env_map.convert_to_plot(pos)
            plt.plot(p_x, p_y, 'x', markersize=6)

        text_start = self.env_map.width + 10
        spacing = int(self.env_map.height / 10)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_start, spacing*1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_start, spacing*2, s) 
        s = f"Done: {self.done}"
        plt.text(text_start, spacing*3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_start, spacing*4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_start, spacing*5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_start, spacing*6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_start, spacing*7, s) 
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(text_start, spacing*8, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, spacing*9, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
  
    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        car_obs = self.car.get_car_state()
        pose = car_obs[0:3]
        scan = self.scan_sim.get_scan(pose)

        observation = np.concatenate([car_obs, scan, [self.reward]])
        return observation


class TrackSim(BaseSim):
    """
    Simulator for Race Tracks, inherits from the base sim and adds a layer for use with a race track for f110

    Important to note the check_done function which checks if the episode is complete
        
    """
    def __init__(self, map_name, sim_conf=None):
        """
        Init function

        Args:
            map_name: name of map to use.
            sim_conf: config file for simulation

        """
        if sim_conf is None:
            path = os.path.dirname(__file__)
            sim_conf = lib.load_conf(path, "std_config")

        env_map = TrackMap(sim_conf, map_name)
        BaseSim.__init__(self, env_map, self.check_done_reward_track_train)
        self.end_distance = sim_conf.end_distance

    def check_done_reward_track_train(self):
        """
        Checks if the race lap is complete

        Returns
            Done flag
        """
        self.reward = 0 # normal
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.colission = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        # horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        # self.y_forces.append(horizontal_force)
        # if horizontal_force > self.car.max_friction_force:
            # self.done = True
            # self.reward = -1
            # self.done_reason = f"Friction limit reached: {horizontal_force} > {self.car.max_friction_force}"
        if self.steps > self.max_steps:
            self.done = True
            self.done_reason = f"Max steps"

        car = [self.car.x, self.car.y]
        cur_end_dis = lib.get_distance(car, self.env_map.start_pose[0:2]) 
        if cur_end_dis < self.end_distance and self.steps > 100:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete, d: {cur_end_dis}"


        return self.done



class ForestSim(BaseSim):
    """
    Simulator for Race Tracks

    Data members:
        map_name: name of the map to be used. Forest yaml file which stores the parameters for the forest. No image is required.

    """
    def __init__(self, map_name, sim_conf=None):
        """
        Init function

        Args:
            map_name: name of forest map to use.
            sim_conf: config file for simulation
        """
        if sim_conf is None:
            path = os.path.dirname(__file__)
            sim_conf = lib.load_conf(path, "std_config")

        env_map = ForestMap(sim_conf, map_name)
        BaseSim.__init__(self, env_map, self.check_done_forest)

    def check_done_forest(self):
        self.reward = 0 # normal
        # check if finished lap
        dx = self.car.x - self.env_map.start_pose[0]
        dx_lim = self.env_map.forest_width * 0.5
        if dx < dx_lim and self.car.y > self.env_map.end_y:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete"

        # check crash
        elif self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        # horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        # check forces
        # if horizontal_force > self.car.max_friction_force:
            # self.done = True
            # self.reward = -1
            # print(f"ThDot: {self.car.th_dot} --> Vel: {self.car.velocity}")
            # self.done_reason = f"Friction: {horizontal_force} > {self.car.max_friction_force}"

        # check steps
        elif self.steps > self.max_steps:
            self.done = True
            self.reward = -1
            self.done_reason = f"Max steps"
        # check orientation
        elif abs(self.car.theta) > 0.66*np.pi:
            self.done = True
            self.done_reason = f"Vehicle turned around"
            self.reward = -1




          

