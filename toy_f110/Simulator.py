import numpy as np 
from matplotlib import pyplot as plt
import os

import toy_f110.LibFunctions as lib
from toy_f110.SimMaps import TrackMap, ForestMap


class CarModel:
    """
    A simple class which holds the state of a car and can update the dynamics based on the bicycle model

    Args:
        sim_conf: a config namespace with relevant car parameters

    """
    def __init__(self, sim_conf):
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
        updates the state of the vehicle according to the kinematic equations for a bicycle model

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
        state = []
        state.append(self.x) #0
        state.append(self.y)
        state.append(self.theta) # 2
        state.append(self.velocity) #3
        state.append(self.steering)  #4

        state = np.array(state)

        return state


class ScanSimulator:
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
        a simple function to get a laser scan reading for a given pose

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
            self.scan_output[i] = self.trace_ray(x, y, scan_theta)

        noise = self.rng.normal(0., self.std_noise, size=self.number_of_beams)
        self.scan_output = self.scan_output + noise

        return self.scan_output

    def trace_ray(self, x, y, theta, noise=True):
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
    """
    def __init__(self, env_map: TrackMap):
        self.env_map = env_map
        self.sim_conf = self.env_map.sim_conf
        self.n_obs = self.env_map.n_obs

        self.timestep = self.sim_conf.time_step
        self.max_steps = self.sim_conf.max_steps

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

    def base_step(self, action, done_fcn):
        """
        Runs the dynamics step for the simulator
        
        Args:
            action(list(2)): [steering, velocity] references which are executed on the vehicle.
            done_fcn: to be removed, checks when done
        """
        self.steps += 1
        d_ref = action[0]
        v_ref = action[1]

        frequency_ratio = 1 # cs updates per planning update
        self.car.prev_loc = [self.car.x, self.car.y]
        for i in range(frequency_ratio): # TODO: remove this stuff.
            acceleration, steer_dot = self.control_system(v_ref, d_ref)
            self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)
            if done_fcn():
                break

        if action[0] != self.action[0]:
            self.action = action
            self.history.velocities.append(self.car.velocity)
            self.history.steering.append(self.car.steering)
            self.history.positions.append([self.car.x, self.car.y])
            self.history.thetas.append(self.car.theta)
        
            self.action_memory.append([self.car.x, self.car.y])
            #TODO: positions and action mem are the same thing

    def control_system(self, v_ref, d_ref):

        kp_a = 10
        a = (v_ref - self.car.velocity) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - self.car.steering) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def base_reset(self):
        self.done = False
        self.done_reason = "Null"
        self.action_memory = []
        self.steps = 0

        self.history.reset_history()

        return self.get_observation()

    def reset_lap(self):
        self.steps = 0
        self.reward = 0
        self.car.prev_loc = [self.car.x, self.car.y]
        self.history.reset_history()
        self.action_memory.clear()
        self.done = False

    def render(self, wait=False):
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)

        xs, ys = self.env_map.convert_positions(self.action_memory)
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
        car_obs = self.car.get_car_state()
        pose = car_obs[0:3]
        scan = self.scan_sim.get_scan(pose)

        observation = np.concatenate([car_obs, scan, [self.reward]])
        return observation


class TrackSim(BaseSim):
    """
    Simulator for Race Tracks

    Args:
        map_name: name of map to use.
        sim_conf: config file for simulation
    """
    def __init__(self, map_name, sim_conf=None):
        if sim_conf is None:
            path = os.path.dirname(__file__)
            sim_conf = lib.load_conf(path, "std_config")

        env_map = TrackMap(sim_conf, map_name)
        BaseSim.__init__(self, env_map)
        self.end_distance = sim_conf.end_distance

    def step(self, action):
        """
        Steps the track sim by a timestep. Updates the dynamics and then gets and observation and checks the done status
        
        Args:
            action(list(2)): [velocity, steering] references which are executed on the vehicle.
            done_fcn: to be removed, checks when done
        Returns:
            observation
            reward: 1, 0, -1 for lap finished, lap not finished, crash respectively.
            done: if lap complete
            info: None currently.
        """
        d_func = self.check_done_reward_track_train
        self.base_step(action, d_func)

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def reset(self, add_obs=True):
        self.car.x = self.env_map.start_pose[0]
        self.car.y = self.env_map.start_pose[1]
        self.car.prev_loc = [self.car.x, self.car.y]
        self.car.velocity = 0
        self.car.steering = 0
        self.car.theta = self.env_map.start_pose[2]

        self.reset_lap()

        #TODO: combine with reset lap that it can be called every lap and do the right thing

        if add_obs:
            self.env_map.add_obstacles()
        
        return self.get_observation()

    def check_done_reward_track_train(self):
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
    """
    def __init__(self, sim_conf, map_name):
        env_map = ForestMap(sim_conf, map_name)
        BaseSim.__init__(self, env_map)

    def step(self, action, dt=None):
        if dt is not None:
            self.dt = dt / 10 # 10 is the current frequency ratio

        # self.env_map.update_obs_cars(self.timestep)
        self.base_step(action, self.check_done_forest)

        # self.check_done_forest()

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def reset(self, add_obs=True):
        self.car.x = self.env_map.start_pose[0]
        self.car.y = self.env_map.start_pose[1]
        self.car.prev_loc = [self.car.x, self.car.y]
        self.car.velocity = 0
        self.car.steering = 0
        self.car.theta = self.env_map.start_pose[2]

        if add_obs:
            self.env_map.generate_forest()
        
        return self.base_reset()

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
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        # check forces
        # if horizontal_force > self.car.max_friction_force:
            # self.done = True
            # self.reward = -1
            # print(f"ThDot: {self.car.th_dot} --> Vel: {self.car.velocity}")
            # self.done_reason = f"Friction: {horizontal_force} > {self.car.max_friction_force}"
        # check steps
        if self.steps > self.max_steps:
            self.done = True
            self.reward = -1
            self.done_reason = f"Max steps"
        # check orientation
        if abs(self.car.theta) > 0.66*np.pi:
            self.done = True
            self.done_reason = f"Vehicle turned around"
            self.reward = -1




          

