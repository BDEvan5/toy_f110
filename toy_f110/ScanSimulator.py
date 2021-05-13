import numpy as np
from numba import njit
from matplotlib import pyplot as plt

import toy_f110.LibFunctions as lib




class ScanSimulator:
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise
        self.rng = np.random.default_rng(seed=12345)

        self.dth = self.fov / (self.number_of_beams -1)

        self.map_height = None
        self.map_width = None
        self.resoltuion = None
        self.orig_x = 0
        self.orig_y = 0

        self.eps = 0.01
        self.max_range = 10

    def reset_n_beams(self, n_beams):
        self.number_of_beams = n_beams
        self.dth = self.fov / (self.number_of_beams -1)

    def init_sim_map(self, env_map):
        self.map_height = env_map.map_height
        self.map_width = env_map.map_width
        self.resoltuion = env_map.resolution
        self.orig_x = env_map.origin[0]
        self.orig_y = env_map.origin[1]

        self.dt = env_map.dt_img

    def scan(self, pose):
        scan = get_scan(pose, self.number_of_beams, self.dth, self.dt, self.fov, self.orig_x, self.orig_y, self.resoltuion, self.map_height, self.map_width, self.eps, self.max_range)

        noise = self.rng.normal(0., self.std_noise, size=self.number_of_beams)
        final_scan = scan + noise
        return final_scan


@njit(cache=True)
def get_scan(pose, num_beams, th_increment, dt, fov, orig_x, orig_y, resolution, height, width, eps, max_range):
    x = pose[0]
    y = pose[1]
    theta = pose[2]

    scan = np.empty(num_beams)

    for i in range(num_beams):
        scan_theta = theta + th_increment * i - fov/2
        scan[i] = trace_ray(x, y, scan_theta, dt, orig_x, orig_y, resolution, height, width, eps, max_range)

    return scan


@njit(cache=True)
def trace_ray(x, y, theta, dt, orig_x, orig_y, resolution, height, width, eps, max_range):
    s = np.sin(theta)
    c = np.cos(theta)

    distance_to_nearest = get_distance_dt(x, y, dt, orig_x, orig_y, resolution, height, width)
    total_distance = distance_to_nearest

    while distance_to_nearest > eps and total_distance <= max_range:
        x += distance_to_nearest * s
        y += distance_to_nearest * c


        distance_to_nearest = get_distance_dt(x, y, dt, orig_x, orig_y, resolution, height, width)
        total_distance += distance_to_nearest

    if total_distance > max_range:
        total_distance = max_range

    return total_distance

@njit(cache=True)
def get_distance_dt(x, y, dt, orig_x, orig_y, resolution, height, width):

    c = int((x -orig_x) / resolution)
    r = int((y-orig_y) / resolution)

    if c >= width or r >= height:
        return 0

    distance = dt[r, c]

    return distance
