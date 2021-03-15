from io import FileIO
import numpy as np 
from scipy import ndimage
from matplotlib import pyplot as plt
import yaml
import csv
from PIL import Image

import toy_f110.LibFunctions as lib

class TrackMap:
    def __init__(self, sim_conf, map_name) -> None:
        self.sim_conf = sim_conf
        self.map_name = map_name 

        # map info
        self.resolution = None
        self.origin = None
        self.n_obs = None 
        self.map_height = None
        self.map_width = None
        self.start_pose = None
        self.obs_size = None
        
        self.map_img = None
        self.dt = None
        self.obs_img = None #TODO: combine to single image with dt for faster scan

        self.load_map()

    def load_map(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())

        try:
            self.resolution = yaml_file['resolution']
            self.origin = yaml_file['origin']
            self.n_obs = yaml_file['n_obs']
            self.obs_size = yaml_file['obs_size']
            map_img_path = 'maps/' + yaml_file['image']
            self.start_pose = np.array(yaml_file['start_pose'])
        except Exception as e:
            print(e)
            raise FileIO("Problem loading map yaml file")

        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        self.obs_img = np.zeros_like(self.map_img) # init's obs img

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt = np.array(dt *self.resolution)
    
    def add_obstacles(self):
        self.obs_img = np.zeros_like(self.obs_img)

        #TODO: copy new code in here to vectorise and improve

        obs_size = self.obs_size
        obs_size = np.array([obs_size, obs_size]) / self.resolution

        buffer = 4
        rands = np.random.randint(5, self.N-5, self.n_obs)
        rands = np.sort(rands)
        diffs = rands[1:] - rands[:-1]
        diffs = np.insert(diffs, 0, buffer+1)
        rands = rands[diffs>buffer]
        rands = rands[rands>8 ]

        n = len(rands)
        obs_locs = []
        for i in range(n):
            pt = self.wpts[rands[i]][:, None]
            obs_locs.append(pt[:, 0])

        for obs in obs_locs:
            for i in range(0, int(obs_size[0])):
                for j in range(0, int(obs_size[1])):
                    x, y = self.xy_to_row_column([obs[0], obs[1]])
                    self.obs_img[y+j, x+i] = 1

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        return c, r

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.map_width -2 or abs(r) > self.map_height -2:
            return True
        val = self.dt[r, c]

        if val < 0.1:
            return True
        if self.obs_img[r, c]:
            return True
        return False

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def render_map(self, figure_n=4, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.map_width])
        plt.ylim([self.map_height, 0])


        if self.obs_img is None:
            plt.imshow(self.map_img, origin='lower')
        else:
            plt.imshow(self.obs_img + self.map_img, origin='lower')

        plt.gca().set_aspect('equal', 'datalim')

        plt.pause(0.0001)
        if wait:
            plt.show()



class ForestMap:
    def __init__(self, sim_conf, map_name) -> None:
        self.sim_conf = sim_conf
        self.map_name = map_name 

        # map info
        self.resolution = None
        self.n_obs = None 
        self.map_height = None
        self.map_width = None
        self.forest_length = None
        self.forest_width = None
        self.start_pose = None
        self.obs_size = None
        self.obstacle_buffer = None
        self.end_y = None
        
        self.map_img = None

        self.load_map()

    def load_map(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())

        try:
            self.resolution = yaml_file['resolution']
            self.n_obs = yaml_file['n_obs']
            self.obs_size = yaml_file['obs_size']
            self.start_pose = np.array(yaml_file['start_pose'])
            self.forest_length = yaml_file['forest_length']
            self.forest_width = yaml_file['forest_width']
            self.obstacle_buffer = yaml_file['obstacle_buffer']
            self.end_y = yaml_file['end_y']
        except Exception as e:
            print(e)
            raise FileIO("Problem loading map yaml file")

        self.map_height = int(self.forest_length / self.resolution)
        self.map_width = int(self.forest_width / self.resolution)
        self.map_img = np.zeros((self.map_width, self.map_height))

    def generate_forest(self):
        self.map_img = np.zeros((self.map_width, self.map_height))
        rands = np.random.random((self.n_obs, 2))
        xs = rands[:, 0] * (self.map_width-self.obs_size) 
        ys = rands[:, 1] * (self.map_height - self.obstacle_buffer*2 - self.obs_size)
        ys = ys + np.ones_like(ys) * self.start_pose[1]
        obs_locations = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        obs_locations = np.array(obs_locations, dtype=np.int)
        obs_size_px = int(self.obs_size/self.resolution)
        for location in obs_locations:
            x, y = location[0], location[1]
            self.map_img[x:x+obs_size_px, y:y+obs_size_px] = 1

    def render_map(self, figure_n=1, wait=False):
        #TODO: draw the track boundaries nicely
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.map_width])
        plt.ylim([0, self.map_height])

        plt.imshow(self.map_img.T, origin='lower')

        xs = np.linspace(0, self.map_width, 10)
        ys = np.ones_like(xs) * self.end_y / self.resolution
        plt.plot(xs, ys, '--')     
        x, y = self.xy_to_row_column(self.start_pose[0:2])
        plt.plot(x, y, '*', markersize=14)

        plt.pause(0.0001)
        if wait:
            plt.show()
            pass

    def xy_to_row_column(self, pt):
        c = int(round(np.clip(pt[0] / self.resolution, 0, self.map_width-2)))
        r = int(round(np.clip(pt[1] / self.resolution, 0, self.map_height-2)))
        return c, r

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        if x_in[0] > self.forest_width or x_in[1] > self.forest_length:
            return True
        x, y = self.xy_to_row_column(x_in)
        if self.map_img[x, y]:
            return True

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)




