import xarray as xr
import time
import os
import sys
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham
import pandas as pd
path = '/project/ychoi/rdimri/'
korea_station_info = pd.read_csv(path + 'korea_urban_site.csv' , header = None)# _201912_distance.csv', header = None)

class Graph():
    def __init__(self):
        self.dist_thres = 3     # not sure, 2 degrees equivalent to 200 km approx.
        self.alti_thres = 1000      # not sure, in meters
        self.use_altitude = True   # True once decided the threshold value
        self.nodes = self._gen_nodes()
        self.node_num = len(self.nodes)                                  # Total number of nodes
        self.altitude = self._load_altitude()                            # Altitude at each station
        # self.node_attr = self._add_node_attr()                         # Add elevation value to the node along with the lat/long information  
        self.nodes = self._add_altitude_nodes()
        self.edge_index, self.edge_attr = self._gen_edges()              # Edge information like connection with other nodes(stations) based on threshold
        # if self.use_altitude:
        #     self._update_edges()                                       # Adding elevation information to the edge features
        self.edge_num = self.edge_index.shape[1]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]    # Matrix to store information about the node connection, directed or not
    
    
    def _gen_nodes(self):
        nodes = OrderedDict()
        folder_path = '/project/ychoi/rdimri/station_wise/'
        start = time.time()
        with open(path + 'korea_urban_site.csv', 'r') as f:# _201912_distance.csv', 'r') as f:
            for line in f:
                index_long, index_lat, index = line.rstrip('\n').split(',') #index, longitude, latitude, index_long, index_lat, dist_grid = line.rstrip('\n').split(',')
                index = int(index)
                file_name = 'staion_' + str(index) + '.nc'
                file_path = str(folder_path + file_name)
                assert os.path.isfile(file_path)
                latitude = xr.open_dataset(file_path).lat
                longitude = xr.open_dataset(file_path).lon
                longitude, latitude = float(longitude), float(latitude)
                nodes.update({index: {'lon': longitude, 'lat': latitude}})
        print(f" toal time gen_nodes : {time.time() - start}")
        print('Updated Graph')
        return nodes
    
    def _add_altitude_nodes(self):
        nodes = OrderedDict()
        folder_path = '/project/ychoi/rdimri/station_wise/'
        start = time.time()
        with open(path + 'korea_urban_site.csv', 'r') as f:#_201912_distance.csv', 'r') as f:
            i = 0
            for line in f:
                index_long, index_lat, index = line.rstrip('\n').split(',') #index, longitude, latitude, index_long, index_lat, dist_grid = line.rstrip('\n').split(',')
                index = int(index)
                file_name = 'staion_' + str(index) + '.nc'
                file_path = str(folder_path + file_name)
                assert os.path.isfile(file_path)
                latitude = xr.open_dataset(file_path).lat
                longitude = xr.open_dataset(file_path).lon
                longitude, latitude = float(longitude), float(latitude)
                altitude = self.altitude[i]
                nodes.update({index: {'altitude': altitude , 'lon': longitude, 'lat': latitude}})
                i = i + 1
        
        print(f" toal time add_altitude_nodes : {time.time() - start}")
        return nodes
        
    
    def _gen_edges(self):
        coords = []
        lonlat = {}
        start = time.time()
        for i in self.nodes: 
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])                  # shape of the output is (number of stations, 2), lat/long column
        dist = distance.cdist(coords, coords, 'euclidean')                               # shape is (no. of stations, no. of stations), that is, (405, 405)    
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        adj[dist <= self.dist_thres] = 1                                                 # threshold condition, put 1 where distace is smaller than or equal to 2
        assert adj.shape == dist.shape
        dist = dist * adj                                                                # replaces values grater than threshold with 0
        edge_index, dist = dense_to_sparse(torch.tensor(dist))                           # gives you the index of source and destination nodes; and the distance brtween them 
        edge_index, dist = edge_index.numpy(), dist.numpy()

        direc_arr = []
        dist_kilometer = []
        for i in range(edge_index.shape[1]):
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = list(self.nodes.items())[src][1]['lat'], list(self.nodes.items())[src][1]['lon']
            dest_lat, dest_lon = list(self.nodes.items())[dest][1]['lat'], list(self.nodes.items())[dest][1]['lat']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon

            u = u * units.meter / units.second
            v = v * units.meter / units.second
            direc = mpcalc.wind_direction(u, v)._magnitude

            direc_arr.append(direc)
            dist_kilometer.append(dist_km)

        direc_arr = np.stack(direc_arr)
        dist_arr = np.stack(dist_kilometer)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        print(f" toal time gen_edges : {time.time() - start}")
        return edge_index, attr
    
    def _load_altitude(self):
        folder_path = '/project/ychoi/rdimri/station_wise/'
        srtm_arr = []
        start = time.time()
        for i in range(185):
            index = list(self.nodes.items())[i][0]
            file_name = 'staion_' + str(index) + '.nc'
            file_path = str(folder_path + file_name)
            assert os.path.isfile(file_path)
            srtm_one = xr.open_dataset(file_path).Band1
            srtm_arr.append(srtm_one)
        
        print(f" toal time load_altitude : {time.time() - start}")
        return np.squeeze(np.array(srtm_arr))
    
if __name__ == '__main__':
    g = Graph()

# g = Graph()
