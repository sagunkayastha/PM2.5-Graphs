import os
import sys

from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data


class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=1,
                       pred_len=24,
                       dataset_num=1,
                       flag = 'Train',
                       ):

        self.flag = flag
        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
            print(flag)
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')

        # self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        # self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        # self.data_start = self._get_time(config['dataset']['data_start'])
        # self.data_end = self._get_time(config['dataset']['data_end'])

        #self.obs_data_fp = '/project/ychoi/rdimri/obs_data.npy'
        #self.mcip_data_fp = '/project/ychoi/rdimri/mcip_data.npy'
        #self.cmaq_data_fp = '/project/ychoi/rdimri/cmaq_data.npy'
        self.model_input_imputed_fp = '/project/ychoi/rdimri/model_input_imputed_new.npy'

        self.graph = graph

        self._load_npy()
        # self._gen_time_arr()
        # self._process_time()
        # self._process_feature()
        self.feature = np.float32(self.feature)
        self.pm25 = np.float32(self.pm25)
        print(self.pm25.shape)
        self._calc_mean_std()
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / (self.feature_std)
        self.pm25 = (self.pm25 - self.pm25_mean) / (self.pm25_std)

    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.pm25 = _add_t(self.pm25, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        # self.time_arr = _add_t(self.time_arr, seq_len)

    def _calc_mean_std(self):
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        self.feature_min = self.feature.min(axis=(0,1))
        self.feature_max = self.feature.max(axis=(0,1))
        self.wind_mean = self.feature_mean[-2:]#[22:24]
        self.wind_std = self.feature_std[-2:]#[22:24]
        self.pm25_mean = self.pm25.mean()
        self.pm25_std = self.pm25.std()
        self.wind_min = self.feature_min[-2:]#[22:24]
        self.wind_max = self.feature_max[-2:]#[22:24]
        self.pm25_min = self.pm25.min()
        self.pm25_max = self.pm25.max()


    def _load_npy(self):
        self.model_input_imputed = np.load(self.model_input_imputed_fp)
        one_mcip = self.model_input_imputed[:,:,15]
        two_mcip = self.model_input_imputed[:,:,17:20]
        three_mcip = self.model_input_imputed[:,:,24:28]
        reduced_mcip = np.concatenate([one_mcip[:,:,np.newaxis], two_mcip, three_mcip], axis = -1)
        if self.flag == 'Train':
            self.feature = reduced_mcip#self.model_input_imputed[:2*8760,:,4:]
            self.pm25 = self.model_input_imputed[:2*8760,:,3]
            print(f"Shape of Training feature set is {self.feature.shape}")
            print(f"Shape of Training PM25 set is {self.pm25.shape}")            
        if self.flag == 'Val':
            self.feature = reduced_mcip#self.model_input_imputed[2*8760:2*8760 + 4380,:,4:]
            self.pm25 = self.model_input_imputed[2*8760:2*8760 + 4380,:,3]
            print(f"Shape of Validation feature set is {self.feature.shape}")
            print(f"Shape of Validation PM25 set is {self.pm25.shape}")
        if self.flag == 'Test':
            self.feature = reduced_mcip#self.model_input_imputed[2*8760 + 4380:,:,4:]
            self.pm25 = self.model_input_imputed[2*8760 + 4380:,:,3]
            print(f"Shape of Testing feature set is {self.feature.shape}")
            print(f"Shape of Testing PM25 set is {self.pm25.shape}")

#     def _get_idx(self, t):
#         t0 = self.data_start
#         return int((t.timestamp - t0.timestamp) / (60 * 60 * 3))

#     # def _get_time(self, time_yaml):c
    
#         arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
#         return arrow_time

    def __len__(self):
        return len(self.pm25)

    def __getitem__(self, index):
        return self.pm25[index], self.feature[index]

if __name__ == '__main__':
    from graph import Graph
    g = Graph()
    train_data = HazeData(g, flag='Train')
    val_data = HazeData(g, flag='Val')
    test_data = HazeData(g, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
