import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset
import numpy as np
from torch import Tensor
import json
from datetime import datetime,timedelta
import os
import pandas as pd

import sys
sys.path.append(".")
from utils.math import z_score, z_inverse
from utils.normalization import normalize_dataset


class TerminalDataset(Dataset):
    def __init__(self, input_dir, n_his, n_pred, is_continous, dates_dist, interval= 5, normalizer= "std"):
        super(TerminalDataset, self).__init__()
        self.interval = interval
        self.n_his = n_his
        self.n_pred = n_pred
        self.terminal_dir_path = os.path.join(input_dir, "terminal_image_frames")
        
        self.is_continuous = is_continous
        self.dates_dist = dates_dist
        self.terminal_data, self.timestamp_dist = self.load_data()

        features, targets, timestamps_all = self.mix_data()
        
        self.normalized_features, self.scaler = normalize_dataset(features, normalizer, column_wise = False)
        self.normalized_targets, _ = normalize_dataset(targets, "std", column_wise = False)
        self.timestamp_all = timestamps_all


    def load_data(self):
        # dates_dist = [[s1, e1], [s2, e2]] 
        if self.is_continuous:
            terminal_data = self.load_terminal(self.terminal_dir_path)
            timestamp_dist= None
        else:
            terminal_data = self.load_terminal(self.terminal_dir_path)
            timestamp_dist=list() 
            for i, date_dist in enumerate(self.dates_dist):
                sdate = date_dist[0]
                edate = date_dist[1]
                new_sdate = datetime.strptime(sdate, "%Y%m%d")
                new_edate = datetime.strptime(edate, "%Y%m%d") + timedelta(days=1)
                timestamp_range = (terminal_data["timestamp"] >= new_sdate) & (terminal_data["timestamp"] < new_edate)
                # timestamps = terminal_data[timestamp_range]
                # print(timestamps)
                num_timestamps = terminal_data[timestamp_range].shape[0]
                if i == 0:
                    dist_range = [0, num_timestamps-1]
                else:
                    start = timestamp_dist[-1][1]+1
                    end = start + num_timestamps-1
                    dist_range = [start, end]
                timestamp_dist.append(dist_range)

            
        return terminal_data,timestamp_dist
            
    
    def load_terminal(self, input_dir_path):
        areas = os.listdir(input_dir_path)
        areas.sort()
        selected_data=list()
        for area in areas:
            area_dir_path = os.path.join(input_dir_path,area)
            dates_dir = os.listdir(area_dir_path)
            dates = self.filter_dates(dates_dir)
            dates.sort()
            for date in dates:
                filepath = os.path.join(area_dir_path, date, f"{date}.json")
                with open(filepath, 'r', encoding='utf-8') as js:
                    # 尝试加载已有的数据
                    area_timestamp_dict = json.load(js)
                for timestamp, vdict in area_timestamp_dict.items():
                    s_timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M")
                    s_area= vdict["area"]
                    s_count = vdict["count"]
                    
                    temp = {"timestamp": s_timestamp, "area":s_area, "count": s_count}
                    selected_data.append(temp)
        
        
        df = pd.DataFrame(selected_data)
        date_area_data = df.pivot_table(columns="area", values="count", aggfunc="first", index="timestamp")
        date_area_data=date_area_data.sort_values(by="timestamp")
        areas = date_area_data.columns
        areas  = sorted(areas, key=lambda area: int(area[1:]))
        date_area_data= date_area_data[areas]
        date_area_data= date_area_data.reset_index(drop=False)
        # print(date_area_data)
        return date_area_data
    
    def generate_dates(self, start_date, end_date):
        # 将字符串日期转换为datetime对象
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        # 生成日期范围内的所有日期
        date_list = []
        delta = timedelta(days=1)
        while start <= end:
            date_list.append(start.strftime("%Y%m%d"))
            start += delta
        return date_list

    # the contained dates
    def filter_dates(self, dir_dates):
        dates=list()
        for date_range in self.dates_dist:
            dates_in_range = self.generate_dates(date_range[0], date_range[1])
            dates.extend(dates_in_range)
        
        filtered_dates = list()
        for date in dates:
            if date not in dir_dates:
                print(f"The date {date} is not in the dataset, please check the dates_dist paramter.\nExpected {dir_dates}, but get dates range {self.dates_dist}")
                exit(1)
            else:
                filtered_dates.append(date)
        
        return filtered_dates
    
    
    def get_index_data(self, index):
        timestamp_list = list()
        # [index, index+1, ..., index + self.his -1]
        his_count = list()
        h_index = index
        
        for _ in range(self.n_his):
            ter_timestamp = self.terminal_data.loc[h_index,"timestamp"]
            ter_timestamp = ter_timestamp.strftime('%Y%m%d%H%M')
            timestamp_list.append(ter_timestamp)
            his_count.append(np.array(list(self.terminal_data.iloc[h_index,1:])))
            h_index += 1

        p_index = h_index
        target_count = list()
        for _ in range(self.n_pred):
            
            ter_timestamp = self.terminal_data.loc[p_index,"timestamp"]
            ter_timestamp = ter_timestamp.strftime('%Y%m%d%H%M')
            timestamp_list.append(ter_timestamp)
            target_count.append(np.array(list(self.terminal_data.iloc[p_index,1:])))
            p_index += 1
        

        return his_count, target_count, timestamp_list

    
    def generate_data(self):
        his_count_all, target_count_all, timestamps_all = list(), list(), list()
        if self.timestamp_dist is not None:
            for timestamp_idx_range in self.timestamp_dist:
                sidx = timestamp_idx_range[0]
                eidx = timestamp_idx_range[1]
                for index in range(sidx, eidx-(self.n_his + self.n_pred-1)+1):
                    his_count, target_count, timestamps= self.get_index_data(index)
                    his_count_all.append(his_count)
                    target_count_all.append(target_count)
                    timestamps_all.append(timestamps)
        else:
            ntimestamps = len(self.terminal_data)
            for index in range(0, ntimestamps- (self.n_his + self.n_pred-1)):
                his_count, target_count, timestamps= self.get_index_data(index)
                his_count_all.append(his_count)
                target_count_all.append(target_count)
                timestamps_all.append(timestamps)

        his_count_all = Tensor(np.array(his_count_all))
        target_count_all = Tensor(np.array(target_count_all))
        timestamps_all = timestamps_all

        return his_count_all, target_count_all, timestamps_all

    def mix_data(self):
        """
        his_count_all : (total_len, n_his, num_nodes)
        target_count_all: (total_len, n_pred, num_nodes)
        flight_count_all: (total_len, n_pred)
        """
        his_count_all, target_count_all, timestamps_all = self.generate_data()
        
        xs = his_count_all.unsqueeze(3) # xs: (total_len, n_his, num_nodes, 1)
        ys = target_count_all.unsqueeze(3) # ys: (total_len, n_pred, num_nodes, 1)

        features = xs # features :(total_len, n_his, num_nodes, in_dim = 1)
        targets = ys # targets : (total_len, n_pred, num_nodes, out_dim = 1)

        return features, targets, timestamps_all


    
    def __getitem__(self, index):
        input = self.normalized_features[index] # (num_nodes, in_dim, n_his)
        target = self.normalized_targets[index] # (num_nodes, out_dim, n_pred)
        timestamps = self.timestamp_all[index]
        
        return input, target, timestamps


    def __len__(self):
        return len(self.normalized_features)
    
if __name__=='__main__':
    tfdata = TerminalDataset(input_dir="data/temp", n_his=6, n_pred=3, is_continous=False, dates_dist=[["20240609", "20240610"],["20240612", "20240613"]], interval = 5)
    print(f"len(tdata): {len(tfdata)}")
    # tfdata.load_terminal("data/processed/terminal_image_frames")
    import time
    start = time.time()
    temp=tfdata[24]
    end = time.time()
    print(f"running time : {end-start}s")
    print(temp[0]) # 历史统计数据
    print(temp[1]) # 目标预测数据
    print(temp[2]) # 时间戳
    print(len(tfdata))