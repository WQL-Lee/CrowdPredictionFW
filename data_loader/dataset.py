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

class TerminalFlightDataset(Dataset):
    def __init__(self, input_dir, n_his, n_pred, is_continous, dates_dist,interval= 5):
        super(TerminalFlightDataset, self).__init__()
        self.mean_std = [22.146185559872638, 16.22383992827241]
        self.flight_mean_std = [8.146901428995541, 16.85464714755638]
        self.flight_max = 157.9657383316001
        self.interval = interval
        self.n_his = n_his
        self.n_pred = n_pred
        self.terminal_dir_path = os.path.join(input_dir, "terminal_image_frames")
        self.flight_dir_path = os.path.join(input_dir, "flight_text_info")
        self.is_continuous = is_continous
        self.dates_dist = dates_dist
    
        self.terminal_data, self.flight_data, self.timestamp_dist = self.load_data()
        
    def load_data(self):
        # dates_dist = [[s1, e1], [s2, e2]] 
        
        if self.is_continuous:
            terminal_data = self.load_terminal(self.terminal_dir_path)
            flight_data = self.load_flight(self.flight_dir_path)
            timestamp_dist= None
            pass
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
            
            flight_data = self.load_flight(self.flight_dir_path)
            
        return terminal_data, flight_data, timestamp_dist
            
    def reindex(self,index, timestamp_dist):
        if self.is_continuous:
            tlen = self.terminal_data.shape[0]
            # if (index - (tlen - self.n_his - self.n_pred + 1)) > 0:    
            #     index -= tlen - self.n_his - self.n_pred + 1
            # return index
        
            if (index - (tlen - self.n_his - self.n_pred + 1)) < 0:
                    pass
            else:
                index -= tlen - self.n_his - self.n_pred + 1
            return index
        # timestamp_dist=[[s1,e1], [s2, e2]]
        # self.timestamp_dist=None
        # self.his = None
        # self.n_pred = None
        for tlist in  timestamp_dist:
            start = tlist[0]
            end= tlist[1]
            # 确定当前时间戳索引是在哪一个时间戳区间中
            if start <= index and index <= end:
                tlen = end- start + 1
                # 从index 开始，其后包含（self.his + self.n_pred）个元素的话，则不需要修改元素索引
                # 否则重索引为从后往前长度为(self.his+ self.n_pred)对应的元素
                if (index - (tlen - self.n_his - self.n_pred + 1)) < 0:
                    break
                else:
                    index -= tlen - self.n_his - self.n_pred + 1
        return index
    
    def load_terminal(self, input_dir_path):
        areas = os.listdir(input_dir_path)
        areas.sort()
        selected_data=list()
        for area in areas:
            area_dir_path = os.path.join(input_dir_path,area)
            dates = os.listdir(area_dir_path)
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
    
    def load_flight(self, input_dir_path):
        dates = os.listdir(input_dir_path)
        dates.sort()
        selected_data = list()
        for date in dates:
            filepath = os.path.join(input_dir_path, date, f"{date}.json")
            with open(filepath, 'r', encoding='utf-8') as js:
                    # 尝试加载已有的数据
                date_timestamp_dict = json.load(js)
            for timestamp, vlist in date_timestamp_dict.items():
                s_timestamp = datetime.strptime(timestamp, "%Y%m%d%H%M")
                temp = {"timestamp": s_timestamp, "feature": vlist}
                selected_data.append(temp)
        date_data = pd.DataFrame(selected_data)
        return date_data
    
    def __getitem__(self, index):
        index = self.reindex(index, self.timestamp_dist)
        timestamp_list = list()
        # [index, index+1, ..., index + self.his -1]
        his_count = list()
        h_index = index-1
        
        for _ in range(self.n_his):
            h_index += 1
            ter_timestamp = self.terminal_data.loc[h_index,"timestamp"]
            ter_timestamp = ter_timestamp.strftime('%Y%m%d%H%M')
            timestamp_list.append(ter_timestamp)
            his_count.append(np.array(list(self.terminal_data.iloc[h_index,1:])))
        
        his_count = z_score(np.stack(his_count, axis=0), self.mean_std[0], self.mean_std[1])


        p_index = h_index
        target_count = list()
        for _ in range(self.n_pred):
            p_index += 1
            ter_timestamp = self.terminal_data.loc[p_index,"timestamp"]
            ter_timestamp = ter_timestamp.strftime('%Y%m%d%H%M')
            timestamp_list.append(ter_timestamp)
            target_count.append(np.array(list(self.terminal_data.iloc[p_index,1:])))
        target_count = z_score(np.stack(target_count, axis=0), self.mean_std[0], self.mean_std[1])
        
        count_valid = np.ones(target_count.shape)

        
        # flight info
        flight_feature = []
        terminal_last =  datetime.strptime(timestamp_list[-1], "%Y%m%d%H%M")
        m_delta = timedelta(minutes=self.interval)
        start_timestamp = terminal_last + m_delta
        re_flight_data = self.flight_data.set_index('timestamp')
        for _ in range(self.n_pred):
            flight_info_slot = []
            for i in range(self.n_his):
                timestamp = start_timestamp + i * m_delta
                flight_info_slot.append(re_flight_data.loc[timestamp, "feature"])
            flight_feature.append(np.stack(flight_info_slot, axis=0))
            start_timestamp += m_delta
            
        flight_feature = (np.stack(flight_feature, axis=0) - self.flight_mean_std[0]) / self.flight_mean_std[1]
        
        his_count = Tensor(his_count).unsqueeze(-1)
        target_count = Tensor(target_count).unsqueeze(-1)
        count_valid = torch.IntTensor(count_valid).unsqueeze(-1)
        flight_feature = torch.FloatTensor(flight_feature)
        return his_count, target_count, count_valid, timestamp_list, flight_feature
        

    def __len__(self):
        records = self.terminal_data.shape[0]
        num_discontinuous= len(self.dates_dist) 
        return records - num_discontinuous*(self.n_his+self.n_pred-1)


class TerminalDataset(Dataset):
    def __init__(self, input_dir, n_his, n_pred, is_continous, dates_dist,interval= 5):
        super(TerminalDataset, self).__init__()
        self.interval = interval
        self.n_his = n_his
        self.n_pred = n_pred
        self.terminal_dir_path = os.path.join(input_dir, "terminal_image_frames")
        self.is_continuous = is_continous
        self.dates_dist = dates_dist
    
        self.terminal_data, self.timestamp_dist = self.load_data()
        self.terminal_max = np.max(self.terminal_data.iloc[:,1:])
        
    def load_data(self):
        # dates_dist = [[s1, e1], [s2, e2]] 
        
        if self.is_continuous:
            terminal_data = self.load_terminal(self.terminal_dir_path)
            timestamp_dist= None
            pass
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
            
    def reindex(self,index, timestamp_dist):
        if self.is_continuous:
            tlen = self.terminal_data.shape[0]
            # if (index - (tlen - self.n_his - self.n_pred + 1)) > 0:    
            #     index -= tlen - self.n_his - self.n_pred + 1
            # return index
        
            if (index - (tlen - self.n_his - self.n_pred + 1)) < 0:
                    pass
            else:
                index -= tlen - self.n_his - self.n_pred + 1
            return index
        # timestamp_dist=[[s1,e1], [s2, e2]]
        # self.timestamp_dist=None
        # self.his = None
        # self.n_pred = None
        for tlist in  timestamp_dist:
            start = tlist[0]
            end= tlist[1]
            # 确定当前时间戳索引是在哪一个时间戳区间中
            if start <= index and index <= end:
                tlen = end- start + 1
                # 从index 开始，其后包含（self.his + self.n_pred）个元素的话，则不需要修改元素索引
                # 否则重索引为从后往前长度为(self.his+ self.n_pred)对应的元素
                if (index - (tlen - self.n_his - self.n_pred + 1)) < 0:
                    break
                else:
                    index -= tlen - self.n_his - self.n_pred + 1
        return index
    
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
    
    def __getitem__(self, index):
        index = self.reindex(index, self.timestamp_dist)
        timestamp_list = list()
        # [index, index+1, ..., index + self.his -1]
        his_count = list()
        h_index = index-1
        
        for _ in range(self.n_his):
            h_index += 1
            ter_timestamp = self.terminal_data.loc[h_index,"timestamp"]
            ter_timestamp = ter_timestamp.strftime('%Y%m%d%H%M')
            timestamp_list.append(ter_timestamp)
            his_count.append(np.array(list(self.terminal_data.iloc[h_index,1:])))

        p_index = h_index
        target_count = list()
        for _ in range(self.n_pred):
            p_index += 1
            ter_timestamp = self.terminal_data.loc[p_index,"timestamp"]
            ter_timestamp = ter_timestamp.strftime('%Y%m%d%H%M')
            timestamp_list.append(ter_timestamp)
            target_count.append(np.array(list(self.terminal_data.iloc[p_index,1:])))
        
        his_count = Tensor(np.array(his_count))
        target_count = Tensor(np.array(target_count))
        return his_count, target_count, timestamp_list

    def __len__(self):
        records = self.terminal_data.shape[0]
        num_discontinuous= len(self.dates_dist) 
        return records - num_discontinuous*(self.n_his+self.n_pred-1)



if __name__=="__main__":
    # # tfdata = TerminalFlightDataset(input_dir="data/processed", n_his=12, n_pred=12, is_continous=False, dates_dist=[["20211003", "20211009"],["20211016","20211028"]], interval = 5)
    # tfdata = TerminalFlightDataset(input_dir="test_data", n_his=12, n_pred=12, is_continous=True, dates_dist=[["20211029", "20211031"]], interval = 5)
    # # tfdata.load_terminal("data/processed/terminal_image_frames")
    # import time
    # start = time.time()
    # temp=tfdata[24]
    # end = time.time()
    # print(f"running time : {end-start}s")
    # print(temp[3]) # 时间戳存在问题
    # print(len(tfdata))


    # tfdata = TerminalFlightDataset(input_dir="data/processed", n_his=12, n_pred=12, is_continous=False, dates_dist=[["20211003", "20211009"],["20211016","20211028"]], interval = 5)
    tfdata = TerminalDataset(input_dir="data/processed", n_his=6, n_pred=3, is_continous=False, dates_dist=[["20240607", "20240607"],["20240609", "20240610"],["20240612", "20240613"]], interval = 5)
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
