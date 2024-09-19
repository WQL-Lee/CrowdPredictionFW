import torch
from glob import glob
import os
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from time import time
from argparse import ArgumentParser
import cv2
import logging
from pathlib import Path
import json

import inferenceMP
import sys
sys.path.append('.')
from utils import read_json, get_subdirectories

class RecongitionModel:
    def __init__(self, model_name, rec_config="recog_model/recong_model_config.json",default_level=logging.INFO):
        rec_config = Path(rec_config)
        if rec_config.is_file():
            self.config = read_json(rec_config)
            self.config_model= self.config["crowd_model"][model_name]
            self.model= self.get_model(model_name)
            self.input_dir = self.config["input_dir"]
            self.saved_dir = self.config["saved_dir"]
            self.vis_dir = self.config["vis_dir"]
            self.model_name = model_name
        else:
            print("Warning: recognitionconfiguration file is not found in {}.".format(rec_config))
            logging.basicConfig(level=default_level)
            
    
    def predict(self, model, img_path):
        confidence = None
        if self.model_name == "MPCount":
            config_mp = self.config_model
            pred_dmap, pred_count =inferenceMP.predict(model, img_path, config_mp["device"], config_mp["unit_size"],config_mp["patch_size"], config_mp["log_para"])
            
            return pred_count, confidence
        else:
            print(f"The model {self.model_name} has not been defined!")
            exit(-1)
            
    
    
    def vis(self):
        pass
    
    def get_model(self, model_name):
        if model_name == "MPCount":
            model = inferenceMP.load_model(self.config_model["pretrained"], self.config_model["device"])
        else:
            print(f"The model {model_name} has not been defined!")
            exit(-1)
        return model
    
    
    def save_crowd(self):
        model = self.get_model("MPCount")
        input_dir = self.config["input_dir"]
        image_dir_paths = get_subdirectories(input_dir,2)
        
        for image_dir_path in image_dir_paths:
            saved_dir_path= image_dir_path.replace("data/mid",self.saved_dir, 1)
            if not os.path.exists(saved_dir_path):
                os.makedirs(saved_dir_path)
            frame_names = os.listdir(image_dir_path)
            frame_names.sort()
            
            path_parts = saved_dir_path.split(os.sep)
            area = path_parts[-2]
            date = path_parts[-1]
            index = 0
            results = list()
            for frame_name in frame_names:
                frame_path = os.path.join(image_dir_path, frame_name)
                pred_count, confidence =self.predict(model, frame_path)

                time = frame_name.split("frame_at_")[1].split(".")[0]
                temp = {"id": index,  "count": pred_count, "conf":confidence, "area": area,  "date": date, "time":time, "frame_name": frame_name, "model": self.model_name}
                print(f"id: {index},  count: {pred_count}, conf:{confidence}, area: {area},  date: {date},  time: {time}, frame_name: {frame_name}, model: {self.model_name}")
                index +=1 
                
                print(f'{frame_path}: {pred_count}')
                results.append(temp)
            with open(os.path.join(saved_dir_path, f"{date}.json"), 'w', encoding='utf-8') as js:
                # 使用json.dump序列化list并存储到文件
                json.dump(results, js, ensure_ascii=False, indent=4)        
    
if __name__=="__main__":
    recong = RecongitionModel("MPCount", rec_config="recog_model/recong_model_config.json")
    recong.save_crowd()
    
