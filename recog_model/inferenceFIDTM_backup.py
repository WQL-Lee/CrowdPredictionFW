from __future__ import division
import warnings
import argparse

from recog_model.FIDTM.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import cv2
import numpy as np
import scipy



from utils.util import *

import logging
# import nni
# from nni.utils import merge_parameter

import json

warnings.filterwarnings('ignore')
import time

logger = logging.getLogger('mnist_AutoML')

img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor_transform = transforms.ToTensor()


def main(config):
    model = get_seg_model()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    if config['pre']:
        if os.path.isfile(config['pre']):
            print("=> loading checkpoint '{}'".format(config['pre']))
            checkpoint = torch.load(config['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            config['start_epoch'] = checkpoint['epoch']
            config['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(config['pre']))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    count_dic = {}
    point_dic = {}
    for id_cam in os.listdir(config['images_path']):
        count_dic[id_cam] = {}
        point_dic[id_cam] = {}
        for img_name in os.listdir(os.path.join(config['images_path'], id_cam)):

            frame = cv2.imread(os.path.join(config['images_path'], id_cam, img_name))

            '''out image'''
            width = frame.shape[1]  # output size
            height = frame.shape[0]  # output size
            try:
                scale_factor = 0.5
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                ori_img = frame.copy()
                if not os.path.exists(f'{config["infer_saves_path"]}/images/all_cam_id/{id_cam}'):
                    os.makedirs(f'{config["infer_saves_path"]}/images/all_cam_id/{id_cam}')
                cv2.imwrite(f'{config["infer_saves_path"]}/images/all_cam_id/{id_cam}/{img_name.split(".")[0]}.jpg', frame)
            except:
                print("Image processing error")
                continue
            frame = frame.copy()
            image = tensor_transform(frame)
            image = img_transform(image).unsqueeze(0)

            with torch.no_grad():
                d6 = model(image)

            count, pred_kpoint = counting(d6)
            point_map = generate_point_map(pred_kpoint)
            box_img = generate_bounding_boxes(pred_kpoint, frame)
            show_fidt = show_fidt_func(d6.data.cpu().numpy())
            # res = np.hstack((ori_img, show_fidt, point_map, box_img))
            res1 = np.hstack((ori_img, show_fidt))
            res2 = np.hstack((box_img, point_map))
            res = np.vstack((res1, res2))

            cv2.putText(res, "Count:" + str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not os.path.exists(f'{config["infer_saves_path"]}/images_results/all_cam_id/{id_cam}'):
                os.makedirs(f'{config["infer_saves_path"]}/images_results/all_cam_id/{id_cam}')
            cv2.imwrite(f'{config["infer_saves_path"]}/images_results/all_cam_id/{id_cam}/{img_name.split(".")[0]}.jpg', res)

            count_dic[f'{id_cam}'].update({f'{img_name.split(".")[0]}': count})
            point_dic[f'{id_cam}'].update({f'{img_name.split(".")[0]}': pred_kpoint.tolist()})
            print("pred:%.3f" % count)
    if not os.path.exists(f'{config["infer_saves_path"]}/count_point'):
        os.makedirs(f'{config["infer_saves_path"]}/count_point')
    with open(f'{config["infer_saves_path"]}/count_point/count_result.json', 'w', encoding='utf-8') as f:
        json.dump(count_dic, f, ensure_ascii=False)
    with open(f'{config["infer_saves_path"]}/count_point/point_result.json', 'w', encoding='utf-8') as f:
        json.dump(point_dic, f, ensure_ascii=False)



def counting(input):
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 100.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1

    '''negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    return count, kpoint


def generate_point_map(kpoint):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)

    return point_map


def generate_bounding_boxes(kpoint, Img_data):
    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048

    if pts.shape[0] > 0:  # Check if there is a human presents in the frame
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

        distances, locations = tree.query(pts, k=4)
        for index, pt in enumerate(pts):
            pt2d = np.zeros(kpoint.shape, dtype=np.float32)
            pt2d[pt[1], pt[0]] = 1.
            if np.sum(kpoint) > 1:
                sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
            sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.04)

            if sigma < 6:
                t = 2
            else:
                t = 2
            Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                     (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return Img_data


def show_fidt_func(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='inference_FIDTM')
    parse.add_argument('-c', '--config', default="config/inference/FIDTM.json", type=str,
                        help='config file path (default: None)')
    args = parse.parse_args()
    config = vars(args)
    with open(config['config'], 'r') as f:
        dic = json.load(f)
    config.update(dic)

    main(config)