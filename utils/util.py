import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import os
import json
import json5


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    try:
        with fname.open('rt') as handle:
                json_data = json.load(handle, object_hook=OrderedDict)
                return json_data
    except json.decoder.JSONDecodeError as e:
        print(f"Using json5 for loading json file")
        with fname.open('rt') as handle:
            json_data = json5.load(handle, object_hook=OrderedDict)
            return json_data


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
    
    
def get_subdirectories(root_dir, level=1):
    """
    递归获取目录下的所有子目录，并返回包含层级信息的路径列表。
    :param root_dir: 要遍历的根目录路径
    :return: 包含层级信息的子目录路径列表
    """
    subdirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            subdir_path = os.path.join(dirpath, dirname)
            # 计算层级：根目录到当前目录的路径分割数量
            current_level = subdir_path.count(os.sep) - root_dir.count(os.sep)
            if current_level == level:
                # 将层级和路径添加到列表中
                subdirs.append(subdir_path)
    subdirs.sort()
    return subdirs
