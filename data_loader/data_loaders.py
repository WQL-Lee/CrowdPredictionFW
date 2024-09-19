from torchvision import datasets, transforms
from base import BaseDataLoader
from .dataset import TerminalFlightDataset
from .dataset import TerminalDataset

class CrowdDataLoader(BaseDataLoader):

    def __init__(self, data_dir, n_his, n_pred, is_continous, dates_dist, interval, batch_size, shuffle=True, validation_split=0.0, num_workers=0, training=True):
        # training参数当前没有发挥作用
        self.data_dir = data_dir
        # self.dataset = TerminalFlightDataset(self.data_dir,n_his, n_pred, is_continous, dates_dist, interval)
        # super(CrowdDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        # test TerminalDataset
        self.dataset = TerminalDataset(self.data_dir,n_his, n_pred, is_continous, dates_dist, interval)
        super(CrowdDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

