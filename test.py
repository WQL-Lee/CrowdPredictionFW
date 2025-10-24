import argparse
import torch
from tqdm import tqdm
import json
import glob
import os
import re

import data_loader.data_loaders as module_data

# import pred_model.structure.CrowdCNNGRU as module_arch

# import pred_model.TGCN.TGCN as module_arch
from trainer.TGCNTrainer import TGCNTrainer

import pred_model.A3TGCN.A3TGCN as module_arch
from trainer.A3TGCNTrainer import A3TGCNTrainer

# import pred_model.AGCRN.AGCRN as module_arch
from trainer.AGCRNTrainer import AGCRNTrainer

import pred_model.loss as module_loss
import pred_model.metric as module_metric
from parse_config import ConfigParser


from utils.math import z_inverse

import torch.serialization
from parse_config import ConfigParser  # 确保能导入该类

# 将 ConfigParser 加入安全全局列表
torch.serialization.add_safe_globals([ConfigParser])

class Testor:
    def __init__(self, model_name, model_arch, model_path, saved_dir, logger, data_loader,loss_fn, metric_fns, device, n_gpu=1):
        self.model_name = model_name
        self.model_path = model_path
        self.data_loader = data_loader
        
        self.model = Testor.load_model(model_path, model_arch)
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)  
        self.loss_fn = loss_fn
        self.metric_fns= metric_fns
        self.device= device
        self.logger = logger
        self.saved_dir = saved_dir
        self.scaler = self.data_loader.dataset.scaler

        
        
    @staticmethod
    def load_model(model_path, model_arch):
        model = model_arch
        checkpoint = torch.load(model_path, weights_only=False)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        return model
    


    
    def test(self):
        if self.model_name == "AGCRN":
            y_pred, y_true, y_time = AGCRNTrainer.test(self.model, self.data_loader, self.device, self.scaler)
        elif self.model_name == 'A3TGCN':
            y_pred, y_true, y_time = A3TGCNTrainer.test(self.model, self.data_loader, self.device, self.scaler)
        elif self.model_name == 'TGCN':
            y_pred, y_true, y_time = TGCNTrainer.test(self.model, self.data_loader, self.device, self.scaler)
        else:
            print("The model has not been specified!")
            exit(-1)

        self.logger.info('*' * 100)
        self.logger.info(self.model_path)
        for t in range(y_true.shape[1]):
            metric_dict = dict()
            for i,met in enumerate(self.metric_fns):
                met_value = met(y_pred[:,t,...],y_true[:,t,...])
                metric_dict[met.__name__] = met_value
        
            formatted_line = " ".join([f"{key}: {value:.4f}" for key, value in metric_dict.items()])
            formatted_line = f"Horizon: {t} " + formatted_line
            self.logger.info(formatted_line)
        
        metric_dict = dict()
        for i,met in enumerate(self.metric_fns):
            met_value = met(y_pred,y_true)
            metric_dict[met.__name__] = met_value

        formatted_line = " ".join([f"{key}: {value:.4f}" for key, value in metric_dict.items()])
        formatted_line = f"Average Horizon: " + formatted_line
        self.logger.info(formatted_line)

        self.logger.info('*' * 100)

        if self.saved_dir:
            self.save(y_pred, y_true, y_time)

    
    def save(self, y_pred, y_true, y_time):
        # y_pred: (B, T, N, 1)
        # y_true: (B, T, N, 1)
        # y_time: (B, T)

        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)
        
        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()

        output_dict = list()
        for i, (i_pred, i_true, i_time) in enumerate(zip(y_pred, y_true, y_time)):
            tmp = {"sample_i": int, "target": list, "prediction": list, "timestamp": list}
            
            tmp['sample_i'] = i
            tmp['target'] = i_true
            tmp['prediction'] = i_pred
            tmp['timestamp'] = i_time
            output_dict.append(tmp)
        
        if not os.path.exists(self.saved_dir):
            os.makedirs(self.saved_dir)
        # output_dict= {'target': y_true, 'prediction':y_pred, 'timestamp': y_time}
        model_index = re.findall(r'\d+\.\d+|\d+', self.model_path)[-1]
        filename = f"{self.model_name}_checkpoint_{int(model_index)}.json"
        with open(os.path.join(self.saved_dir, filename), 'w', encoding='utf-8') as js:
            json.dump(output_dict, js, indent=4, ensure_ascii=False)

        
        
        
        


def main(config):
    logger = config.get_logger('test')
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    model_arch = config.init_obj('arch', module_arch)
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    logger.info(model_arch) 
    
    if os.path.isdir(config["model_path"]):
        model_paths = glob.glob(os.path.join(config["model_path"], "*.pth"))
        model_paths.sort()
    else:
        model_paths = [config["model_path"]]
    
    model_name = config['name']
    n_gpu = config['n_gpu']
    saved_dir = config["saved_dir"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for _, model_path in enumerate(model_paths): 
        testor =Testor(model_name, model_arch, model_path, saved_dir, logger, data_loader, loss_fn, metric_fns, device, n_gpu)
        testor.test()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config/test/A3TGCN.jsonc", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-tr', '--train', default=False, type = bool,
                       help = 'decide to train/test mode')

    config = ConfigParser.from_args(args)
    main(config)
