import argparse
import torch
from tqdm import tqdm
import json
import glob
import os
import re

import data_loader.data_loaders as module_data
# import pred_model.structure.CrowdCNNGRU as module_arch
import pred_model.TGCN.TGCN as module_arch
import pred_model.loss as module_loss
import pred_model.metric as module_metric
from parse_config import ConfigParser

from utils.math import z_inverse

class Inference:
    def __init__(self, model_name, model, data_loader,loss_fn, metric_fns, device):
        self.model_name = model_name
        self.data_loader = data_loader
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fns= metric_fns
        self.device= device
    
    def inference(self):
        if self.model_name == "Crowd_CNN_GRU":
            total_loss, total_metrics, output_dict = self.test_crowd_cnn_gru()
        elif self.model_name == "TGCN":
            total_loss, total_metrics, output_dict = self.test_tgcn()
        else:
            print("The model has not been specified!")
            exit(-1)
        return total_loss, total_metrics, output_dict
    
    def test_crowd_cnn_gru(self):
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        output_dict= {}
        device = self.device
        mean_std = self.data_loader.dataset.mean_std
        with torch.no_grad():
            for batch_idx, (x_g, t_g, cam_valid, time_stamp, flight) in enumerate(self.data_loader):
                tmp={}
                x_g, t_g, cam_valid,flight = x_g.to(device), t_g.to(device), cam_valid.to(device), flight.to(device)
                output = self.model.decode(x_g,flight[:,:,:,:])
                # output = model.multi_pred(x_g,flight[:,:,:,:])

                #
                # save sample images, or do something with output here
                #

                tmp['target']=z_inverse(t_g[:,:,:,:].cpu().numpy(),mean_std[0],mean_std[1]).tolist()
                tmp['prediction'] = z_inverse(output.cpu().numpy(),mean_std[0],mean_std[1]).tolist()
                tmp['time_stamp'] = time_stamp[-12:]
                # tmp['flight'] = flight[:,-1:,:,:]
                output_dict[batch_idx]=tmp.copy()
                # computing loss, metrics on test set
                loss = self.loss_fn(output, t_g[:,:,:,:])
                batch_size = x_g.shape[0]
                total_loss += loss.item() * batch_size
                # for i, metric in enumerate(metric_fns):
                #     total_metrics[i] += metric(output, t_g) * batch_size
                for i,met in enumerate(self.metric_fns):
                    total_metrics[i] += met( t_g[:,:,:,:].cpu().numpy(),output.cpu().numpy(), mean_std,
                                                                cam_valid[:,:,:,:].cpu().numpy())
        return total_loss, total_metrics, output_dict
    
    def test_tgcn(self):
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        output_dict= {}
        device = self.device
        max_value = self.data_loader.dataset.terminal_max
        with torch.no_grad():
            for batch_idx, (his_data, targets, timestamp) in enumerate(self.data_loader):
                tmp={}
                his_data = his_data/max_value
                targets = targets/max_value
                his_data = his_data.to(device)
                targets = targets.to(device)

                outputs = self.model(his_data)

                outputs = outputs * max_value
                targets = targets* max_value
                tmp['target']=targets.cpu().detach().numpy().tolist()
                tmp['prediction'] = outputs.cpu().detach().numpy().tolist()
                tmp['time_stamp'] = timestamp[:-3]
                # tmp['flight'] = flight[:,-1:,:,:]
                output_dict[batch_idx]=tmp.copy()
                # computing loss, metrics on test set
                loss = self.loss_fn(outputs, targets, self.model)
                batch_size = targets.shape[0]
                total_loss += loss.item() * batch_size

                for i,met in enumerate(self.metric_fns):
                    total_metrics[i] += met(outputs, targets).cpu().detach().numpy()
        return total_loss, total_metrics, output_dict

    
        
    


def main(config):
    logger = config.get_logger('test')
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    

    if os.path.isdir(config["model_path"]):
        model_paths = glob.glob(os.path.join(config["model_path"], "*.pth"))
        model_paths.sort()
    else:
        model_paths = [config["model_path"]]

    saved_dir = config["saved_dir"]
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    for i, model_path in enumerate(model_paths):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        infer = Inference(model.name, model, data_loader, loss_fn, metric_fns, device)
        total_loss, total_metrics, output_dict = infer.inference()
        model_index = re.findall(r'\d+\.\d+|\d+', model_path)[-1]
        filename = f"{model.name}_checkpoint_{int(model_index)}.json"
        with open(os.path.join(saved_dir, filename), 'w', encoding='utf-8') as js:
            json.dump(output_dict, js, ensure_ascii=False)

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        print('-'*100)
        print(model_path)
        logger.info(log)
        print('-'*100)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config/test/TGCN.jsonc", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
