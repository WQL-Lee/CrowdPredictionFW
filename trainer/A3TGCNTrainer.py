import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time


class A3TGCNTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        self.train_per_epoch = len(data_loader)
        if valid_data_loader is not None:
            self.valid_per_epoch = len(valid_data_loader)
        
        self.model_name = model.name
        self.scaler = self.data_loader.dataset.scaler

        # self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
       
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        total_loss = 0.0
        self.model.train()
        edge_index = self.model.edge_index

        for batch_idx, (inputs, targets, _) in enumerate(self.data_loader):
            # inputs = his_data

            inputs = inputs.permute(0,2,3,1)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            edge_index = edge_index.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, edge_index) # outputs: (batch_sz, num_node, n_pred)
            outputs = outputs.unsqueeze(3)
            outputs = outputs.permute(0,2,1,3)

            loss = self.criterion(outputs, targets,self.model)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} learning rate:{} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.optimizer.state_dict()['param_groups'][0]['lr'],
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        
        # log = self.train_metrics.result()
        log = None

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log={'val_'+k : v for k, v in val_log.items()}
        
        self.logger.info('**************************************')
        self.logger.info('Train Epoch {}: Averaged Loss: {:.6f}'.format(epoch, total_loss/self.train_per_epoch))
        if self.do_validation:
            self.logger.info('Valid Epoch {}: Averaged Loss: {:.6f}'.format(epoch, val_log['loss']))
            for key, value in val_log.items():
                self.logger.info('    {:10s}: {:.4f}'.format(str(key), value))
        self.logger.info('**************************************')

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log  

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """


        self.model.eval()
        self.valid_metrics.reset()
        edge_index = self.model.edge_index

        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(self.valid_data_loader):

                inputs = inputs.permute(0,2,3,1)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                edge_index = edge_index.to(self.device)

                outputs = self.model(inputs, edge_index) # outputs: (batch_sz, num_node, n_pred)
                outputs = outputs.unsqueeze(3)
                outputs = outputs.permute(0,2,1,3)

                # targets = self.scaler.inverse_transform(targets)
                loss = self.criterion(outputs, targets, self.model)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    met_value = met(outputs, targets).cpu().item()
                    self.valid_metrics.update(met.__name__, met_value)   

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    

    @staticmethod
    def test(model, data_loader, device, scaler):
        model.eval()
        model = model.to(device)
        edge_index = model.edge_index
        y_pred = []
        y_true = []
        y_time = []
        with torch.no_grad():
            for batch_idx, (inputs, targets,timestamp) in enumerate(data_loader):
                n_his = inputs.shape[1]
                
                # inputs: (B, T, N, C) -> (B, N, T, C)
                inputs = inputs.permute(0,2,3,1)
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                edge_index = edge_index.to(device)


                outputs = model(inputs, edge_index)
                outputs = outputs.unsqueeze(3)
                outputs = outputs.permute(0,2,1,3)
                
                # outputs (B, T, N, 1)
                # targets (B, T, N, 1)
                targets = scaler.inverse_transform(targets)
                outputs = scaler.inverse_transform(outputs)
                
                y_true.append(targets)
                y_pred.append(outputs)
                y_time.append(timestamp[n_his:])


        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred =scaler.inverse_transform(torch.cat(y_pred, dim=0))

        return y_pred, y_true, y_time
       