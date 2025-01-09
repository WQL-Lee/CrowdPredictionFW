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
        
        self.model_name = model.name
        self.scaler = self.data_loader.dataset.scaler

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
       
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        edge_index = self.model.edge_index

        for batch_idx, (inputs, targets, _) in enumerate(self.data_loader):
            # inputs = his_data

            # inputs = inputs/max_value
            # targets = targets/max_value
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
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                met_value=met(outputs, targets).cpu().item() 
                self.train_metrics.update(met.__name__,  met_value)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} learning rate:{} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.optimizer.state_dict()['param_groups'][0]['lr'],
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if batch_idx == self.len_epoch:
                break
        
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        #
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()

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
