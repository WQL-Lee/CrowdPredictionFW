import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time


class AGCRNTrainer(BaseTrainer):
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
       
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        ## handling the max_value, which is used for normlization
        max_value = self.data_loader.dataset.terminal_max
        num_nodes = self.model.num_node
        n_his = self.data_loader.dataset.n_his

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (his_data, targets, flight_data, _) in enumerate(self.data_loader):
            flight = flight_data.unsqueeze(1)
            flight = flight.unsqueeze(3)
            flight = flight.repeat(1,num_nodes, 1, n_his)
            inputs = torch.concat((his_data, flight), 2)
            # inputs = his_data

            inputs = inputs/max_value
            targets = targets/max_value
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = targets.unsqueeze(3)
            targets = targets.permute(0, 2, 1, 3)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets, self.model)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                met_value=met(outputs, targets).cpu().item() * max_value
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
        ## handling the max_value, which is used for normlization
        max_value = self.data_loader.dataset.terminal_max
        num_nodes = self.model.num_node
        n_his = self.data_loader.dataset.n_his


        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (his_data, targets,flight_data, _) in enumerate(self.valid_data_loader):

                flight = flight_data.unsqueeze(1)
                flight = flight.unsqueeze(3)
                flight = flight.repeat(1,num_nodes, 1, n_his)
                inputs = torch.concat((his_data, flight), 2)
                # inputs = his_data

                inputs = inputs/max_value
                targets = targets/max_value
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = targets.unsqueeze(3)
                targets = targets.permute(0, 2, 1, 3)


                # his_data = his_data/ max_value
                # targets = targets/ max_value
                # his_data, targets = his_data.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, self.model)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    met_value = met(outputs, targets).cpu().item() * max_value
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
