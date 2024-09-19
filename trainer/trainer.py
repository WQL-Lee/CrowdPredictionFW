import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import time


class Trainer(BaseTrainer):
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

        

    def train_default(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))


            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    def train_crowd_cnn_gru(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        mean_std = self.data_loader.dataset.mean_std
        for batch_idx, (x_g, t_g, cam_valid, _, flight) in enumerate(self.data_loader):

            x_g,t_g,cam_valid,flight=x_g.to(self.device),t_g.to(self.device),cam_valid.to(self.device),flight.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x_g,t_g[:,:,:,:], flight[:,:,:,:])
            output = torch.where(cam_valid[:,:,:,:]==True,output, t_g[:,:,:,:])
            loss = self.criterion(output, t_g[:,:,:,:])

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                met_value=met(t_g[:,:,:,:].cpu().detach().numpy(),output.cpu().detach().numpy(), mean_std, cam_valid[:,:,:,:].cpu().detach().numpy())
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
    
    def train_tgcn(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        ## handling the max_value, which is used for normlization
        max_value = self.data_loader.dataset.terminal_max

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (his_data, targets, _) in enumerate(self.data_loader):
            his_data = his_data/max_value
            targets = targets/max_value
            his_data = his_data.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(his_data)
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
    
    def valid_tgcn(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        ## handling the max_value, which is used for normlization
        max_value = self.data_loader.dataset.terminal_max


        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (his_data, targets,_) in enumerate(self.valid_data_loader):
                his_data = his_data/ max_value
                targets = targets/ max_value
                his_data, targets = his_data.to(self.device), targets.to(self.device)

                outputs = self.model(his_data)
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
    
    def _train_epoch(self, epoch):
        if not self.model_name:
            log = self.train_default(epoch)
        elif self.model_name == "Crowd_CNN_GRU":
            start = time.time()
            log = self.train_crowd_cnn_gru(epoch)
            end = time.time()
            print(f"Epoch {epoch}: Time cost: {end-start}s")
        elif self.model_name == "TGCN": 
            start = time.time()
            log = self.train_tgcn(epoch)
            end = time.time()
            print(f"Epoch {epoch}: Time cost: {end-start}s")
        else:
            print("The training model has not specified!")
            exit(-1)
        return log

    def _valid_epoch(self, epoch):
        if self.model_name == "TGCN":
            val_log = self.valid_tgcn(epoch)
        else:
            print("The training model has not specified!")
            exit(-1)

        return val_log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
