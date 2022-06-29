import csv
import os
import torch
import time
import numpy as np
from tqdm import tqdm
from metrics import DiceScore


class EpochLogger:
    r"""
    Keeps a log of metrics in the current epoch.
    """
    def __init__(self):
        self.log = {'train loss': 0., 'train dice': 0., 'val loss': 0., 'val dice': 0.}

    def update_log(self, metrics, phase):
        r"""
        Update the metrics in the current epoch, this method is called at every step of the epoch.
        :param metrics: dict
            Metrics to update: loss, PSNR and SSIM.
        :param phase: str
            Phase of the current epoch: training (train) or validation (val).
        :return: None
        """
        for key, value in metrics.items():
            self.log[' '.join([phase, key])] += value

    def get_log(self, n_samples, phase):
        r"""
        Returns the average of the monitored metrics in the current moment,
        given the number of evaluated samples.
        :param n_samples: int
            Number of evaluated samples.
        :param phase: str
            Phase of the current epoch: training (train) or validation (val).
        :return:
        """
        log = {
            phase + ' loss': self.log[phase + ' loss'] / n_samples,
            phase + ' dice': self.log[phase + ' dice'] / n_samples
        }
        return log


class FileLogger(object):
    r"""
    Keeps a log of the whole training and validation process.
    The results are recorded in a CSV files.

    Args:
        ile_path (string): path of the csv file.
    """
    def __init__(self, file_path):
        """
        Creates the csv record file.
        :param f
        """
        self.file_path = file_path
        header = ['epoch', 'lr', 'train loss', 'train dice', 'val loss', 'val dice']

        with open(self.file_path, 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(header)

    def __call__(self, epoch_log):
        """
        Updates the CSV record file.
        :param epoch_log: dict
            Log of the current epoch.
        :return:
        """

        # Format log file:
        # Epoch and learning rate:
        log = ['{:03d}'.format(epoch_log['epoch']), '{:.5e}'.format(epoch_log['learning rate'])]

        # Training loss, DICE:
        log.extend(['{:.5e}'.format(epoch_log['train loss']), '{:.5f}'.format(epoch_log['train dice'])])

        # Validation loss, DICE
        # Validation might not be done at all epochs, in that case the default calue is zero.
        log.extend(['{:.5e}'.format(epoch_log.get('val loss', 0.)), '{:.5f}'.format(epoch_log.get('val dice', 0.))])

        with open(self.file_path, 'a') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(log)


def fit_model(model, data_loaders, criterion, optimizer, scheduler, device, n_epochs,
              val_freq, checkpoint_dir, model_name, verbose=False):
    """
    Training of the denoiser model.
    :param model: torch Module
        Neural network to fit.
    :param data_loaders: dict
        Dictionary with torch DataLoaders with training and validation datasets.
    :param criterion: torch Module
        Loss function.
    :param optimizer: torch Optimizer
        Gradient descent optimization algorithm.
    :param scheduler: torch lr_scheduler
        Learning rate scheduler.
    :param device: torch device
        Device used during training (CPU/GPU).
    :param val_freq: int
        Interval of the validation process.
    :param n_epochs: int
        Number of epochs to fit the model.
    :param checkpoint_dir: str
        Path to the directory where the model checkpoints and CSV log files will be stored.
    :param model_name: str
        Prefix name of the trained model saved in checkpoint_dir.
    :param verbose: bool
        Show progress bar.
    :return:
    """
    dice = DiceScore(to_one_hot=True, n_classes=2, apply_activation=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logfile_path = os.path.join(checkpoint_dir,  ''.join([model_name, '_logfile.csv']))
    model_path = os.path.join(checkpoint_dir, ''.join([model_name, '-{:03d}-{:.5e}-{:.5f}{}.pth']))
    file_logger = FileLogger(logfile_path)
    best_model_path, best_dice, best_loss = '', -np.inf, np.inf
    since = time.time()

    for epoch in range(1, n_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        epoch_logger = EpochLogger()
        epoch_log = dict()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val' and epoch % val_freq == 0:
                model.eval()
            else:
                break

            if verbose:
                if phase == 'train':
                    print('\nEpoch: {}/{} - Learning rate: {:.4e}'.format(epoch, n_epochs, lr))
                    description = 'Training - Loss:{:.5e} - DICE:{:.5f}'
                else:
                    description = 'Validation - Loss:{:.5e} - DICE:{:.5f}'
                iterator = tqdm(enumerate(data_loaders[phase], 1), total=len(data_loaders[phase]), ncols=85)
                iterator.set_description(description.format(0, 0))
            else:
                iterator = enumerate(data_loaders[phase], 1)

            for step, (inputs, targets) in iterator:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                metrics = {'loss': loss.item(), 'dice': dice(outputs, targets).item()}
                epoch_logger.update_log(metrics, phase)
                log = epoch_logger.get_log(step, phase)
                if verbose:
                    iterator.set_description(description.format(log[phase + ' loss'], log[phase + ' dice']))

            if phase == 'val':
                save_model = False
                if log['val dice'] > best_dice:
                    best_dice, save_model = log['val dice'], True
                if log['val loss'] < best_loss:
                    best_loss, save_model = log['val loss'], True

                if save_model:
                    best_model_path = model_path.format(epoch, log['val loss'], log['val dice'], '-improve')

                torch.save(model.state_dict(), best_model_path)

            elif scheduler is not None:         # Apply another scheduler at epoch level.
                scheduler.step()

            epoch_log = {**epoch_log, **log}

        # Save the current epoch metrics in a CVS file.
        epoch_data = {'epoch': epoch, 'learning rate': lr, **epoch_log}
        file_logger(epoch_data)

    # Save the last model and report training time.
    best_model_path = model_path.format(epoch, log['val loss'], log['val dice'], 'final')
    torch.save(model.state_dict(), best_model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best PSNR: {:4f}'.format(best_dice))
