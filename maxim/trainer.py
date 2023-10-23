import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients

from .dataloader import MxIFReader
from .networks import Generator, weights_init

from scipy import stats as st
from skimage.metrics import structural_similarity as ssim


class Trainer:
    def __init__(self, marker_panel, input_markers, output_markers, results_dir, lr=0.002, seed=1):
        """
        Trainer class for training and evaluating a protein marker imputation model.

        Args:
            marker_panel (list): A list of marker names in the same order as the channels in the MXIF images.
            input_markers (list): A list of marker names to be used as input to the model.
            output_markers (list): A list of marker names to be used as output to the model.
            results_dir (str): Directory to store the results.
            lr (float, optional): Learning rate for the adam optimizer. Defaults to 0.002.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
        """
        self.marker_panel = marker_panel
        self.input_markers = input_markers
        self.output_markers = output_markers
        self.results_dir = results_dir
        self.lr = lr
        self.seed = seed

        self.counter = 0
        self.lowest_loss = np.Inf

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

        self.model_g = None
        self.optimizer = None
        self.loss_l1 = None
        self.loss_mse = None

        os.makedirs(self.results_dir, exist_ok=True)

    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Args:
            seed (int): Random seed.
        """

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if str(self.device.type) == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_data_loader(self, data_csv_path, percent=100, img_size=256, batch_size=64, num_workers=4):
        """
        Initializes the data loader for training and validation data.

        Args:
            data_csv_path (str): Path to the data CSV file.
            percent (int, optional): Percentage of data to use. Defaults to 100.
            img_size (int, optional): Size of the input images. Defaults to 256.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """
        self.train_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='train', marker_panel=self.marker_panel,
                                        input_markers=self.input_markers, output_markers=self.output_markers,
                                        training=True, img_size=img_size, percent=percent)
        self.train_loader = MxIFReader.get_data_loader(self.train_dataset, batch_size=batch_size, training=True,
                                                       num_workers=num_workers)

        self.valid_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='valid', marker_panel=self.marker_panel,
                                        input_markers=self.input_markers, output_markers=self.output_markers,
                                        training=False, img_size=img_size)
        self.valid_loader = MxIFReader.get_data_loader(self.train_dataset, batch_size=batch_size, training=False,
                                                       num_workers=num_workers)


    def init_model(self, is_train=False):
        """
        Initializes the marker imputation model.

        Args:
            is_train (bool, optional): If True, move the model to one of the gpu if available. Defaults to False.
        """
        self.model_g = Generator(in_channels=len(self.input_markers), out_channels=len(self.output_markers), init_features=32)
        self.model_g = self.model_g.apply(weights_init)
        if is_train:
            self.model_g = self.model_g.to(device=self.device)

    def init_optimizer(self):
        """Initializes the optimizer."""
        self.optimizer = optim.Adam(self.model_g.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def init_loss_function(self):
        """Initializes the loss functions."""
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()

    def load_model(self, ckpt_path):
        """
        Loads the model from a checkpoint file.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        ckpt = torch.load(ckpt_path)
        ckpt_clean = {}
        for key in ckpt.keys():
            ckpt_clean.update({key.replace('module.', ''): ckpt[key]})
        self.model_g.load_state_dict(ckpt_clean, strict=True)
        self.model_g = self.model_g.to(device=self.device)

    def train(self, data_csv_path, percent=100, img_size=256, batch_size=64, num_workers=4, max_epochs=200,
              minimum_epochs=50, patience=25):
        """
        Trains the marker imputation model.

        Args:
            data_csv_path (str): Path to the data CSV file.
            percent (int, optional): Percentage of data to use. Defaults to 100.
            img_size (int, optional): Size of the input images. Defaults to 256.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 200.
            minimum_epochs (int, optional): Minimum number of epochs before early stopping. Defaults to 50.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 25.

        Returns:
            dict: Dictionary containing training and validation total loss, L1 loss, and MSE loss.
        """
        self.counter = 0
        self.lowest_loss = np.Inf
        self.set_seed(seed=self.seed)
        self.init_data_loader(data_csv_path, percent=percent, img_size=img_size, batch_size=batch_size, num_workers=num_workers)
        self.init_model(is_train=True)
        self.init_optimizer()
        self.init_loss_function()

        result_dict = {'train_loss': [], 'valid_loss': [], 'train_l1': [], 'valid_l1': [], 'train_mse': [], 'valid_mse': []}
        for epoch in range(max_epochs):
            start_time = time.time()

            train_loss, train_l1, train_mse = self.train_loop(self.train_loader)
            print('\rTrain Epoch: {}, train_loss: {:.4f}, train_l1: {:.4f}, train_mse: {:.4f}      '.format(epoch, train_loss, train_l1, train_mse))
            result_dict['train_loss'].append(train_loss)
            result_dict['train_l1'].append(train_l1)
            result_dict['train_mse'].append(train_mse)

            valid_loss, valid_l1, valid_mse = self.valid_loop(self.valid_loader)
            print('\rValid Epoch: {}, valid_loss: {:.4f}, valid_l1: {:.4f}, valid_mse: {:.4f}     '.format(epoch, valid_loss, valid_l1, valid_mse))
            result_dict['valid_loss'].append(valid_loss)
            result_dict['valid_l1'].append(valid_l1)
            result_dict['valid_mse'].append(valid_mse)

            if self.lowest_loss > valid_loss:
                print('--------------------Saving best model--------------------')
                torch.save(self.model_g.state_dict(), os.path.join(self.results_dir, 'checkpoint.pt'))
                self.lowest_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                print('Loss is not decreased in last %d epochs' % self.counter)

            if (self.counter > patience) and (epoch >= minimum_epochs):
                break

            total_time = time.time() - start_time
            print('Time to process epoch({}): {:.4f} minutes                             \n'.format(epoch, total_time/60))
            pd.DataFrame.from_dict(result_dict).to_csv(os.path.join(self.results_dir, 'training_stats.csv'), index=False)
        return result_dict

    def train_loop(self, data_loader):
        """
        Training loop for a single epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the training data.

        Returns:
            float: Average training loss.
            float: Average L1 loss.
            float: Average MSE loss.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        self.model_g.train()
        batch_count = len(data_loader)
        for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
            self.model_g.zero_grad()

            input_batch = input_batch.to(self.device)
            output_batch = output_batch.to(self.device)

            generated_output_batch = self.model_g(input_batch)

            error_l1 = self.loss_l1(output_batch, generated_output_batch)
            error_mse = self.loss_mse(output_batch, generated_output_batch)
            error = error_l1 + error_mse

            error.backward()
            self.optimizer.step()

            print('Training - [%d/%d]\tL1 Loss: %.06f \tMSE Loss: %.06f                                                            '
                  % (batch_idx, batch_count, error_l1.item(), error_mse.item()), end='\r')
            total_error += error.item()
            total_error_l1 += error_l1.item()
            total_error_mse += error_mse.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def valid_loop(self, data_loader):
        """
        Validation loop for a single epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the validation data.

        Returns:
            float: Average validation loss.
            float: Average L1 loss.
            float: Average MSE loss.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        self.model_g.eval()
        batch_count = len(data_loader)
        with torch.no_grad():
            for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)

                generated_output_batch = self.model_g(input_batch)

                error_l1 = self.loss_l1(output_batch, generated_output_batch)
                error_mse = self.loss_mse(output_batch, generated_output_batch)
                error = error_l1 + error_mse

                print('Validation - [%d/%d]\tL1 Loss: %.06f \tMSE Loss: %.06f                      '
                    % (batch_idx, batch_count, error_l1.item(), error_mse.item()), end='\r')
                total_error += error.item()
                total_error_l1 += error_l1.item()
                total_error_mse += error_mse.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def eval(self, data_csv_path, split_name='test', img_size=256, batch_size=64, num_workers=4):
        """
        Evaluates the trained model on the test data.

        Args:
            data_csv_path (str): Path to the data CSV file.
            split_name (str, optional): Name of the data split to evaluate. Defaults to 'test'.
            img_size (int, optional): Size of the input images. Defaults to 256.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """
        self.set_seed(self.seed)
        dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name, marker_panel=self.marker_panel,
                                        input_markers=self.input_markers, output_markers=self.output_markers,
                                        training=False, img_size=img_size)
        data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size, training=False, num_workers=num_workers)

        self.init_model()
        self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint.pt'))
        eval_dir_name = '%s_%d_%d' % (split_name, img_size, img_size)
        self.eval_loop(data_loader, eval_dir_name)

    def eval_loop(self, data_loader, eval_dir_name):
        """
        Evaluation loop for the test data.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the test data.
            eval_dir_name (str): Name of the directory to save the evaluation results.
        """
        stats_dict = {"Image_Name":[], "MAE":[], "MSE": [], "SSIM": [], "PSNR": [], "RMSE": [], "Corr": [], "p-value": []}
        stats_dict_zeros = {"Image_Name": [], "MAE": [], "MSE": [], "SSIM": [], "PSNR": [], "RMSE": []}
        stats_dict_mean = {"Image_Name": [], "MAE": [], "MSE": [], "SSIM": [], "PSNR": [], "RMSE": []}

        os.makedirs(os.path.join(self.results_dir, eval_dir_name), exist_ok=True)
        batch_count = len(data_loader)
        self.model_g.eval()
        with torch.no_grad():
            for batch_idx, (input_batch, output_batch, image_name_batch, img_dims) in enumerate(data_loader):

                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)
                generated_batch = self.model_g(input_batch)

                for i, image_name in enumerate(image_name_batch):
                    img_dim = [img_dims[0][i].item(), img_dims[1][i].item()]
                    image_name = os.path.basename(image_name)
                    image_name, ext = os.path.splitext(image_name)

                    print('%d/%d - (%d) %s' % (batch_idx, batch_count, i, image_name))
                    input = input_batch[i, :, :, :].detach().cpu().numpy()
                    real = output_batch[i, :, :, :].detach().cpu().numpy()
                    generated = generated_batch[i, :, :, :].detach().cpu().numpy()

                    input = input[:, :img_dim[0], :img_dim[1]]
                    real = real[:, :img_dim[0], :img_dim[1]]
                    generated = generated[:, :img_dim[0], :img_dim[1]]

                    output = np.concatenate([real, generated], axis=0)
                    np.save(os.path.join(self.results_dir, eval_dir_name, image_name + '.npy'), output)

                    input = input * 255.0
                    real = real * 255.0
                    generated = generated * 255.0

                    zero_image = np.zeros_like(real)
                    mean_image = np.mean(input, axis=0)

                    stats = self.pixel_metrics(real, generated, max_val=255, baseline=False)
                    stats_zeros = self.pixel_metrics(real, zero_image, max_val=255, baseline=True)
                    stats_mean = self.pixel_metrics(real, mean_image, max_val=255, baseline=True)

                    stats_dict["Image_Name"].append(image_name)
                    stats_dict_zeros["Image_Name"].append(image_name)
                    stats_dict_mean["Image_Name"].append(image_name)
                    for key in stats.keys():
                        stats_dict[key].append(stats[key])
                    for key in stats_zeros.keys():
                        stats_dict_zeros[key].append(stats_zeros[key])
                    for key in stats_mean.keys():
                        stats_dict_mean[key].append(stats_mean[key])

            pd.DataFrame.from_dict(stats_dict).to_csv(os.path.join(self.results_dir, '%s_stats.csv' % eval_dir_name), index=False)
            pd.DataFrame.from_dict(stats_dict_zeros).to_csv(os.path.join(self.results_dir, '%s_stats_zero.csv' % eval_dir_name), index=False)
            pd.DataFrame.from_dict(stats_dict_mean).to_csv(os.path.join(self.results_dir, '%s_stats_mean.csv' % eval_dir_name), index=False)

    @staticmethod
    def pixel_metrics(real, generated, max_val=255, baseline=False):
        """
        Computes pixel-level evaluation metrics between the ground truth and generated images.

        Args:
            real (numpy.ndarray): Ground truth images.
            generated (numpy.ndarray): Generated images.
            max_val (float, optional): Maximum pixel value. Defaults to 255.
            baseline (bool, optional): If True, compares against a baseline image. Defaults to False.

        Returns:
            dict: Dictionary containing pixel-level evaluation metrics.
        """
        real= np.squeeze(real)
        generated = np.squeeze(generated)
        stats = {}
        stats["MAE"] = np.mean(np.abs(real - generated))
        stats["MSE"] = np.mean((real - generated) ** 2)
        stats["RMSE"] = np.sqrt(stats["MSE"])
        stats["PSNR"] = 20 * np.log10(max_val) - 10.0 * np.log10(stats["MSE"])
        stats["SSIM"] = ssim(real, generated, data_range=max_val)
        if not baseline:
            corr, p_value = st.pearsonr(real.flatten(), generated.flatten())
            stats['Corr'] = corr
            stats['p-value'] = p_value
        return stats

    def attributions(self, data_csv_path, split_name='test', img_size=256, batch_size=32, num_workers=4):
        """
         Computes and saves attributions for the test data.

         Args:
             data_csv_path (str): Path to the data CSV file.
             split_name (str, optional): Name of the data split to evaluate. Defaults to 'test'.
             img_size (int, optional): Size of the input images. Defaults to 256.
             batch_size (int, optional): Batch size. Defaults to 32.
             num_workers (int, optional): Number of workers for data loading. Defaults to 4.
         """
        self.set_seed(self.seed)
        dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name, marker_panel=self.marker_panel,
                                        input_markers=self.input_markers, output_markers=self.output_markers,
                                        training=False, img_size=img_size)
        data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size, training=False, num_workers=num_workers)

        self.init_model()
        self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint.pt'))
        attr_dir_name = 'attributions_%s_%d_%d' % (split_name, img_size, img_size)
        self.attributions_loop(data_loader, attr_dir_name)

    def attributions_loop(self, data_loader, attr_dir_name):
        """
        Attribution computation loop for the test data.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the test data.
            attr_dir_name (str): Name of the directory to save the attributions.
        """
        os.makedirs(os.path.join(self.results_dir, attr_dir_name), exist_ok=True)
        image_path_list = []
        attr_array = None
        attr_array_pos = None
        attr_array_neg = None
        batch_count = len(data_loader)
        ig = IntegratedGradients(self.interpretable_model)
        self.model_g.eval()
        for batch_idx, (input_batch, output_batch, image_name_batch, img_dims) in enumerate(data_loader):
            input_batch = input_batch.to(self.device)
            input_batch.requires_grad_()
            attr, _ = ig.attribute(input_batch, baselines=torch.zeros_like(input_batch, device=self.device), target=0,
                                   return_convergence_delta=True)

            for i, image_name in enumerate(image_name_batch):
                image_path_list.append(image_name)
                image_name = os.path.basename(image_name)
                image_name, ext = os.path.splitext(image_name)
                print('%d/%d - %s' % (batch_idx, batch_count, image_name))

                img_dim = [img_dims[0][i].item(), img_dims[1][i].item()]

                attr = attr.detach().cpu().numpy()
                attr_ = attr[i, :, :img_dim[0], :img_dim[1]]
                np.save(os.path.join(self.results_dir, attr_dir_name, image_name + '_attr.npy'), attr_)
                pos_attr_ = np.maximum(attr_, 0)
                neg_attr_ = np.maximum((-1)*attr_, 0)
                attr_ = np.expand_dims(np.sum(np.sum(np.abs(attr_), axis=-1), axis=-1), axis=0)
                pos_attr_ = np.expand_dims(np.sum(np.sum(pos_attr_, axis=-1), axis=-1), axis=0)
                neg_attr_ = np.expand_dims(np.sum(np.sum(neg_attr_, axis=-1), axis=-1), axis=0)

                if attr_array is None:
                    attr_array_pos = pos_attr_
                    attr_array_neg = neg_attr_
                    attr_array = attr_
                else:
                    attr_array = np.concatenate((attr_array,attr_), axis=0)
                    attr_array_pos = np.concatenate((attr_array_pos, pos_attr_), axis=0)
                    attr_array_neg = np.concatenate((attr_array_neg, neg_attr_), axis=0)

        marker_names = data_loader.dataset.input_markers
        df = pd.DataFrame(attr_array, columns=marker_names)
        df_pos = pd.DataFrame(attr_array_pos, columns=marker_names)
        df_neg = pd.DataFrame(attr_array_neg, columns=marker_names)
        df['image_path'] = image_path_list
        df_pos['image_path'] = image_path_list
        df_neg['image_path'] = image_path_list
        df_pos.to_csv(os.path.join(self.results_dir, '%s_attributions_pos.csv' % attr_dir_name), index=False)
        df_neg.to_csv(os.path.join(self.results_dir, '%s_attributions_neg.csv' % attr_dir_name), index=False)
        df.to_csv(os.path.join(self.results_dir, '%s_attributions_abs.csv' % attr_dir_name), index=False)

    def interpretable_model(self, batch):
        pred = self.model_g(batch)
        pred = nn.AdaptiveAvgPool2d((1,1))(pred)
        return pred
