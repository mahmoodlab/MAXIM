import torch
import torch.nn as nn
import torch.optim as optim

from .trainer import Trainer
from .networks import Generator, Discriminator, weights_init

class TrainerCGAN(Trainer):
    def __init__(self, marker_panel, input_markers, output_markers, results_dir, lr=0.002, seed=1):
        """
        Trainer class for Conditional Generative Adversarial Network (CGAN) training.

        Args:
            marker_panel (list): List of marker names in the same order as the channels in the MXIF images.
            input_markers (list): List of marker names to be used as input to the model.
            output_markers (list): List of marker names to be used as output from the model.
            results_dir (str): Directory to save the training results.
            lr (float): Learning rate for the optimizer (default: 0.002).
            seed (int): Random seed for reproducibility (default: 1).
        """
        super().__init__(marker_panel, input_markers, output_markers, results_dir)

        self.model_d = None
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()
        self.loss_bce = None
        self.optimizer_d = None

    def init_model(self, is_train=False):
        """
        Initializes the generator and discriminator models.

        Args:
            is_train (bool): Whether the model is in training mode (default: False).
        """
        self.model_g = Generator(in_channels=len(self.input_markers), out_channels=len(self.output_markers), init_features=32)
        self.model_g = self.model_g.apply(weights_init)

        self.model_d = Discriminator(real_channels=len(self.input_markers), gen_channels=len(self.output_markers))
        self.model_d = self.model_d.apply(weights_init)

        if is_train:
            self.model_g = self.model_g.to(device=self.device)
            self.model_d = self.model_d.to(device=self.device)

    def init_optimizer(self):
        """
        Initializes the optimizers for generator and discriminator.
        """
        self.optimizer = optim.Adam(self.model_g.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def init_loss_function(self):
        """
        Initializes the loss functions for generator and discriminator.
        """
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

    def train_loop(self, data_loader):
        """
        Performs a single training loop over the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for training data.

        Returns:
            float: Average error across all batches.
            float: Average L1 loss across all batches.
            float: Average MSE loss across all batches.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        self.model_g.train()
        self.model_d.train()
        batch_count = len(data_loader)
        for batch_idx, (input_batch, output_batch, _, _,) in enumerate(data_loader):
            input_batch = input_batch.to(self.device)
            output_batch = output_batch.to(self.device)

            # Train discriminator
            with torch.cuda.amp.autocast():
                y_fake = self.model_g(input_batch)
                D_real = self.model_d(input_batch, output_batch)
                D_fake = self.model_d(input_batch, y_fake.detach())
                D_real_loss = self.loss_bce(D_real, torch.ones_like(D_real))
                D_fake_loss = self.loss_bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            self.model_d.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.optimizer_d)
            self.d_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = self.model_d(input_batch, y_fake)
                G_fake_loss = self.loss_bce(D_fake, torch.ones_like(D_fake))
                L1 = self.loss_l1(y_fake, output_batch)
                L2 = self.loss_mse(y_fake, output_batch)
                G_loss = G_fake_loss + L1 * 100

            self.optimizer.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.optimizer)
            self.g_scaler.update()

            print('Training - [%d/%d] - D_Loss: %.06f - G_Loss: %.06f - L1_Loss: %.06f - G_Fake_Loss: %.06f - D_Fake_Loss: %.06f ' %
                (batch_idx, batch_count, D_loss.item(), G_loss.item(), L1.item(), G_fake_loss.item(), D_fake_loss.item()), end='\r')

            total_error += L1.item() + L2.item()
            total_error_l1 += L1.item()
            total_error_mse += L2.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def valid_loop(self, data_loader):
        """
        Performs a validation loop over the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for validation data.

        Returns:
            float: Average error across all batches.
            float: Average L1 loss across all batches.
            float: Average MSE loss across all batches.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        self.model_g.eval()
        self.model_d.eval()
        batch_count = len(data_loader)
        with torch.no_grad():
            for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)

                # Train discriminator
                with torch.cuda.amp.autocast():
                    y_fake = self.model_g(input_batch)
                    D_real = self.model_d(input_batch, output_batch)
                    D_fake = self.model_d(input_batch, y_fake.detach())
                    D_real_loss = self.loss_bce(D_real, torch.ones_like(D_real))
                    D_fake_loss = self.loss_bce(D_fake, torch.zeros_like(D_fake))
                    D_loss = (D_real_loss + D_fake_loss) / 2

                # Train generator
                with torch.cuda.amp.autocast():
                    D_fake = self.model_d(input_batch, y_fake)
                    G_fake_loss = self.loss_bce(D_fake, torch.ones_like(D_fake))
                    L1 = self.loss_l1(y_fake, output_batch)
                    L2 = self.loss_mse(y_fake, output_batch)
                    G_loss = G_fake_loss + L1 * 100

                print('Validation - [%d/%d] - D_Loss: %.06f - G_Loss: %.06f - L1_Loss: %.06f - G_Fake_Loss: %.06f - D_Fake_Loss: %.06f ' %
                    (batch_idx, batch_count, D_loss.item(), G_loss.item(), L1.item(), G_fake_loss.item(), D_fake_loss.item()), end='\r')

                total_error += L1.item() + L2.item()
                total_error_l1 += L1.item()
                total_error_mse += L2.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count
