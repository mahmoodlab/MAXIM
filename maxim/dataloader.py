import numpy as np
import pandas as pd

import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torchvision.transforms as transforms

class MxIFReader(Dataset):
    def __init__(self, data_csv_path, split_name, marker_panel, input_markers, output_markers, training=False, img_size=256, percent=100):
        """
        Custom PyTorch dataset for reading and preprocessing multi-channel multiplexed images.

        Args:
            data_csv_path (str): Path to a CSV file with Image_Path and Split_Name columns.
            split_name (str): Name of the split (e.g., 'train', 'valid', 'test', 'inference').
            marker_panel (list): A list of marker names in the same order as the channels in the MXIF images
            input_markers (list): A list of marker names to be used as input to the model.
            output_markers (list): A list of marker names to be used as output to the model.
            training (bool, optional): If True, preprocesses the images for training; otherwise, for validation. Defaults to False.
            img_size (int, optional): The size of the images to be returned by the data loader. Defaults to 256.
            percent (int, optional): Percentage of samples to include from the data. Defaults to 100%.
        """
        self.data_csv_path = data_csv_path
        self.split_name = split_name
        self.marker_panel = marker_panel
        self.input_markers = input_markers
        self.output_markers = output_markers
        self.training = training
        self.img_size = img_size

        # Read the CSV file and filter rows based on split_name
        df = pd.read_csv(self.data_csv_path)
        df = df[df['Split_Name'] == split_name]
        self.x = df['Image_Path'].tolist()

        if percent < 100:
            # Randomly select a subset of samples based on percent
            rand_perm = np.random.permutation(len(self.x))
            sample_count = int(len(self.x)*percent/100)
            rand_perm = rand_perm[:sample_count]
            self.x = [self.x[idx] for idx in rand_perm]

        # Get the channel indexes for input and output markers
        self.input_channel_indexes = []
        for marker_name in self.input_markers:
            self.input_channel_indexes.append(self.marker_panel.index(marker_name.upper()))

        self.output_channel_indexes = []
        for marker_name in self.output_markers:
            self.output_channel_indexes.append(self.marker_panel.index(marker_name.upper()))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns a tuple containing the input and output images, image path, and image dimensions.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing input_img, output_img, img_path, img_dim.
        """
        img_path = self.x[idx]
        img = np.load(img_path)  # Load the image as a numpy array (N, H, W) where N is the number of markers
        img = torch.tensor(img / 255.0)  # Convert to tensor and normalize to [0, 1]
        img_dim = [img.shape[1], img.shape[2]]

        if self.training:
            img = self.preprocess_train(img, self.img_size)
        else:
            img = self.preprocess_valid(img, self.img_size)

        input_img = torch.zeros((len(self.input_channel_indexes), img.shape[1], img.shape[2]))
        output_img = torch.zeros((len(self.output_channel_indexes), img.shape[1], img.shape[2]))

        for i in range(len(self.input_channel_indexes)):
            input_img[i, :, :] = img[self.input_channel_indexes[i], :, :]

        for i in range(len(self.output_channel_indexes)):
            output_img[i, :, :] = img[self.output_channel_indexes[i], :, :]

        return input_img, output_img, img_path, img_dim

    @staticmethod
    def preprocess_train(img, target_size):
        """
        Preprocesses the image for training by applying random crops, horizontal, and vertical flips.

        Args:
            img (torch.Tensor): Input image tensor.
            target_size (int): Target size for random crop.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        preprocess = transforms.Compose([
            transforms.RandomCrop(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        img = preprocess(img)
        img = torch.rot90(img, np.random.randint(4), [1, 2])
        return img

    @staticmethod
    def preprocess_valid(img, target_size):
        """
        Preprocesses the image for validation by applying center crop and padding if necessary.

        Args:
            img (torch.Tensor): Input image tensor.
            target_size (int): Target size for center crop.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if target_size > img.shape[1]:
            rows = target_size - img.shape[1]
            cols = target_size - img.shape[2]
            img = f.pad(img, (0, cols, 0, rows), "constant", 0)
        preprocess = transforms.Compose([
            transforms.CenterCrop(target_size)
        ])
        img = preprocess(img)
        return img

    @staticmethod
    def get_data_loader(dataset, batch_size=4, training=False, num_workers=4):
        """
        Returns a data loader for the given dataset.

        Args:
            dataset (MxIFReader): Custom dataset object.
            batch_size (int, optional): Batch size. Defaults to 4.
            training (bool, optional): If True, creates a training data loader; otherwise, creates a validation data loader. Defaults to False.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.

        Returns:
            torch.utils.data.DataLoader: Data loader object.
        """
        if training:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), drop_last=False, num_workers=num_workers)
        return loader
