import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct
import pandas as pd
import h5py
import sys

# Function to apply DCT on each axis
def apply_dct(data):
    # Convert signal to a NumPy array (if not already)
    signal_array = np.array(data)
    # Apply DCT along each axis (0 for columns, 1 for rows)
    dct_result = dct(signal_array, axis=0, norm='ortho')  # Apply DCT to columns (X, Y, Z)
    return dct_result

def load_file_to_torch(path: str, activity: str) -> torch.Tensor:
    with h5py.File(path, 'r') as hf:
        data = []

        # Iterate over datasets
        for dataset_name in hf.keys():
            dataset = hf[dataset_name]

            # Extract label metadata
            label = dataset.attrs.get('activity', 'No Label')

            if label != activity:
                continue

            data.append(torch.tensor(dataset[:], dtype=torch.float32))  # Extracting data from dataset

    return data


class Dataset():
    def __init__(self, root, activities):
        self.root = root
        self.activities = activities
        self.dataset, self.labels = self.load_file(root, activities)
        self.length = self.dataset.shape[0]  # Number of samples
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx, :, :]  # Retrieve one sample
        target = self.labels[idx]  # Retrieve corresponding label
        return step, target

    def minmax_normalize(self):
        '''Min-max normalize dataset.'''
        for i in range(self.dataset.shape[1]):  # Loop over channels
            self.dataset[:, i, :] = (self.dataset[:, i, :] - self.dataset[:, i, :].min()) / (
                self.dataset[:, i, :].max() - self.dataset[:, i, :].min())

    def load_file(self, path: str, activities: list) -> (torch.Tensor, torch.Tensor):
        with h5py.File(path, 'r') as hf:
            data = []
            labels = []
            a_keys = list(activities.keys())

            # Iterate over datasets
            for dataset_name in hf.keys():
                dataset = hf[dataset_name]

                # Extract label metadata
                label = dataset.attrs.get('activity', 'No Label')
                if label not in a_keys:
                    continue

                # double each row to have 6 axes
                dataset = np.repeat(dataset, 2, axis=1)

                data.append(dataset[:])
                labels.append(activities[label])

        # shorten all samples to the length of the shortest sample
        min_length = 256
        data = [sample[:min_length, :] for sample in data if sample.shape[0] >= min_length]

        #replace nan values with 0
        data = np.nan_to_num(data)

        # Stack data to create a 3D array: [batch_size, sequence_length, num_axes]
        data = np.stack(data, axis=0)  # Shape: [batch_size, sequence_length, num_axes]

        # Transpose to match Conv1d input: [batch_size, num_axes, sequence_length]
        dataset = np.transpose(data, (0, 2, 1))  # Shape: [batch_size, num_axes, sequence_length]

        dataset = torch.from_numpy(dataset).float()
        labels = torch.tensor(labels, dtype=torch.long)

        # Print shape for debugging
        print(f"Loaded dataset shape: {dataset.shape}")

        return dataset, labels


if __name__ == '__main__':
    dataset = Dataset('./data')
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()
