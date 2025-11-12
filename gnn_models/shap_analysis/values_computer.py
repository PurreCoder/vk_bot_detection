import torch
import numpy as np


class ValuesComputer:
    def __init__(self, model, feature_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.to(device)
        self.model.eval()

    def sample_data(self, data, size):
        if len(data) > size:
            idx = np.random.choice(len(data), size, replace=False)
            data = data[idx]
        return data

    def prepare_data(self, data, background_size, test_size):
        # Prepare background data (reference dataset)
        background_data = data.cpu().x.detach().numpy()[data.train_mask]
        background_data = self.sample_data(background_data, background_size)

        # Prepare test data
        test_data = data.cpu().x.detach().numpy()[data.test_mask]
        test_data = self.sample_data(test_data, test_size)

        print(f"Background data shape: {background_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        return background_data, test_data
