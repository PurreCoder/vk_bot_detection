import torch
import numpy as np


class ValuesComputer:
    def __init__(self, model, feature_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.to(device)
        self.model.eval()

    def prepare_data(self, data, background_size, test_size):
        # Prepare background data (reference dataset)
        background_data = data.cpu().x.detach().numpy()[data.train_mask]
        if len(background_data) > background_size:
            background_idx = np.random.choice(len(background_data), background_size, replace=False)
            background_data = background_data[background_idx]

        # Prepare test data
        test_data = data.x.detach().numpy()[data.test_mask]
        if len(test_data) > test_size:
            test_idx = np.random.choice(len(test_data), test_size, replace=False)
            test_data = test_data[test_idx]

        print(f"Background data shape: {background_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        return background_data, test_data
