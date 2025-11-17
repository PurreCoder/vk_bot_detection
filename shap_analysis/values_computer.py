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

    def prepare_data(self, data, mask, size):
        data = data.x.detach().cpu().numpy()[mask.cpu().numpy()]
        data = self.sample_data(data, size)
        return data

    def get_attribute_values(self, data, test_size=None):
        """Compute attribute values using Integrated Gradients or DeepLift"""

        if test_size is None:
            test_data = data[data.train_mask]
        else:
            test_data = self.prepare_data(data, data.train_mask, test_size)

        test_tensor = torch.tensor(test_data, requires_grad=True).to(self.device)

        return self.get_values_for_test(test_tensor), test_data

    def get_values_for_test(self, test_tensor):
        pass