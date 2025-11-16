import numpy as np
import shap
import torch
from torch_geometric.data import Data
from gnn_models.model_1.model import Model as my_model
from gnn_models.data_producer import DataProducer
from gnn_models.shap_analysis.values_computer import ValuesComputer


class KernelValuesComputer(ValuesComputer):
    def _model_predict(self, x_array):
        """
        Wrapper function for SHAP that properly handles 2D numpy array input.
        SHAP expects: input shape (n_samples, n_features), output shape (n_samples, n_classes)
        """

        with torch.no_grad():
            # Convert to tensor
            x_tensor = torch.tensor(x_array, dtype=torch.float).to(self.device)

            # For GNN models, we need to create a proper graph structure
            n_samples = x_tensor.shape[0]

            # Create edge_index: simple fully connected graph for the batch
            if n_samples > 1:
                edge_index, edge_attr = DataProducer(my_model).build_edges(x_array)
            else:
                edge_index = torch.LongTensor([list(range(n_samples)), list(range(n_samples))]).to(self.device)
                #edge_attr = torch.tensor([1] * n_samples, dtype=torch.float).to(self.device)

            if edge_index is None or edge_index.numel() == 0:
                edge_index = torch.LongTensor([list(range(n_samples)), list(range(n_samples))]).to(self.device)
                # edge_index = torch.tensor([list(range(n_samples)), list(range(n_samples))], dtype=torch.long).t().to(self.device)
                #edge_attr = torch.tensor([1] * n_samples, dtype=torch.float).to(self.device)

            # Create data object
            data = Data(x=x_tensor, edge_index=edge_index, feature_names=self.feature_names).to(self.device)

            # Get model predictions
            outputs = self.model(data)

            # Convert from log_softmax to probabilities
            probabilities = torch.exp(outputs)

            # Return as numpy array with shape (n_samples, n_classes)
            return probabilities.cpu().numpy()

    def compute_in_batches(self, kernel_explainer, test_data, n_samples=100):
        PART_SIZE = 1
        n_parts = test_data.shape[0] // PART_SIZE
        test_parts = np.array_split(test_data, n_parts)

        shap_parts = []
        for part in test_parts:
            shap_part = kernel_explainer.shap_values(part, nsamples=n_samples)
            shap_parts.append(shap_part)

        return np.concatenate(shap_parts)

    def get_shap_values(self, data, background_size=40, test_size=None, n_samples=100):
        """Compute SHAP values using Kernel Explainer"""
        print("Computing SHAP values using Kernel Explainer...")

        if test_size is None:
            test_size = data.test_mask.sum().item()

        # Prepare background data (reference dataset)
        background_data = self.prepare_data(data, data.train_mask, background_size)
        # Prepare test data (what we calculate values for)
        test_data = self.prepare_data(data, data.test_mask, test_size)

        print(f"Background data shape: {background_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        explainer = shap.KernelExplainer(self._model_predict, background_data)
        shap_values = self.compute_in_batches(explainer, test_data, n_samples)
        shap_values = shap_values[:, :, 1]

        return shap_values, test_data
