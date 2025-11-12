import torch
from captum.attr import IntegratedGradients, DeepLiftShap
from torch_geometric.data import Data
from gnn_models.model_1.model import Model as my_model
from gnn_models.data_producer import DataProducer
from gnn_models.shap_analysis.values_computer import ValuesComputer


class DeepValuesComputer(ValuesComputer):

    def get_values(self, data, background_size=None, test_size=None):
        """Compute pseudo-SHAP values using DeepLift"""
        print("Computing pseudo-SHAP values using DeepLift...")

        if background_size is None:
            background_size = data.train_mask.sum().item()
        if test_size is None:
            test_size = data.test_mask.sum().item()

        background_data, test_data = self.prepare_data(data, background_size, test_size)
        test_tensor = torch.tensor(test_data).to(self.device)
        #edge_index, edge_attr = DataProducer(my_model).build_edges(test_data)

        #baseline = torch.tensor(test_data.mean(axis=0)).repeat(test_size, 1).to(self.device)
        #baseline = torch.zeros(test_size, background_size).to(self.device)
        #baseline_edge_index = torch.tensor([[i, i] for i in range(2)], dtype=torch.long).t().to(self.device).repeat(test_size, 1)

        self.model.eval()
        dl = DeepLiftShap(self.model.to(self.device))
        attributions = dl.attribute(
            inputs=test_tensor,#(test_tensor, edge_index),
            #baselines=torch.zeros(test_size, len(self.feature_names)).to(self.device),
            baselines=torch.tensor(test_data.mean(axis=0)).repeat(test_size, 1).to(self.device),
            target=1)

        return attributions.detach().cpu().numpy(), test_data
