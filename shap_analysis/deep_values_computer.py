import torch
from captum.attr import DeepLiftShap
from gnn_models.model_1.model import Model as my_model
from data_processing.data_processor import DataProcessor
from shap_analysis.values_computer import ValuesComputer


class DeepValuesComputer(ValuesComputer):

    def get_values_for_test(self, test_data):
        """Compute attribute values using Deep Lift"""

        print("Computing attribute values using Deep Lift...")

        edge_index, _ = DataProcessor(my_model).build_edges(test_data)
        edge_index = edge_index.to(self.device)
        test_tensor = torch.Tensor(test_data).requires_grad_(True).to(self.device)

        baseline = torch.tensor(test_data.mean(axis=0)).repeat(test_tensor.size(0), 1).to(self.device)

        self.model.eval()
        explainer = DeepLiftShap(self.model.to(self.device))
        attributions = explainer.attribute(
            inputs=test_tensor,
            additional_forward_args=(edge_index, None),
            baselines=baseline,
            target=1
        )

        return attributions.detach().cpu().numpy()

