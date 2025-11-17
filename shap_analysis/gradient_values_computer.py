import torch
from captum.attr import IntegratedGradients
from gnn_models.model_1.model import Model as my_model
from data_processing.data_processor import DataProcessor
from shap_analysis.values_computer import ValuesComputer


class GradientValuesComputer(ValuesComputer):

    def get_values_for_test(self, test_data):
        """Compute attribute values using Integrated Gradients"""

        print("Computing attribute values using Integrated Gradients...")

        edge_index, _ = DataProcessor(my_model).build_edges(test_data)
        test_tensor = torch.Tensor(test_data).requires_grad_(True).to(self.device)
        edge_index = edge_index.to(self.device)

        # baseline = torch.tensor(test_data.mean(axis=0)).repeat(test_size, 1).to(self.device)

        self.model.eval()
        ig = IntegratedGradients(self.model.to(self.device))
        attributions = ig.attribute(
            inputs=test_tensor,
            additional_forward_args=edge_index,
            # baselines=baseline,
            target=1,
            internal_batch_size=test_tensor.size(0)
        )

        return attributions.detach().cpu().numpy()
