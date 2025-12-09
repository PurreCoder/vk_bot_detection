import numpy as np
import shap
import torch
from shap import Explanation
from data_processing.data_processor import DataProcessor
from shap_analysis.gradient_values_computer import GradientValuesComputer
from gnn_models.model_1.model import Model as my_model


def modify_input_hook(module, model_input):
    model_input = model_input[0]
    edge_index, _ = DataProcessor(my_model).get_connections_normalized(model_input.detach().cpu().numpy())
    edge_index = edge_index.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model_input, edge_index

def modify_output_hook(module, model_input, model_output):
    return torch.exp(model_output)  # [:,1]


class ExplanationConstructor:
    def __init__(self, model, graph_data, data_to_explain):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.background_data = graph_data.x.detach().cpu()[graph_data.train_mask.detach().cpu().numpy()].requires_grad_(True).to(self.device)
        self.data_to_explain = data_to_explain
        self.model = model.requires_grad_(True).to(self.device)
        self.background_data.to(self.device)

    def create_explanation(self):
        input_hook_handle = self.model.register_forward_pre_hook(modify_input_hook)
        output_hook_handle = self.model.register_forward_hook(modify_output_hook)

        tensor_to_explain = torch.Tensor(self.data_to_explain).requires_grad_(True).to(self.device)

        self.model.eval()
        explainer = shap.GradientExplainer(self.model, self.background_data)
        explanation = explainer(tensor_to_explain)

        input_hook_handle.remove()
        output_hook_handle.remove()

        return explanation

    def get_shap_values(self):
        computer = GradientValuesComputer(self.model, my_model.feature_names)
        shap_values = computer.get_values_for_test(self.data_to_explain)
        #print("Shap:")
        #print(shap_values)
        return shap_values

    def inject_explanation(self, explanation, attr_name, new_value):
        required_attr = ('values',
                        'base_values',
                        'data',
                        'display_data',
                        'instance_names',
                        'feature_names',
                        'output_names',
                        'output_indexes',
                        'lower_bounds',
                        'upper_bounds',
                        'error_std',
                        'main_effects',
                        'hierarchical_values',
                        'clustering',
                        'compute_time')
        attr_dict = {attr: getattr(explanation, attr) for attr in required_attr}
        attr_dict[attr_name] = new_value
        new_expl = Explanation(**attr_dict)
        return new_expl

    def alter_explanation(self, explanation):
        shap_values = self.get_shap_values()
        explanation = self.inject_explanation(explanation, 'values', shap_values)
        return explanation

    def construct_prediction(self):
        explanation = self.create_explanation()
        explanation = self.inject_explanation(explanation, 'values', explanation[:, :, 1])
        #explanation = self.alter_explanation(explanation)
        explanation = self.inject_explanation(explanation, 'data', self.data_to_explain)
        explanation.base_values = np.zeros(explanation.values.shape[1])

        return explanation
