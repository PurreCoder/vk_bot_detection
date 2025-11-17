import torch
import config
from copy import deepcopy
from data_processing.file_manager import load_all_users, prepare_clean_folder, save_json_data
from gnn_models.gnn import BotGNN
from gnn_models.metrics_calc import compute_metrics
from gnn_models.model_transductive.model_trainer import ModelTrainer
from data_processing.data_processor import DataProcessor
from gnn_models.model_1.model import Model as my_model
from shap_analysis.explanation_pipeline import explain_false_predictions
from shap_analysis.gradient_values_computer import GradientValuesComputer
from shap_analysis.deep_values_computer import DeepValuesComputer
from shap_analysis.kernel_values_computer import KernelValuesComputer
from viz.graph_viz import *
from viz.shap_viz import SHAPVisualizer


class ModelTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        bots_users, humans_users = load_all_users(config.DATA_SOURCE['BOTS_FILE'], config.DATA_SOURCE['HUMANS_FILE'])
        self.processor = DataProcessor(my_model)
        graph_data, ids = self.processor.prepare_full_graph_data(bots_users, humans_users)

        self.models = {
            'GCN': BotGNN(graph_data.num_features, 64, 'GCN'),
            'GAT': BotGNN(graph_data.num_features, 64, 'GAT'),
            'SAGE': BotGNN(graph_data.num_features, 64, 'SAGE')
        }

        self.results = {}

        self.process_models(graph_data, ids)

    def process_models(self, graph_data, ids):
        for model in self.models.values():
            self.evaluate_model(model, graph_data, ids)

            self.best_model_name = max(self.results, key=self.results.get)
            self.best_model = self.models[self.best_model_name]

            self.visualize_model(deepcopy(model), graph_data)

        # Сохранение лучшей модели
        torch.save(self.best_model.state_dict(), config.MODELS_SAVES['TRANSDUCTIVE_SAVE'])
        print(f"\nЛучшая модель сохранена: {max(self.results, key=self.results.get)}")

        visualize_parameters_comparison([f'saves/{cur_model.model_type}.png' for cur_model in self.models.values()])

        prepare_clean_folder('saves/dependence_plots')
        prepare_clean_folder('saves/explanation_plots')

        explain_false_predictions(self.processor, deepcopy(self.models['GAT']), graph_data, SHAPVisualizer(my_model.feature_names))

        attribution_method = ('gradient', 'deeplift', 'kernel')[0]
        attr_kwargs = dict(background_size=40, test_size=None, n_samples=100) # actually applied only for kernel
        self.add_shap_analysis(self.models['SAGE'], graph_data, attribution_method, **attr_kwargs)

    def evaluate_model(self, model, graph_data, ids):
        model_name = model.model_type
        print(f"\n=== Обучение {model_name} ===")
        trainer = ModelTrainer(model, self.device)
        trainer.train(graph_data, epochs=25)

        # getting labels and model predictions for data
        y_true, y_pred = trainer.predict_labels_for_test(graph_data)

        # logging model mistakes
        test_ids = [its_id for its_id, its_bit in zip(ids, graph_data.test_mask) if its_bit]
        self.log_model_mistakes(y_true, y_pred, test_ids, model_name)

        # calculating metrics
        metrics = compute_metrics(y_true, y_pred)
        self.results[model_name] = metrics['accuracy']

        print(f"\n{model_name} Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} Precision: {metrics['precision']:.4f}")
        print(f"{model_name} Recall: {metrics['recall']:.4f}")
        print(f"{model_name} F1-Score: {metrics['f1']:.4f}")

        # ROC-AUC
        # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        # roc_auc = auc(fpr, tpr)
        # print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
        # plot_roc_curve(fpr, tpr, roc_auc)

    def log_model_mistakes(self, y_true, y_pred, test_ids, model_name):
        wrong_preds = {}
        for i in range(y_true.shape[0]):
            if y_true[i] != y_pred[i]:
                wrong_preds[str(test_ids[i])] = int(y_pred[i])

        save_json_data(wrong_preds, f'saves/{model_name}_mistakes.json')

    def visualize_model(self, model, graph_data):
        model.eval()
        with torch.enable_grad():
            feature_weights = model.get_feature_weights(graph_data)
        top_features_idx = np.argsort(feature_weights)[-10:]
        top_features_names = [my_model.feature_names[i] for i in top_features_idx]

        visualize_menu(graph_data, self.results,
                       feature_weights=feature_weights[top_features_idx],
                       feature_names=top_features_names,
                       filename=f'saves/{model.model_type}.png',
                       use_3d=False)

    def get_shap_values(self, model, graph_data, method, **kwargs):
        if method == 'kernel':
            computer = KernelValuesComputer(model, my_model.feature_names)
            shap_values, test_data = computer.get_shap_values(graph_data, **kwargs)
            return shap_values, test_data

        if method == 'gradient':
            computer = GradientValuesComputer(model, my_model.feature_names)
        else:
            computer = DeepValuesComputer(model, my_model.feature_names)

        test_data = graph_data.x[graph_data.test_mask].detach().cpu().numpy()
        shap_values = computer.get_values_for_test(test_data)

        return shap_values, test_data

    def add_shap_analysis(self, model, graph_data, method="gradient", **kwargs):
        """Add SHAP analysis for the chosen model using chosen method"""
        print(f"\n=== SHAP Analysis for {model.model_type} ===")

        if method not in ('gradient', 'deeplift', 'kernel'):
            print("Shap analysis failed: Passed invalid method")
            return

        shap_values, test_data = self.get_shap_values(deepcopy(model), graph_data, method, **kwargs)

        if shap_values is None:
            print("❌ Failed to compute SHAP values")
            return

        # general shap analysis
        shap_visualizer = SHAPVisualizer(my_model.feature_names)
        shap_visualizer.create_comprehensive_report(shap_values, test_data)

        print("\nSHAP Analysis Complete!")
        print("Check 'saves' directory for visualizations.")
