import torch
from sklearn.metrics import roc_curve, auc
from gnn_models.gnn import BotGNN
from gnn_models.metrics_calc import compute_metrics
from gnn_models.model_transductive.model_trainer import ModelTrainer
from gnn_models.data_producer import DataProducer
from gnn_models.model_1.model import Model as my_model
from gnn_models.shap_analysis.gradient_values_computer import GradientValuesComputer
from gnn_models.shap_analysis.deep_values_computer import DeepValuesComputer
from gnn_models.shap_analysis.kernel_values_computer import KernelValuesComputer
from gnn_models.viz.graph_viz import *
from gnn_models.viz.roc_viz import plot_roc_curve
from gnn_models.viz.shap_viz import SHAPVisualizer


class ModelTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        self.producer = DataProducer(my_model)
        bots_users, humans_users = self.producer.load_all_data('data/for_model_1/bots_data.json',
                                                      'data/for_model_1/humans_data.json')
        graph_data = self.producer.prepare_full_graph_data(bots_users, humans_users)

        self.models = {
            'GCN': BotGNN(graph_data.num_features, 64, 2, 'GCN'),
            'GAT': BotGNN(graph_data.num_features, 64, 2, 'GAT'),
            'GraphSAGE': BotGNN(graph_data.num_features, 64, 2, 'SAGE')
        }

        self.results = {}

        for model_name, model in self.models.items():
            self.test_model(model_name, model, graph_data)

        # Сохранение лучшей модели
        torch.save(self.best_model.state_dict(), 'best_bot_detector_gnn.pth')
        print(f"\nЛучшая модель сохранена: {max(self.results, key=self.results.get)}")

        visualize_parameters_comparison([f'saves/{cur_model.model_type}.png' for cur_model in self.models.values()])

        self.add_shap_analysis(self.models['GraphSAGE'], graph_data)


    def test_model(self, model_name, model, graph_data):
        print(f"\n=== Обучение {model_name} ===")
        trainer = ModelTrainer(model, self.device)
        trainer.train(graph_data, epochs=25)

        y_true, y_pred = trainer.predict_labels_for_test(graph_data)
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

        self.best_model_name = max(self.results, key=self.results.get)
        self.best_model = self.models[self.best_model_name]

        self.visualize_model(model, graph_data)

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

    def add_shap_analysis(self, model, graph_data):
        """Add SHAP analysis for the chosen model"""
        print(f"\n=== SHAP Analysis for {model.model_type} ===")

        shap_visualizer = SHAPVisualizer(my_model.feature_names)
        computer = KernelValuesComputer(self.best_model, my_model.feature_names)
        #computer = GradientValuesComputer(model, my_model.feature_names)
        #computer = DeepValuesComputer(model, my_model.feature_names)

        BACKGROUND_SIZE = 30
        TEST_SIZE = 110
        N_SAMPLES = 100
        shap_values, test_data = computer.get_shap_values(graph_data, 10, 10, 78)

        if shap_values is None:
            print("❌ Failed to compute SHAP values")
            return

        shap_visualizer.create_comprehensive_report(shap_values, test_data)

        print("\nSHAP Analysis Complete!")
        print("Check 'saves' directory for visualizations.")
