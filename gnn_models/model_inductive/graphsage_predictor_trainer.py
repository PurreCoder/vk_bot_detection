import torch
import config
from data_processing.file_manager import ensure_file_deleted, ensure_file_deleted, load_all_users
from gnn_models.gnn import BotGNN
from gnn_models.metrics_calc import compute_metrics
from gnn_models.model_transductive.model_trainer import ModelTrainer
from data_processing.data_processor import DataProcessor
from gnn_models.model_1.model import Model as my_model
from viz.graph_viz import *


class GraphSAGETrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        ensure_file_deleted(config.MODELS_SAVES['INDUCTIVE_SAVE'])
        ensure_file_deleted(config.SCALER_SAVE_FILE)

        bots_users, humans_users = load_all_users(config.DATA_SOURCE['BOTS_FILE'], config.DATA_SOURCE['HUMANS_FILE'])

        if config.FLAGS['FILTER_DEACTIVATED']:
            is_not_deactivated = lambda _user: _user.get('deactivated', '') == ''
            filter_deactivated = lambda _users: list(filter(is_not_deactivated, _users))

            bots_users = filter_deactivated(bots_users)
            print(f'\nВыделено {len(bots_users)} ботов в соответствии с фильтром')
            humans_users = filter_deactivated(humans_users)
            print(f'Выделено {len(humans_users)} людей в соответствии с фильтром\n')

        if config.BOTS_TO_USERS > 1:
            common_size = min(len(bots_users), len(humans_users))
            humans_limit = int(common_size / config.BOTS_TO_USERS)
            if len(humans_users) > humans_limit:
                humans_users = humans_users[:humans_limit]

            if len(bots_users) > common_size:
                bots_users = bots_users[:common_size]

            print(f'Общая выборка будет содержать {len(humans_users)} людей и {len(bots_users)} ботов\n')

        self.processor = DataProcessor(my_model, config.SCALER_SAVE_FILE)
        all_features, all_labels, all_ids = self.processor.get_all_features(bots_users, humans_users)
        graph_data, ids = self.processor.prepare_full_graph_data(all_features, all_labels, all_ids)

        self.model= BotGNN(graph_data.num_features, 64, 'SAGE')

        self.test_model(self.model, graph_data)

        torch.save(self.model.state_dict(), config.MODELS_SAVES['INDUCTIVE_SAVE'])
        print("\nИндуктивная модель сохранена")


    def test_model(self, model, graph_data):
        model_name = model.model_type

        print(f"\n=== Обучение {model_name} ===")
        trainer = ModelTrainer(model, self.device)
        trainer.train(graph_data, epochs=25)

        # getting labels and model predictions for data
        y_true, y_pred = trainer.predict_labels_for_test(graph_data)

        # calculating metrics
        metrics = compute_metrics(y_true, y_pred)

        print(f"\n{model_name} Accuracy: {metrics['accuracy']:.4f}")
        print(f"{model_name} Precision: {metrics['precision']:.4f}")
        print(f"{model_name} Recall: {metrics['recall']:.4f}")
        print(f"{model_name} F1-Score: {metrics['f1']:.4f}")

        self.visualize_model(model, graph_data)

    def visualize_model(self, model, graph_data):
        model.eval()
        with torch.enable_grad():
            feature_weights = model.get_feature_weights(graph_data)
        top_features_idx = np.argsort(feature_weights)[-10:]
        top_features_names = [my_model.feature_names[i] for i in top_features_idx]

        visualize_feature_importance(feature_weights[top_features_idx], top_features_names, 'saves/inductive_training.png')
