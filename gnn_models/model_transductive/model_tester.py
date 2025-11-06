import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gnn_models.bot_gnn import BotGNN
from gnn_models.bot_detector_trainer import BotDetectorTrainer
from gnn_models.data_producer import DataProducer
from gnn_models.model_1.model import Model as my_model
from gnn_models.graph_viz import *


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

        visualize_parameters_comparison([f'saves/{cur_model.model_type}.png' for cur_model in self.models.values()])

        # Сохранение лучшей модели
        torch.save(self.best_model.state_dict(), 'best_bot_detector_gnn.pth')
        print(f"\nЛучшая модель сохранена: {max(self.results, key=self.results.get)}")


    def test_model(self, model_name, model, graph_data):
        print(f"\n=== Обучение {model_name} ===")
        trainer = BotDetectorTrainer(model, self.device)
        losses = trainer.train(graph_data, epochs=35)
        accuracy = trainer.test(graph_data)
        self.results[model_name] = accuracy

        print(f"{model_name} Точность: {accuracy:.4f}")

        self.best_model_name = max(self.results, key=self.results.get)
        self.best_model = self.models[self.best_model_name]

        self.visualize_model(model, graph_data)

    def visualize_model(self, model, graph_data):
        model.eval()
        with torch.enable_grad():
            feature_weights = model.get_feature_weights(graph_data)
        top_features_idx = np.argsort(feature_weights)[-10:]
        top_features_names = [self.producer.feature_names[i] for i in top_features_idx]

        visualize_menu(graph_data, self.results,
                       feature_weights=feature_weights[top_features_idx],
                       feature_names=top_features_names,
                       filename=f'saves/{model.model_type}.png',
                       use_3d=False)


if __name__ == "__main__":
    modelTester = ModelTester()