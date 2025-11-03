import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from gnn_models.bot_gnn import BotGNN
from gnn_models.bot_detector_trainer import BotDetectorTrainer
from gnn_models.bot_detector_gnn import BotDetectorGNN
from gnn_models.model_1.model import Model as my_model
from gnn_models.graph_viz import *


class ModelTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        self.detector = BotDetectorGNN(my_model)
        bots_users, humans_users = self.detector.load_data('data/for_model_1/bots_data.json',
                                                      'data/for_model_1/humans_data.json')

        graph_data = self.build_graph(bots_users, humans_users)

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


    def build_graph(self, bots_users, humans_users):
        # Извлечение признаков
        bots_features, bots_labels, bots_ids = self.detector.extract_features(bots_users, 0)  # 0 - бот
        humans_features, humans_labels, humans_ids = self.detector.extract_features(humans_users, 1)  # 1 - человек

        # Построение графа
        graph_data, all_ids = self.detector.build_graph(
            bots_features, humans_features, bots_labels, humans_labels, bots_ids, humans_ids
        )

        # Разделение на train/test
        train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)

        indices = list(range(graph_data.num_nodes))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=True)  # random_state=42

        train_mask[train_idx] = True
        test_mask[test_idx] = True

        graph_data.train_mask = train_mask
        graph_data.test_mask = test_mask

        print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")

        return graph_data


    def test_model(self, model_name, model, graph_data):
        print(f"\n=== Обучение {model_name} ===")
        trainer = BotDetectorTrainer(model, self.device)
        losses = trainer.train(graph_data, epochs=15)
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
        top_features_names = [self.detector.feature_names[i] for i in top_features_idx]

        visualize_menu(graph_data, self.results,
                       feature_weights=feature_weights[top_features_idx],
                       feature_names=top_features_names,
                       use_3d=True)


if __name__ == "__main__":
    modelTester = ModelTester()