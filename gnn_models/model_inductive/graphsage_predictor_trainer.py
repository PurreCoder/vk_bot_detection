import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from gnn_models.bot_gnn import BotGNN
from gnn_models.bot_detector_trainer import BotDetectorTrainer
from gnn_models.data_producer import DataProducer
from gnn_models.model_1.model import Model as my_model
from gnn_models.graph_viz import *


class GraphSAGETrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        try:
            os.remove('saves/inductive_gnn.pth')
        except FileNotFoundError:
            pass
        try:
            os.remove('saves/scaler.pkl')
        except FileNotFoundError:
            pass

        producer = DataProducer(my_model, 'saves/scaler.pkl')
        bots_users, humans_users = producer.load_all_data('data/for_model_1/bots_data.json',
                                                      'data/for_model_1/humans_data.json')
        graph_data = producer.prepare_full_graph_data(bots_users, humans_users)

        self.model= BotGNN(graph_data.num_features, 64, 2, 'SAGE')

        self.test_model('GraphSAGE', self.model, graph_data)

        # Сохранение лучшей модели
        torch.save(self.model.state_dict(), 'saves/inductive_gnn.pth')
        print("\nИндуктивная модель сохранена.")


    def test_model(self, model_name, model, graph_data):
        print(f"\n=== Обучение {model_name} ===")
        trainer = BotDetectorTrainer(model, self.device)
        losses = trainer.train(graph_data, epochs=35)
        accuracy = trainer.test(graph_data)

        print(f"Точность: {accuracy:.4f}")

        self.visualize_model(model, graph_data)

    def visualize_model(self, model, graph_data):
        model.eval()
        with torch.enable_grad():
            feature_weights = model.get_feature_weights(graph_data)
        top_features_idx = np.argsort(feature_weights)[-10:]
        top_features_names = [my_model.feature_names[i] for i in top_features_idx]

        visualize_feature_importance(feature_weights[top_features_idx], top_features_names, 'saves/inductive_training.png')


if __name__ == "__main__":
    modelTrainer = GraphSAGETrainer()