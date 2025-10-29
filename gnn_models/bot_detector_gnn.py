import json
import os

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


class BotDetectorGNN:
    def __init__(self, cls):
        self.model = cls
        self.feature_names = cls.feature_names
        self.node_features = None
        self.edge_index = None
        self.labels = None

    def load_data(self, bots_file='data/for_model_1/bots_data.json', humans_file='data/for_model_1/humans_data.json'):
        """Загружает и объединяет данные ботов и людей"""
        print("Загрузка данных...")

        with open(bots_file, 'r', encoding='utf-8') as f:
            bots_data = json.load(f)
        with open(humans_file, 'r', encoding='utf-8') as f:
            humans_data = json.load(f)

        bots_users = bots_data.get('users', [])
        humans_users = humans_data.get('users', [])

        print(f"Загружено ботов: {len(bots_users)}")
        print(f"Загружено людей: {len(humans_users)}")

        return bots_users, humans_users

    def extract_features(self, bots_users, humans_users):
        return self.model.extract_features(bots_users, humans_users)

    def build_graph(self, bots_features, humans_features, bots_labels, humans_labels, bots_ids, humans_ids):
        """Строит граф на основе сходства пользователей"""
        print("Построение графа...")

        # Объединяем все данные
        all_features = np.vstack([bots_features, humans_features])
        all_labels = bots_labels + humans_labels
        all_ids = bots_ids + humans_ids

        # Нормализуем признаки
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)

        # Создаем матрицу смежности на основе косинусного сходства
        n_nodes = len(all_features)
        edges = []
        edge_attributes = []

        # Вычисляем попарные сходства (можно оптимизировать для больших данных)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                similarity = np.dot(all_features[i], all_features[j]) / (
                        np.linalg.norm(all_features[i]) * np.linalg.norm(all_features[j]) + 1e-8
                )
                if similarity > 0.95:  # Порог сходства для создания ребра
                    edges.append([i, j])
                    edge_attributes.append(similarity)

        # Преобразуем в формат PyTorch Geometric
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        node_features = torch.tensor(all_features, dtype=torch.float)
        labels = torch.tensor(all_labels, dtype=torch.long)

        print(f"Граф построен: {n_nodes} узлов, {edge_index.shape[1]} ребер")

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels), all_ids