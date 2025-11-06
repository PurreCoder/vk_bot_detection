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


class DataProducer:
    def __init__(self, cls, file_to_save=None):
        self.model = cls
        self.file_to_save = file_to_save
        self.feature_names = cls.feature_names
        self.node_features = None
        self.edge_index = None
        self.labels = None

    def load_data(self, users_file):
        with open(users_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        users = data.get('users', [])
        return users

    def load_all_data(self, bots_file, humans_file):
        """Загружает и объединяет данные ботов и людей"""
        print("Загрузка данных...")

        bots_users = self.load_data(bots_file)
        humans_users = self.load_data(humans_file)

        print(f"Загружено ботов: {len(bots_users)}")
        print(f"Загружено людей: {len(humans_users)}")

        return bots_users, humans_users

    def extract_features(self, users):
        return self.model.extract_features(users)

    def extract_features_and_ids(self, users):
        return self.model.extract_features_and_ids(users)

    def extract_data(self, users, label):
        features, ids = self.model.extract_features_and_ids(users)
        return features, [label] * len(ids), ids

    def merge_features(self, bots_features, humans_features, bots_labels, humans_labels, bots_ids, humans_ids):
        all_features = np.vstack([bots_features, humans_features])
        all_labels = bots_labels + humans_labels
        all_ids = bots_ids + humans_ids
        return all_features, all_labels, all_ids

    def scale_features(self, all_features):
        if self.file_to_save is None:
            scaler = StandardScaler()
        else:
            try:
                import pickle
                with open(self.file_to_save, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as e:
                scaler = StandardScaler()

        # Нормализуем признаки
        all_features = scaler.fit_transform(all_features)

        if self.file_to_save is not None:
            # Сохраняем scaler при обучении
            import pickle
            with open(self.file_to_save, 'wb') as scaler_file:
                pickle.dump(scaler, scaler_file)

        return all_features

    def build_graph(self, all_features, all_labels):
        """Строит граф на основе сходства пользователей"""

        print("Построение графа...")

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
                if similarity > self.model.SIMILARITY_THRESHOLD:  # Порог сходства для создания ребра
                    edges.append([i, j])
                    edge_attributes.append(similarity)

        # Преобразуем в формат PyTorch Geometric
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        node_features = torch.tensor(all_features, dtype=torch.float)
        node_features.requires_grad = True
        labels = torch.tensor(all_labels, dtype=torch.long)

        print(f"Граф построен: {n_nodes} узлов, {edge_index.shape[1]} ребер")

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    def make_train_test_split(self, graph_data):
        # Разделение на train/test
        train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)

        indices = list(range(graph_data.num_nodes))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=True)

        train_mask[train_idx] = True
        test_mask[test_idx] = True

        graph_data.train_mask = train_mask
        graph_data.test_mask = test_mask

        print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")

        return graph_data

    def prepare_full_graph_data(self, bots_users, humans_users):
        # Извлечение признаков
        bots_features, bots_labels, bots_ids = self.extract_data(bots_users, 0)  # 0 - бот
        humans_features, humans_labels, humans_ids = self.extract_data(humans_users, 1)  # 1 - человек

        with open("saves/used_humans_ids.txt", 'w', encoding='utf-8') as f:
            f.write(str(humans_ids))
        with open("saves/used_bots_ids.txt", 'w', encoding='utf-8') as f:
            f.write(str(bots_ids))

        all_features, all_labels, all_ids = self.merge_features(bots_features, humans_features,
                                                                bots_labels, humans_labels,
                                                                bots_ids, humans_ids)

        # Масштабирование
        all_features = self.scale_features(all_features)

        # Построение графа
        graph_data = self.build_graph(all_features, all_labels)

        # Разбиение на train/test
        graph_data = self.make_train_test_split(graph_data)

        return graph_data
