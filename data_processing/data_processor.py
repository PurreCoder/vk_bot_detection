import numpy as np
import torch
import config
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from data_processing.feature_processor import FeatureProcessor
from data_processing.file_manager import save_array


class DataProcessor(FeatureProcessor):
    def __init__(self, cls, file_to_save=None):
        super(DataProcessor, self).__init__(cls, file_to_save)
        self.model = cls
        self.file_to_save = file_to_save
        self.node_features = None

    def build_edges(self, all_features):
        """Строит графовую струтуру на основе косинусового свойства"""
        n_nodes = len(all_features)
        edges_from = []
        edges_to = []
        edge_attributes = []

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                similarity = np.dot(all_features[i], all_features[j]) / (
                        np.linalg.norm(all_features[i]) * np.linalg.norm(all_features[j]) + 1e-8
                )
                if similarity > self.model.SIMILARITY_THRESHOLD:
                    edges_from.append(i)
                    edges_to.append(j)
                    edge_attributes.append(similarity)

        edge_index = torch.LongTensor([edges_from, edges_to])
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

        return edge_index, edge_attr

    def build_graph(self, all_features, all_labels):
        """Строит граф на основе сходства пользователей"""

        print("Построение графа...")

        edge_index, edge_attr = self.build_edges(all_features)

        print(f"Граф построен: {len(all_features)} узлов, {edge_index.shape[1]} ребер")

        # Преобразуем в формат PyTorch Geometric
        node_features = torch.tensor(all_features, dtype=torch.float)
        node_features.requires_grad = True
        labels = torch.tensor(all_labels, dtype=torch.long)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    def make_train_test_split(self, graph_data):
        """Фиксирует разделение graph_data на train/test"""
        train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)

        indices = list(range(graph_data.num_nodes))
        train_idx, test_idx = train_test_split(indices, test_size=0.2)

        train_mask[train_idx] = True
        test_mask[test_idx] = True

        graph_data.train_mask = train_mask
        graph_data.test_mask = test_mask

        print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")

        return graph_data

    def prepare_full_graph_data(self, all_features, all_labels, all_ids):
        """Получение готового объекта Data на основе списков признаков, метод и идентификаторов"""

        # Масштабирование
        all_features = self.scale_features(all_features)

        # Построение графа
        graph_data = self.build_graph(all_features, all_labels)

        # Разбиение на train/test
        graph_data = self.make_train_test_split(graph_data)

        return graph_data, all_ids
