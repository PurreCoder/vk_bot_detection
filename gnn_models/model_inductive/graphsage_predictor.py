import numpy as np
import torch
import config
from torch_geometric.data import Data
from data_processing.file_manager import deserialize_object
from gnn_models.gnn import BotGNN


class GraphSAGEPredictor:
    def __init__(self, model_path, feature_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_names = feature_names

        self.model = BotGNN(
            num_features=len(feature_names),
            hidden_channels=64,
            model_type='SAGE'
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.scaler = self.load_scaler(config.SCALER_SAVE_FILE)

    def load_scaler(self, scaler_path):
        """Загружает сохраненный StandardScaler"""
        scaler = deserialize_object(scaler_path)
        if scaler is None:
            print("Scaler not found, using new one")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        return scaler

    def predict_new_nodes(self, new_features, existing_graph_data=None, k_neighbors=5):
        """
        Предсказание для новых вершин

        Args:
            new_features: numpy array [n_new_nodes, n_features] - признаки новых узлов
            existing_graph_data: Data объект обученного графа (опционально)
            k_neighbors: количество ближайших соседей для построения связей
        """
        # Масштабируем признаки
        new_features_scaled = self.scaler.fit_transform(new_features)
        new_features_tensor = torch.tensor(new_features_scaled, dtype=torch.float).to(self.device)

        if existing_graph_data is None:
            # Если нет существующего графа, создаем минимальный граф из новых узлов
            return self._predict_isolated(new_features_tensor)
        else:
            # Интегрируем новые узлы в существующий граф
            return self._predict_with_existing_graph(new_features_tensor, existing_graph_data, k_neighbors)

    def _predict_isolated(self, new_features_tensor):
        """Предсказание для изолированных узлов (без графа)"""
        with torch.no_grad():
            edge_index = torch.empty(2, 0, dtype=torch.long).to(self.device)

            predictions = self.model(new_features_tensor, edge_index)
            probabilities = torch.exp(predictions)

            return {
                'predictions': predictions.argmax(dim=1).cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'is_isolated': True
            }

    def _predict_with_existing_graph(self, new_features_tensor, existing_graph_data, k_neighbors):
        """Предсказание с интеграцией в существующий граф"""
        existing_graph_data = existing_graph_data.to(self.device)

        # Находим k ближайших соседей для каждого нового узла
        similarity_matrix = self._compute_similarity(new_features_tensor, existing_graph_data.x)

        # Строим новые ребра
        new_edges = self._find_k_neighbors(similarity_matrix, k_neighbors)

        # Объединяем старый и новый граф
        combined_data = self._merge_graphs(
            existing_graph_data, new_features_tensor, new_edges
        )

        # Предсказание для новых узлов
        with torch.no_grad():
            all_predictions = self.model(combined_data.x, combined_data.edge_index)
            new_nodes_predictions = all_predictions[-new_features_tensor.size(0):]
            probabilities = torch.exp(new_nodes_predictions)

            return {
                'predictions': new_nodes_predictions.argmax(dim=1).cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'is_isolated': False,
                'neighbor_indices': new_edges.cpu().numpy()
            }

    def _compute_similarity(self, new_features, existing_features):
        """Вычисляет косинусное сходство между новыми и существующими узлами"""
        from sklearn.metrics.pairwise import cosine_similarity
        new_np = new_features.cpu().numpy()
        existing_np = existing_features.cpu().detach().numpy()
        return cosine_similarity(new_np, existing_np)

    def _find_k_neighbors(self, similarity_matrix, k):
        """Находит k ближайших соседей для каждого нового узла"""
        n_new = similarity_matrix.shape[0]
        edges = []

        for i in range(n_new):
            # Индексы топ-k наиболее похожих существующих узлов
            top_k_indices = np.argsort(similarity_matrix[i])[-k:]

            for j in top_k_indices:
                # Ребро от нового узла к существующему
                edges.append([i + similarity_matrix.shape[1], j])  # новые узлы добавляются в конец
                edges.append([j, i + similarity_matrix.shape[1]])  # неориентированный граф

        return torch.tensor(edges, dtype=torch.long).t().to(self.device)

    def _merge_graphs(self, existing_data, new_features, new_edges):
        """Объединяет существующий граф с новыми узлами"""
        # Объединяем признаки
        combined_x = torch.cat([existing_data.x, new_features], dim=0)

        # Объединяем ребра
        if existing_data.edge_index is not None:
            # Смещаем индексы новых ребер
            new_edges_offset = new_edges.clone()
            combined_edge_index = torch.cat([existing_data.edge_index, new_edges_offset], dim=1)
        else:
            combined_edge_index = new_edges

        return Data(x=combined_x, edge_index=combined_edge_index)
