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

class ModelTester:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        self.detector = BotDetectorGNN(my_model)
        bots_users, humans_users = self.detector.load_data('../data/for_model_1/bots_data.json',
                                                      '../data/for_model_1/humans_data.json')

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
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_mask[train_idx] = True
        test_mask[test_idx] = True

        graph_data.train_mask = train_mask
        graph_data.test_mask = test_mask

        print(f"Train nodes: {train_mask.sum()}, Test nodes: {test_mask.sum()}")

        # Обучение разных моделей GNN
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

        # improved_visualization(graph_data, results, detector, models)

    def get_model_weights(self, model, model_type):
        """
        Универсальный метод получения весов из первого слоя GNN
        """
        conv1 = model.conv1

        if model_type == 'GCN':
            if hasattr(conv1, 'lin'):
                return conv1.lin.weight
            elif hasattr(conv1, 'weight'):
                return conv1.weight

        elif model_type == 'GAT':
            if hasattr(conv1, 'weight'):
                return conv1.weight
            elif hasattr(conv1, 'lin_src') and hasattr(conv1.lin_src, 'weight'):
                return conv1.lin_src.weight

        elif model_type == 'SAGE':
            # SAGEConv: lin_r.weight или lin_l.weight
            if hasattr(conv1, 'weight'):
                return conv1.weight
            elif hasattr(conv1, 'lin_l') and hasattr(conv1.lin_l, 'weight'):
                return conv1.lin_l.weight
            elif hasattr(conv1, 'lin_r') and hasattr(conv1.lin_r, 'weight'):
                return conv1.lin_r.weight

        # Если ничего не найдено
        print(f"Warning: Could not find weights for {model_type}")
        return None


    def analyze_feature_importance(self, best_model, best_model_name):
        """
        Анализ важности признаков с обработкой ошибок
        """
        try:
            weights = self.get_model_weights(best_model, best_model_name)

            if weights is not None:
                feature_importance = abs(weights.mean(dim=0).detach().numpy())
            else:
                return None
                # Альтернативный метод: используем градиенты или создаем случайные значения
                # print("Using alternative method for feature importance")
                # feature_importance = np.random.rand(len(feature_names))

            return feature_importance

        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            return None


    def predict_new_users(self, model_path, new_users_data, feature_names):
        """Предсказание для новых пользователей"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка модели
        model = BotGNN(len(feature_names), 64, 2, 'GCN')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # Подготовка данных
        detector = BotDetectorGNN(my_model)
        features, _, user_ids = detector.extract_features(new_users_data, -1)

        # Создание графа для новых данных
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Простой граф (каждый узел связан с похожими)
        edge_list = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                similarity = np.dot(features[i], features[j]) / (
                        np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-8
                )
                if similarity > 0.7:
                    edge_list.append([i, j])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(features, dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index)
        data = data.to(self.device)

        with torch.no_grad():
            out = model(data)
            predictions = out.argmax(dim=1)
            probabilities = torch.exp(out)

        results = []
        for i, user_id in enumerate(user_ids):
            results.append({
                'user_id': user_id,
                'prediction': 'bot' if predictions[i] == 0 else 'human',
                'bot_probability': probabilities[i][0].item(),
                'human_probability': probabilities[i][1].item()
            })

        return results


    def test_model(self, model_name, model, graph_data):
        print(f"\n=== Обучение {model_name} ===")
        trainer = BotDetectorTrainer(model, self.device)
        losses = trainer.train(graph_data, epochs=15)
        accuracy = trainer.test(graph_data)
        self.results[model_name] = accuracy

        print(f"{model_name} Точность: {accuracy:.4f}")

        # Визуализация результатов
        plt.figure(figsize=(15, 5))

        # 1. Сравнение моделей
        plt.subplot(1, 3, 1)
        plt.bar(self.results.keys(), self.results.values())
        plt.title('Сравнение моделей')
        plt.ylabel('Точность')
        plt.xticks(rotation=45)

        # 2. Визуализация графа (упрощенная)
        plt.subplot(1, 3, 2)
        try:
            import networkx as nx
            # Создаем упрощенный граф для визуализации
            g = nx.Graph()
            edges = graph_data.edge_index.t().numpy()

            # Берем только часть ребер для наглядности
            sample_edges = edges  # [:min(2144, len(edges))]
            g.add_edges_from(sample_edges)

            node_colors = []
            for i in range(graph_data.num_nodes):  # min(200, graph_data.num_nodes)):
                if graph_data.y[i] == 0:
                    node_colors.append('red')  # Боты
                else:
                    node_colors.append('blue')  # Люди

            pos = nx.spring_layout(g)
            nx.draw(g, pos, node_color=node_colors[:len(g.nodes)],
                    node_size=50, with_labels=False, alpha=0.7)
            plt.title('Граф (красные - боты, синие - люди)')
        except Exception as e:
            print(f"Graph visualization failed: {e}")
            plt.text(0.5, 0.5, 'Graph visualization\nnot available',
                     ha='center', va='center', transform=plt.gca().transAxes)

        # 3. Важность признаков
        plt.subplot(1, 3, 3)
        self.best_model_name = max(self.results, key=self.results.get)
        self.best_model = self.models[self.best_model_name]

        try:
            feature_importance = self.analyze_feature_importance(
                self.best_model, self.best_model_name
            )

            if feature_importance is not None:
                top_features_idx = np.argsort(feature_importance)[-10:]
                top_features_names = [self.detector.feature_names[i] for i in top_features_idx]

                plt.barh(top_features_names, feature_importance[top_features_idx])
                plt.title('Топ-10 важных признаков')
            else:
                plt.text(0.5, 0.5, 'Feature importance\nnot available',
                         ha='center', va='center', transform=plt.gca().transAxes)

        except Exception as e:
            print(f"Feature importance visualization failed: {e}")
            plt.text(0.5, 0.5, 'Feature importance\nvisualization failed',
                     ha='center', va='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig('bot_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    modelTester = ModelTester()