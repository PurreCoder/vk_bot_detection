import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


class BotGNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, model_type='GCN'):
        super(BotGNN, self).__init__()

        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_features = num_features

        if model_type == 'GCN':
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

        elif model_type == 'GAT':
            self.conv1 = GATConv(num_features, hidden_channels, heads=4, concat=True)
            self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
            self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)

        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.feature_projection = nn.Linear(num_features, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.model_type == 'GAT':
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = F.elu(self.conv3(x, edge_index))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))

        if batch is not None:
            x = global_mean_pool(x, batch)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def _weight_based_importance(self):
        """Метод на основе весов модели с исправлениями"""
        try:
            weights = None

            if self.model_type == 'GCN':
                # Для GCN берем веса из первого слоя
                if hasattr(self.conv1, 'lin') and hasattr(self.conv1.lin, 'weight'):
                    weights = self.conv1.lin.weight
                elif hasattr(self.conv1, 'weight') and self.conv1.weight is not None:
                    weights = self.conv1.weight

            elif self.model_type == 'GAT':
                # Для GAT используем веса из первого слоя
                if hasattr(self.conv1, 'lin_src') and hasattr(self.conv1.lin_src, 'weight'):
                    weights = self.conv1.lin_src.weight
                else:
                    # Fallback на feature_projection
                    weights = self.feature_projection.weight

            elif self.model_type == 'SAGE':
                # Для SAGE используем оба преобразования
                weights_l, weights_r = None, None

                if hasattr(self.conv1, 'lin_l') and hasattr(self.conv1.lin_l, 'weight'):
                    weights_l = self.conv1.lin_l.weight
                if hasattr(self.conv1, 'lin_r') and hasattr(self.conv1.lin_r, 'weight'):
                    weights_r = self.conv1.lin_r.weight

                if weights_l is not None and weights_r is not None:
                    weights = (weights_l + weights_r) / 2
                elif weights_l is not None:
                    weights = weights_l
                elif weights_r is not None:
                    weights = weights_r

            # Если все попытки провалились, используем classifier weights
            if weights is None or weights.shape[1] != self.num_features:
                weights = self.classifier[0].weight  # Первый слой classifier

            # Нормализованная важность признаков
            feature_importance = torch.abs(weights).mean(dim=0)
            feature_importance = feature_importance / feature_importance.sum()

            return feature_importance.detach().cpu().numpy()

        except Exception as e:
            print(f"Error in weight-based importance: {e}")
            return np.ones(self.num_features) / self.num_features


    def _gradient_based_importance(self, data):
        """Метод на основе градиентов с правильной настройкой градиентов"""
        try:
            self.eval()
            x = data.x.clone().requires_grad_(True)

            # Создаем новый data объект для прямого прохода
            grad_data = data.clone()
            grad_data.x = x

            out = self.forward(grad_data)

            # Используем сумму выходов для градиентов (избегаем unused tensors)
            # Это гарантирует, что все элементы x будут использованы в вычислении градиентов
            output_sum = out.sum()

            # Вычисляем градиенты
            gradients = torch.autograd.grad(
                outputs=output_sum,
                inputs=x,
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )[0]

            feature_importance = torch.abs(gradients).mean(dim=0)

            # Нормализация
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
            else:
                feature_importance = torch.ones_like(feature_importance) / feature_importance.size(0)

            return feature_importance.detach().cpu().numpy()

        except Exception as e:
            print(f"Error in gradient-based importance: {e}")
            # Fallback на weight-based метод
            return self._weight_based_importance()

    def get_feature_weights(self, data):
        return self._gradient_based_importance(data)
