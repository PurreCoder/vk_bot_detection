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

    def get_feature_importance(self):
        """
                Универсальный метод получения важности признаков
        """
        try:
            # Пробуем разные источники весов по порядку
            if self.model_type == 'GCN':
                if hasattr(self.conv1, 'lin') and hasattr(self.conv1.lin, 'weight'):
                    weights = self.conv1.lin.weight
                else:
                    weights = self.conv1.weight

            elif self.model_type == 'GAT':
                # Для GAT используем веса из feature_projection как fallback
                weights = self.feature_projection.weight

            elif self.model_type == 'SAGE':
                if hasattr(self.conv1, 'lin_r') and hasattr(self.conv1.lin_r, 'weight'):
                    weights = self.conv1.lin_r.weight
                elif hasattr(self.conv1, 'lin_l') and hasattr(self.conv1.lin_l, 'weight'):
                    weights = self.conv1.lin_l.weight
                else:
                    # Для SAGE также используем feature_projection как fallback
                    weights = self.feature_projection.weight

            # Если все попытки провалились, используем classifier weights
            if weights is None or weights.shape[1] != self.num_features:
                weights = self.classifier[0].weight  # Первый слой classifier

            feature_importance = abs(weights.mean(dim=0).detach().cpu().numpy())
            return feature_importance

        except Exception as e:
            print(f"Error in get_feature_importance: {e}")
            # Возвращаем равномерное распределение как последний fallback
            return np.ones(self.num_features) / self.num_features

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

        # Глобальное среднее pooling для классификации
        if batch is not None:
            x = global_mean_pool(x, batch)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)