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


class BotDetectorTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, data, epochs=15):
        self.model.train()
        train_losses = []

        data = data.to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())

            if epoch % 5 == 0:
                accuracy = self.test(data)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        return train_losses

    def test(self, data):
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data)
            pred = out.argmax(dim=1)
            test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            test_accuracy = test_correct / data.test_mask.sum()

        return test_accuracy.item()