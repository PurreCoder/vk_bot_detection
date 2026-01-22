import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class ModelTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=75)

    def train(self, data, epochs=15):
        self.model.train()
        train_losses = []
        train_acc = []

        data = data.to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            accuracy = self.test(data)
            self.model.train()
            train_acc.append(accuracy)

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        return train_losses, train_acc

    def test(self, data):
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data)
            pred = out.argmax(dim=1)
            test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            test_accuracy = test_correct / data.test_mask.sum()

        return test_accuracy.item()

    def predict_labels_for_test(self, data, probs=False):
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            out = self.model(data)

        if probs:
            pred = torch.exp(out)[:,1]
        else:
            pred = out.argmax(dim=1)
        test_mask = data.test_mask
        y_true = data.y[test_mask]
        y_pred = pred[test_mask]

        return y_true.cpu().numpy(), y_pred.cpu().numpy()
