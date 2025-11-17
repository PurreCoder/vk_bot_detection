import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processing.file_manager import deserialize_object, serialize_object


class FeatureProcessor:
    def __init__(self, cls, file_to_save=None):
        self.model = cls
        self.file_to_save = file_to_save
        self.node_features = None

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
            scaler = deserialize_object(self.file_to_save)
            if scaler is None:
                scaler = StandardScaler()

        # Нормализуем признаки
        all_features = scaler.fit_transform(all_features)

        if self.file_to_save is not None:
            # Сохраняем scaler при обучении
            serialize_object(scaler, self.file_to_save)

        return all_features
