import numpy as np
from sklearn.preprocessing import StandardScaler

from data_processing.feature_scaler import FeatureScalerSingleton
from data_processing.file_manager import deserialize_object, serialize_object
from gnn_models.model_1.limits import model_limits


class FeatureProcessor:
    def __init__(self, cls, file_to_save=None):
        self.model = cls
        self.file_to_save = file_to_save

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

    def get_all_features(self, user_list1, user_list2):
        # Извлечение признаков
        bots_features, bots_labels, bots_ids = self.extract_data(user_list1, 0)  # 0 - бот
        humans_features, humans_labels, humans_ids = self.extract_data(user_list2, 1)  # 1 - человек

        # Слияние данных от ботов и подлинных пользователей
        all_features, all_labels, all_ids = self.merge_features(bots_features, humans_features,
                                                                bots_labels, humans_labels,
                                                                bots_ids, humans_ids)

        return all_features, all_labels, all_ids

    def scale_with_save(self, all_features):
        if self.file_to_save is None:
            scaler = FeatureScalerSingleton(self.model, model_limits)
        else:
            scaler = deserialize_object(self.file_to_save)
            if scaler is None:
                scaler = FeatureScalerSingleton(self.model, model_limits)

        all_features = self.scale(scaler, all_features)

        if self.file_to_save is not None:
            serialize_object(scaler, self.file_to_save)

        return all_features

    def scale(self, scaler, all_features):
        all_features = scaler.fit_transform(all_features)
        return all_features
