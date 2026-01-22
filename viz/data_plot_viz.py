import umap
import matplotlib.pyplot as plt

import config
from data_processing.data_filter import sieve_deactivated, balance_users
from data_processing.data_processor import DataProcessor
from data_processing.file_manager import load_all_users
from gnn_models.model_1.model import Model as my_model


def get_data():
    bots_users, humans_users = load_all_users(f"../{config.DATA_SOURCE['BOTS_FILE']}", f"../{config.DATA_SOURCE['HUMANS_FILE']}")

    bots_users, humans_users = sieve_deactivated(bots_users, humans_users)
    bots_users, humans_users = balance_users(bots_users, humans_users)

    processor = DataProcessor(my_model)
    data, labels, _ = processor.get_all_features(bots_users, humans_users)
    return data, labels


def main():
    data, labels = get_data()

    reducer = umap.UMAP(n_neighbors=10, metric='cosine', min_dist=0.0, n_components=2, n_jobs=-1)
    embedding = reducer.fit_transform(data)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the VK Users/Bots dataset')
    plt.show()


if __name__ == '__main__':
    main()
