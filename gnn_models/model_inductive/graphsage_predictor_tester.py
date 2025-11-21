import config
from data_processing.data_processor import DataProcessor
from data_processing.file_manager import load_all_users, load_users
from gnn_models.model_1.model import Model
from gnn_models.model_inductive.graphsage_predictor import GraphSAGEPredictor
from gnn_models.model_1.model import Model as my_model


def make_predictions():
    # Старые данные
    original_bots_users, original_humans_users = load_all_users(config.DATA_SOURCE['BOTS_FILE'], config.DATA_SOURCE['HUMANS_FILE'])
    processor = DataProcessor(my_model, config.SCALER_SAVE_FILE)
    old_features, old_labels, old_ids = processor.get_all_features(original_bots_users, original_humans_users)
    original_graph_data, original_ids = processor.prepare_full_graph_data(old_features, old_labels, old_ids)

    # Новые данные
    new_users = load_users(config.DATA_SOURCE['INFERENCE_FILE'])
    new_features, ids = processor.extract_features_and_ids(new_users)

    # Инференс новых узлов
    predictor = GraphSAGEPredictor(config.MODELS_SAVES['INDUCTIVE_SAVE'], Model.feature_names)

    # Вариант 1: Без существующего графа
    # result = predictor.predict_new_nodes(new_users_features)

    # Вариант 2: С интеграцией в обученный граф
    result = predictor.predict_new_nodes(
        new_features,
        existing_graph_data=original_graph_data,
        k_neighbors=3
    )
    print("Предсказания с графом:")
    predictions = result['predictions']
    probs = result['probabilities']
    for ans, prob, user_id in zip(predictions, probs, ids):
        print(f"vk.com/id{user_id}: {'человек' if ans else 'бот'} с вероятностью {prob[ans]:.2f}")
