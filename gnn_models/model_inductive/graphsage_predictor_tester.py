from gnn_models.data_producer import DataProducer
from gnn_models.model_1.model import Model
from gnn_models.model_inductive.graphsage_predictor import GraphSAGEPredictor
from gnn_models.model_1.model import Model as my_model


def make_predictions():
    producer = DataProducer(my_model, 'saves/scaler.pkl')
    # Старые данные
    original_bots_users, original_humans_users = producer.load_all_data('data/for_model_1/bots_data.json',
                                                  'data/for_model_1/humans_data.json')
    original_graph_data, original_ids = producer.prepare_full_graph_data(original_bots_users, original_humans_users)

    # Новые данные
    new_users = producer.load_data('data/for_inference/users_data.json')
    new_features, ids = producer.extract_features_and_ids(new_users)

    # Инференс новых узлов
    predictor = GraphSAGEPredictor('saves/inductive_gnn.pth', Model.feature_names)

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
