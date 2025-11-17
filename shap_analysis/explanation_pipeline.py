from copy import deepcopy
from config import DATA_SOURCE
from data_processing.file_manager import load_all_users, load_json_data
from shap_analysis.explanation_constructor import ExplanationConstructor


def explain_false_predictions(processor, model, graph_data, shap_visualizer):
    print("\nLoading false predictions...")
    model_mistakes = load_json_data(f'saves/{model.model_type}_mistakes.json')
    ids_mistaken = list(map(int, model_mistakes.keys()))
    bots_users, humans_users = load_all_users(DATA_SOURCE['BOTS_FILE'], DATA_SOURCE['HUMANS_FILE'])
    list_to_filter = bots_users + humans_users
    filtered_list = [entry for entry in list_to_filter if entry.get('id', 0) in ids_mistaken]

    list_to_explain, id_list = processor.extract_features_and_ids(filtered_list)
    list_to_explain_scaled = processor.scale_features(list_to_explain)

    print("\nExplaining false predictions...")
    constructor = ExplanationConstructor(deepcopy(model), graph_data, list_to_explain_scaled)
    explanations = constructor.construct_prediction()
    explanations = constructor.inject_explanation(explanations, 'data', list_to_explain)
    explanations.feature_names = shap_visualizer.feature_names

    shap_visualizer.plot_explanations(model.model_type, id_list, explanations)
