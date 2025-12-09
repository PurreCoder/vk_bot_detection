from copy import deepcopy
import numpy as np
import config
from data_processing.feature_scaler import FeatureScalerSingleton
from data_processing.file_manager import load_all_users, load_json_data
from gnn_models.model_1.limits import model_limits
from shap_analysis.explanation_constructor import ExplanationConstructor


def load_mistakes_to_explain(model, processor):
    print("\nLoading false predictions...")
    model_mistakes = load_json_data(f'saves/{model.model_type}_mistakes.json')
    ids_mistaken = list(map(int, model_mistakes.keys()))
    bots_users, humans_users = load_all_users(config.DATA_SOURCE['BOTS_FILE'], config.DATA_SOURCE['HUMANS_FILE'])
    list_to_filter = bots_users + humans_users
    mistakes_filter = lambda _entry: _entry.get('id', 0) in ids_mistaken
    filtered_list = list(filter(mistakes_filter, list_to_filter))
    list_to_explain, id_list = processor.extract_features_and_ids(filtered_list)
    return list_to_explain, id_list

def shuffle_mistakes(list_to_explain, id_list):
    pairs = list(zip(list_to_explain, id_list))
    from random import shuffle
    shuffle(pairs)
    list_to_explain, id_list = zip(*pairs)
    list_to_explain = np.array(list_to_explain)
    return list_to_explain, id_list

def explain_false_predictions(processor, model, graph_data, shap_visualizer):
    list_to_explain, id_list = load_mistakes_to_explain(model, processor)

    # shuffling because we will only save top-15 mistakes
    # not just trimming array to top-15 rn in order to preserve data to build proper graph structure
    list_to_explain, id_list = shuffle_mistakes(list_to_explain, id_list)

    list_to_explain_scaled = processor.scale_with_save(list_to_explain)

    print("\nExplaining false predictions...")
    constructor = ExplanationConstructor(deepcopy(model), graph_data, list_to_explain_scaled)
    explanations = constructor.construct_prediction()
    explanations = constructor.inject_explanation(explanations, 'data', list_to_explain)
    explanations.feature_names = shap_visualizer.feature_names

    shap_visualizer.plot_explanations(model.model_type, id_list, explanations)
