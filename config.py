from enum import Enum
SCALING_STRATEGY = Enum('Scaling', 'STANDARD MIXED')

ACCESS_TOKEN = ""

GRAPH_DATA_LOGS = dict(
    BOTS_IDS_FILE = 'saves/used_bots_ids.txt',
    HUMANS_IDS_FILE = 'saves/used_humans_ids.txt'
)

DATA_SOURCE = dict(
    BOTS_FILE = 'data/for_model_2/bots_data.json',
    HUMANS_FILE = 'data/for_model_2/humans_data.json',
    INFERENCE_FILE = 'data/for_inference/users_data.json',
)

# bots/users classes ratio
BOTS_TO_USERS = 1 # has to be NO LESS than 1

PARAMS_SOURCE = 'gnn_models/model_1/params.csv'

MODELS_SAVES = dict(
    TRANSDUCTIVE_SAVE = 'saves/best_bot_detector_gnn.pth',
    INDUCTIVE_SAVE = 'saves/inductive_gnn.pth'
)

SCALER_SAVE_FILE = 'saves/scaler.pkl'
# (outdated, but can be brought back to code)
SCALING = SCALING_STRATEGY.MIXED

FLAGS = dict(
    # forbid deactivated users from the graph? (except new nodes of inductive model!)
    FILTER_DEACTIVATED = False,
    # remove feature "deactivated" from the features vector?
    REMOVE_DEACTIVATED = False
)

PLOT_SETTINGS = dict(
    MAX_NODES = 3000,
    MAX_EDGES = 170000
)
