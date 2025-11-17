import shap
import torch
import xgboost

import config
from data_collection.vk_data_collector import VKDataCollector
from gnn_models.model_inductive.graphsage_predictor_tester import make_predictions
from gnn_models.model_inductive.graphsage_predictor_trainer import GraphSAGETrainer
from gnn_models.model_transductive.model_tester import ModelTester


def check_predictions():
    collector = VKDataCollector(config.ACCESS_TOKEN)
    #id_list = [272655352, 184278050, 309319362]
    id_list = [98947872,
               560901763,
               562121454,
               562782278,
               568925934,
               569516195,
               571088829,
               573134360,
               578138997,
               581000831,
               422046288, 422041826, 422039815, 422035299, 422034314, 422030585, 422020816, 422016591, 422008651,
               421997796]

    collector.collect_users_data(
        user_ids=id_list,
        n_users=len(id_list),
        params_file=config.PARAMS_SOURCE,
        output_file=config.DATA_SOURCE['INFERENCE_FILE'],
        delay=0.8
    )

    make_predictions()


if __name__ == "__main__":
    ModelTester()
    #GraphSAGETrainer()
    #check_predictions()
