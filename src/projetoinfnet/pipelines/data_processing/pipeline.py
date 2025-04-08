import mlflow
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    preprocess_shuttles,
    preprocess_single_dataset, 
    save_filtered_data,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Cria o pipeline de processamento de dados.""" 
    # Adiciona contexto de run do MLflow conforme solicitado
    with mlflow.start_run(run_name="PreparacaoDados"):
        # Loga uma tag para identificar facilmente este tipo de run
        mlflow.set_tag("pipeline_type", "data_processing")
        return pipeline(
            [
                node(
                    func=preprocess_shuttles,
                    inputs=["dataset_dev", "dataset_prod"],
                    outputs="shuttles_filtered",
                    name="preprocess_shuttles_node",
                ),
                node(
                    func=save_filtered_data,
                    inputs="shuttles_filtered",
                    outputs="data_filtered",
                    name="save_filtered_data_node",
                ),
                node(
                    func=split_data,
                    inputs="data_filtered",
                    outputs=["train_data", "test_data"],
                    name="split_data_node",
                ),
                node(
                    func=save_filtered_data,
                    inputs="train_data",
                    outputs="base_train",
                    name="save_train_data_node",
                ),
                node(
                    func=save_filtered_data,
                    inputs="test_data",
                    outputs="base_test",
                    name="save_test_data_node",
                ),
                node(
                    func=create_model_input_table,
                    inputs="base_train", # Entrada alterada para apenas base_train
                    outputs="model_input_table",
                    name="create_model_input_table_node",
                ),
                node( # Nó adicionado para pré-processar dados de produção separadamente
                    func=preprocess_single_dataset,
                    inputs="dataset_prod",
                    outputs="preprocessed_prod_data",
                    name="preprocess_prod_data_node",
                ),
            ]
        # Parêntese de fechamento para definição do pipeline dentro do bloco 'with'
        )
