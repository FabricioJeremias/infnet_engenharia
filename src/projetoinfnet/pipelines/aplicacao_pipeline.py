from kedro.pipeline import Pipeline, node, pipeline
import mlflow
from .aplicacao_nodes import apply_model_to_production

def create_pipeline(**kwargs) -> Pipeline:
    """Cria o pipeline de aplicação para aplicar o modelo aos dados de produção."""

    # Nota: O gerenciamento de runs do MLflow pode ser melhor tratado via hooks ou plugin Kedro-MLflow
    # para integração mais robusta, mas está encapsulado aqui por simplicidade baseada na solicitação.
    with mlflow.start_run(run_name="PipelineAplicacao"):
        # Loga uma tag para identificar facilmente este tipo de run
        mlflow.set_tag("pipeline_type", "application")

        return pipeline(
            [
                node(
                    func=apply_model_to_production,
                    # Entrada: modelo final do data_science e dados prod pré-processados
                    inputs=["final_model", "preprocessed_prod_data"],
                    # Saída: dataframe com previsões
                    outputs="production_predictions",
                    name="apply_model_to_prod_node",
                ),
            ],
            # Define entradas para o pipeline geral
            inputs=["final_model", "preprocessed_prod_data"],
            # Define saídas para o pipeline geral
            outputs=["production_predictions"],
            namespace="aplicacao", # Adiciona namespace para clareza
        )
