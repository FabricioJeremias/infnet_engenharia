import mlflow
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_logistic_regression,
    train_decision_tree,
    select_best_model,
    save_model_justification,
)

def create_pipeline(**kwargs) -> Pipeline:
    """Cria o pipeline de ciência de dados.""" 
    # Adiciona contexto de run do MLflow conforme solicitado
    with mlflow.start_run(run_name="Treinamento"):
        # Loga uma tag para identificar facilmente este tipo de run
        mlflow.set_tag("pipeline_type", "data_science")
        
        # Define a estrutura do pipeline dentro do contexto
        data_science_pipeline = pipeline(
            [
                node(
                    func=train_logistic_regression,
                    inputs=["base_train", "base_test"],
                    outputs="lr_model",
                    name="train_lr_node",
                ),
                node(
                    func=train_decision_tree,
                    inputs=["base_train", "base_test"], 
                    outputs="dt_model",
                    name="train_dt_node",
                ),
                node(
                    func=select_best_model,
                    inputs=["lr_model", "dt_model", "base_test"],
                    outputs=[
                        "final_model",
                        "best_model_justification_string",
                        "lr_f1_metric",      
                        "lr_logloss_metric", 
                        "dt_f1_metric",      
                        "dt_logloss_metric"  
                    ],
                    name="select_best_model_node",
                ),
                node(
                    func=save_model_justification,
                    inputs="best_model_justification_string", # Entrada é a string de select_best_model
                    outputs="model_justification", # Saída corresponde à entrada do catálogo
                    name="save_model_justification_node",
                ),
            ],
            # Define entradas e saídas para o pipeline geral se necessário,
            # caso contrário, o Kedro as infere.
            # inputs=["base_train", "base_test"], # Exemplo se necessário
            # outputs=["final_model", "model_justification"], # Exemplo se necessário
            # namespace="data_science", # Namespace removido
        ) # Parêntese de fechamento para definição do pipeline
        
        return data_science_pipeline # Retorna o pipeline definido
