import logging
import pandas as pd
import mlflow
from pycaret.classification import predict_model
from sklearn.metrics import log_loss, f1_score

logger = logging.getLogger(__name__)
TARGET_COLUMN = "shot_made_flag"

def apply_model_to_production(model: object, prod_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o modelo treinado aos dados de produção pré-processados,
    registra métricas e retorna as previsões.

    Args:
        model: O modelo de classificação treinado (carregado pelo Kedro).
        prod_data: Dados de produção pré-processados.

    Returns:
        DataFrame contendo os dados de produção originais junto com as previsões.
    """
    if prod_data.empty:
        logger.warning("Dados de produção estão vazios. Pulando previsão.")
        # Retorna dataframe vazio com colunas de previsão esperadas
        return pd.DataFrame(columns=list(prod_data.columns) + ['prediction_label', 'prediction_score'])

    logger.info(f"Aplicando modelo aos dados de produção (shape: {prod_data.shape})...")

    # Faz previsões
    # Garante verbose=False se não quiser a saída padrão do PyCaret durante a previsão
    predictions_df = predict_model(model, data=prod_data, verbose=False)
    logger.info(f"Previsões geradas (shape: {predictions_df.shape}).")

    # Registra métricas no MLflow dentro da run "PipelineAplicacao"
    # Verifica se a coluna alvo existe nos dados de produção para cálculo de métricas
    if TARGET_COLUMN in predictions_df.columns:
        try:
            prod_logloss = log_loss(predictions_df[TARGET_COLUMN], predictions_df['prediction_score'])
            prod_f1 = f1_score(predictions_df[TARGET_COLUMN], predictions_df['prediction_label'])

            mlflow.log_metric("prod_logloss", prod_logloss)
            mlflow.log_metric("prod_f1_score", prod_f1)
            logger.info(f"Produção - Log Loss: {prod_logloss}")
            logger.info(f"Produção - F1 Score: {prod_f1}")
        except Exception as e:
            logger.error(f"Erro ao calcular ou registrar métricas de produção: {e}", exc_info=True)
            # Continua mesmo se métricas falharem
    else:
        logger.warning(f"Coluna alvo '{TARGET_COLUMN}' não encontrada nos dados de produção. Pulando cálculo de métricas.")


    # Retorna o dataframe com previsões (será salvo como artefato pelo Kedro)
    # Seleciona colunas relevantes se necessário, ex: features originais + prediction_label + prediction_score
    # Por agora, retorna o dataframe completo gerado por predict_model
    return predictions_df
