import logging
import mlflow
import pandas as pd
import pandas as pd
from pycaret.classification import setup, create_model, predict_model, pull, finalize_model, save_model
from sklearn.metrics import log_loss, f1_score

logger = logging.getLogger(__name__)

TARGET_COLUMN = "shot_made_flag" # Define a coluna alvo

def _setup_pycaret(data: pd.DataFrame, target: str):
    """Configura o ambiente PyCaret."""
    numeric_features = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']
    # Se houvesse features categóricas conhecidas, seriam listadas aqui:
    # categorical_features = ['feature_cat_1', 'feature_cat_2']

    logger.info(f"Configurando PyCaret com features numéricas explícitas: {numeric_features}")

    # Argumento silent=True removido pois não é suportado na versão atual do PyCaret
    # Logging do MLflow desabilitado para evitar AttributeError
    return setup(data=data,
                 target=target,
                 session_id=123,
                 log_experiment=False,
                 experiment_name='kobe_shot_prediction',
                 numeric_features=numeric_features,
                 # categorical_features=categorical_features, # Descomentar se houver categóricas
                 verbose=False)

def train_logistic_regression(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """Treina um modelo de Regressão Logística usando PyCaret e registra métricas."""
    logger.info("Configurando PyCaret para Regressão Logística...")
    clf_setup = _setup_pycaret(train_data, TARGET_COLUMN)

    logger.info("Treinando modelo de Regressão Logística...")
    lr_model = create_model('lr', verbose=False)

    logger.info("Avaliando modelo de Regressão Logística...")
    predictions = predict_model(lr_model, data=test_data, verbose=False)
    logloss = log_loss(predictions[TARGET_COLUMN], predictions['prediction_score'])
    mlflow.log_metric("lr_logloss", logloss) # Registra log loss conforme solicitado
    logger.info(f"Regressão Logística - Log Loss: {logloss}")

    # Finaliza e salva o modelo (opcional, mas boa prática)
    final_lr = finalize_model(lr_model)
    return final_lr

def train_decision_tree(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """Treina um modelo de Árvore de Decisão usando PyCaret e registra métricas."""
    logger.info("Configurando PyCaret para Árvore de Decisão...")
    # Precisa de setup separado para cada modelo se parâmetros diferirem ou para evitar problemas de estado
    clf_setup = _setup_pycaret(train_data, TARGET_COLUMN)

    logger.info("Treinando modelo de Árvore de Decisão...")
    dt_model = create_model('dt', verbose=False)

    logger.info("Avaliando modelo de Árvore de Decisão...")
    predictions = predict_model(dt_model, data=test_data, verbose=False)
    logloss = log_loss(predictions[TARGET_COLUMN], predictions['prediction_score'])
    f1 = f1_score(predictions[TARGET_COLUMN], predictions['prediction_label']) # Usa prediction_label para F1
    mlflow.log_metric("dt_logloss", logloss) # Registra log loss conforme solicitado
    mlflow.log_metric("dt_f1_score", f1) # Registra F1 score conforme solicitado
    logger.info(f"Árvore de Decisão - Log Loss: {logloss}")
    logger.info(f"Árvore de Decisão - F1 Score: {f1}")

    # Finaliza e salva o modelo
    final_dt = finalize_model(dt_model)
    # save_model(final_dt, 'decision_tree_model_pycaret') # Salvar é tratado pelo catálogo Kedro

    return final_dt

def select_best_model(lr_model, dt_model, test_data: pd.DataFrame) -> tuple[object, str, float, float, float, float]:
    """Seleciona o melhor modelo baseado no F1 score, registra a escolha e retorna as métricas."""

    # Avalia LR
    lr_predictions = predict_model(lr_model, data=test_data, verbose=False)
    lr_f1 = f1_score(lr_predictions[TARGET_COLUMN], lr_predictions['prediction_label'])
    lr_logloss = log_loss(lr_predictions[TARGET_COLUMN], lr_predictions['prediction_score'])

    # Avalia DT
    dt_predictions = predict_model(dt_model, data=test_data, verbose=False)
    dt_f1 = f1_score(dt_predictions[TARGET_COLUMN], dt_predictions['prediction_label'])
    dt_logloss = log_loss(dt_predictions[TARGET_COLUMN], dt_predictions['prediction_score'])

    logger.info(f"Final LR F1: {lr_f1}, LogLoss: {lr_logloss}")
    logger.info(f"Final DT F1: {dt_f1}, LogLoss: {dt_logloss}")

    # Registra métricas de comparação
    mlflow.log_metric("final_lr_f1", lr_f1)
    mlflow.log_metric("final_lr_logloss", lr_logloss)
    mlflow.log_metric("final_dt_f1", dt_f1)
    mlflow.log_metric("final_dt_logloss", dt_logloss)

    # Critério de seleção: F1 maior geralmente é melhor para tarefas de classificação, especialmente se as classes estiverem desbalanceadas.
    if dt_f1 >= lr_f1:
        best_model_obj = dt_model
        best_model_name = "Decision Tree"
        justification = f"Árvore de Decisão selecionada devido a F1 score maior ou igual ({dt_f1:.4f} vs {lr_f1:.4f})." # Traduzido
    else:
        best_model_obj = lr_model
        best_model_name = "Logistic Regression"
        justification = f"Regressão Logística selecionada devido a F1 score maior ({lr_f1:.4f} vs {dt_f1:.4f})." # Traduzido

    logger.info(justification)
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_param("selection_justification", justification) # Registra justificação conforme solicitado
    return best_model_obj, justification, lr_f1, lr_logloss, dt_f1, dt_logloss

def save_model_justification(model_justification: str) -> pd.DataFrame:
    """Salva a justificação da seleção do modelo em um arquivo CSV.

    Args:
        model_justification: Justificativa textual para seleção do modelo.
    Returns:
        DataFrame Pandas contendo a justificação.
    """
    return pd.DataFrame([{"justification": model_justification}])
