import pandas as pd
from typing import Tuple
import logging # Garante que logging seja importado
import mlflow # Garante que mlflow seja importado
import os # Garante que os seja importado
from sklearn.model_selection import train_test_split # Garante que train_test_split seja importado

logger = logging.getLogger(__name__) # Define o logger para o módulo

# Função preprocess_companies removida

def load_shuttles_to_csv(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Carrega shuttles para csv porque não é possível carregar excel diretamente no spark (Comentário original mantido).
    """
    return shuttles

def preprocess_single_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa um único dataset (dev ou prod)."""
    logger.info("--- Entrando em preprocess_single_dataset ---") # Log de entrada
    logger.info(f"Tipo do dataset de entrada: {type(dataset)}")
    if not isinstance(dataset, pd.DataFrame):
        logger.error(f"Dataset de entrada não é um DataFrame Pandas! Tipo: {type(dataset)}")
        raise TypeError("Entrada para preprocess_single_dataset deve ser um DataFrame Pandas")

    logger.info(f"Shape do dataset de entrada: {dataset.shape}")
    if dataset.empty:
        logger.warning("DataFrame de entrada está vazio. Retornando DataFrame vazio.")
        return pd.DataFrame() # Retorna vazio se a entrada for vazia

    logger.info(f"Colunas do dataset de entrada: {list(dataset.columns)}")

    selected_columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]

    # Verifica se todas as colunas selecionadas existem antes de selecionar
    missing_cols = [col for col in selected_columns if col not in dataset.columns]
    if missing_cols:
        logger.error(f"ERRO: Colunas ausentes no dataset de entrada para pré-processamento: {missing_cols}. Disponíveis: {list(dataset.columns)}")
        raise ValueError(f"Colunas necessárias ausentes para pré-processamento: {missing_cols}")

    try:
        dataset_filtered = dataset[selected_columns].dropna(subset=selected_columns)
        logger.info(f"Shape do dataset filtrado: {dataset_filtered.shape}")
        if dataset_filtered.empty:
            logger.warning("DataFrame de saída está vazio após filtrar/remover NA.")
    except Exception as e:
        logger.error(f"Erro durante filtragem/remoção de NA: {e}", exc_info=True)
        raise # Re-levanta a exceção após logar

    logger.info("--- Saindo de preprocess_single_dataset ---") # Log de saída
    return dataset_filtered

def preprocess_shuttles(dataset_dev: pd.DataFrame, dataset_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Combina datasets dev e prod, seleciona colunas relevantes e remove linhas com valores ausentes.

    Args:
        dataset_dev: Dados brutos do dataset de desenvolvimento.
        dataset_prod: Dados brutos do dataset de produção.
    Returns:
        Dados pré-processados (combinados, colunas filtradas, sem NaNs nas colunas selecionadas).
    """
    logger.info("Combinando datasets dev e prod...")
    shuttles = pd.concat([dataset_dev, dataset_prod], ignore_index=True)
    logger.info(f"Shape do dataset combinado: {shuttles.shape}")

    # Colunas requeridas pelo usuário + variável alvo
    selected_columns = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"]
    logger.info(f"Selecionando colunas: {selected_columns}")

    # Verifica colunas ausentes antes da seleção
    available_columns = list(shuttles.columns)
    missing_cols = [col for col in selected_columns if col not in available_columns]
    if missing_cols:
        logger.error(f"Colunas requeridas ausentes: {missing_cols}. Disponíveis: {available_columns}")
        raise ValueError(f"Colunas requeridas ausentes: {missing_cols}")

    shuttles_selected = shuttles[selected_columns]
    logger.info(f"Shape após seleção de colunas: {shuttles_selected.shape}")

    # Remove linhas com valores ausentes *apenas nas colunas selecionadas*
    shuttles_filtered = shuttles_selected.dropna(subset=selected_columns)
    logger.info(f"Shape após remover NaNs nas colunas selecionadas: {shuttles_filtered.shape}")

    if shuttles_filtered.empty:
        logger.warning("DataFrame resultante está vazio após filtrar/remover NA.")

    # Loga a dimensão resultante conforme solicitado
    logger.info(f"Dimensão resultante do dataset filtrado (linhas, colunas): {shuttles_filtered.shape}")
    # Loga dimensão no MLflow também
    mlflow.log_param("filtered_data_rows", shuttles_filtered.shape[0])
    mlflow.log_param("filtered_data_cols", shuttles_filtered.shape[1])

    return shuttles_filtered

def save_filtered_data(shuttles_filtered: pd.DataFrame) -> pd.DataFrame:
    """Retorna os dados filtrados (salvar é tratado pelo Kedro).

    Args:
        shuttles_filtered: Dados filtrados para shuttles.
    Returns:
        Dados filtrados.
    """
    # Código de depuração removido
    return shuttles_filtered # Retorna para o Kedro salvar baseado no catálogo

def split_data(data: pd.DataFrame, test_ratio: float = 0.2, target_col: str = "shot_made_flag", random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide os dados em conjuntos de treinamento e teste, tentando amostragem estratificada.
    Registra parâmetros e métricas no MLflow.

    Args:
        data: DataFrame a ser dividido.
        test_ratio: Proporção do dataset a incluir na divisão de teste.
        target_col: Nome da coluna alvo para estratificação.
        random_state: Controla o embaralhamento aplicado aos dados antes da divisão.

    Returns:
        Tupla contendo DataFrames de treinamento e teste.
    """
    logger.info(f"Dividindo dados. Shape de entrada: {data.shape}")
    mlflow.log_param("test_ratio", test_ratio)
    mlflow.log_param("random_state_split", random_state)

    if target_col not in data.columns:
        logger.warning(f"Coluna alvo '{target_col}' não encontrada para estratificação. Realizando divisão aleatória simples.")
        mlflow.log_param("split_type", "random")
        train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_state)
    else:
        try:
            # Tenta divisão estratificada
            train_data, test_data = train_test_split(
                data,
                test_size=test_ratio,
                random_state=random_state,
                stratify=data[target_col] # Estratifica baseado na coluna alvo
            )
            logger.info("Realizada divisão estratificada.")
            mlflow.log_param("split_type", "stratified")
            mlflow.log_param("stratify_column", target_col)
        except ValueError as e:
            # Fallback para divisão aleatória se estratificação falhar (ex: poucas amostras em uma classe)
            logger.warning(f"Divisão estratificada falhou ({e}). Voltando para divisão aleatória simples.")
            mlflow.log_param("split_type", "random_fallback")
            train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=random_state)

    train_size = len(train_data)
    test_size = len(test_data)
    logger.info(f"Shape dos dados de treino: {train_data.shape}")
    logger.info(f"Shape dos dados de teste: {test_data.shape}")

    # Loga métricas
    mlflow.log_metric("train_data_rows", train_size)
    mlflow.log_metric("test_data_rows", test_size)

    return train_data, test_data

# Função preprocess_reviews removida

def create_model_input_table(
    shuttles: pd.DataFrame # Argumentos de entrada companies e reviews removidos
) -> pd.DataFrame:
    """Cria a tabela de entrada do modelo a partir dos dados de shuttles.
       (Simplificado pois companies e reviews não são usados).

    Args:
        shuttles: Dados pré-processados para shuttles (contém divisão treino/teste).
    Returns:
        Tabela de entrada do modelo (atualmente apenas os dados de shuttles).
    """
    # Como companies e reviews foram removidos, esta função pode precisar
    # de maior simplificação ou ajuste baseado nos requisitos reais.
    # Por agora, apenas retorna os dados de shuttles diretamente.
    # Renomeia colunas se necessário para consistência downstream
    # shuttles = shuttles.rename(columns={"id": "shuttle_id"}) # Exemplo se necessário
    model_input_table = shuttles.dropna() # Mantém dropna por consistência
    return model_input_table
