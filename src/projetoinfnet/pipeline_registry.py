"""Pipelines do projeto.""" # Traduzido

from kedro.pipeline import Pipeline

# Importa todas as funções de criação de pipeline explicitamente
from projetoinfnet.pipelines import data_processing as dp
from projetoinfnet.pipelines import data_science as ds
from projetoinfnet.pipelines import aplicacao_pipeline as ap
from projetoinfnet.pipelines import reporting as rp # Importa o pipeline reporting

def register_pipelines() -> dict[str, Pipeline]:
    """Registra os pipelines do projeto.

    Returns:
        Um mapeamento de nomes de pipeline para objetos ``Pipeline``.
    """

    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    aplicacao_pipeline = ap.create_pipeline()
    reporting_pipeline = rp.create_pipeline() # Cria a instância do pipeline reporting

    # Constrói o dicionário diretamente usando nomes completos
    pipelines = {
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "aplicacao": aplicacao_pipeline,
        "reporting": reporting_pipeline, # Adiciona o pipeline reporting ao registro
        "__default__": data_processing_pipeline + data_science_pipeline + reporting_pipeline, # Adiciona reporting ao default
    }

    return pipelines
