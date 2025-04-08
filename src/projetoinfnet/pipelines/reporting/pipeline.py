from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compare_passenger_capacity_exp,
    compare_passenger_capacity_go,
    create_confusion_matrix,
    plot_model_comparison_metrics, 
)


def create_pipeline(**kwargs) -> Pipeline:
    """Cria o pipeline de reporting, focando na comparação de métricas.""" # Descrição atualizada
    return pipeline(
        [
            node(
                func=plot_model_comparison_metrics,
                inputs=[
                    "lr_f1_metric",      # Entrada vinda do pipeline data_science
                    "lr_logloss_metric", # Entrada vinda do pipeline data_science
                    "dt_f1_metric",      # Entrada vinda do pipeline data_science
                    "dt_logloss_metric"  # Entrada vinda do pipeline data_science
                ],
                outputs="model_comparison_metrics_plot", # Saída para o catálogo
                name="plot_model_metrics_node",
            ),
        ]
    )
