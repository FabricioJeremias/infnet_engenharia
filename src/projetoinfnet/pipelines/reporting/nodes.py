import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession


# This function uses plotly.express
def compare_passenger_capacity_exp(preprocessed_shuttles: SparkDataFrame):
    spark = SparkSession.builder.appName("PassengerCapacityComparison").getOrCreate()

    # Register the DataFrame as a temporary table
    preprocessed_shuttles.createOrReplaceTempView("shuttles")

    # Perform the grouping and aggregation using SQL
    query = """
            SELECT shuttle_type, AVG(passenger_capacity) as passenger_capacity
            FROM shuttles
            GROUP BY shuttle_type
        """
    grouped_data = spark.sql(query)
    # Convert Spark DataFrame to Pandas for visualization
    pandas_grouped_data = grouped_data.toPandas()
    return pandas_grouped_data


def compare_passenger_capacity_go(preprocessed_shuttles: SparkDataFrame):
    spark = SparkSession.builder.appName("PassengerCapacityComparison").getOrCreate()

    # Register the DataFrame as a temporary table
    preprocessed_shuttles.createOrReplaceTempView("shuttles")

    # Perform the grouping and aggregation using SQL
    query = """
        SELECT shuttle_type, AVG(passenger_capacity) as avg_passenger_capacity
        FROM shuttles
        GROUP BY shuttle_type
    """
    grouped_data = spark.sql(query)

    # Convert Spark DataFrame to Pandas for visualization
    pandas_grouped_data = grouped_data.toPandas()

    # Create the Plotly figure
    fig = go.Figure(
        [
            go.Bar(
                x=pandas_grouped_data["shuttle_type"],
                y=pandas_grouped_data["avg_passenger_capacity"],
            )
        ]
    )

    return fig


def create_confusion_matrix(companies: pd.DataFrame):
    actuals = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    predicted = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    data = {"y_Actual": actuals, "y_Predicted": predicted}
    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(
        df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
    )
    sn.heatmap(confusion_matrix, annot=True)
    return plt


def plot_model_comparison_metrics(lr_f1: float, lr_logloss: float, dt_f1: float, dt_logloss: float) -> plt.Figure:
    """
    Gera um gráfico de barras comparando F1 Score e Log Loss para
    Regressão Logística e Árvore de Decisão.

    Args:
        lr_f1: F1 score da Regressão Logística.
        lr_logloss: Log Loss da Regressão Logística.
        dt_f1: F1 score da Árvore de Decisão.
        dt_logloss: Log Loss da Árvore de Decisão.

    Returns:
        Objeto matplotlib Figure contendo os gráficos.
    """
    metrics_data = {
        'Model': ['Logistic Regression', 'Decision Tree'],
        'F1 Score': [lr_f1, dt_f1],
        'Log Loss': [lr_logloss, dt_logloss]
    }
    df_metrics = pd.DataFrame(metrics_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Comparação de Métricas dos Modelos')

    # Gráfico F1 Score
    sn.barplot(x='Model', y='F1 Score', data=df_metrics, ax=axes[0])
    axes[0].set_title('F1 Score (Maior é Melhor)')
    axes[0].set_ylim(bottom=0) # Garante que o eixo y comece em 0
    for index, value in enumerate(df_metrics['F1 Score']):
        axes[0].text(index, value + 0.01, f'{value:.4f}', ha='center') # Adiciona valor no topo da barra

    # Gráfico Log Loss
    sn.barplot(x='Model', y='Log Loss', data=df_metrics, ax=axes[1])
    axes[1].set_title('Log Loss (Menor é Melhor)')
    axes[1].set_ylim(bottom=0) # Garante que o eixo y comece em 0
    for index, value in enumerate(df_metrics['Log Loss']):
        axes[1].text(index, value + 0.01, f'{value:.4f}', ha='center') # Adiciona valor no topo da barra


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta layout para evitar sobreposição do título
    return fig
