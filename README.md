# projetoinfnet

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Visão Geral (Padrão Kedro)

Projeto Kedro gerado usando `kedro 0.19.12`.

Documentação disponível em [documentação do Kedro](https://docs.kedro.org).

## Visão Geral do Projeto (Específica)

Este projeto tem como objetivo desenvolver um modelo preditivo para determinar se um arremesso de basquete realizado por Kobe Bryant ("Black Mamba") resultou em acerto ou erro (`shot_made_flag`). O projeto utiliza dados históricos de arremessos e implementa um fluxo de trabalho de Machine Learning usando Kedro, Pandas, PyCaret, Scikit-learn e MLflow.

O projeto inclui os seguintes pipelines principais:
*   **`data_processing`**: Carrega, combina, limpa e divide os dados brutos em conjuntos de treino e teste, além de preparar os dados de produção.
*   **`data_science`**: Treina dois modelos de classificação (Regressão Logística e Árvore de Decisão), avalia-os, seleciona o melhor modelo com base no F1-Score e registra métricas e parâmetros no MLflow.
*   **`aplicacao`**: Aplica o modelo final treinado aos dados de produção pré-processados, registra métricas de produção no MLflow e salva as previsões.

## Diagrama do Pipeline

O diagrama abaixo ilustra o fluxo de dados e as etapas dos pipelines do projeto:

```mermaid
graph TD
    subgraph Aquisição_de_Dados
        A[Dados Brutos: dev e prod] --> B[Salvar em /data/01_raw/]
    end

    subgraph Pipeline_Data_Processing
        B --> C[Processar e Dividir Dados]
        C --> D[Salvar: base_train e base_test]
        B --> E[Pré-processar Prod]
        E --> F[Salvar: preprocessed_prod]
    end

    subgraph Pipeline_Data_Science
        D --> G[Treinar Modelos: LR e DT]
        G --> H[Selecionar Melhor Modelo]
        H --> I[Salvar: final_model]
    end

    subgraph Pipeline_Aplicacao
        F --> J[Aplicar Modelo]
        I --> J
        J --> K[Salvar: predictions]
    end

    subgraph MLflow_Tracking
        G --> L[MLflow]
        H --> L
        J --> L
    end

    style L fill:#f9f,stroke:#333,stroke-width:2px
```

## Detalhamento das Etapas

### 1. Aquisição e Preparação Inicial dos Dados
*   Os datasets `dataset_kobe_dev.parquet` e `dataset_kobe_prod.parquet` são obtidos e colocados no diretório `data/01_raw/`.

### 2. Pipeline `data_processing` (Run MLflow: "PreparacaoDados")
*   **Carregar e Combinar:** Os datasets `dev` e `prod` são carregados e combinados.
*   **Filtrar e Limpar:** As colunas relevantes (`lat`, `lon`, `minutes_remaining`, `period`, `playoffs`, `shot_distance`, `shot_made_flag`) são selecionadas. Linhas com valores nulos nessas colunas são removidas. O resultado é salvo como `data_filtered.parquet`. A dimensão resultante é logada no MLflow.
*   **Dividir Treino/Teste:** O `data_filtered` é dividido em `base_train.parquet` (80%) e `base_test.parquet` (20%), usando amostragem estratificada (se possível) pela coluna `shot_made_flag`. Parâmetros (proporção, tipo de split) e métricas (tamanho das bases) são logados no MLflow.
*   **Pré-processar Produção:** O dataset `dataset_prod` original é pré-processado separadamente (seleção de colunas, remoção de nulos) e salvo como `preprocessed_prod_data.parquet`.

### 3. Pipeline `data_science` (Run MLflow: "Treinamento")
*   **Treinar Regressão Logística:** Um modelo de Regressão Logística é treinado usando PyCaret com `base_train` e avaliado com `base_test`. A métrica Log Loss é logada no MLflow. O modelo é salvo como `lr_model.pkl`.
*   **Treinar Árvore de Decisão:** Um modelo de Árvore de Decisão é treinado usando PyCaret com `base_train` e avaliado com `base_test`. As métricas Log Loss e F1-Score são logadas no MLflow. O modelo é salvo como `dt_model.pkl`.
*   **Selecionar Melhor Modelo:** Os modelos LR e DT são comparados com base no F1-Score no `base_test`. Métricas de comparação são logadas. O melhor modelo é selecionado e salvo como `final_model.pkl`.
*   **Salvar Justificação:** A justificativa da seleção é logada como parâmetro no MLflow e salva em `model_justification.csv`.

### 4. Pipeline `aplicacao` (Run MLflow: "PipelineAplicacao")
*   **Carregar Dados e Modelo:** Carrega `preprocessed_prod_data.parquet` e `final_model.pkl`.
*   **Aplicar Modelo:** Aplica o `final_model` aos dados de produção pré-processados.
*   **Logar Métricas:** Calcula e loga as métricas Log Loss e F1-Score no MLflow, *se* a coluna `shot_made_flag` estiver presente nos dados de produção.
*   **Salvar Previsões:** Salva o DataFrame resultante (dados de produção + colunas `prediction_label` e `prediction_score`) como `production_predictions.parquet`. Este arquivo pode ser logado como artefato no MLflow.


## Como Executar Pipeline Kedro

Você pode executar o projeto Kedro com:

```
kedro run
```
Para executar um pipeline específico:
```
kedro run --pipeline <nome_do_pipeline>
```
Exemplo:
```
kedro run --pipeline data_processing
kedro run --pipeline data_science
kedro run --pipeline aplicacao
```

## Como Executar o Dashboard Streamlit

Este projeto inclui um dashboard interativo construído com Streamlit para visualizar e interagir com as predições do modelo.

1.  **Certifique-se de que as dependências estão instaladas:**
    Instale todas as dependências, incluindo o Streamlit:
    ```bash
    # Execute a partir da pasta raiz do projeto (projetoinfnet)
    pip install -r requirements.txt
    ```

2.  **Execute o Dashboard:**
    Navegue até a pasta raiz do projeto (`projetoinfnet`) no seu terminal e execute o seguinte comando:
    ```bash
    streamlit run src/projetoinfnet/dashboard.py
    ```
    Isso iniciará o servidor Streamlit e abrirá o dashboard no seu navegador padrão.

## Como Acessar a Interface do MLflow

Este projeto utiliza MLflow para rastrear experimentos. Para visualizar os resultados na interface web do MLflow:

1.  **Navegue até o diretório raiz do projeto** (`projetoinfnet`) no seu terminal.
2.  **Execute o comando:**
    ```bash
    mlflow ui
    ```
3.  **Abra seu navegador** e acesse o endereço fornecido no terminal (normalmente `http://127.0.0.1:5000`).

    Você poderá ver as execuções dos pipelines (`data_processing`, `data_science`, `aplicacao`), seus parâmetros, métricas e artefatos registrados.

## Como Testar seu Projeto Kedro

Dê uma olhada nos arquivos `src/tests/test_run.py` e `src/tests/pipelines/data_science/test_pipeline.py` para instruções sobre como escrever seus testes. Execute os testes da seguinte forma:

```
pytest
```

Para configurar o limite de cobertura (coverage threshold), veja o arquivo `.coveragerc`.

## Dependências do Projeto

Para ver e atualizar os requisitos de dependência do seu projeto, use `requirements.txt`. Instale os requisitos do projeto com `pip install -r requirements.txt`.

[Mais informações sobre dependências do projeto](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## Como Trabalhar com Kedro e Notebooks

> Nota: Usar `kedro jupyter` ou `kedro ipython` para executar seu notebook fornece estas variáveis no escopo: `catalog`, `context`, `pipelines` e `session`.
>
> Jupyter, JupyterLab e IPython já estão incluídos nos requisitos do projeto por padrão, então, uma vez que você tenha executado `pip install -r requirements.txt`, você não precisará de etapas extras antes de usá-los.

### Jupyter
Para usar notebooks Jupyter no seu projeto Kedro, você precisa instalar o Jupyter:

```
pip install jupyter
```

Após instalar o Jupyter, você pode iniciar um servidor de notebook local:

```
kedro jupyter notebook
```

### JupyterLab
Para usar o JupyterLab, você precisa instalá-lo:

```
pip install jupyterlab
```

Você também pode iniciar o JupyterLab:

```
kedro jupyter lab
```

### IPython
E se você quiser executar uma sessão IPython:

```
kedro ipython
```

### Como ignorar células de saída do notebook no `git`
Para remover automaticamente todo o conteúdo das células de saída antes de fazer commit para o `git`, você pode usar ferramentas como [`nbstripout`](https://github.com/kynan/nbstripout). Por exemplo, você pode adicionar um hook em `.git/config` com `nbstripout --install`. Isso executará `nbstripout` antes que qualquer coisa seja commitada para o `git`.

> *Nota:* Suas células de saída serão mantidas localmente.

## Empacotar seu Projeto Kedro

[Mais informações sobre como construir a documentação do projeto e empacotar seu projeto](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
