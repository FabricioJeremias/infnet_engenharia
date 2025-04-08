import streamlit as st
import pandas as pd
from pycaret.classification import predict_model 
from pathlib import Path # Usar pathlib para caminhos
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

# --- Configuração da Página ---
st.set_page_config(page_title="Dashboard de Predição de Arremessos", layout="wide")

# --- Carregamento do Modelo via Kedro Catalog ---
# Obter o diretório raiz do projeto Kedro
# Assumindo que dashboard.py está em src/projetoinfnet/
project_path = Path(__file__).resolve().parents[2] # Sobe dois níveis: projetoinfnet -> src -> raiz

@st.cache_resource # Cacheia o contexto e o modelo para performance
def load_kedro_model():
    """Carrega o modelo usando o DataCatalog do Kedro."""
    try:
        # Inicializa o projeto Kedro para acessar o contexto e o catálogo
        bootstrap_project(project_path)
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            # Carrega o modelo definido como "final_model" no catalog.yml
            model = catalog.load("final_model")
            # Carrega dados de treino para inferir features (se necessário)
            train_df_sample = catalog.load("base_train").head(1).drop(columns=['shot_made_flag'], errors='ignore')
            return model, train_df_sample, None # Retorna modelo, dados de exemplo, e None para erro
    except Exception as e:
        st.error(f"Erro ao carregar o modelo via Kedro Catalog: {e}")
        st.error("Verifique a configuração do Kedro, o catalog.yml e as permissões.")
        return None, None, e # Retorna None para modelo/dados, e o erro

model, train_df_sample, load_error = load_kedro_model()

# --- Título e Descrição ---
st.title("🏀 Dashboard de Predição de Arremessos - Kobe Bryant")
st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para prever a probabilidade
de um arremesso de Kobe Bryant ser convertido (Made = 1) ou errado (Missed = 0),
baseado nas características do arremesso.

**Instruções:** Preencha os campos a esquerda com as informações do arremesso e clique em "Prever".
""")



# --- Coleta de Inputs do Usuário ---
if model is not None and load_error is None: # Verifica se o modelo foi carregado sem erro
    st.sidebar.header("Características do Arremesso")

    # Identificar as features esperadas pelo modelo PyCaret
    # Uma forma comum é verificar as colunas usadas no setup original,
    # ou inspecionar o pipeline interno do modelo.
    # Para PyCaret >= 3.0, model.feature_names_in_ pode funcionar.
    # Para versões anteriores, pode ser mais complexo.
    # Vamos tentar obter as features do pipeline interno.
    try:
        # Tentativa comum para obter features de pipelines sklearn/pycaret
        if hasattr(model, 'feature_names_in_'):
             features = model.feature_names_in_
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][-1], 'feature_names_in_'):
             # Acessa o último step do pipeline (geralmente o estimador)
             features = model.steps[-1][-1].feature_names_in_
        else:
             # Fallback: Usar as colunas dos dados de treino carregados via catálogo
             if train_df_sample is not None:
                 features = train_df_sample.columns.tolist()
                 st.warning("Não foi possível determinar as features diretamente do modelo. Usando colunas de 'base_train' carregado via catálogo. Certifique-se que estas são as features corretas após o pré-processamento do PyCaret.")
             else:
                 st.error("Não foi possível carregar dados de exemplo ('base_train') para inferir features.")
                 features = [] # Define como lista vazia para evitar mais erros

        # Remover a coluna alvo da lista de features de entrada
        TARGET_COLUMN = "shot_made_flag"
        if TARGET_COLUMN in features:
            features.remove(TARGET_COLUMN)

        if features:
            # st.write("Features identificadas para input:", features) # Debug opcional
            pass
        else:
             st.error("Nenhuma feature identificada. O dashboard não pode continuar.")


    except AttributeError as attr_err:
        st.error(f"Erro ao acessar 'feature_names_in_' no modelo: {attr_err}")
        st.error("Não foi possível determinar as features necessárias automaticamente.")
        features = [] # Define como lista vazia

    # Usar a lista 'features' original inferida
    features_to_use = features


    input_data = {}
    # Usar a lista de features decidida acima (features_to_use)
    if features_to_use:
        # Criar inputs dinamicamente
        for feature in features_to_use: # Iterar sobre features_to_use
            # Heurística simples para tipo de input (pode precisar de ajuste)
            feature_norm = feature.strip().lower() # Normalizar nome para verificação

            if 'minutes_remaining' in feature_norm or 'seconds_remaining' in feature_norm or 'shot_distance' in feature_norm or 'period' in feature_norm:
                 # Usar o nome original 'feature' para o label e chave do dicionário
                 input_data[feature] = st.sidebar.number_input(f"{feature}", value=0, step=1)
            # Tratar 'playoffs' como numérico (0 ou 1)
            elif 'playoffs' in feature_norm:
                 input_data[feature] = st.sidebar.number_input(f"{feature} (0=Não, 1=Sim)", value=0, min_value=0, max_value=1, step=1)
            # Incluir lat e lon na condição para number_input float
            elif 'loc_x' in feature_norm or 'loc_y' in feature_norm or 'lat' in feature_norm or 'lon' in feature_norm:
                 input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.1, format="%.1f")
            else: # Fallback para outras colunas (se houver)
                 st.warning(f"Feature não mapeada explicitamente: {feature}. Tratando como texto por padrão.")
                 input_data[feature] = st.sidebar.text_input(f"{feature}", value="")

        # Botão para fazer a predição
        predict_button = st.button("Prever Resultado do Arremesso")

        # --- Predição e Exibição ---
        if predict_button:
            # Bloco try...except para criação do DataFrame e predição
            input_df = None # Inicializar input_df
            try:
                # Criar DataFrame diretamente dos inputs (tipos básicos dos widgets)
                input_df = pd.DataFrame([input_data])
                # Garantir a ordem das colunas conforme 'features_to_use'
                input_df = input_df[features_to_use]
                # Resetar o índice pode ajudar em alguns casos com pipelines
                input_df = input_df.reset_index(drop=True)

                # Realizar a predição usando predict_model do PyCaret
                with st.spinner("Realizando predição..."):
                    # Usar predict_model que retorna um DataFrame com colunas adicionais
                    prediction_df = predict_model(model, data=input_df, verbose=False)

                # Exibir resultados se a predição for bem-sucedida
                st.subheader("Resultado da Predição:")
                # Extrair label e score do DataFrame retornado por predict_model
                pred_label = prediction_df['prediction_label'].iloc[0]
                pred_score = prediction_df['prediction_score'].iloc[0] # Probabilidade da classe 1 (Made)

                if pred_label == 1:
                    st.success(f"**Predição: Arremesso Convertido (Made)**")
                    st.metric(label="Probabilidade de Acerto", value=f"{pred_score:.2%}")
                else:
                    st.error(f"**Predição: Arremesso Perdido (Missed)**")
                    st.metric(label="Probabilidade de Acerto", value=f"{pred_score:.2%}") # Mostra a prob de ser 1

                st.write("---")
                st.write("Detalhes da Predição (Output Completo do Modelo):")
                st.dataframe(prediction_df) # Exibir o DataFrame completo retornado

            except KeyError as ke:
                # Verificar se o KeyError ainda ocorre
                st.error(f"Erro de Chave durante a predição com predict_model: {ke}")
                st.error("Isso geralmente significa que uma coluna esperada pelo modelo está faltando nos dados de entrada.")
                # Mostrar colunas fornecidas se o DataFrame foi criado
                if input_df is not None:
                     st.error(f"Colunas fornecidas ao predict_model: {input_df.columns.tolist()}")
                else:
                     st.error("DataFrame de entrada não pôde ser criado.")
            except Exception as e:
                st.error(f"Erro durante a criação do DataFrame ou predição: {e}")
                st.error("Verifique os tipos de dados inseridos e se correspondem ao esperado pelo modelo.")
    else:
        st.warning("Não foi possível criar os campos de entrada pois as features não foram identificadas.")

elif load_error:
    # Se houve erro no carregamento, já exibimos a mensagem na função load_kedro_model
    st.error("A aplicação não pode continuar devido a um erro no carregamento do modelo.")
else:
    # Caso inesperado onde model é None mas não houve erro explícito
    st.error("O modelo não está disponível. A aplicação não pode continuar.")


st.sidebar.info("Projeto INFNET. Fabrício Jeremias")
