import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from google.cloud import bigquery
from sklearn import metrics as mt
from sklearn.preprocessing import StandardScaler, RobustScaler
import sklearn.model_selection as ms

## Auxiliares

def save_picked_file(file, name):
    """
    Salva um objeto Python em formato `.pkl` usando compressão.

    Args:
        file (object): Objeto a ser serializado.
        name (str): Caminho e nome base do arquivo (sem extensão).

    Returns:
        list[str]: Lista com o caminho do arquivo salvo.
    """
    return joblib.dump(file, f'{name}.pkl', compress=3)


def load_picked_file(name):
    """
    Carrega um objeto Python de um arquivo `.pkl`.

    Args:
        name (str): Caminho e nome base do arquivo (sem extensão).

    Returns:
        object: Objeto desserializado do arquivo.
    """
    return joblib.load(f'{name}.pkl')


def save_parquet_file(file, name):
    """
    Salva um DataFrame em formato `.parquet`.

    Args:
        file (pandas.DataFrame): DataFrame a ser salvo.
        name (str): Caminho e nome base do arquivo (sem extensão).

    Returns:
        None
    """
    table = pa.Table.from_pandas(file)
    return pq.write_table(table, f'{name}.parquet')


def load_parquet_file(name):
    """
    Carrega um arquivo `.parquet` e o converte em um DataFrame.

    Args:
        name (str): Caminho e nome base do arquivo (sem extensão).

    Returns:
        pandas.DataFrame: Tabela carregada do arquivo.
    """
    table = pq.read_table(f'{name}.parquet')
    return table.to_pandas()


def load_data_bigquery(projeto, query):
    """
    Executa uma query no BigQuery e retorna os dados como um DataFrame.

    Args:
        projeto (str): Nome do projeto no GCP.
        query (str): Comando SQL a ser executado.

    Returns:
        pandas.DataFrame: Dados retornados pela consulta.
    """
    client = bigquery.Client(project=projeto)
    query_job = client.query(query)

    df_list = []
    for row in query_job.result():
        df_list.append(dict(row))

    return pd.DataFrame(df_list)


def create_table_bigquery(projeto, dataset, table_name, data):
    """
    Cria ou sobrescreve uma tabela no BigQuery com os dados de um DataFrame.

    Args:
        projeto (str): Nome do projeto no GCP.
        dataset (str): Nome do dataset no BigQuery.
        table_name (str): Nome da tabela a ser criada.
        data (pandas.DataFrame): Dados a serem carregados.

    Returns:
        str: Confirmação da conclusão do job.
    """
    client = bigquery.Client(project=projeto)
    table_ref = client.dataset(dataset).table(table_name)
    job = client.load_table_from_dataframe(data, table_ref)
    job.result()
    
    return "Done!"


def salvar_grafico(nome_arquivo, pasta_destino, figura=None, formato='png', dpi=300):
    """
    Salva a figura atual do Matplotlib em um diretório especificado.

    Args:
        nome_arquivo (str): Nome do arquivo (sem extensão).
        pasta_destino (str): Caminho para salvar o gráfico.
        figura (matplotlib.figure.Figure, optional): Figura a ser salva. Se None, usa a figura atual.
        formato (str, optional): Formato do arquivo (ex: 'png', 'jpg').
        dpi (int, optional): Resolução da imagem.

    Returns:
        str: Caminho completo do arquivo salvo.
    """
    if figura is None:
        figura = plt.gcf()
    
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    
    caminho_completo = os.path.join(pasta_destino, f"{nome_arquivo}.{formato}")
    figura.savefig(caminho_completo, format=formato, dpi=dpi, bbox_inches='tight')
    print(f"Gráfico salvo em: {caminho_completo}")

    return None


def padronizar_coluna_data(df, coluna):
    """
    Padroniza os valores de uma coluna de data para o formato 'YYYY-MM-DD'.

    Suporta formatos como 'YYYY-WW' (semana do ano) e 'YYYY-MM-DD'.

    Args:
        df (pandas.DataFrame): DataFrame de entrada.
        coluna (str): Nome da coluna contendo datas.

    Returns:
        pandas.DataFrame: DataFrame com a coluna convertida para datetime.
    """

    def parse_data(valor):
        if isinstance(valor, str) and '-' in valor:
            partes = valor.split('-')
            if len(partes) == 2 and partes[1].isdigit() and len(partes[1]) == 2:
                ano = int(partes[0])
                semana = int(partes[1])
                return pd.to_datetime(f'{ano}-W{semana}-1', format='%Y-W%W-%w')
            else:
                try:
                    return pd.to_datetime(valor)
                except:
                    return pd.NaT
        return pd.NaT

    df[coluna] = df[coluna].apply(parse_data)
    df[coluna] = df[coluna].dt.strftime('%Y-%m-%d')
    df[coluna] = pd.to_datetime(df[coluna], errors='coerce')
    return df


def statistics(data):
    """
    Calcula estatísticas descritivas para variáveis numéricas de um DataFrame.

    Inclui medidas de tendência central, dispersão, forma e presença de nulos.

    Args:
        data (pandas.DataFrame): DataFrame com colunas numéricas.

    Returns:
        pandas.DataFrame: Tabela com estatísticas para cada variável numérica.
    """
    num_data = data.select_dtypes(include=['int', 'float'])

    mean = num_data.apply(np.mean)
    q50 = num_data.quantile(0.5)
    q25 = num_data.quantile(0.25)
    q75 = num_data.quantile(0.75)
    range_ = num_data.apply(lambda x: x.max() - x.min())
    count = num_data.count()

    min_ = num_data.apply(min)
    max_ = num_data.apply(max)
    std = num_data.apply(np.std)
    skew = num_data.apply(lambda x: x.skew())
    kurtosis = num_data.apply(lambda x: x.kurtosis())

    metrics = pd.DataFrame({
        'count': count,
        'null_values': num_data.shape[0] - count,
        'type': num_data.dtypes,
        'range': range_,
        'min': min_,
        'quant25': q25,
        'median': q50,
        'quant75': q75,
        'max': max_,
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis
    })

    return np.round(metrics, 1)


## Plots

def plot_matrix(data, columns_features, n_rows, n_cols, plot, sort_by= None, plot_kwargs = {}, loop_feature = None, figure = None, figsize = (15,15), label = True, save_image = False, nome_imagem='imagem', formato='png', dpi=700, pasta_destino='images/'):

    """
    Cria uma matriz de gráficos para comparar visualmente múltiplas variáveis.

    Args:
        data (pd.DataFrame): Conjunto de dados de entrada.
        columns_features (list): Lista de colunas a serem plotadas.
        n_rows (int): Número de linhas da grade de plots.
        n_cols (int): Número de colunas da grade de plots.
        plot (callable): Função de plotagem (ex: sns.barplot, sns.lineplot).
        sort_by (str, optional): Coluna usada para ordenar os dados.
        plot_kwargs (dict, optional): Argumentos adicionais passados para a função de plot.
        loop_feature (str, optional): Parâmetro da função de plotagem que será atualizado com o nome da feature.
        figure (matplotlib.figure.Figure, optional): Figura existente para reutilização.
        figsize (tuple, optional): Tamanho da figura (largura, altura).
        label (bool, optional): Se True, adiciona rótulos aos gráficos.
        save_image (bool, optional): Se True, salva o gráfico gerado como imagem.
        nome_imagem (str, optional): Nome do arquivo de imagem a ser salvo.
        formato (str, optional): Formato do arquivo (ex: 'png').
        dpi (int, optional): Resolução da imagem salva.
        pasta_destino (str, optional): Caminho para salvar o gráfico.

    Returns:
        None
    """

    grid = gridspec.GridSpec(n_rows, n_cols)

    if figure:
        figure
    else:
        plt.figure(figsize=figsize)

    for r in range(0, n_rows):
        for c in range(0, n_cols ):
            if (c + r*n_cols) >= len(columns_features):
                break
            else:
                feature = columns_features[ (c + r*n_cols) ]

                if sort_by:
                    data = data.sort_values(f'{sort_by}',ascending = False)
                else:
                    data = data.sort_values(f'{feature}',ascending = False)

                if loop_feature:
                    plot_kwargs[loop_feature] = feature
                    
                plt.subplot(grid[r, c])
                plt.title(f'{feature}')
                g = plot(data = data, **plot_kwargs)

                plt.xticks(rotation = 30)

                if label:
                    if plot.__name__ == 'lineplot':
                        if loop_feature == 'y':
                            y_col = feature
                            x_col = plot_kwargs['x']
                        else:
                            y_col = plot_kwargs['y']
                            x_col = feature

                        for i, row in data.iterrows():
                            g.annotate(str(np.round(row[y_col], 1)), (row[x_col], row[y_col]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
                    else:
                        for i in g.containers:
                            g.bar_label(i, color = 'black',label_type='edge')
                else:
                    pass

    plt.tight_layout()

    if save_image:
        salvar_grafico(nome_imagem, pasta_destino, figura=plt.gcf(), formato=formato, dpi=dpi)

def analise_multivariada(df, atributos_numericos, nome_arquivo='analise_multivariada', metodo='pearson', dpi=700):
    """
    Gera um mapa de calor de correlação entre variáveis numéricas e salva a imagem.

    Args:
        df (pd.DataFrame): Conjunto de dados de entrada.
        atributos_numericos (list): Lista de colunas numéricas a serem avaliadas.
        nome_arquivo (str, optional): Nome do arquivo da imagem gerada.
        metodo (str, optional): Método de correlação ('pearson', 'kendall', 'spearman').
        dpi (int, optional): Resolução da imagem.

    Returns:
        None
    """
    # Selecionar colunas numéricas + risco
    colunas = atributos_numericos
    num_attributes = df[colunas]

    # Calcular correlação
    correlations = num_attributes.corr(method=metodo)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.title('Mapa de Correlações', fontsize=15)
    sns.heatmap(correlations, annot=True, fmt='.2f')

    # Salvar figura
    salvar_grafico(nome_arquivo, '../images/', figura=plt.gcf(), formato='png', dpi=dpi)

## Data Preparation

def split_dataset(data, test_size, target):
    """
    Divide o dataset em conjuntos de treino e teste, mantendo X e y alinhados.

    Args:
        data (pd.DataFrame): DataFrame original com as features e o target.
        test_size (float): Proporção do conjunto de teste.
        target (str): Nome da coluna alvo.

    Returns:
        tuple: Dois DataFrames, um com os dados de treino e outro com os dados de teste,
               contendo ambos features e target lado a lado.
    """
    dados = data.copy()

    A1, B1, A2, B2 = ms.train_test_split(dados.drop(target, axis=1), dados[target], test_size=test_size, random_state=0)
    
    A = pd.concat([A1,A2], axis=1)
    B = pd.concat([B1,B2], axis=1)

    return A, B

def target_encoding(df, column, train = True, final=False) :
    """
    Aplica codificação target encoding em uma variável categórica com base na média da variável 'risco'.

    Args:
        df (pd.DataFrame): Conjunto de dados original.
        column (str): Nome da coluna categórica a ser codificada.
        train (bool): Define se está em modo de treinamento ou aplicação do encoding.
        final (bool): Define se o encoder deve ser salvo como definitivo (True) ou temporário (False).

    Returns:
        pd.DataFrame: DataFrame com a coluna codificada via target encoding.
    """
    data = df.copy()

    if train:
        media_por_categoria = data[[column,'risco']].groupby(column).agg(total=('risco','mean')).reset_index()
        media_por_categoria['total'] = round(media_por_categoria['total'],2)
        media_por_categoria = media_por_categoria.set_index(column)['total'].to_dict()

        data[column] = data[column].map(media_por_categoria)

        if final:
            save_picked_file(media_por_categoria, f'../params/preparation/target_encoding_{column}')
        else:
            save_picked_file(media_por_categoria, f'../params/preparation/tmp/target_encoding_{column}')

    else:
        
        if final:
            media_por_categoria = load_picked_file(f'../params/preparation/target_encoding_{column}')
        else:
            media_por_categoria = load_picked_file(f'../params/preparation/tmp/target_encoding_{column}')

        data[column] = data[column].map(media_por_categoria)
        data.loc[data[column].isna(), column] = data[column].mean()

    return data

def frequency_encoding(df, column, train=True, final=False):
    """
    Aplica codificação por frequência em uma coluna categórica.

    A codificação por frequência substitui cada categoria pelo valor proporcional de sua ocorrência no dataset.
    Durante o treinamento, as frequências são calculadas e salvas para uso posterior. Em fase de inferência, 
    as categorias desconhecidas são substituídas pela média das frequências.

    Args:
        df (pd.DataFrame): Conjunto de dados contendo a coluna a ser codificada.
        column (str): Nome da coluna categórica.
        train (bool): Indica se a função está sendo executada em modo de treinamento.
        final (bool): Define se os parâmetros devem ser salvos como definitivos ou temporários.

    Returns:
        pd.DataFrame: DataFrame com a coluna transformada via frequency encoding.
    """

    data = df.copy()

    if train:
        # Calcula a frequência relativa da categoria
        freq_map = data[column].value_counts(normalize=True).round(4).to_dict()
        
        # Aplica o encoding
        data[column] = data[column].map(freq_map)

        # Salva o dicionário para uso posterior
        if final:
            save_picked_file(freq_map, f'../params/preparation/freq_encoding_{column}')
        else:
            save_picked_file(freq_map, f'../params/preparation/tmp/freq_encoding_{column}')

    else:

        # Carrega o dicionário salvo
        if final:
            freq_map = load_picked_file(f'../params/preparation/freq_encoding_{column}')
        else:
            freq_map = load_picked_file(f'../params/preparation/tmp/freq_encoding_{column}')
        
        # Aplica o encoding
        data[column] = data[column].map(freq_map)

        # Trata categorias novas com a média das frequências conhecidas
        data.loc[data[column].isna(), column] = np.mean(list(freq_map.values()))

    return data


def cyclical_encoding(df, column, final = False):
    """
    Aplica codificação cíclica a colunas com dados periódicos (como mês ou ano).

    Transforma a variável em duas novas colunas usando funções seno e cosseno para preservar a natureza circular dos dados.

    Args:
        df (pd.DataFrame): Conjunto de dados original.
        column (str): Nome da coluna periódica (ex: 'mes_ciclo_previsto' ou 'ano_ciclo_previsto').
        final (bool): Parâmetro reservado para consistência com outras funções (não utilizado nesta).

    Returns:
        pd.DataFrame: DataFrame com duas novas colunas adicionadas: {coluna}_sin e {coluna}_cos.
    """

    data = df.copy()

    match column:
        case 'ano_ciclo_previsto':
            max_value = df[column].max()
        case 'mes_ciclo_previsto':
            max_value = 12     

    data[f'{column}_sin'] = data[f'{column}'].apply(lambda x: np.sin(x*(2*np.pi/max_value)))
    data[f'{column}_cos'] = data[f'{column}'].apply(lambda x: np.cos(x*(2*np.pi/max_value)))

    return data

def preparacao_dos_dados(df, dict_preparation, is_train=True, final = False):
    """
    Aplica múltiplos tipos de codificação em colunas de um DataFrame de acordo com um dicionário de preparação.

    Esta função é responsável por iterar sobre o dicionário de preparação fornecido, onde cada chave é uma
    coluna e o valor associado é o tipo de codificação desejada (target, frequency ou cyclical).

    Args:
        df (pd.DataFrame): DataFrame original com as colunas a serem transformadas.
        dict_preparation (dict): Dicionário com os nomes das colunas e os tipos de codificação a aplicar.
        is_train (bool): Define se o processo está em modo de treinamento ou aplicação.
        final (bool): Define se os parâmetros devem ser salvos/carregados como definitivos ou temporários.

    Returns:
        pd.DataFrame: DataFrame com todas as transformações aplicadas conforme especificado.
    """
    data = df.copy()

    for column, preparation in dict_preparation.items():
        try:
            match preparation:
                case 'target_encoding':
                    data = target_encoding(data, column, train=is_train, final=final)
                case 'frequency_encoding':
                    data = frequency_encoding(data, column, train=is_train, final=final)
                case 'cyclical_encoding':
                    data = cyclical_encoding(data, column, final=final)  
        except:
            pass

    return data      

## Machine Learning

def calcular_importancia_variaveis(model, dados, dict_preparation, columns_to_drop, target_col):
    """
    Calcula a importância das variáveis de um modelo após a preparação dos dados.

    Args:
        model (sklearn.BaseEstimator): Modelo de aprendizado de máquina que possui o atributo `feature_importances_`.
        dados (pd.DataFrame): DataFrame contendo os dados brutos.
        dict_preparation (dict): Dicionário com os tipos de codificação para cada coluna.
        columns_to_drop (list): Lista de colunas a serem removidas antes do treino.
        target_col (str): Nome da coluna alvo.

    Returns:
        pd.DataFrame: DataFrame com as features e suas respectivas importâncias ordenadas.
    """

    # Preparar dados
    dados_preparados = preparacao_dos_dados(dados, dict_preparation, is_train=True)

    X = dados_preparados.drop(columns=columns_to_drop, axis=1)
    y = dados_preparados[target_col]

    # Treinar modelo
    model.fit(X, y)

    # Calcular importância
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = np.asarray(X.columns.tolist())

    # Criar DataFrame
    features_importances = pd.DataFrame({
        'feature': feature_names[indices],
        'importance': np.round(importances[indices], 4)
    })

    return features_importances

def seleciona_features(features_importances, treadshot):
    """
    Seleciona variáveis com importância acima de um determinado limiar.

    Args:
        features_importances (pd.DataFrame): DataFrame com as colunas 'feature' e 'importance'.
        treadshot (float): Limiar mínimo de importância para que a feature seja selecionada.

    Returns:
        list: Lista de nomes das features selecionadas.
    """

    features_selected = features_importances.loc[features_importances['importance'] >= treadshot, 'feature'].tolist()

    return features_selected

def prepare_fit_and_predict(modelo, dados_treino, dados_teste, dict_preparation, features_selected, TARGET):
    """
    Prepara os dados, treina o modelo e realiza a predição.

    Args:
        modelo (sklearn.BaseEstimator): Modelo a ser treinado.
        dados_treino (pd.DataFrame): DataFrame com dados de treino.
        dados_teste (pd.DataFrame): DataFrame com dados de teste.
        dict_preparation (dict): Dicionário com os tipos de codificação para cada coluna.
        features_selected (list): Lista de features selecionadas para o treino.
        TARGET (str): Nome da coluna alvo.

    Returns:
        tuple: y_real (valores reais), y_hat (preditos), modelo treinado.
    """

    ## preparação dos dados de treino
    dados_treino_transformed = preparacao_dos_dados(dados_treino, dict_preparation, is_train=True)
    X_treino = dados_treino_transformed[features_selected].copy()
    y_treino = dados_treino_transformed[TARGET].copy()

    ## preparacao dos dados de teste
    dados_teste_transformed = preparacao_dos_dados(dados_teste, dict_preparation, is_train=False)
    X_teste = dados_teste_transformed[features_selected].copy()
    y_real = dados_teste_transformed[TARGET].copy()

    ## ajuste do modelo
    modelo.fit(X_treino, y_treino)

    ## predicao do modelo
    y_hat = modelo.predict(X_teste)

    return y_real, y_hat, modelo

def cross_validation(k_folds, modelo, nome_modelo, dados, dict_preparation, features_selected, TARGET):
    """
    Realiza validação cruzada com k-folds, calcula métricas médias e treina o modelo final em todos os dados.

    Args:
        k_folds (sklearn.model_selection.StratifiedKFold): Objeto k-fold configurado.
        modelo (sklearn.BaseEstimator): Modelo a ser validado.
        nome_modelo (str): Nome do modelo para fins de identificação.
        dados (pd.DataFrame): Dataset completo com features e target.
        dict_preparation (dict): Dicionário com os tipos de codificação a aplicar.
        features_selected (list): Lista de features a serem utilizadas no modelo.
        TARGET (str): Nome da coluna alvo.

    Returns:
        pd.DataFrame: DataFrame contendo as métricas médias da validação cruzada.
    """

    # armazena métricas dos modelos
    df_results = pd.DataFrame()

    ## armazena métricas dos modelos para cada fold
    metricas_folds = []

    ## para cada fold
    for fold, (train_idx, val_idx) in enumerate(k_folds.split(dados, dados[TARGET])):

        ## seleciona segmento de dados
        dados_treino = dados.iloc[train_idx]
        dados_validacao = dados.iloc[val_idx]
        
        ## preparação, ajuste e predição
        y_real, y_hat, _ = prepare_fit_and_predict(modelo, dados_treino, dados_validacao, dict_preparation, features_selected, TARGET)

        ## avalia modelos
        df_aux = classification_metrics(nome_modelo, y_real, y_hat)
        df_aux['fold'] = fold + 1

        ## armazena resultados
        metricas_folds.append(df_aux)

    # calcula média de cada fold
    df_modelo = pd.concat(metricas_folds, axis=0)
    df_modelo_mean = df_modelo.drop(columns=['fold']).groupby("Model Name").mean(numeric_only=True).reset_index()
    df_results = pd.concat([df_results, df_modelo_mean], axis=0)

    # treina o modelo final em todos os dados
    dados_transformed = preparacao_dos_dados(dados, dict_preparation, True)

    X_treino = dados_transformed[features_selected]
    y_treino = dados_transformed[TARGET]
    modelo.fit(X_treino, y_treino)
    
    ## salvando modelo
    save_picked_file(file=modelo, name=f'../models/tmp/modelo_{nome_modelo}_cv')

    return df_results

## Avalicação

def classification_metrics(model_name, y_true, y_pred):
    accuracy = mt.accuracy_score(y_true, y_pred)
    precision = mt.precision_score(y_true, y_pred, zero_division=0)
    recall = mt.recall_score(y_true, y_pred)
    f1 = mt.f1_score(y_true, y_pred)

    return pd.DataFrame({
        'Model Name': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })

