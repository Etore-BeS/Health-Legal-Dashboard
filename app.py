import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

# Função para carregar e preprocessar os dados
@st.cache_data
def load_data():
    unimed_df = pd.read_csv('unimed.csv', parse_dates=['data_publicacao'])
    unimed_df['ano'] = unimed_df['data_publicacao'].dt.year
    unimed_df['mes'] = unimed_df['data_publicacao'].dt.month_name()
    unimed_df['Year'] = unimed_df['data_publicacao'].dt.year  # For consistency in plots

    pmc_df = pd.read_csv('cmed_10anos_top30.csv')
    pmc_df['mes'] = pmc_df['mes'].astype(str).str.zfill(2)  # Ensure month is two digits
    pmc_df['data'] = pd.to_datetime(pmc_df['ano'].astype(str) + pmc_df['mes'], format='%Y%m')
    
    # Convert 'pf_0' and 'pmc_0' to numeric
    pmc_df['pf_0'] = pd.to_numeric(pmc_df['pf_0'], errors='coerce')
    pmc_df['pmc_0'] = pd.to_numeric(pmc_df['pmc_0'], errors='coerce')    
    return unimed_df, pmc_df

# --- Data Loading and Caching ---
@st.cache_data
def load_natjus_data(file_path):
    """Loads and parses the JSONL data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.json_normalize(data)
    # Convert date column to datetime
    df['par_data_emissao'] = pd.to_datetime(df['par_data_emissao'])
    return df

# Helper function for styled metric-like display
def styled_metric(column, label, value, font_size="1.2rem", label_size="0.8rem"):
    column.markdown(
        f"""
        <div style="text-align: center; padding: 5px; border: 1px solid #e0e0e0; border-radius: 5px; margin-bottom: 10px;">
            <div style="font-size: {label_size}; color: #666;">{label}</div>
            <div style="font-size: {font_size}; font-weight: bold;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

unimed_df, pmc_df = load_data()

# Navegação
st.sidebar.title('Navegação')
panel = st.sidebar.radio('Ir para', ['Análise Geral', 'Análise Financeira', 'Relatórios', 'Análise Natjus'])


if panel == 'Análise Geral':
    st.title('Painel de Casos Legais de Saúde')
    st.header('Análise Geral')

    # Sidebar Filtros
    st.sidebar.title('Filtros')

    # Filtro de Intervalo de Datas
    min_date = unimed_df['data_publicacao'].min().date()
    max_date = unimed_df['data_publicacao'].max().date()
    date_range = st.sidebar.date_input('Intervalo de Datas', [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filtro de 'Assunto Agravo'
    assunto_options = unimed_df['assunto_agravo'].unique()
    assunto_selected = st.sidebar.multiselect('Assunto Agravo', options=assunto_options, default=assunto_options)

    # Filtro de 'Descritor'
    descritor_options = unimed_df['descritor'].unique()
    descritor_selected = st.sidebar.multiselect('Descritor', options=descritor_options, default=descritor_options)

    # Aplicar Filtros
    df_filtered = unimed_df[
        (unimed_df['data_publicacao'].dt.date >= date_range[0]) &
        (unimed_df['data_publicacao'].dt.date <= date_range[1]) &
        (unimed_df['assunto_agravo'].isin(assunto_selected))
    ].copy()
    
    
    st.markdown('---')
    col1, col2, = st.columns(2)


    with col1:
        # 1. Principais Causas de Ação Legal (descritor)
        st.subheader('1. Principais Causas de Ação Legal')

        # Calcular as 10 principais causas
        top_causes = df_filtered['descritor'].value_counts().reset_index()
        top_causes.columns = ['Descritor', 'Contagem']

        # Função para truncar rótulos
        def truncate_label(label, max_length=30):
            return (label[:max_length] + '...') if len(label) > max_length else label

        top_causes['Descritor_Truncated'] = top_causes['Descritor'].apply(truncate_label)

        fig1 = px.bar(
            top_causes.head(10),
            x='Descritor_Truncated',
            y='Contagem',
            hover_data=['Descritor'],  # Mostrar rótulo completo no hover
            title='Top 10 Causas de Ação Legal'
        )

        fig1.update_layout(
            xaxis_title='Descritor',
            yaxis_title='Contagem',
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig1, use_container_width=True)


        # 2. Distribuição de Resultados (ação)
        st.subheader('2. Distribuição de Resultados')
        outcome_distribution = df_filtered['acao'].value_counts().reset_index()
        outcome_distribution.columns = ['Resultado', 'Contagem']
        fig2 = px.pie(outcome_distribution, names='Resultado', values='Contagem', title='Distribuição de Resultados')
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Tendências Temporais (Número de Processos ao Longo do Tempo)
        st.subheader('3. Tendências Temporais')
        temporal_trends = df_filtered.groupby('Year').size().reset_index(name='Número de Casos')
        fig3 = px.line(temporal_trends, x='Year', y='Número de Casos', markers=True, title='Número de Casos ao Longo dos Anos')
        st.plotly_chart(fig3, use_container_width=True)

        # 4. Análise de Tutela
        st.subheader('4. Análise de Tutela')
        tutela_counts = df_filtered['tutela'].value_counts().reset_index()
        tutela_counts.columns = ['Tutela Concedida', 'Contagem']
        fig4 = px.bar(tutela_counts, x='Tutela Concedida', y='Contagem', title='Tutela Concedida vs. Não Concedida')
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Correlação Entre Duração do Caso e Resultado
        st.subheader('5. Correlação Entre Duração do Caso e Resultado')
        fig5 = px.box(df_filtered, x='acao', y='tempo_processo_mes', points='all',
                    labels={'acao': 'Resultado', 'tempo_processo_mes': 'Duração do Processo (Meses)'},
                    title='Duração do Processo por Resultado')
        st.plotly_chart(fig5, use_container_width=True)

        # 6. Análise Monetária
        st.subheader('6. Análise Monetária')

        # Identificar outliers usando o método IQR
        Q1 = df_filtered['valor'].quantile(0.25)
        Q3 = df_filtered['valor'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Identificar outliers
        outliers_valor = df_filtered[df_filtered['valor'] > upper_bound]

        # Contagem de outliers
        outlier_count_valor = outliers_valor.shape[0]

        # Excluir outliers do gráfico
        valor_filtered = df_filtered['valor'].copy()
        valor_filtered = valor_filtered[valor_filtered <= upper_bound]

        # Plotar histograma incluindo 'valor' == 0 mas excluindo outliers
        fig6 = px.histogram(valor_filtered, x='valor', nbins=50, title='Distribuição dos Valores dos Casos (Excluindo Outliers)',
                        labels={'valor': 'Valor do Caso (R$)'}, 
                        range_x=(0, upper_bound))
        st.plotly_chart(fig6, use_container_width=True)

        # Exibir contagem de outliers
        st.markdown(f"**Nota:** {outlier_count_valor} outlier(s) foram excluídos do gráfico para garantir uma visualização clara.")

        # 7. Advogados Mais Ativos
        st.subheader('7. Advogados Mais Ativos')
        # Dividir e explodir a coluna 'adv'
        df_attorneys = df_filtered.copy()
        df_attorneys['adv'] = df_attorneys['adv'].fillna('')
        df_attorneys['attorneys_list'] = df_attorneys['adv'].str.split(';')
        df_attorneys_exploded = df_attorneys.explode('attorneys_list')
        attorney_counts = df_attorneys_exploded['attorneys_list'].str.strip().value_counts().reset_index()
        attorney_counts.columns = ['Advogado', 'Número de Casos']
        fig7 = px.bar(attorney_counts.head(10), x='Advogado', y='Número de Casos', title='Top 10 Advogados')
        st.plotly_chart(fig7, use_container_width=True)

        # 9. Análise de Varas
        st.subheader('9. Análise de Varas')
        court_counts = df_filtered['vara_nome'].value_counts().reset_index()
        court_counts.columns = ['Vara', 'Número de Casos']
        fig9 = px.bar(court_counts.head(10), x='Vara', y='Número de Casos', title='Top 10 Varas por Número de Casos')
        st.plotly_chart(fig9, use_container_width=True)


    with col2:    
        # 8. Tendências Mensais e Anuais
        st.subheader('8. Tendências Mensais e Anuais')
        monthly_trends = df_filtered.groupby(df_filtered['data_publicacao'].dt.to_period('M')).size().reset_index(name='Número de Casos')
        monthly_trends['Mês'] = monthly_trends['data_publicacao'].astype(str)
        fig10 = px.line(monthly_trends, x='Mês', y='Número de Casos', markers=True, title='Número de Casos por Mês')
        st.plotly_chart(fig10, use_container_width=True)

        # 9. Indicadores de Complexidade dos Casos
        st.subheader('9. Indicadores de Complexidade dos Casos')

        # Excluir casos com tempo_processo_mes = 0
        complexity_zero_count = df_filtered[df_filtered['tempo_processo_mes'] == 0].shape[0]
        df_complexity = df_filtered[df_filtered['tempo_processo_mes'] > 0]

        # Calcular duração média por assunto_agravo
        complexity = df_complexity.groupby('assunto_agravo')['tempo_processo_mes'].mean().reset_index()
        complexity.columns = ['Assunto Agravo', 'Duração Média (Meses)']

        # Ordenar por duração média e selecionar os 20 principais para melhor visualização
        complexity = complexity.sort_values(by='Duração Média (Meses)', ascending=False).head(20)

        # Função para abreviar nomes de 'Assunto Agravo'
        def abbreviate_text(text, max_length=20):
            return text if len(text) <= max_length else text[:17] + '...'

        complexity['Assunto Agravo Abreviado'] = complexity['Assunto Agravo'].apply(lambda x: abbreviate_text(x, 20))

        # Plotar gráfico de barras com rótulos abreviados
        fig9 = px.bar(
            complexity,
            x='Assunto Agravo Abreviado',
            y='Duração Média (Meses)',
            title='Duração Média dos Casos por Assunto Agravo (Top 20)',
            labels={'Assunto Agravo Abreviado': 'Assunto Agravo', 'Duração Média (Meses)': 'Duração Média (Meses)'},
            hover_data=['Assunto Agravo', 'Duração Média (Meses)']
        )

        fig9.update_layout(
            xaxis_tickangle=-45,
            yaxis_title='Duração Média (Meses)',
            xaxis_title='Assunto Agravo'
        )

        st.plotly_chart(fig9, use_container_width=True)

        # Exibir a contagem de tempo_processo_mes = 0
        st.markdown(f"**Número de casos com tempo_processo_mes = 0:** {complexity_zero_count}")


        # 10. Taxa de Sucesso das Tutelas
        st.subheader('10. Taxa de Sucesso das Tutelas')
        injunctions = df_filtered.groupby(['tutela', 'acao']).size().reset_index(name='Número de Casos')
        fig10 = px.bar(injunctions, x='acao', y='Número de Casos', color='tutela', barmode='group', title='Taxa de Sucesso das Tutelas por Resultado')
        st.plotly_chart(fig10, use_container_width=True)

        # 11. Análise de Honorários Legais
        st.subheader('11. Análise de Honorários Legais')

        # Remover casos com valor = 0 e multa = 0 para análise significativa
        df_fees = df_filtered[(df_filtered['valor'] > 0) & (df_filtered['multa'] > 0)]

        # Identificar outliers usando o método IQR para 'valor'
        Q1_valor = df_fees['valor'].quantile(0.25)
        Q3_valor = df_fees['valor'].quantile(0.75)
        IQR_valor = Q3_valor - Q1_valor
        upper_bound_valor = Q3_valor + 1.5 * IQR_valor

        # Identificar outliers em 'valor'
        outliers_valor = df_fees[df_fees['valor'] > upper_bound_valor]

        # Filtrar outliers para o gráfico
        df_fees_plot = df_fees[df_fees['valor'] <= upper_bound_valor]

        fig11 = px.scatter(
            df_fees_plot,
            x='valor',
            y='multa',
            trendline='ols',
            labels={'valor': 'Valor do Caso (R$)', 'multa': 'Multa (R$)'},
            title='Valor da Multa vs. Valor do Caso (Excluindo Outliers)',
            hover_data=['processo', 'vara_nome', 'adv']
        )

        # Adicionar detalhes da linha de tendência
        fig11.update_layout(
            xaxis_title='Valor do Caso (R$)',
            yaxis_title='Multa (R$)',
            xaxis=dict(range=[0, upper_bound_valor * 1.05]),  # Adicionar alguma margem
            yaxis=dict(range=[0, df_fees_plot['multa'].max() * 1.05])
        )

        st.plotly_chart(fig11, use_container_width=True)

        # Exibir informações sobre o outlier
        if not outliers_valor.empty:
            st.markdown(f"**Outliers Excluídos do Gráfico:** {outliers_valor.shape[0]} casos com valor > {upper_bound_valor:.2f} R$")
            st.markdown("Esses outliers são excluídos para garantir que o gráfico permaneça legível. Eles ainda são considerados em outras análises.")


        # 12. Distribuição de Assunto Agravo ao Longo do Tempo
        st.subheader('12. Distribuição de Assunto Agravo ao Longo do Tempo')

        # Agregar dados
        assunto_time = df_filtered.groupby(['Year', 'assunto_agravo']).size().reset_index(name='Número de Casos')

        # Truncar rótulos de 'assunto_agravo' para exibição
        def truncate_label(label, max_length=20):
            return (label[:max_length] + '...') if len(label) > max_length else label

        assunto_time['Assunto_Agravo_Truncated'] = assunto_time['assunto_agravo'].apply(truncate_label)

        fig12 = px.line(
            assunto_time,
            x='Year',
            y='Número de Casos',
            color='Assunto_Agravo_Truncated',
            hover_data=['assunto_agravo'],
            title='Assunto Agravo ao Longo do Tempo',
            labels={'Assunto_Agravo_Truncated': 'Assunto Agravo'}
        )

        # Mover legenda para fora do gráfico
        fig12.update_layout(
            legend=dict(
                title='Assunto Agravo',
                orientation="h",
                yanchor="bottom",
                y=1.6,
                xanchor="right",
                x=1
            ),
            xaxis_title='Ano',
            yaxis_title='Número de Casos'
        )

        st.plotly_chart(fig12, use_container_width=True)

        # 13. Duração do Caso por Vara
        st.subheader('13. Duração do Caso por Vara')
        duration_court = df_filtered.groupby('vara_nome')['tempo_processo_mes'].mean().reset_index()
        duration_court.columns = ['Vara', 'Duração Média (Meses)']
        fig17 = px.bar(duration_court.sort_values(by='Duração Média (Meses)', ascending=False).head(10),
                    x='Vara', y='Duração Média (Meses)', title='Top 10 Varas por Duração Média do Caso')
        st.plotly_chart(fig17, use_container_width=True)

        # 14. Análise Comparativa Entre Anos com Exclusão de Valores Extremos
        st.subheader('14. Análise Comparativa Entre Anos')
        
        # Passo 1: Identificar Outliers na Coluna 'valor'
        Q1_valor = df_filtered['valor'].quantile(0.25)
        Q3_valor = df_filtered['valor'].quantile(0.75)
        IQR_valor = Q3_valor - Q1_valor
        lower_bound_valor = Q1_valor - 1.5 * IQR_valor
        upper_bound_valor = Q3_valor + 1.5 * IQR_valor
        
        # Identificar outliers
        outliers = df_filtered[(df_filtered['valor'] < lower_bound_valor) | (df_filtered['valor'] > upper_bound_valor)]
        outlier_count = outliers.shape[0]
        
        # Excluir outliers
        df_no_outliers = df_filtered[(df_filtered['valor'] >= lower_bound_valor) & (df_filtered['valor'] <= upper_bound_valor)]
        
        # Passo 2: Recalcular as Métricas por Ano sem Outliers
        yearly_comparison = df_no_outliers.groupby('Year').agg({
            'valor': 'mean',
            'tempo_processo_mes': 'mean',
            'processo': 'count'
        }).reset_index()
        yearly_comparison.columns = ['Ano', 'Valor Médio do Caso (R$)', 'Duração Média (Meses)', 'Número de Casos']
        
        # Passo 3: Visualizar os Dados Atualizados
        # Criar figura com eixo y secundário
        fig14 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Adicionar barra para 'Número de Casos'
        fig14.add_trace(
            go.Bar(
                x=yearly_comparison['Ano'],
                y=yearly_comparison['Número de Casos'],
                name='Número de Casos',
                marker_color='rgba(55, 128, 191, 0.7)'  # Azul
            ),
            secondary_y=False,
        )
        
        # Adicionar linha para 'Duração Média (Meses)'
        fig14.add_trace(
            go.Scatter(
                x=yearly_comparison['Ano'],
                y=yearly_comparison['Duração Média (Meses)'],
                name='Duração Média (Meses)',
                mode='lines+markers',
                marker=dict(color='firebrick'),
                line=dict(color='firebrick', width=2)
            ),
            secondary_y=True,
        )
        
        # Atualizar layout para melhor estética
        fig14.update_layout(
            title_text='Análise Comparativa Anual',
            legend=dict(x=0.01, y=0.99),
            barmode='group',
            height=600,
            width=1200  # Ajustar conforme necessário
        )
        
        # Definir títulos dos eixos y
        fig14.update_yaxes(title_text="Número de Casos", secondary_y=False)
        fig14.update_yaxes(title_text="Duração Média (Meses)", secondary_y=True)
        
        # Ajustar título do eixo x
        fig14.update_xaxes(title_text="Ano")
        
        # Exibir o plot
        st.plotly_chart(fig14, use_container_width=True)
        
        # Exibir informações sobre os outliers
        st.markdown(f"**Total de outliers excluídos na análise:** {outlier_count}")
        
        if not outliers.empty:
            # Exibir os anos com outliers
            outlier_years = outliers['Year'].unique()
            st.markdown(f"**Outliers Detectados:** Alguns casos em **{', '.join(map(str, outlier_years))}** tiveram valores de **'valor'** extremamente altos que foram excluídos da análise.")
            st.markdown("Os anos permaneceram na análise, mas os casos com valores extremos foram removidos para evitar distorções nas médias.")

if panel == 'Análise Financeira':
    st.title('Análise Financeira de Medicamentos')
    st.header('Análise com Base em Dados Históricos de Preços')

    # Sidebar filters
    st.sidebar.title('Filtros')

    # Read data
    df = pd.read_parquet('df_optimized.parquet')

    # Ensure 'data' is datetime
    df['data'] = pd.to_datetime(df['data'])

    # Select 'farmaco'
    farmacos = df['farmaco'].dropna().unique()
    farmacos = sorted(farmacos)
    farmaco_selected = st.sidebar.selectbox('Selecione o Fármaco', options=farmacos)

    # Filter data for the selected 'farmaco'
    df_farmaco = df[df['farmaco'] == farmaco_selected].copy()

    # Proceed only if data is available
    if df_farmaco.empty:
        st.warning('Nenhum dado disponível para o fármaco selecionado.')
    else:
        # Calculate mean_price
        df_farmaco['mean_price'] = (df_farmaco['menor_preco'] + df_farmaco['maior_preco']) / 2

        # Sort data by date
        df_farmaco = df_farmaco.sort_values('data')

        # Convert date to month-year period
        df_farmaco['Mes_Ano'] = df_farmaco['data'].dt.to_period('M')

        # Handle multiple series for the same 'laboratorio'
        # Use 'apresentacao' as a unique identifier
        overall_mean_price = df_farmaco['mean_price'].mean()

        # Identify laboratories with multiple series
        lab_series_counts = df_farmaco.groupby('laboratorio')['apresentacao'].nunique()
        labs_with_multiple_series = lab_series_counts[lab_series_counts > 1].index.tolist()

        # For each lab with multiple series, select the one with mean price closest to overall mean
        df_selected_series = pd.DataFrame()
        for lab in df_farmaco['laboratorio'].unique():
            df_lab = df_farmaco[df_farmaco['laboratorio'] == lab]
            unique_series = df_lab['apresentacao'].unique()
            if len(unique_series) > 1:
                # Multiple series exist, select the one closest to overall mean price
                mean_prices = []
                for series in unique_series:
                    df_series = df_lab[df_lab['apresentacao'] == series]
                    series_mean_price = df_series['mean_price'].mean()
                    mean_prices.append((series, series_mean_price))
                # Find the series with mean price closest to overall mean
                selected_series = min(mean_prices, key=lambda x: abs(x[1] - overall_mean_price))[0]
                # Select this series data
                df_selected_series = pd.concat([df_selected_series, df_lab[df_lab['apresentacao'] == selected_series]])
            else:
                # Only one series, include it
                df_selected_series = pd.concat([df_selected_series, df_lab])

        # Replace df_farmaco with the selected series data
        df_farmaco = df_selected_series

        # Recalculate overall mean price (optional)
        overall_mean_price = df_farmaco['mean_price'].mean()

        # Create time_idx using the ordinal attribute
        df_farmaco['time_idx'] = df_farmaco['Mes_Ano'].apply(lambda x: x.ordinal) - df_farmaco['Mes_Ano'].min().ordinal

        # Group by Mes_Ano and calculate mean, min, max prices across laboratories
        df_grouped = df_farmaco.groupby('Mes_Ano').agg({
            'menor_preco': 'min',
            'maior_preco': 'max',
            'mean_price': 'mean'
        }).reset_index()

        # Create time_idx for grouped data
        df_grouped['time_idx'] = df_grouped['Mes_Ano'].apply(lambda x: x.ordinal) - df_grouped['Mes_Ano'].min().ordinal

        # Prepare future periods for predictions (next 3 months)
        forecast_horizon = 6  # Changed to 3 months
        last_period = df_farmaco['Mes_Ano'].max()
        future_periods = [last_period + i for i in range(1, forecast_horizon + 1)]
        future_time_idx = [df_farmaco['time_idx'].max() + i for i in range(1, forecast_horizon + 1)]

        from sklearn.ensemble import RandomForestRegressor

        # Function to create lagged features
        def create_lagged_features(df, target_col, lags):
            df_lagged = df.copy()
            for lag in range(1, lags + 1):
                df_lagged[f'lag_{lag}'] = df_lagged[target_col].shift(lag)
            df_lagged = df_lagged.dropna()
            return df_lagged

        n_lags = 3  # Number of lags to use

        # Predictions for laboratories
        predictions_labs = pd.DataFrame()
        laboratorios = df_farmaco['laboratorio'].unique()

        for i, lab in enumerate(laboratorios):
            df_lab = df_farmaco[df_farmaco['laboratorio'] == lab].copy()
            if len(df_lab) < n_lags + forecast_horizon:
                st.warning(f'Dados insuficientes para o laboratório {lab}. Previsão não será realizada.')
                continue
            # Sort by time_idx
            df_lab = df_lab.sort_values('time_idx')
            # Prepare lagged features
            df_lab_lagged = create_lagged_features(df_lab, 'mean_price', n_lags)
            # Features and target
            X_lab = df_lab_lagged[[f'lag_{lag}' for lag in range(1, n_lags + 1)]]
            y_lab = df_lab_lagged['mean_price']
            # Fit model
            model_lab = RandomForestRegressor()
            model_lab.fit(X_lab, y_lab)
            # Prepare for recursive forecasting
            last_known_lags = df_lab['mean_price'].iloc[-n_lags:].values
            # Predict future prices recursively
            future_predictions = []
            for _ in range(forecast_horizon):
                X_input = last_known_lags[-n_lags:].reshape(1, -1)
                y_pred = model_lab.predict(X_input)[0]
                future_predictions.append(y_pred)
                # Update last_known_lags
                last_known_lags = np.append(last_known_lags, y_pred)
            # Prepare DataFrame with predictions
            df_pred = pd.DataFrame({
                'Mes_Ano': future_periods,
                'mean_price': future_predictions,
                'laboratorio': lab,
                'tipo': 'Previsão'
            })
            # Append to predictions DataFrame
            predictions_labs = pd.concat([predictions_labs, df_pred], ignore_index=True)

        # Predictions for max, min, mean prices
        # Prepare data
        df_grouped = df_grouped.sort_values('time_idx')

        # For Mean Price
        df_grouped_lagged = create_lagged_features(df_grouped, 'mean_price', n_lags)
        X_mean = df_grouped_lagged[[f'lag_{lag}' for lag in range(1, n_lags + 1)]]
        y_mean = df_grouped_lagged['mean_price']
        model_mean = RandomForestRegressor()
        model_mean.fit(X_mean, y_mean)
        # Recursive forecasting
        last_known_lags_mean = df_grouped['mean_price'].iloc[-n_lags:].values
        future_predictions_mean = []
        for _ in range(forecast_horizon):
            X_input = last_known_lags_mean[-n_lags:].reshape(1, -1)
            y_pred = model_mean.predict(X_input)[0]
            future_predictions_mean.append(y_pred)
            last_known_lags_mean = np.append(last_known_lags_mean, y_pred)

        # For Min Price
        df_grouped_lagged_min = create_lagged_features(df_grouped, 'menor_preco', n_lags)
        X_min = df_grouped_lagged_min[[f'lag_{lag}' for lag in range(1, n_lags + 1)]]
        y_min = df_grouped_lagged_min['menor_preco']
        model_min = RandomForestRegressor()
        model_min.fit(X_min, y_min)
        # Recursive forecasting
        last_known_lags_min = df_grouped['menor_preco'].iloc[-n_lags:].values
        future_predictions_min = []
        for _ in range(forecast_horizon):
            X_input = last_known_lags_min[-n_lags:].reshape(1, -1)
            y_pred = model_min.predict(X_input)[0]
            future_predictions_min.append(y_pred)
            last_known_lags_min = np.append(last_known_lags_min, y_pred)

        # For Max Price
        df_grouped_lagged_max = create_lagged_features(df_grouped, 'maior_preco', n_lags)
        X_max = df_grouped_lagged_max[[f'lag_{lag}' for lag in range(1, n_lags + 1)]]
        y_max = df_grouped_lagged_max['maior_preco']
        model_max = RandomForestRegressor()
        model_max.fit(X_max, y_max)
        # Recursive forecasting
        last_known_lags_max = df_grouped['maior_preco'].iloc[-n_lags:].values
        future_predictions_max = []
        for _ in range(forecast_horizon):
            X_input = last_known_lags_max[-n_lags:].reshape(1, -1)
            y_pred = model_max.predict(X_input)[0]
            future_predictions_max.append(y_pred)
            last_known_lags_max = np.append(last_known_lags_max, y_pred)

        # Prepare DataFrame with predictions
        df_pred_prices = pd.DataFrame({
            'Mes_Ano': future_periods,
            'menor_preco': future_predictions_min,
            'maior_preco': future_predictions_max,
            'mean_price': future_predictions_mean,
            'tipo': 'Previsão'
        })

        # Combine historical data and predictions
        df_grouped['tipo'] = 'Histórico'
        df_prices = pd.concat([df_grouped[['Mes_Ano', 'menor_preco', 'maior_preco', 'mean_price', 'tipo']],
                               df_pred_prices],
                              ignore_index=True)

        # Prepare data for plotting laboratories
        df_farmaco['tipo'] = 'Histórico'
        df_labs_all = pd.concat([df_farmaco[['Mes_Ano', 'mean_price', 'laboratorio', 'tipo']],
                                 predictions_labs[['Mes_Ano', 'mean_price', 'laboratorio', 'tipo']]],
                                ignore_index=True)

        # Plotting 'Preços' by laboratory
        import plotly.graph_objects as go
        import plotly.express as px

        st.subheader('Evolução dos Preços por Laboratório')

        fig_prices = go.Figure()

        # Get list of labs
        labs = df_labs_all['laboratorio'].unique()

        # Generate a color palette
        colors = px.colors.qualitative.Plotly

        for i, lab in enumerate(labs):
            df_lab = df_labs_all[df_labs_all['laboratorio'] == lab]
            df_hist = df_lab[df_lab['tipo'] == 'Histórico']
            df_pred = df_lab[df_lab['tipo'] == 'Previsão']
            color = colors[i % len(colors)]
            # Plot historical data
            fig_prices.add_trace(go.Scatter(
                x=df_hist['Mes_Ano'].dt.to_timestamp(),
                y=df_hist['mean_price'],
                mode='lines+markers',
                name=f'{lab}',
                line=dict(color=color, dash='solid'),
                legendgroup=lab
            ))
            # Plot predictions
            if not df_pred.empty:
                fig_prices.add_trace(go.Scatter(
                    x=pd.PeriodIndex(df_pred['Mes_Ano']).to_timestamp(),
                    y=df_pred['mean_price'],
                    mode='lines+markers',
                    name=f'{lab}',
                    line=dict(color=color, dash='dash'),
                    legendgroup=lab,
                    showlegend=False
                ))

        # Add legend entries for line styles
        fig_prices.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='black', dash='solid'),
            name='Histórico',
            showlegend=True
        ))
        fig_prices.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Previsão',
            showlegend=True
        ))

        # Adjust layout to place legend below the plot
        fig_prices.update_layout(
            xaxis_title='Data',
            yaxis_title='Preço Médio (R$)',
            title='Evolução dos Preços por Laboratório',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.25,
                xanchor='left',
                x=0,
                title_text='Laboratórios',
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(b=200),
            width=900,
            height=600
        )

        st.plotly_chart(fig_prices, use_container_width=True)

        # Plotting max, min, mean prices
        st.subheader('Evolução do Preço Máximo, Mínimo e Médio')

        fig_stats = go.Figure()

        df_hist = df_prices[df_prices['tipo'] == 'Histórico']
        df_pred = df_prices[df_prices['tipo'] == 'Previsão']

        # Plot 'menor_preco' (Min Price)
        fig_stats.add_trace(go.Scatter(
            x=df_hist['Mes_Ano'].dt.to_timestamp(),
            y=df_hist['menor_preco'],
            mode='lines+markers',
            name='Preço Mínimo',
            line=dict(color='blue', dash='solid'),
            legendgroup='Preço Mínimo'
        ))
        fig_stats.add_trace(go.Scatter(
            x=pd.PeriodIndex(df_pred['Mes_Ano']).to_timestamp(),
            y=df_pred['menor_preco'],
            mode='lines+markers',
            name='Preço Mínimo',
            line=dict(color='blue', dash='dash'),
            legendgroup='Preço Mínimo',
            showlegend=False
        ))

        # Plot 'maior_preco' (Max Price)
        fig_stats.add_trace(go.Scatter(
            x=df_hist['Mes_Ano'].dt.to_timestamp(),
            y=df_hist['maior_preco'],
            mode='lines+markers',
            name='Preço Máximo',
            line=dict(color='red', dash='solid'),
            legendgroup='Preço Máximo'
        ))
        fig_stats.add_trace(go.Scatter(
            x=pd.PeriodIndex(df_pred['Mes_Ano']).to_timestamp(),
            y=df_pred['maior_preco'],
            mode='lines+markers',
            name='Preço Máximo',
            line=dict(color='red', dash='dash'),
            legendgroup='Preço Máximo',
            showlegend=False
        ))

        # Plot 'mean_price' (Mean Price)
        fig_stats.add_trace(go.Scatter(
            x=df_hist['Mes_Ano'].dt.to_timestamp(),
            y=df_hist['mean_price'],
            mode='lines+markers',
            name='Preço Médio',
            line=dict(color='green', dash='solid'),
            legendgroup='Preço Médio'
        ))
        fig_stats.add_trace(go.Scatter(
            x=pd.PeriodIndex(df_pred['Mes_Ano']).to_timestamp(),
            y=df_pred['mean_price'],
            mode='lines+markers',
            name='Preço Médio',
            line=dict(color='green', dash='dash'),
            legendgroup='Preço Médio',
            showlegend=False
        ))

        # Add legend entries for line styles
        fig_stats.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='black', dash='solid'),
            name='Histórico',
            showlegend=True
        ))
        fig_stats.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Previsão',
            showlegend=True
        ))

        # Adjust layout to place legend below the plot
        fig_stats.update_layout(
            xaxis_title='Data',
            yaxis_title='Preço (R$)',
            title='Evolução do Preço Máximo, Mínimo e Médio',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.25,
                xanchor='left',
                x=0,
                title_text='Preços',
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(b=200),
            width=900,
            height=600
        )

        st.plotly_chart(fig_stats, use_container_width=True)

        # Additional Analyses
        st.subheader('Estatísticas Descritivas dos Preços')
        stats_summary = df_farmaco[['menor_preco', 'maior_preco', 'mean_price']].describe()
        st.table(stats_summary)

        # Plot histograms
        st.subheader('Distribuição dos Preços')
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_farmaco['menor_preco'],
            name='Menor Preço',
            opacity=0.5
        ))
        fig_hist.add_trace(go.Histogram(
            x=df_farmaco['maior_preco'],
            name='Maior Preço',
            opacity=0.5
        ))
        fig_hist.add_trace(go.Histogram(
            x=df_farmaco['mean_price'],
            name='Preço Médio',
            opacity=0.5
        ))
        fig_hist.update_layout(
            barmode='overlay',
            xaxis_title='Preço (R$)',
            yaxis_title='Frequência',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.25,
                xanchor='left',
                x=0,
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(b=150),
            width=900,
            height=600
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Plot box plots
        st.subheader('Box Plot dos Preços')
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df_farmaco['menor_preco'],
            name='Menor Preço'
        ))
        fig_box.add_trace(go.Box(
            y=df_farmaco['maior_preco'],
            name='Maior Preço'
        ))
        fig_box.add_trace(go.Box(
            y=df_farmaco['mean_price'],
            name='Preço Médio'
        ))
        fig_box.update_layout(
            yaxis_title='Preço (R$)',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.25,
                xanchor='left',
                x=0,
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(b=150),
            width=900,
            height=600
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Explanation of the Predictor
        st.subheader('Modelo de Previsão Utilizado')
        st.write("""
        **Modelo Utilizado:** `RandomForestRegressor` com características defasadas (lagged features).

        **Motivação da Escolha:**

        - O `RandomForestRegressor` é um modelo de aprendizado de máquina robusto que pode capturar relações não lineares nos dados.
        - Ao usar características defasadas, o modelo considera valores anteriores para prever futuros, capturando tendências temporais.
        - A abordagem de previsão recursiva permite que previsões anteriores influenciem as subsequentes, proporcionando uma previsão mais dinâmica.
        - O `RandomForestRegressor` é menos propenso a overfitting e pode lidar bem com dados ruidosos e complexos.
        - A biblioteca scikit-learn foi utilizada, garantindo a confiabilidade e eficiência computacional do modelo.

        **Como Funciona:**

        - **Características Defasadas:** Para cada ponto de dados, utilizamos os preços médios dos 3 meses anteriores como entradas (lags) para prever o preço futuro.
        - **Previsão Recursiva:** Para prever os próximos 6 meses, usamos as previsões anteriores como entradas para as próximas previsões, permitindo que o modelo capture a dinâmica temporal.
        - **Aplicação aos Laboratórios e Preços Agregados:** Essa abordagem foi aplicada tanto para prever os preços de cada laboratório individualmente quanto para os preços máximos, mínimos e médios agregados.

        **Vantagens:**

        - Captura de padrões complexos nos dados históricos.
        - Flexibilidade para modelar relações não lineares.
        - Robustez contra outliers e variabilidade nos dados.

        **Considerações Finais:**

        - Embora o `RandomForestRegressor` não seja especificamente um modelo de séries temporais, a inclusão de características defasadas permite que ele seja adaptado para previsões temporais.
        - Para aplicações futuras, especialmente com dados mais extensos, pode-se considerar modelos especializados em séries temporais, como ARIMA ou Prophet.
        """)


if panel == 'Relatórios':
    HTML_DIR = 'html_files'  # your folder with HTML files
    html_files = [f for f in os.listdir(HTML_DIR) if f.endswith('.html')]
    selected_file = st.selectbox("Select an HTML file to view", html_files)
    file_path = os.path.join(HTML_DIR, selected_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
    
if panel == 'Análise Natjus':
        # --- Load Data ---
    try:
        df = load_natjus_data('public_natjus_silver_export_2025-10-14_162219.jsonl')
    except FileNotFoundError:
        st.error("The data file 'sample-2025-10-14_62926.json' was not found. Please make sure it's in the same directory as the app.")

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    selected_state = st.sidebar.multiselect(
        "Filter by State",
        options=sorted(df['par_estado'].unique()),
        default=sorted(df['par_estado'].unique())
    )
    selected_conclusion = st.sidebar.multiselect(
        "Filter by Conclusion",
        options=df['par_conclusao'].unique(),
        default=df['par_conclusao'].unique()
    )

    # Apply filters
    filtered_df = df[
        (df['par_estado'].isin(selected_state)) &
        (df['par_conclusao'].isin(selected_conclusion))
    ]

    # --- Dashboard Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Resumo", "Visão de Pacientes", "Visão de Medicamentos", "Relatório NATJUS", "Análise Temporal", "Análise por CID","Amostra de Dados"])

    with tab1:
        # --- Overview Section ---
        st.header("Dashboard Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Cases", value=len(filtered_df))
        with col2:
            st.metric(label="Favorable Conclusions", value=len(filtered_df[filtered_df['par_conclusao'] == 'FAVORAVEL']))
        with col3:
            st.metric(label="Unfavorable Conclusions", value=len(filtered_df[filtered_df['par_conclusao'] == 'DESFAVORAVEL']))

        # --- Charts ---
        col1, col2 = st.columns(2)
        with col1:
            # Conclusion Distribution
            conclusion_counts = filtered_df['par_conclusao'].value_counts()
            fig = px.pie(
                values=conclusion_counts.values,
                names=conclusion_counts.index,
                title="Conclusion Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Cases by State
            state_counts = filtered_df['par_estado'].value_counts()
            fig = px.bar(
                x=state_counts.index,
                y=state_counts.values,
                title="Cases by State",
                labels={'x': 'State', 'y': 'Number of Cases'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # --- Patient Analysis Section ---
        st.header("Patient Analysis")
        # Slider to set maximum allowed age
        max_age_default = min(120, int(np.nanpercentile(filtered_df['pac_idade'], 99)))  # sensible default
        max_age = st.sidebar.slider(
            "Maximum Patient Age to Include",
            min_value=1,
            max_value=130,
            value=max_age_default,
            step=1,
            help="Use this to exclude outliers or incorrect ages."
        )

        # Apply max age filter (keep NaNs if you want to show them elsewhere)
        age_filtered_df = filtered_df[(filtered_df['pac_idade'] >= 1) & (filtered_df['pac_idade'] <= max_age)].copy()

        col1, col2 = st.columns(2)
        with col1:
            # Age Distribution with cap
            age_counts = (
                age_filtered_df
                .assign(pac_idade=lambda d: d['pac_idade'].round().astype(int))
                .groupby('pac_idade', as_index=False)
                .size()
                .rename(columns={'size': 'count'})
                .sort_values('pac_idade')
            )

            fig = px.bar(
                age_counts,
                x='pac_idade',
                y='count',
                title='Patient Age Distribution',
                labels={'pac_idade': 'Age', 'count': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            excluded = len(filtered_df) - len(age_filtered_df)
            if excluded > 0:
                st.caption(f"{excluded} record(s) excluded with pac_idade > {max_age}.")
                
        with col2:
            # Gender Distribution
            gender_counts = filtered_df['pac_sexo'].value_counts()
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Patient Gender Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)


        # Clinical Descriptions Word Cloud
        st.subheader("Most Common Clinical Descriptions")
        text = ' '.join(filtered_df['pac_desc_clinica'].dropna())
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("No clinical descriptions to display for the current selection.")

    with tab3:
        # --- Medication Analysis Section ---
        st.header("Medication Analysis")
        # Top 10 Medications
        medication_counts = filtered_df['med_principio_ativo'].value_counts().nlargest(20)
        fig = px.bar(
            x=medication_counts.index,
            y=medication_counts.values,
            title="Top 20 Most Requested Medications",
            labels={'x': 'Medication', 'y': 'Number of Requests'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Treemap for Medication Distribution ---
        st.subheader("Medication Distribution Treemap")
        # To avoid clutter, let's use the top 30 medications for the treemap
        medication_counts_treemap = filtered_df['med_principio_ativo'].value_counts().nlargest(30).reset_index()
        medication_counts_treemap.columns = ['Medication', 'Count']

        fig_treemap = px.treemap(
            medication_counts_treemap,
            path=['Medication'],
            values='Count',
            title='Proportional Distribution of Top 30 Medications',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with tab4:
        # --- NATJUS Report Section ---
        st.header("NATJUS Report Details")

        if not filtered_df.empty:
            # Dropdown to select a case
            case_options = filtered_df['par_numero'].tolist()
            selected_case = st.selectbox("Select a Case Number to view the report:", options=case_options)

            if selected_case:
                case_details = filtered_df.loc[filtered_df['par_numero'] == selected_case].iloc[0]

                st.subheader(f"Resumo do Caso: {case_details['par_numero']}")

                # Top badges/metrics row
                c1, c2, c3, c4 = st.columns(4)

                # Apply styled_metric to each column
                styled_metric(c1, "Conclusão", case_details['par_conclusao'])
                styled_metric(c2, "Idade", int(case_details['pac_idade']) if pd.notna(case_details['pac_idade']) else "—")
                styled_metric(c3, "Sexo", case_details['pac_sexo'] if pd.notna(case_details['pac_sexo']) else "—")
                styled_metric(c4, "Data", case_details['par_data_emissao'].strftime('%d/%m/%Y'))

                # Three info sections with nice layout cards
                with st.container():
                    st.markdown("#### Parecer")
                    p1, p2, p3, p4 = st.columns(4)
                    styled_metric(p1, "Tribunal", case_details['par_tribunal'])
                    styled_metric(p2, "Esfera", case_details['par_esfera'])
                    styled_metric(p3, "Estado", case_details['par_estado'])
                    styled_metric(p4, "Município", case_details['par_municipio'])

                with st.container():
                    st.markdown("#### Paciente")
                    pa1, pa2, pa3 = st.columns(3)
                    styled_metric(pa1, "Idade", int(case_details['pac_idade']) if pd.notna(case_details['pac_idade']) else '—')
                    styled_metric(pa2, "Sexo", case_details['pac_sexo'] if pd.notna(case_details['pac_sexo']) else '—')
                    styled_metric(pa3, "CID-10", case_details['pac_cid10'] if pd.notna(case_details['pac_cid10']) else '—')
                    
                    with st.expander("Descrição Clínica", expanded=False):
                        st.write(case_details['pac_desc_clinica'] if pd.notna(case_details['pac_desc_clinica']) else "—")

                with st.container():
                    st.markdown("#### Medicamento")
                    med1, med2, med3 = st.columns(3)
                    styled_metric(med1, "Princípio Ativo", case_details['med_principio_ativo'] if pd.notna(case_details['med_principio_ativo']) else '—')
                    styled_metric(med2, "Nome Comercial", case_details['med_nome_comercial'] if pd.notna(case_details['med_nome_comercial']) else '—')
                    styled_metric(med3, "Fabricante", case_details['med_fabricante'] if pd.notna(case_details['med_fabricante']) else '—')

                    med4, med5, med6 = st.columns(3)
                    styled_metric(med4, "Dosagem", case_details['med_dosagem'] if pd.notna(case_details['med_dosagem']) else "Não Informado")
                    styled_metric(med5, "Via Admin.", case_details['med_via_admin'] if pd.notna(case_details['med_via_admin']) else '—')
                    styled_metric(med6, "Registrado na ANVISA", 'Sim' if bool(case_details.get('med_registro_anvisa', False)) else 'Não')

                    with st.expander("Posologia", expanded=False):
                        st.write(case_details['med_posologia'] if pd.notna(case_details['med_posologia']) else '—')

                # Display the justification in an expander
                with st.expander("Leia toda a justificativa", expanded=True):
                    st.write(case_details['par_justificativa'])

                # Display PDF link
                st.markdown(f"**PDF Link:** [Clique para Ver]({case_details['par_pdf_url']})")
        else:
            st.warning("No cases match the current filter selection.")

    with tab5:
        st.header("Análise de Séries Temporais de Medicamentos")

        # Build options for convenience (exact choices), but also offer free-text search
        medication_options = sorted(
            filtered_df['med_principio_ativo'].dropna().unique()
        )

        c1, c2 = st.columns([2, 3])
        with c1:
            selected_medication = st.selectbox(
                "Selecione um medicamento (exatamente):",
                options=["—"] + medication_options,
                index=0
            )
        with c2:
            query_text = st.text_input(
                "Ou pesquise por substring (contém):",
                placeholder="Digite parte do nome do princípio ativo, e.g., 'imatinib'"
            )

        # Decide which filter to apply: substring search takes priority if provided
        if query_text:
            # Case-insensitive contains
            mask = filtered_df['med_principio_ativo'].fillna("").str.contains(query_text, case=False, regex=False)
            selection_df = filtered_df[mask].copy()  # ← Keep all columns here
            search_label = query_text
        elif selected_medication != "—":
            selection_df = filtered_df[filtered_df['med_principio_ativo'] == selected_medication].copy()
            search_label = selected_medication
        else:
            selection_df = pd.DataFrame()
            search_label = None

        if selection_df.empty:
            st.info("Nenhuma linha encontrada. Tente um termo diferente ou escolha uma opção exata.")
        else:
            # Create time series for plotting (separate variable)
            monthly_counts = (selection_df
                            .set_index('par_data_emissao')
                            .resample('M')
                            .size()
                            .reset_index(name='count')
                            .rename(columns={'par_data_emissao': 'Month'}))

            fig = px.line(
                monthly_counts,
                x='Month',
                y='count',
                title=f"Solicitações de Medicamentos por Mês — {search_label}",
                labels={'Month': 'Data', 'count': 'Número de Solicitações'}
            )
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

            # KPIs
            monthly_counts['Year'] = monthly_counts['Month'].dt.year
            k1, k2, k3, k4 = st.columns(4)
            styled_metric(k1, "Total de Solicitações", int(monthly_counts['count'].sum()))
            styled_metric(k2, "Meses Ativos", int((monthly_counts['count'] > 0).sum()))
            styled_metric(k3, "Média por Mês", f"{monthly_counts['count'].mean():.1f}")
            styled_metric(k4, "Mês de Pico", int(monthly_counts['count'].max()))

            # Seasonality
            mo = monthly_counts.copy()
            mo['month_name'] = mo['Month'].dt.month_name(locale='pt_BR')
            mo['month_num'] = mo['Month'].dt.month
            by_moy = mo.groupby(['month_num','month_name'], as_index=False)['count'].sum().sort_values('month_num')
            fig = px.bar(by_moy, x='month_name', y='count', title=f"Distribuição por Mês do Ano — {search_label}")
            st.plotly_chart(fig, use_container_width=True)

            # Year-over-year
            yoy = monthly_counts.copy()
            yoy['month'] = yoy['Month'].dt.month
            yoy['year'] = yoy['Month'].dt.year
            yoy_pivot = yoy.pivot_table(index='month', columns='year', values='count', aggfunc='sum').fillna(0)
            fig = px.line(yoy_pivot, x=yoy_pivot.index, y=yoy_pivot.columns, markers=True,
                        labels={'index':'Mês', 'value':'Solicitações', 'variable':'Ano'},
                        title=f"Análise por Mês e Ano — {search_label}")
            st.plotly_chart(fig, use_container_width=True)

            # Top states/tribunals (use selection_df, not time_series_df)
            geo1, geo2 = st.columns(2)
            with geo1:
                top_states = selection_df['par_estado'].value_counts().head(10)
                st.bar_chart(top_states, use_container_width=True)
            with geo2:
                top_trib = selection_df['par_tribunal'].value_counts().head(10)
                st.bar_chart(top_trib, use_container_width=True)

            # Co-occurring actives
            sep = ';'
            co = (selection_df['med_principio_ativo']
                .dropna().str.split(sep)
                .explode().str.strip().str.lower()
                .value_counts().head(15))
            st.subheader("Substâncias co-ocorrentes")
            st.write(co)

            # Demographics over time
            demo = selection_df.copy()
            demo['Month'] = demo['par_data_emissao'].dt.to_period('M').dt.to_timestamp()
            demo_age = demo.groupby('Month')['pac_idade'].median().reset_index()
            fig = px.line(demo_age, x='Month', y='pac_idade', title='Média de Idade ao Longo do Tempo')
            st.plotly_chart(fig, use_container_width=True)

            gender_share = (demo.groupby(['Month','pac_sexo']).size()
                            .reset_index(name='count'))
            gender_share['share'] = gender_share.groupby('Month')['count'].transform(lambda s: s / s.sum())
            fig = px.area(gender_share, x='Month', y='share', color='pac_sexo',
                        title='Participação por Gênero ao Longo do Tempo', groupnorm='fraction')
            st.plotly_chart(fig, use_container_width=True)

            # Share of total requests
            all_monthly = filtered_df.set_index('par_data_emissao').resample('M').size().rename('all_count')
            sel_monthly = selection_df.set_index('par_data_emissao').resample('M').size().rename('sel_count')
            share = pd.concat([all_monthly, sel_monthly], axis=1).fillna(0).reset_index().rename(columns={'par_data_emissao':'Month'})
            share['share'] = (share['sel_count'] / share['all_count']).replace([np.inf, np.nan], 0)
            fig = px.line(share, x='Month', y='share', title=f"Participação no Total de Solicitações — {search_label}",
                        labels={'share':'Participação', 'Month':'Mês'})
            st.plotly_chart(fig, use_container_width=True)

            # Top municipalities comparison
            left, right = st.columns(2)
            with left:
                top_mun_sel = selection_df['par_municipio'].value_counts().head(10).reset_index()
                top_mun_sel.columns = ['Município', 'count']
                st.subheader("Top Municípios (Selecionado)")
                st.bar_chart(top_mun_sel.set_index('Município')['count'], use_container_width=True)
            with right:
                top_mun_all = filtered_df['par_municipio'].value_counts().head(10).reset_index()
                top_mun_all.columns = ['Município', 'count']
                st.subheader("Top Municípios (Geral)")
                st.bar_chart(top_mun_all.set_index('Município')['count'], use_container_width=True)

            # Anomaly detection
            mc = monthly_counts.copy()
            mc['roll_mean'] = mc['count'].rolling(6, min_periods=3).mean()
            mc['roll_std']  = mc['count'].rolling(6, min_periods=3).std()
            mc['z'] = (mc['count'] - mc['roll_mean']) / mc['roll_std']
            spikes = mc[(mc['z'] > 2) | (mc['z'] < -2)].dropna()
            if not spikes.empty:
                st.subheader("Meses Anômalos (|z| > 2)")
                st.dataframe(spikes[['Month','count','z']].round(2), use_container_width=True)

            # Data completeness heatmap
            heat = monthly_counts.copy()
            heat['year'] = heat['Month'].dt.year
            heat['month'] = heat['Month'].dt.month
            pivot = heat.pivot(index='year', columns='month', values='count').fillna(0)
            fig = px.imshow(pivot, text_auto=True, aspect='auto',
                            labels=dict(x="Mês", y="Ano", color="Contagem"),
                            title=f"Integridade de Dados — {search_label}")
            st.plotly_chart(fig, use_container_width=True)

            # Conclusions over time
            conc_by_month = (selection_df
                            .groupby([pd.Grouper(key='par_data_emissao', freq='M'),'par_conclusao'])
                            .size().reset_index(name='count'))
            fig = px.area(conc_by_month, x='par_data_emissao', y='count', color='par_conclusao',
                        title='Conclusões ao Longo do Tempo para Ativos Selecionados', groupnorm='fraction')
            st.plotly_chart(fig, use_container_width=True)

            # Show matched unique med_principio_ativo values
            with st.expander("Mostrar valores únicos de med_principio_ativo coincidentes"):
                st.dataframe(
                    selection_df[['med_principio_ativo']].drop_duplicates().sort_values('med_principio_ativo'),
                    use_container_width=True
                )
            
    with tab6:
        st.header("Análise baseada em CID")

        # Build CID options from the filtered set
        cid_options = (
            filtered_df['pac_cid10']
            .dropna()
            .astype(str)
            .str.strip()
            .replace({'': np.nan})
            .dropna()
            .unique()
        )
        cid_options = sorted(cid_options)

        left, right = st.columns([2, 3])
        with left:
            selected_cid = st.selectbox(
                "Selecione um CID-10",
                options=["—"] + cid_options,
                index=0,
                help="Analise conclusões, substâncias ativas, posologia, etc. para este CID"
            )

        # Opcional: filtro de texto livre para corresponder a padrões de CID mais amplos (ex: C50, E11, M54.5)
        with right:
            cid_query = st.text_input(
                "Ou pesquise por substring de CID (contém):",
                placeholder="Ex: C50, E11, M54.5"
            )

        # Decide filtering strategy
        if cid_query:
            cid_mask = filtered_df['pac_cid10'].astype(str).str.contains(cid_query, case=False, regex=False)
            cid_df = filtered_df[cid_mask].copy()
            cid_label = cid_query
        elif selected_cid != "—":
            cid_df = filtered_df[filtered_df['pac_cid10'].astype(str).str.strip() == selected_cid].copy()
            cid_label = selected_cid
        else:
            cid_df = pd.DataFrame()
            cid_label = None

        if cid_df.empty:
            st.info("Nenhum caso encontrado para o CID selecionado. Tente um CID diferente ou um padr o de pesquisa diferente.")
        else:
            st.subheader(f"Resumo para o CID: {cid_label}")

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            styled_metric(k1, "Total de Casos", int(len(cid_df)))
            styled_metric(k2, "Ativos Únicos", int(cid_df['med_principio_ativo'].dropna().nunique()))
            styled_metric(k3, "Posologias Únicas", int(cid_df['med_posologia'].dropna().nunique()))
            styled_metric(k4, "Intervalo de Tempo",
                    cid_df['par_data_emissao'].min().strftime('%d/%m/%Y') + " – " + cid_df['par_data_emissao'].max().strftime('%d/%m/%Y')
                    if pd.api.types.is_datetime64_any_dtype(cid_df['par_data_emissao'])
                    else "—")

            st.divider()

            # 1) Conclusion distribution
            col1, col2 = st.columns(2)
            with col1:
                concl_counts = cid_df['par_conclusao'].value_counts(dropna=False).reset_index()
                concl_counts.columns = ['Conclusão', 'count']
                fig = px.pie(concl_counts, values='count', names='Conclusão',
                            title="Conclusões para o CID selecionado",
                            color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)

            # 2) Time trend of requests for this CID
            with col2:
                ts = cid_df.set_index('par_data_emissao').resample('M').size().reset_index(name='count')
                ts.rename(columns={'par_data_emissao': 'Month'}, inplace=True)
                fig = px.line(ts, x='Month', y='count',
                            title="Solicitações por Mês (CID selecionado)",
                            labels={'Month': 'Mês', 'count': 'Solicitações'})
                fig.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # 3) Top active ingredients (handles multi-active strings if needed)
            st.markdown("#### Principais Princípios Ativos")
            # If multi-actives are separated by ';' (adjust separator if needed)
            sep = ';'
            actives = (cid_df['med_principio_ativo'].dropna()
                    .astype(str).str.split(sep).explode()
                    .str.strip().str.lower())
            top_actives = actives.value_counts().head(15).reset_index()
            top_actives.columns = ['Princípio Ativo', 'count']
            fig = px.bar(top_actives, x='Princípio Ativo', y='count',
                        title="Top 15 Princípios Ativos",
                        labels={'count': 'Solicitações'})
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            # 4) Posologia highlights
            st.markdown("#### Posologia (3 mais recentes)")

            # Frases a excluir
            excluir = {"conforme prescrição médica", "padrão", "vide nt em anexo"}

            # Garantir que temos a coluna de data como datetime
            cid_df = cid_df.copy()
            cid_df['par_data_emissao'] = pd.to_datetime(cid_df['par_data_emissao'], errors='coerce')

            # Filtrar posologia válida
            poso_df = cid_df[['par_data_emissao', 'med_posologia', 'med_principio_ativo', 'pac_sexo', 'pac_idade']].copy()
            poso_df['med_posologia'] = poso_df['med_posologia'].astype(str).str.strip()

            # Remover vazios/NaN e genéricos (case-insensitive)
            poso_df = poso_df[poso_df['med_posologia'].str.len() > 0]
            poso_df = poso_df[~poso_df['med_posologia'].str.lower().isin(excluir)]

            # Ordenar por data (recente primeiro)
            poso_df = poso_df.sort_values('par_data_emissao', ascending=False)

            # Pegar as 3 mais recentes
            poso_recent = poso_df.head(3)

            if not poso_recent.empty:
                cols = st.columns(min(3, len(poso_recent)))
                for (idx, row), col in zip(poso_recent.iterrows(), cols):
                    with col:
                        st.markdown("##### ")  # pequeno espaçamento
                        st.markdown(f"**{row['par_data_emissao'].strftime('%d/%m/%Y') if pd.notna(row['par_data_emissao']) else '—'}**")
                        st.caption(f"{row['med_principio_ativo'] if pd.notna(row['med_principio_ativo']) else '—'}")
                        idade = int(row['pac_idade']) if pd.notna(row['pac_idade']) else "—"
                        sexo = row['pac_sexo'] if pd.notna(row['pac_sexo']) else "—"
                        st.caption(f"Sexo: {sexo} | Idade: {idade}")
                        st.write(f"Posologia: {row['med_posologia']}")
            else:
                st.caption("Sem posologia recente válida (não genérica) para este CID.")
            
            st.divider()

            # 5) Cross breakdowns
            st.markdown("#### Que conclusões para cada princípio ativo?")
            cross1 = (cid_df.assign(med_principio_ativo=cid_df['med_principio_ativo'].fillna('—'))
                    .groupby(['med_principio_ativo', 'par_conclusao']).size()
                    .reset_index(name='count'))
            if not cross1.empty:
                fig = px.bar(cross1, x='med_principio_ativo', y='count', color='par_conclusao',
                            barmode='group',
                            title="Conclusão por Princípio Ativo",
                            labels={'med_principio_ativo':'Princípio Ativo', 'count':'Solicitações'})
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Sem distribuição de conclusões por princípio ativo para este CID.")

            st.markdown("#### Conclusões por Via Administrativa (se disponível)")
            cross2 = (cid_df.assign(med_via_admin=cid_df['med_via_admin'].fillna('—'))
                    .groupby(['med_via_admin', 'par_conclusao']).size()
                    .reset_index(name='count'))
            if not cross2.empty:
                fig = px.bar(cross2, x='med_via_admin', y='count', color='par_conclusao',
                            barmode='group',
                            title="Conclusão por Via Administrativa",
                            labels={'med_via_admin':'Via Administrativa', 'count':'Solicitações'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Sem dados suficientes por via administrativa.")

            st.divider()

            # 6) Geography for this CID
            geo1, geo2 = st.columns(2)
            with geo1:
                top_states = cid_df['par_estado'].value_counts().head(10).reset_index()
                top_states.columns = ['Estado', 'count']
                fig = px.bar(top_states, x='Estado', y='count', title="Top Estados")
                st.plotly_chart(fig, use_container_width=True)
            with geo2:
                top_muns = cid_df['par_municipio'].value_counts().head(10).reset_index()
                top_muns.columns = ['Município', 'count']
                fig = px.bar(top_muns, x='Município', y='count', title="Top Municípios")
                st.plotly_chart(fig, use_container_width=True)

            # Optional: table of matched cases and download
            with st.expander("Ver casos correspondentes (amostra)"):
                show_cols = ['par_numero', 'par_data_emissao', 'par_estado', 'par_municipio',
                            'par_conclusao', 'med_principio_ativo', 'med_posologia', 'med_via_admin']
                st.dataframe(cid_df[show_cols].sort_values('par_data_emissao', ascending=False).head(200),
                            use_container_width=True)

            st.download_button(
                "Baixar linhas filtradas (CSV)",
                cid_df.to_csv(index=False).encode('utf-8'),
                file_name=f"natjus_cid_{str(cid_label).replace(' ','_')}.csv",
                mime="text/csv"
            )

    with tab7:
        # --- Case Details Section ---
        st.header("Amostra de Dados")
        start = random.randint(0, len(filtered_df) - 50)
        st.dataframe(filtered_df[start:start+50])