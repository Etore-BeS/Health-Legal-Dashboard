import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
from collections import Counter
from plotly.colors import n_colors
import sklearn
import os

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

unimed_df, pmc_df = load_data()

# Navegação
st.sidebar.title('Navegação')
panel = st.sidebar.radio('Ir para', ['Análise Geral', 'Análise Financeira', 'Relatórios'])


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
    