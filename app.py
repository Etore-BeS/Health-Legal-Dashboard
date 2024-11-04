import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
from collections import Counter

st.set_page_config(layout="wide")

# Função para carregar e preprocessar os dados
@st.cache_data
def load_data():
    unimed_df = pd.read_csv('unimed.csv', parse_dates=['data_publicacao'])
    unimed_df['ano'] = unimed_df['data_publicacao'].dt.year
    unimed_df['mes'] = unimed_df['data_publicacao'].dt.month_name()
    unimed_df['Year'] = unimed_df['data_publicacao'].dt.year  # For consistency in plots

    pmc_df = pd.read_csv('pmc_2015_2024.csv')
    pmc_df['mes'] = pmc_df['mes'].astype(str).str.zfill(2)  # Ensure month is two digits
    pmc_df['data'] = pd.to_datetime(pmc_df['ano'].astype(str) + pmc_df['mes'], format='%Y%m')
    
    # Convert 'pf_0' and 'pmc_0' to numeric
    pmc_df['pf_0'] = pd.to_numeric(pmc_df['pf_0'], errors='coerce')
    pmc_df['pmc_0'] = pd.to_numeric(pmc_df['pmc_0'], errors='coerce')
    return unimed_df, pmc_df

unimed_df, pmc_df = load_data()

# Navegação
st.sidebar.title('Navegação')
panel = st.sidebar.radio('Ir para', ['Análise Geral', 'Análise Financeira'])


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

elif panel == 'Análise Financeira':
    st.title('Análise Financeira de Medicamentos')
    st.header('Análise com Base em Dados Históricos de Preços')

    # Sidebar Filtros para Análise Financeira
    st.sidebar.title('Filtros')

    # Selecionar Medicamentos
    medications = pmc_df['substancia'].unique()
    medications_selected = st.sidebar.multiselect('Selecionar Medicamentos', options=medications, default=medications[:5])

    # Selecionar Laboratórios
    laboratories = pmc_df['laboratorio'].unique()
    laboratories_selected = st.sidebar.multiselect('Selecionar Laboratórios', options=laboratories, default=laboratories[:5])

    # Filtro de Intervalo de Datas
    min_date = pmc_df['data'].min()
    max_date = pmc_df['data'].max()
    date_range = st.sidebar.date_input('Intervalo de Datas', [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filtrar Dados com Base nas Seleções
    df_filtered = pmc_df[
        (pmc_df['substancia'].isin(medications_selected)) &
        (pmc_df['laboratorio'].isin(laboratories_selected)) &
        (pmc_df['data'] >= pd.to_datetime(date_range[0])) &
        (pmc_df['data'] <= pd.to_datetime(date_range[1]))
    ].copy()

    # Verificar se há dados disponíveis
    if df_filtered.empty:
        st.warning('Nenhum dado disponível para os filtros selecionados.')
    else:
        # Criar colunas para layout
        col1, col2 = st.columns(2)

        # 1. Evolução de Preços ao Longo do Tempo
        with col1:
            st.subheader('1. Evolução de Preços ao Longo do Tempo')

            # Agregar dados
            price_trends = df_filtered.groupby(['substancia', 'data'])[['pf_0', 'pmc_0']].mean().reset_index()

            # Plotar a evolução dos preços para cada medicamento
            fig = px.line(
                price_trends,
                x='data',
                y=['pf_0', 'pmc_0'],
                color='substancia',
                labels={'value': 'Preço (R$)', 'data': 'Data', 'variable': 'Tipo de Preço'},
                title='Evolução de Preços ao Longo do Tempo',
                markers=True
            )
            fig.update_layout(
                xaxis_title='Data',
                yaxis_title='Preço (R$)',
                legend_title='Medicamento',
                xaxis=dict(rangeslider=dict(visible=True), type='date'),
                legend=dict(
                    x=0,
                    y=0,
                    font=dict(
                        size=8
                    ),
                    orientation='v'
                )
                
            )
            st.plotly_chart(fig, use_container_width=True)

        # 2. Evolução Média de Preços por Medicamento ao Longo dos Anos (Novo Plot)
        with col2:
            st.subheader('2. Evolução Média de Preços por Medicamento ao Longo dos Anos')

            # Extrair o ano da coluna 'data'
            df_filtered['Ano'] = df_filtered['data'].dt.year

            # Calcular o preço médio por medicamento por ano
            avg_price_per_year = df_filtered.groupby(['substancia', 'Ano'])['pmc_0'].mean().reset_index()

            # Plotar a evolução média de preços
            fig_avg_price = px.line(
                avg_price_per_year,
                x='Ano',
                y='pmc_0',
                color='substancia',
                labels={'pmc_0': 'Preço Médio (R$)', 'Ano': 'Ano', 'substancia': 'Medicamento'},
                title='Evolução Média de Preços por Medicamento ao Longo dos Anos',
                markers=True
            )
            fig_avg_price.update_layout(
                xaxis_title='Ano',
                yaxis_title='Preço Médio (R$)',
                legend_title='Medicamento',
                xaxis=dict(dtick=1),  # Garante que cada ano seja mostrado no eixo x
                legend=dict(
                    x=0,
                    y=1,
                    font=dict(
                        size=8
                    ),
                    orientation='v'
                )
            )
            st.plotly_chart(fig_avg_price, use_container_width=True)

        st.markdown('---')

        # 3. Comparação de Preços no Início e no Fim de Casos Legais
        st.subheader('3. Comparação de Preços no Início e no Fim de Casos Legais')

        # Mesclar unimed_df e pmc_df com base em 'descritor' e 'substancia'
        merged_df = unimed_df.merge(
            pmc_df[['substancia', 'data', 'pf_0', 'pmc_0']],
            left_on='descritor',
            right_on='substancia',
            how='left',
            suffixes=('', '_pmc')
        )

        # Garantir que as colunas de data são datetime
        merged_df['data_publicacao'] = pd.to_datetime(merged_df['data_publicacao'])
        merged_df['data'] = pd.to_datetime(merged_df['data'])

        # Criar 'data_sentenca' como 'data_publicacao' + 'tempo_processo_mes'
        merged_df['data_sentenca'] = merged_df['data_publicacao'] + pd.to_timedelta(merged_df['tempo_processo_mes'] * 30, unit='D')

        # Função para obter preço mais próximo de uma determinada data
        def get_closest_price(substancia, target_date):
            med_prices = pmc_df[pmc_df['substancia'] == substancia]
            if med_prices.empty:
                return np.nan
            med_prices = med_prices.copy()  # Evitar SettingWithCopyWarning
            med_prices['date_diff'] = (med_prices['data'] - target_date).abs()
            closest_price = med_prices.loc[med_prices['date_diff'].idxmin()]
            return closest_price['pmc_0']

        # Aplicar função para obter preços em 'data_publicacao' e 'data_sentenca'
        merged_df['pmc_inicio'] = merged_df.apply(lambda row: get_closest_price(row['descritor'], row['data_publicacao']), axis=1)
        merged_df['pmc_sentenca'] = merged_df.apply(lambda row: get_closest_price(row['descritor'], row['data_sentenca']), axis=1)

        # Calcular mudança de preço
        merged_df['mudanca_preco'] = merged_df['pmc_sentenca'] - merged_df['pmc_inicio']
        merged_df['mudanca_preco_perc'] = (merged_df['mudanca_preco'] / merged_df['pmc_inicio']) * 100

        # Visualizar distribuição da mudança de preço
        fig3 = px.histogram(
            merged_df,
            x='mudanca_preco_perc',
            nbins=50,
            title='Distribuição da Mudança Percentual de Preço Entre Início e Fim de Casos Legais',
            labels={'mudanca_preco_perc': 'Mudança de Preço (%)'}
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Exibir estatísticas resumidas
        st.write('**Estatísticas Resumidas da Mudança de Preço (%):**')
        st.write(merged_df['mudanca_preco_perc'].describe())
        
        st.markdown('---')

        # 4. Comparando Preço na Data da Sentença com Valor do Caso
        st.subheader('4. Comparando Preço na Data da Sentença com Valor do Caso')

        # Garantir que 'valor' é numérico
        merged_df['valor'] = pd.to_numeric(merged_df['valor'], errors='coerce')

        # Remover linhas com dados faltantes
        comparison_df = merged_df.dropna(subset=['valor', 'pmc_sentenca'])

        # Calcular razão de valor do caso para preço do medicamento
        comparison_df['razao_valor_preco'] = comparison_df['valor'] / comparison_df['pmc_sentenca']

        # Visualizar a relação
        fig4 = px.scatter(
            comparison_df,
            x='pmc_sentenca',
            y='valor',
            hover_data=['substancia', 'processo'],
            labels={'pmc_sentenca': 'Preço do Medicamento na Data da Sentença (R$)', 'valor': 'Valor do Caso (R$)'},
            title='Valor do Caso vs. Preço do Medicamento na Data da Sentença'
        )
        fig4.add_shape(
            type='line',
            x0=comparison_df['pmc_sentenca'].min(),
            y0=comparison_df['pmc_sentenca'].min(),
            x1=comparison_df['pmc_sentenca'].max(),
            y1=comparison_df['pmc_sentenca'].max(),
            line=dict(color='Red', dash='dash'),
            name='y = x'
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Exibir insights
        st.write('**Insights:**')
        st.write('- Pontos acima da linha vermelha tracejada indicam casos onde o valor do caso é maior que o preço do medicamento.')
        st.write('- Pontos abaixo da linha indicam casos onde o valor do caso é menor que o preço do medicamento.')

        # Opcionalmente, fornecer uma tabela de casos com altas razões
        high_ratio_cases = comparison_df[comparison_df['razao_valor_preco'] > 10]
        if not high_ratio_cases.empty:
            st.write('**Casos com Alta Razão Valor para Preço (>10):**')
            st.dataframe(high_ratio_cases[['processo', 'substancia', 'valor', 'pmc_sentenca', 'razao_valor_preco']])
