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

if panel == 'Análise Financeira':
    st.title('Análise Financeira de Medicamentos')
    st.header('Análise com Base em Dados Históricos de Preços')

    # Sidebar Filtros para Análise Financeira
    st.sidebar.title('Filtros')

    # Selecionar Substância
    substances = pmc_df['substancia'].dropna().unique()
    substance_selected = st.sidebar.selectbox('Selecionar Substância', options=substances)

    # Filtrar dados pela substância selecionada
    df_substance = pmc_df[pmc_df['substancia'] == substance_selected].copy()

    # Selecionar Laboratórios
    laboratories = df_substance['laboratorio'].dropna().unique()
    laboratories_selected = st.sidebar.multiselect('Selecionar Laboratórios', options=laboratories, default=laboratories)

    # Filtro de Intervalo de Datas
    min_date = df_substance['data'].min()
    max_date = df_substance['data'].max()
    date_range = st.sidebar.date_input('Intervalo de Datas', [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filtrar Dados com Base nas Seleções
    df_filtered = df_substance[
        (df_substance['laboratorio'].isin(laboratories_selected)) &
        (df_substance['data'] >= pd.to_datetime(date_range[0])) &
        (df_substance['data'] <= pd.to_datetime(date_range[1]))
    ].copy()

    # Verificar se há dados disponíveis
    if df_filtered.empty:
        st.warning('Nenhum dado disponível para os filtros selecionados.')
    else:
        # Adicionar uma coluna de mês-ano para agrupamentos mensais
        df_filtered['Mes_Ano'] = df_filtered['data'].dt.to_period('M').dt.to_timestamp()

        # Calcular a média mensal para cada preço
        monthly_mean_pf = df_filtered.groupby('Mes_Ano')['pf_0'].mean().reset_index()
        monthly_mean_pmc = df_filtered.groupby('Mes_Ano')['pmc_0'].mean().reset_index()

        # Paleta de cores menos destacadas para os laboratórios
        num_labs = df_filtered['laboratorio'].nunique()
        if num_labs > 1:
            colors = n_colors('rgb(200, 200, 200)', 'rgb(100, 100, 100)', num_labs, colortype='rgb')
        elif num_labs == 1:
            colors = ['rgb(150, 150, 150)']  # Cor padrão para um único laboratório
        else:
            colors = []  # Nenhuma cor se não houver laboratórios

        # Gráfico 1: Evolução do Preço de Fábrica (PF)
        st.subheader(f'Evolução do Preço de Fábrica (PF) para {substance_selected}')
        
        fig_pf = go.Figure()

        # Adicionar as linhas dos laboratórios com cores menos destacadas
        for idx, lab in enumerate(df_filtered['laboratorio'].unique()):
            lab_data = df_filtered[df_filtered['laboratorio'] == lab]
            if num_labs > 1:
                color = colors[idx]
            else:
                color = colors[0] if colors else 'rgb(150, 150, 150)'  # Fallback se colors estiver vazio
            fig_pf.add_trace(go.Scatter(
                x=lab_data['data'],
                y=lab_data['pf_0'],
                mode='lines+markers',
                name=f'{lab} (PF)',
                line=dict(width=1, color=color, dash='dot'),
                opacity=0.6
            ))

        # Adicionar a linha de média mensal em vermelho vivo
        fig_pf.add_trace(go.Scatter(
            x=monthly_mean_pf['Mes_Ano'],
            y=monthly_mean_pf['pf_0'],
            mode='lines+markers',
            name='Média Mensal (PF)',
            line=dict(width=3, color='red')
        ))

        # Configurações do layout do gráfico PF
        fig_pf.update_layout(
            xaxis_title='Data',
            yaxis_title='Preço de Fábrica (R$)',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.3,
                xanchor='center',
                x=0.5
            ),
            margin=dict(b=150),  # Aumentar a margem inferior para acomodar a legenda
            width=800,  # Tamanho fixo do gráfico
            height=700,  # Aumentar a altura do gráfico
            title=f'Preço de Fábrica (PF) por Laboratório e Média Mensal',
        )

        st.plotly_chart(fig_pf, use_container_width=True)

        # Gráfico 2: Evolução do Preço Máximo ao Consumidor (PMC)
        st.subheader(f'Evolução do Preço Máximo ao Consumidor (PMC) para {substance_selected}')
        
        fig_pmc = go.Figure()

        # Adicionar as linhas dos laboratórios com cores menos destacadas
        for idx, lab in enumerate(df_filtered['laboratorio'].unique()):
            lab_data = df_filtered[df_filtered['laboratorio'] == lab]
            if num_labs > 1:
                color = colors[idx]
            else:
                color = colors[0] if colors else 'rgb(150, 150, 150)'  # Fallback se colors estiver vazio
            fig_pmc.add_trace(go.Scatter(
                x=lab_data['data'],
                y=lab_data['pmc_0'],
                mode='lines+markers',
                name=f'{lab} (PMC)',
                line=dict(width=1, color=color, dash='dot'),
                opacity=0.6
            ))

        # Adicionar a linha de média mensal em vermelho vivo
        fig_pmc.add_trace(go.Scatter(
            x=monthly_mean_pmc['Mes_Ano'],
            y=monthly_mean_pmc['pmc_0'],
            mode='lines+markers',
            name='Média Mensal (PMC)',
            line=dict(width=3, color='red')
        ))

        # Configurações do layout do gráfico PMC
        fig_pmc.update_layout(
            xaxis_title='Data',
            yaxis_title='Preço Máximo ao Consumidor (R$)',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.3,
                xanchor='center',
                x=0.5
            ),
            margin=dict(b=150),  # Aumentar a margem inferior para acomodar a legenda
            width=800,  # Tamanho fixo do gráfico
            height=700,  # Aumentar a altura do gráfico
            title=f'Preço Máximo ao Consumidor (PMC) por Laboratório e Média Mensal',
        )

        st.plotly_chart(fig_pmc, use_container_width=True)

        # **Novo: Calcular Estatísticas para o Último Mês Disponível**
        # Determinar o último mês disponível
        ultimo_mes = df_filtered['Mes_Ano'].max()
        df_ultimo_mes = df_filtered[df_filtered['Mes_Ano'] == ultimo_mes].copy()

        # Formatar a data do último mês para exibir como 'Mês/Ano'
        ultimo_mes_str = ultimo_mes.strftime('%B/%Y')  # Exemplo: "Setembro/2023"

        # Calcular estatísticas para o último mês (PF)
        df_ultimo_pf = df_ultimo_mes.dropna(subset=['pf_0'])
        if not df_ultimo_pf.empty:
            min_pf = df_ultimo_pf['pf_0'].min()
            lab_min_pf = df_ultimo_pf.loc[df_ultimo_pf['pf_0'].idxmin(), 'laboratorio']
            max_pf = df_ultimo_pf['pf_0'].max()
            lab_max_pf = df_ultimo_pf.loc[df_ultimo_pf['pf_0'].idxmax(), 'laboratorio']
            mean_pf = df_ultimo_pf['pf_0'].mean()
            num_labs_pf = df_ultimo_pf['laboratorio'].nunique()
        else:
            min_pf = max_pf = mean_pf = num_labs_pf = lab_min_pf = lab_max_pf = 'Dados indisponíveis'

        # Calcular estatísticas para o último mês (PMC)
        df_ultimo_pmc = df_ultimo_mes.dropna(subset=['pmc_0'])
        if not df_ultimo_pmc.empty:
            min_pmc = df_ultimo_pmc['pmc_0'].min()
            lab_min_pmc = df_ultimo_pmc.loc[df_ultimo_pmc['pmc_0'].idxmin(), 'laboratorio']
            max_pmc = df_ultimo_pmc['pmc_0'].max()
            lab_max_pmc = df_ultimo_pmc.loc[df_ultimo_pmc['pmc_0'].idxmax(), 'laboratorio']
            mean_pmc = df_ultimo_pmc['pmc_0'].mean()
            num_labs_pmc = df_ultimo_pmc['laboratorio'].nunique()
        else:
            min_pmc = max_pmc = mean_pmc = num_labs_pmc = lab_min_pmc = lab_max_pmc = 'Dados indisponíveis'

        # Criar tabela resumo com o mês e ano
        resumo = pd.DataFrame({
            'Categoria': ['Preço de Fábrica (PF)', 'Preço Máximo ao Consumidor (PMC)'],
            'Mês/Ano': [ultimo_mes_str, ultimo_mes_str],
            'Preço Mínimo (R$)': [min_pf, min_pmc],
            'Laboratório com Preço Mínimo': [lab_min_pf, lab_min_pmc],
            'Preço Máximo (R$)': [max_pf, max_pmc],
            'Laboratório com Preço Máximo': [lab_max_pf, lab_max_pmc],
            'Preço Médio (R$)': [mean_pf, mean_pmc],
            'Número de Laboratórios Ofertantes': [num_labs_pf, num_labs_pmc]
        })

        st.subheader('Resumo dos Preços do Último Mês Disponível')
        st.table(resumo)