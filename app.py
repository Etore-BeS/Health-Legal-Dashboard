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

# Navigation
st.sidebar.title('Navigation')
panel = st.sidebar.radio('Go to', ['General Analysis', 'Financial Analysis'])


if panel == 'General Analysis':
    st.title('Healthcare Legal Cases Dashboard')
    st.header('General Analysis')

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
        # 1. Top Causes of Legal Action (descritor)
        st.subheader('1. Top Causes of Legal Action')

        # Calculate top 10 causes
        top_causes = df_filtered['descritor'].value_counts().reset_index()
        top_causes.columns = ['Descritor', 'Count']

        # Function to truncate labels
        def truncate_label(label, max_length=30):
            return (label[:max_length] + '...') if len(label) > max_length else label

        top_causes['Descritor_Truncated'] = top_causes['Descritor'].apply(truncate_label)

        fig1 = px.bar(
            top_causes.head(10),
            x='Descritor_Truncated',
            y='Count',
            hover_data=['Descritor'],  # Show full label on hover
            title='Top 10 Causes of Legal Action'
        )

        fig1.update_layout(
            xaxis_title='Descritor',
            yaxis_title='Count',
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig1, use_container_width=True)


        # 2. Outcome Distribution (acao)
        st.subheader('2. Outcome Distribution')
        outcome_distribution = df_filtered['acao'].value_counts().reset_index()
        outcome_distribution.columns = ['Outcome', 'Count']
        fig2 = px.pie(outcome_distribution, names='Outcome', values='Count', title='Outcome Distribution')
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Temporal Trends (Number of Processes Over Time)
        st.subheader('3. Temporal Trends')
        temporal_trends = df_filtered.groupby('Year').size().reset_index(name='Number of Cases')
        fig3 = px.line(temporal_trends, x='Year', y='Number of Cases', markers=True, title='Number of Cases Over Years')
        st.plotly_chart(fig3, use_container_width=True)

        # 4. Analysis of Tutela
        st.subheader('4. Analysis of Tutela')
        tutela_counts = df_filtered['tutela'].value_counts().reset_index()
        tutela_counts.columns = ['Tutela Granted', 'Count']
        fig4 = px.bar(tutela_counts, x='Tutela Granted', y='Count', title='Tutela Granted vs. Not Granted')
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Correlation Between Case Duration and Outcome
        st.subheader('5. Correlation Between Case Duration and Outcome')
        fig5 = px.box(df_filtered, x='acao', y='tempo_processo_mes', points='all',
                    labels={'acao': 'Outcome', 'tempo_processo_mes': 'Process Duration (Months)'},
                    title='Process Duration by Outcome')
        st.plotly_chart(fig5, use_container_width=True)

        # 6. Monetary Analysis
        st.subheader('6. Monetary Analysis')

        # Identify outliers using the IQR method
        Q1 = df_filtered['valor'].quantile(0.25)
        Q3 = df_filtered['valor'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers_valor = df_filtered[df_filtered['valor'] > upper_bound]

        # Count of outliers
        outlier_count_valor = outliers_valor.shape[0]

        # Exclude outliers from the plot
        valor_filtered = df_filtered['valor'].copy()
        valor_filtered = valor_filtered[valor_filtered <= upper_bound]

        # Plot histogram including 'valor' == 0 but excluding outliers
        fig6 = px.histogram(valor_filtered, x='valor', nbins=50, title='Distribution of Case Values (Excluding Outliers)',
                        labels={'valor': 'Case Value (R$)'}, 
                        range_x=(0, upper_bound))
        st.plotly_chart(fig6, use_container_width=True)

        # Display count of outliers
        st.markdown(f"**Note:** {outlier_count_valor} outlier(s) were excluded from the plot to ensure a clear visualization.")


        # 7. Most Active Attorneys
        st.subheader('7. Most Active Attorneys')
        # Split and explode the 'adv' column
        df_attorneys = df_filtered.copy()
        df_attorneys['adv'] = df_attorneys['adv'].fillna('')
        df_attorneys['attorneys_list'] = df_attorneys['adv'].str.split(';')
        df_attorneys_exploded = df_attorneys.explode('attorneys_list')
        attorney_counts = df_attorneys_exploded['attorneys_list'].str.strip().value_counts().reset_index()
        attorney_counts.columns = ['Attorney', 'Number of Cases']
        fig7 = px.bar(attorney_counts.head(10), x='Attorney', y='Number of Cases', title='Top 10 Attorneys')
        st.plotly_chart(fig7, use_container_width=True)

        # 9. Analysis of Courts
        st.subheader('9. Analysis of Courts')
        court_counts = df_filtered['vara_nome'].value_counts().reset_index()
        court_counts.columns = ['Court', 'Number of Cases']
        fig9 = px.bar(court_counts.head(10), x='Court', y='Number of Cases', title='Top 10 Courts by Number of Cases')
        st.plotly_chart(fig9, use_container_width=True)


    with col2:    
        # 8. Monthly and Yearly Trends
        st.subheader('8. Monthly and Yearly Trends')
        monthly_trends = df_filtered.groupby(df_filtered['data_publicacao'].dt.to_period('M')).size().reset_index(name='Number of Cases')
        monthly_trends['Month'] = monthly_trends['data_publicacao'].astype(str)
        fig10 = px.line(monthly_trends, x='Month', y='Number of Cases', markers=True, title='Monthly Number of Cases')
        st.plotly_chart(fig10, use_container_width=True)

        # 9. Case Complexity Indicators
        st.subheader('9. Case Complexity Indicators')

        # Exclude cases with tempo_processo_mes = 0
        complexity_zero_count = df_filtered[df_filtered['tempo_processo_mes'] == 0].shape[0]
        df_complexity = df_filtered[df_filtered['tempo_processo_mes'] > 0]

        # Calculate average duration by assunto_agravo
        complexity = df_complexity.groupby('assunto_agravo')['tempo_processo_mes'].mean().reset_index()
        complexity.columns = ['Assunto Agravo', 'Average Duration (Months)']

        # Sort by average duration and select top 10 for better visualization
        complexity = complexity.sort_values(by='Average Duration (Months)', ascending=False).head(20)

        # Function to abbreviate 'Assunto Agravo' names
        def abbreviate_text(text, max_length=20):
            return text if len(text) <= max_length else text[:17] + '...'

        complexity['Assunto Agravo Abbreviated'] = complexity['Assunto Agravo'].apply(lambda x: abbreviate_text(x, 20))

        # Plot bar chart with abbreviated labels
        fig9 = px.bar(
            complexity,
            x='Assunto Agravo Abbreviated',
            y='Average Duration (Months)',
            title='Average Case Duration by Assunto Agravo (Top 10)',
            labels={'Assunto Agravo Abbreviated': 'Assunto Agravo', 'Average Duration (Months)': 'Average Duration (Months)'},
            hover_data=['Assunto Agravo', 'Average Duration (Months)']
        )

        fig9.update_layout(
            xaxis_tickangle=-45,
            yaxis_title='Average Duration (Months)',
            xaxis_title='Assunto Agravo'
        )

        st.plotly_chart(fig9, use_container_width=True)

        # Display the count of tempo_processo_mes = 0
        st.markdown(f"**Number of cases with `tempo_processo_mes` = 0:** {complexity_zero_count}")


        # 10. Success Rate of Injunctions
        st.subheader('10. Success Rate of Injunctions')
        injunctions = df_filtered.groupby(['tutela', 'acao']).size().reset_index(name='Number of Cases')
        fig10 = px.bar(injunctions, x='acao', y='Number of Cases', color='tutela', barmode='group', title='Success Rate of Injunctions by Outcome')
        st.plotly_chart(fig10, use_container_width=True)

        # 11. Analysis of Legal Fees
        st.subheader('11. Analysis of Legal Fees')

        # Remove cases with valor = 0 and multa = 0 for meaningful analysis
        df_fees = df_filtered[(df_filtered['valor'] > 0) & (df_filtered['multa'] > 0)]

        # Identify outliers using the IQR method for 'valor'
        Q1_valor = df_fees['valor'].quantile(0.25)
        Q3_valor = df_fees['valor'].quantile(0.75)
        IQR_valor = Q3_valor - Q1_valor
        upper_bound_valor = Q3_valor + 1.5 * IQR_valor

        # Identify outliers in 'valor'
        outliers_valor = df_fees[df_fees['valor'] > upper_bound_valor]

        # Filter out outliers for plotting
        df_fees_plot = df_fees[df_fees['valor'] <= upper_bound_valor]

        fig11 = px.scatter(
            df_fees_plot,
            x='valor',
            y='multa',
            trendline='ols',
            labels={'valor': 'Case Value (R$)', 'multa': 'Fine (R$)'},
            title='Fine Amount vs. Case Value (Excluding Outliers)',
            hover_data=['processo', 'vara_nome', 'adv']
        )

        # Add trendline details
        fig11.update_layout(
            xaxis_title='Case Value (R$)',
            yaxis_title='Fine (R$)',
            xaxis=dict(range=[0, upper_bound_valor * 1.05]),  # Add some padding
            yaxis=dict(range=[0, df_fees_plot['multa'].max() * 1.05])
        )

        st.plotly_chart(fig11, use_container_width=True)

        # Display information about the outlier
        if not outliers_valor.empty:
            st.markdown(f"**Outliers Excluded from Plot:** {outliers_valor.shape[0]} cases with `valor` > {upper_bound_valor:.2f} R$")
            st.markdown("These outliers are excluded to ensure the plot remains readable. They are still accounted for in other analyses.")


        # 12. Assunto Agravo Distribution Over Time
        st.subheader('12. Assunto Agravo Distribution Over Time')

        # Aggregate data
        assunto_time = df_filtered.groupby(['Year', 'assunto_agravo']).size().reset_index(name='Number of Cases')

        # Truncate 'assunto_agravo' labels for display
        def truncate_label(label, max_length=20):
            return (label[:max_length] + '...') if len(label) > max_length else label

        assunto_time['Assunto_Agravo_Truncated'] = assunto_time['assunto_agravo'].apply(truncate_label)

        fig12 = px.line(
            assunto_time,
            x='Year',
            y='Number of Cases',
            color='Assunto_Agravo_Truncated',
            hover_data=['assunto_agravo'],
            title='Assunto Agravo Over Time',
            labels={'Assunto_Agravo_Truncated': 'Assunto Agravo'}
        )

        # Move legend outside the plot
        fig12.update_layout(
            legend=dict(
                title='Assunto Agravo',
                orientation="h",
                yanchor="bottom",
                y=1.6,
                xanchor="right",
                x=1
            ),
            xaxis_title='Year',
            yaxis_title='Number of Cases'
        )

        st.plotly_chart(fig12, use_container_width=True)

        # 13. Case Duration by Court
        st.subheader('13. Case Duration by Court')
        duration_court = df_filtered.groupby('vara_nome')['tempo_processo_mes'].mean().reset_index()
        duration_court.columns = ['Court', 'Average Duration (Months)']
        fig17 = px.bar(duration_court.sort_values(by='Average Duration (Months)', ascending=False).head(10),
                    x='Court', y='Average Duration (Months)', title='Top 10 Courts by Average Case Duration')
        st.plotly_chart(fig17, use_container_width=True)

        # 14. Comparative Analysis Between Years com Exclusão de Valores Extremos
        st.subheader('14. Comparative Analysis Between Years')
        
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
        yearly_comparison.columns = ['Year', 'Average Case Value (R$)', 'Average Duration (Months)', 'Number of Cases']
        
        # Passo 3: Visualizar os Dados Atualizados
        # Criar figura com eixo y secundário
        fig14 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Adicionar barra para 'Number of Cases'
        fig14.add_trace(
            go.Bar(
                x=yearly_comparison['Year'],
                y=yearly_comparison['Number of Cases'],
                name='Number of Cases',
                marker_color='rgba(55, 128, 191, 0.7)'  # Azul
            ),
            secondary_y=False,
        )
        
        # Adicionar linha para 'Average Duration (Months)'
        fig14.add_trace(
            go.Scatter(
                x=yearly_comparison['Year'],
                y=yearly_comparison['Average Duration (Months)'],
                name='Average Duration (Months)',
                mode='lines+markers',
                marker=dict(color='firebrick'),
                line=dict(color='firebrick', width=2)
            ),
            secondary_y=True,
        )
        
        # Atualizar layout para melhor estética
        fig14.update_layout(
            title_text='Yearly Comparative Analysis',
            legend=dict(x=0.01, y=0.99),
            barmode='group',
            height=600,
            width=1200  # Ajuste conforme necessário
        )
        
        # Definir títulos dos eixos y
        fig14.update_yaxes(title_text="Number of Cases", secondary_y=False)
        fig14.update_yaxes(title_text="Average Duration (Months)", secondary_y=True)
        
        # Ajustar título do eixo x
        fig14.update_xaxes(title_text="Year")
        
        # Exibir o plot
        st.plotly_chart(fig14, use_container_width=True)
        
        # Exibir informações sobre os outliers
        st.markdown(f"**Total de outliers excluídos na análise:** {outlier_count}")
        
        if not outliers.empty:
            # Exibir os anos com outliers
            outlier_years = outliers['Year'].unique()
            st.markdown(f"**Outlier Detected:** Alguns casos em **{', '.join(map(str, outlier_years))}** tiveram valores de **'valor'** extremamente altos que foram excluídos da análise.")
            st.markdown("Os anos permaneceram na análise, mas os casos com valores extremos foram removidos para evitar distorções nas médias.")

elif panel == 'Financial Analysis':
    st.title('Financial Analysis of Medications')
    st.header('Analysis Based on Historical Price Data')

    # Sidebar Filters for Financial Analysis
    st.sidebar.title('Filters')

    # Select Medications
    medications = pmc_df['substancia'].unique()
    medications_selected = st.sidebar.multiselect('Select Medications', options=medications, default=medications[:5])

    # Select Laboratories
    laboratories = pmc_df['laboratorio'].unique()
    laboratories_selected = st.sidebar.multiselect('Select Laboratories', options=laboratories, default=laboratories[:5])

    # Date Range Filter
    min_date = pmc_df['data'].min()
    max_date = pmc_df['data'].max()
    date_range = st.sidebar.date_input('Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter Data Based on Selections
    df_filtered = pmc_df[
        (pmc_df['substancia'].isin(medications_selected)) &
        (pmc_df['laboratorio'].isin(laboratories_selected)) &
        (pmc_df['data'] >= pd.to_datetime(date_range[0])) &
        (pmc_df['data'] <= pd.to_datetime(date_range[1]))
    ].copy()

    # Check if data is available
    if df_filtered.empty:
        st.warning('No data available for the selected filters.')
    else:
        # Create columns for layout
        col1, col2 = st.columns(2)

        # 1. Price Evolution Over Time
        with col1:
            st.subheader('1. Price Evolution Over Time')

            # Aggregate data
            price_trends = df_filtered.groupby(['substancia', 'data'])[['pf_0', 'pmc_0']].mean().reset_index()

            # Plot the evolution of prices for each medication
            fig = px.line(
                price_trends,
                x='data',
                y=['pf_0', 'pmc_0'],
                color='substancia',
                labels={'value': 'Price (R$)', 'data': 'Date', 'variable': 'Price Type'},
                title='Price Evolution Over Time',
                markers=True
            )
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (R$)',
                legend_title='Medication',
                xaxis=dict(rangeslider=dict(visible=True), type='date')
            )
            st.plotly_chart(fig, use_container_width=True)

        # 2. Laboratory Price Margins
        with col2:
            st.subheader('2. Laboratory Price Margins')

            # Calculate price margin (pmc_0 - pf_0)
            df_filtered['price_margin'] = df_filtered['pmc_0'] - df_filtered['pf_0']

            # Aggregate data
            lab_price_margins = df_filtered.groupby(['laboratorio', 'data'])['price_margin'].mean().reset_index()

            # Plot price margins over time
            fig2 = px.line(
                lab_price_margins,
                x='data',
                y='price_margin',
                color='laboratorio',
                labels={'price_margin': 'Price Margin (R$)', 'data': 'Date'},
                title='Laboratory Price Margins Over Time',
                markers=True
            )
            fig2.update_layout(
                xaxis_title='Date',
                yaxis_title='Price Margin (R$)',
                legend_title='Laboratory',
                xaxis=dict(rangeslider=dict(visible=True), type='date')
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('---')

        # 3. Price Comparison at Start and End of Legal Cases
        st.subheader('3. Price Comparison at Start and End of Legal Cases')

        # Merge unimed_df and pmc_df on 'descritor' and 'substancia'
        merged_df = unimed_df.merge(
            pmc_df[['substancia', 'data', 'pf_0', 'pmc_0']],
            left_on='descritor',
            right_on='substancia',
            how='left',
            suffixes=('', '_pmc')
        )

        # Ensure date columns are datetime
        merged_df['data_publicacao'] = pd.to_datetime(merged_df['data_publicacao'])
        merged_df['data'] = pd.to_datetime(merged_df['data'])

        # Create 'data_sentenca' as 'data_publicacao' + 'tempo_processo_mes'
        merged_df['data_sentenca'] = merged_df['data_publicacao'] + pd.to_timedelta(merged_df['tempo_processo_mes'] * 30, unit='D')

        # Function to get price closest to a given date
        def get_closest_price(substancia, target_date):
            med_prices = pmc_df[pmc_df['substancia'] == substancia]
            if med_prices.empty:
                return np.nan
            med_prices = med_prices.copy()  # Avoid SettingWithCopyWarning
            med_prices['date_diff'] = (med_prices['data'] - target_date).abs()
            closest_price = med_prices.loc[med_prices['date_diff'].idxmin()]
            return closest_price['pmc_0']

        # Apply function to get prices at 'data_publicacao' and 'data_sentenca'
        merged_df['pmc_inicio'] = merged_df.apply(lambda row: get_closest_price(row['descritor'], row['data_publicacao']), axis=1)
        merged_df['pmc_sentenca'] = merged_df.apply(lambda row: get_closest_price(row['descritor'], row['data_sentenca']), axis=1)

        # Calculate price change
        merged_df['price_change'] = merged_df['pmc_sentenca'] - merged_df['pmc_inicio']
        merged_df['price_change_perc'] = (merged_df['price_change'] / merged_df['pmc_inicio']) * 100

        # Visualize price change distribution
        fig3 = px.histogram(
            merged_df,
            x='price_change_perc',
            nbins=50,
            title='Distribution of Price Change Percentage Between Start and End of Legal Cases',
            labels={'price_change_perc': 'Price Change (%)'}
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Display summary statistics
        st.write('**Summary Statistics of Price Change (%):**')
        st.write(merged_df['price_change_perc'].describe())
        
        st.markdown('---')

        # 4. Comparing Price at Sentence Date with Case Value
        st.subheader('4. Comparing Price at Sentence Date with Case Value')

        # Ensure 'valor' is numeric
        merged_df['valor'] = pd.to_numeric(merged_df['valor'], errors='coerce')

        # Remove rows with missing data
        comparison_df = merged_df.dropna(subset=['valor', 'pmc_sentenca'])

        # Calculate ratio of case value to medication price
        comparison_df['value_to_price_ratio'] = comparison_df['valor'] / comparison_df['pmc_sentenca']

        # Visualize the relationship
        fig4 = px.scatter(
            comparison_df,
            x='pmc_sentenca',
            y='valor',
            hover_data=['substancia', 'processo'],
            labels={'pmc_sentenca': 'Medication Price at Sentence Date (R$)', 'valor': 'Case Value (R$)'},
            title='Case Value vs. Medication Price at Sentence Date'
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

        # Display insights
        st.write('**Insights:**')
        st.write('- Points above the red dashed line indicate cases where the case value is higher than the medication price.')
        st.write('- Points below the line indicate cases where the case value is lower than the medication price.')

        # Optionally, provide a table of cases with high ratios
        high_ratio_cases = comparison_df[comparison_df['value_to_price_ratio'] > 10]
        if not high_ratio_cases.empty:
            st.write('**Cases with High Value to Price Ratio (>10):**')
            st.dataframe(high_ratio_cases[['processo', 'substancia', 'valor', 'pmc_sentenca', 'value_to_price_ratio']])
