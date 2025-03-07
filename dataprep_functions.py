from IPython.display import Markdown, display, clear_output

import pandas as pd
import numpy as np


# plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_squared_error, r2_score
import optuna

from scipy.stats import gaussian_kde, boxcox, skew
import logging


# finds columns with nulls and returns a dataframe with the columns that have nulls, their data types, the number of nulls, and the percent of the column that is null
def nulls_data(df): 

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    percent_null = (null_counts/len(df)) * 100
    null_table = pd.DataFrame({'column': null_counts.index, 'data type': df[null_counts.index].dtypes, 'number of nulls': null_counts.values, 
                               'precent null': percent_null.values})
    null_table = null_table.sort_values(by='precent null', ascending=False).reset_index(drop=True)

    return null_table

# this functions goes through each column and prints the head and description. It requests input for the types of data and and notes on intial impressions and thoughts

def review_columns(df): 

    concepts_of_a_plan = pd.DataFrame(columns=['column', 'datatype','type', 'description_notes'])

    for col in df.columns: 

        clear_output(wait=True)
        print('='*20)
        print(f'Reviewing {col}')
        print('='*20)

        print(f'\n{'__'*20}\nhead: {col}\n{'__'*20}')
        print(df[col].head())
        print(f'\n{'__'*20}\ndescription:{col}\n{'__'*20}')
        print(df[col].describe())
        print('\n \n')
        print(f'number of unique vales {col}: {df[col].nunique()}')

        type_input =input(f'enter type for "{col}":')
        description_notes_input =input(f'description notes "{col}":')

        add_row = pd.DataFrame({'column': [col], 'datatype': df[col].dtype,'type': [type_input], 'description_notes': [description_notes_input]})
        concepts_of_a_plan = pd.concat([concepts_of_a_plan, add_row])


    return concepts_of_a_plan


def column_plots(df1, df2, nodc, nrows, ncolumns, pltype, width, height): 

    fig = make_subplots(
        rows=nrows, cols=ncolumns, 
        subplot_titles=[f'{col}' for col in df1['column']],
        horizontal_spacing=0.15, vertical_spacing=0.15,
        specs=[[{'type': pltype}] * ncolumns for _ in range(nrows)] 
    )
    
    row, col = 1, 1
    for index, row_data in df1.iterrows():
        column_name = row_data['column']

        if nodc == 'n' or nodc == 'o': 

            bar_fig = px.bar(df2.groupby(column_name).size().reset_index(name='count').sort_values('count', ascending=False),
                             x=column_name, y='count', text_auto='.1s', title=column_name)

            for trace in bar_fig.data:
                fig.add_trace(trace, row=row, col=col)

        elif nodc == 'c': 

            kde = gaussian_kde(df2[column_name], bw_method=0.1) 
            x_range = np.linspace(df2[column_name].min() - 1, df2[column_name].max() + 1, 10000)
            kde_values = kde(x_range)
            kde_values_rescaled = kde_values * len(df2[column_name])

            fig.add_trace(go.Histogram(x=df2[column_name], histnorm='density', nbinsx=1000, name='Histogram', opacity=0.7, marker=dict(color='lightseagreen')), row=row, col=col)
            fig.add_trace(go.Scatter(x=x_range, y=kde_values_rescaled, mode='lines', name='KDE', line=dict(color='blue')), row=row, col=col)

            skew_score = skew(df2[column_name])
            direction_skew = None
            
            if skew_score < 0: 

                direction_skew = 'left'

            else: 

                direction_skew = 'right'


            text = f'{column_name} skewed {direction_skew}: {round(skew_score, 2)}'

            fig.add_annotation(
                x=0.5, y=-0.25, 
                xref="x domain", yref="y domain",  
                text=text,
                showarrow=False,
                font=dict(size=12),
                align="center",
                borderpad=10, 
                row=row, col=col
            )


        elif nodc == 'd': 

            # Box plot
            box_fig = px.box(df2, y=column_name, title=f'Box plot for {column_name}')
            for trace in box_fig.data:
                fig.add_trace(trace, row=row, col=col)
                
            q1 = df2[column_name].describe()['25%']
            q3 = df2[column_name].describe()['75%']
            iqr = q3 - q1
            outliers = q3 + 1.5 * iqr
            ratio_outliers = (len(df2[df2[column_name] > outliers]) / len(df2))*100

            text = f'{column_name} IQR: {outliers}, ratio of outliers: {round(ratio_outliers, 2)}'
            

            fig.add_annotation(
                x=0.5, y=-0.1,  
                xref="x domain", yref="y domain",  
                text=text,
                showarrow=False,
                font=dict(size=12),
                align="center",
                borderpad=10, 
                row=row, col=col
            )


        col += 1
        if col > ncolumns:
            col = 1
            row += 1
    

    fig.update_layout(
        showlegend=True,
        template='plotly_white',
        width=width, 
        height=height,
        margin=dict(t=50, b=300, l=50, r=50), 
    )

    fig.show()



def outliers(df, column): 

    q1 = df[column].describe()['25%']
    q3 = df[column].describe()['75%']
    iqr = q3 - q1

    outliers = q3 + 1.5 * iqr

    df[column] = np.where(df[column] > outliers, outliers, df[column])