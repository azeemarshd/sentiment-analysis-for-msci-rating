import pandas as pd
import random
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import textwrap
import plotly.io as pio
import numpy as np
from dash.dependencies import State
import plotly.io as pio
import os

# Function to add <br> tags without breaking words
def wrap_text(text, width=100):
    return '<br>'.join(textwrap.wrap(text, width=width, break_long_words=False))

# Load all the provided CSV files
nyon_2022 = pd.read_csv('./data/combined_data_v1/nyon_2022.csv')
nyon_2023 = pd.read_csv('./data/combined_data_v1/nyon_2023.csv')
rolle_2022 = pd.read_csv('./data/combined_data_v1/rolle_2022.csv')
rolle_2023 = pd.read_csv('./data/combined_data_v1/rolle_2023.csv')
vevey_2022 = pd.read_csv('./data/combined_data_v1/vevey_2022.csv')
vevey_2023 = pd.read_csv('./data/combined_data_v1/vevey_2023.csv')

# Add a municipality column to each dataframe
nyon_2022['city'] = 'Nyon'
nyon_2023['city'] = 'Nyon'
rolle_2022['city'] = 'Rolle'
rolle_2023['city'] = 'Rolle'
vevey_2022['city'] = 'Vevey'
vevey_2023['city'] = 'Vevey'

# Add a year column to each dataframe
nyon_2022['year'] = 2022
nyon_2023['year'] = 2023
rolle_2022['year'] = 2022
rolle_2023['year'] = 2023
vevey_2022['year'] = 2022
vevey_2023['year'] = 2023

# Combine all dataframes into one and add the necessary columns
combined_data = pd.concat([nyon_2022, nyon_2023, rolle_2022, rolle_2023, vevey_2022, vevey_2023], ignore_index=True)

# Convert the date column to datetime
combined_data['date'] = pd.to_datetime(combined_data['date'])

combined_data['hover_text'] = combined_data.apply(
    lambda row: f"ESG CLASS: {row['esg_predictor'].upper()}<br>" + wrap_text(row['text'], width=100),
    axis=1
)
# add jitter to the sentiment_pred column
jitter_strength = 0.1
combined_data['sentiment_pred_jittered'] = combined_data['sentiment_pred'] + np.random.uniform(-jitter_strength, jitter_strength, combined_data.shape[0])



# Define colors for each ESG category
category_colors = {
    'gouvernance': '#1f77b4',  # Dark Blue
    'social': '#2ca02c',       # Dark Green
    'environnemental': '#d62728',  # Dark Red
    'non-esg': '#9467bd'       # Dark Purple
}

city_colors = {
    'Nyon': '#ff4646',  # red
    'Rolle': '#65d881',  # green
    'Vevey': '#6fafef'   # blue
}

# Function to generate random colors
def generate_random_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    # Title centered over the entire layout
    html.Div([
        html.H1("Sentiment Analysis by City and ESG Category", 
                style={'textAlign': 'center',
                       'color': '#4A4A4A',
                       'font-family': 'Arial, sans-serif',
                       'font-size':24,
                       'margin-bottom': '10px'})
    ]),

    # Dropdowns centered
    html.Div([
        dcc.Dropdown(
            id='esg-category-dropdown',
            options=[{'label': cat, 'value': cat} for cat in combined_data['esg_predictor'].unique()],
            value='gouvernance',
            multi=False,
            style={'width': '300px', 'margin-right': '10px'}
        ),
        dcc.Dropdown(
            id='date-dropdown',
            options=[{'label': str(month), 'value': str(month)} for month in combined_data['date'].dt.to_period('M').unique()] +
                    [{'label': '2022', 'value': '2022'}, {'label': '2023', 'value': '2023'}],
            value=str(combined_data['date'].dt.to_period('M').unique()[0]),
            multi=False,
            style={'width': '300px', 'margin-left': '10px'}
        )
    ], style={'textAlign': 'center', 'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'center'}),

    # Layout for text area and graph side by side
    html.Div([
        # Text area and clipboard on the left, centered vertically
        html.Div([
            dcc.Textarea(
                id="textarea_id",
                value="",
                style={"height": 100, "width": "300px", "margin": "0px 100px 20px -50px"}
            ),
            dcc.Clipboard(
                target_id="textarea_id",
                title="Copy",
                style={
                    "display": "inline-block",
                    "fontSize": 20,
                    "verticalAlign": "top",
                    "margin": "0px 100px 20px -50px",
                    "textAlign": "center",
                },
            ),
        ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'}),

        # Graph on the right, shifted towards the left
        html.Div([
            dcc.Graph(id='box-plot', style={'margin-left': '-175px', 'margin-right': '0px', 'width': '95%'})
        ], style={'flex': '2'})
        
    ], style={'display': 'flex', 'justify-content': 'center'})
])


@app.callback(
    Output('box-plot', 'figure'),
    [Input('esg-category-dropdown', 'value'),
     Input('date-dropdown', 'value')]
)
def update_graph(selected_category, selected_date):
    if selected_date in ['2022', '2023']:
        filtered_data = combined_data[
            (combined_data['esg_predictor'] == selected_category) &
            (combined_data['year'] == int(selected_date))
        ]
    else:
        filtered_data = combined_data[
            (combined_data['esg_predictor'] == selected_category) &
            (combined_data['date'].dt.to_period('M') == selected_date)
        ]
    
    fig = go.Figure()
    
    for city in filtered_data['city'].unique():
        city_data = filtered_data[filtered_data['city'] == city]
        fig.add_trace(go.Box(
            y=city_data['sentiment_pred_jittered'],
            x=city_data['city'],
            name=city,
            marker_color=city_colors[city],
            boxpoints='all',  # shows all points
            hovertemplate='%{customdata}<extra></extra>',
            customdata=city_data['hover_text'],
            marker=dict(
                line=dict(
                    color='black',  # outline color
                    width=1  # outline width
                )
            )
        ))
    
    fig.update_layout(
        title={
            'text': f'Sentiment Rating for {selected_category} in {selected_date}',
            'y':0.9,  # Slightly move the title upwards
            'x':0.5,  # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Rating',
        xaxis_title='City',
        width=1000,
        height=600
    )
    
    # Save the figure to an HTML file
    pio.write_html(fig, file='saved_dashboard.html', auto_open=False)
    
    return fig


# @app.callback(
#     Output('textarea_id', 'value'),
#     [Input('box-plot', 'hoverData')],
#     [State('textarea_id', 'value')]
# )
# def display_hover_text(hoverData, current_text):
#     if hoverData is None:
#         return current_text
#     point_data = hoverData['points'][0]
#     hover_text = point_data['customdata']
#     return hover_text

@app.callback(
    Output('textarea_id', 'value'),
    [Input('box-plot', 'clickData')],
    [State('textarea_id', 'value')]
)
def update_textarea(clickData, current_text):
    if clickData is None:
        return current_text
    point_data = clickData['points'][0]
    hover_text = point_data['customdata']
    return hover_text

if __name__ == '__main__':
    app.run_server(debug=True)
