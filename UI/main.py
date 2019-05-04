import urllib

import dash
from dash.dependencies import Input, Output, State
import dash_table
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import base64
import io
import datetime
from src.BandA.Normalization import normalize
from UI import utils
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div(
    [
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-data-upload'),
    ])


# TODO use own parser @yuri
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    global df
    global df_updated

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        dcc.Store(id='memory-output'),
        dcc.Dropdown(
            options=[
                {'label': 'Fast', 'value': 'a'},
                {'label': 'Regular', 'value': 'b'},
                {'label': 'Full', 'value': 'c'},
            ],
            placeholder="Select a preset",
            id='outlier_preset',
            style={'width': "30%"}
        ),
        dcc.Dropdown(
            options=[
                {'label': 'Isolation Forest', 'value': 0},
                {'label': 'Cluster-based Local Outlier Factor', 'value': 1},
                {'label': 'Minimum Covariance Determinant (MCD)', 'value': 2},
                {'label': 'Principal Component Analysis (PCA)', 'value': 3},
                {'label': 'Angle-based Outlier Detector (ABOD)', 'value': 4},
                {'label': 'Histogram-base Outlier Detection (HBOS)', 'value': 5},
                {'label': 'K Nearest Neighbors (KNN)', 'value': 6},
                {'label': 'Local Outlier Factor (LOF)', 'value': 7},
                {'label': 'Feature Bagging', 'value': 8},
                {'label': 'One-class SVM (OCSVM)', 'value': 9},
            ],
            multi=True,
            placeholder="Select at least 2",
            id='outlier_custom_setting',
            style={'width': "50%"}
        ),
        html.Button('Detect outliers!', id='submit_outlier'),
        dcc.Dropdown(
            multi=True,
            placeholder="Select columns to normalize",
            id='normalize_selection',
            style={'width': "50%"}
        ),
        dcc.Input(
            id='normalize_range',
            placeholder='Range (i.e. "0,1")',
            type='text',
            value=''
        ),
        html.Button('Normalize!', id='submit_normalize'),
        dash_table.DataTable(
            id='datatable',
            columns=[
                {"name": i, "id": i, "deletable": True} for i in df.columns
            ],
            data=df.to_dict('records'),
            editable=True,
            filtering=True,
            sorting=True,
            sorting_type="multi",
            row_selectable="multi",
            row_deletable=True,
            selected_rows=[],
            pagination_mode="fe",
            pagination_settings={
                "displayed_pages": 1,
                "current_page": 0,
                "page_size": 50,
            },
            navigation="page",
        ),
        html.A(html.Button('Download current data', id='download-button'), id='download-link',
               download="cleandata.csv",
               href="",
               target="_blank"),
        html.Div(id='datatable-interactivity-container')
    ], style={'rowCount': 2, 'width': "85%", 'margin-left': 'auto', 'margin-right': 'auto'}
    )


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_table(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output('memory-output', 'data'),
    [Input('submit_outlier', 'n_clicks'),
     Input('submit_normalize', 'n_clicks')],
    state=[State('outlier_custom_setting', 'value'),
           State('normalize_selection', 'value'),
           State('normalize_range', 'value'),
           State('datatable', 'derived_virtual_data'),
           ])
def process_input(outlier_submit, normalize_submit, outlier_setting, normalize_selection,
                  normalize_range, data):
    # TODO use current derived_virtual_data
    df = pd.DataFrame(data)
    ctx = dash.callback_context
    button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_clicked == 'submit_outlier':
        if outlier_setting is not None:
            df_updated = utils.handle_outlier_dash(df, outlier_setting)
            return df_updated.to_dict("records")
    if button_clicked == 'submit_normalize' is not None and normalize_range is not None and normalize_submit is not None:
        df_updated = normalize(df, normalize_selection, tuple(int(i) for i in normalize_range.split(',')))
        return df_updated.to_dict("records")


@app.callback(Output('datatable', 'data'),
              [Input('memory-output', 'data')])
def update_datatable(data):
    if data is None:
        raise PreventUpdate

    return data


@app.callback(Output('datatable', 'columns'),
              [Input('datatable', 'data')])
def update_columns(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    return [
        {"name": i, "id": i, "deletable": True} for i in df.columns
    ]


@app.callback(Output('datatable', 'style_data_conditional'),
              [Input('datatable', 'data')])
def update_datatable_styling(columns):
    if columns is None:
        raise PreventUpdate
    return [
        {
            'if': {
                'column_id': 'probability',
                'filter': 'probability > num(0.9999995)'
            },
            'backgroundColor': '#a3524e',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'prediction',
                'filter': 'prediction eq num(1)'
            },
            'backgroundColor': '#8b0000',
            'color': 'white',
        }
    ]


@app.callback(
    Output('download-link', 'href'),
    [Input('datatable', 'derived_virtual_data'),
     Input('download-button', 'n_clicks')])
def update_download_link(data, n_clicks):
    if n_clicks is not None:
        df_download = pd.DataFrame(data)
        return "data:text/csv;charset=utf-8," + \
               urllib.parse.quote(df_download.to_csv(index=False, encoding='utf-8'))


@app.callback(Output('normalize_selection', 'options'),
              [Input('datatable', 'data')])
def on_data_set_table(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    eligible_features = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        {"label": i, "value": i} for i in eligible_features
    ]


@app.callback(Output('outlier_custom_setting', 'value'),
              [Input('outlier_preset', 'value')])
def on_data_set_table(value):
    if value == 'a':
        return [0, 1, 2, 3]
    if value == 'b':
        return [0, 1, 2, 3, 4, 5, 6, 7]
    if value == 'c':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


if __name__ == '__main__':
    app.run_server(debug=True)
