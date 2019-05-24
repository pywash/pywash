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

from sklearn.preprocessing import StandardScaler

from UI.layout import *
from src.BandA.Normalization import normalize
from src.BandB.DataTypes import discover_types
from src.BandB.MissingValues import handle_missing
from src.SharedDataFrame import SharedDataFrame
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
        html.Div(id='tabs_container', children=[dcc.Tabs(id='tabs')]),
        html.Div(id='output-data-upload'),
    ])


class DataSet:
    """ Class to keep track of all uploaded datasets """

    # TODO, move this class
    def __init__(self):
        self.datasets = {}

    def add_dataset(self, filename, sdf: SharedDataFrame):
        self.datasets.update({filename: sdf})

    def get_datasets(self):
        return self.datasets

    def get_dataset(self, filename):
        return self.datasets.get(filename)


UI_data = DataSet()


@app.callback(Output('tabs_container', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('tabs', 'children')])
def upload_data(contents: list, filenames: list, dates: list, current_tabs: list):
    if filenames is None:
        if current_tabs is None:
            created_tabs = [dcc.Tab(id='main', label='Main', value='main')]
        else:
            created_tabs = current_tabs
        return dcc.Tabs(id='tabs', children=created_tabs)
    if filenames is not None:
        print("loading datasets: " + str(filenames))
        # Load the datasets into the Dataset object for storage
        for i in range(len(filenames)):
            print(contents)
            new_dataset = SharedDataFrame(file_path=filenames[i],
                                          contents=contents.pop(),
                                          verbose=True)
            UI_data.add_dataset(new_dataset.name, new_dataset)
            filenames[i] = new_dataset.name
        # Add filenames to the tabs
        created_tabs = [dcc.Tab(label=name, value=name)
                        for name in filenames]
        # TODO Use dataset name instead of filepath as header
        # TODO Test for duplicates and separate them
        current_tabs.extend(created_tabs)
        return dcc.Tabs(id='tabs', value='main', children=current_tabs)


'''
@app.callback(Output('cleaning-tabs-container', 'children'),
              [Input('tabs-cleaning', 'value')],)
def render_cleaningtab(tab):
    if tab == 'BandB':
        return layout_bandB()
    if tab == 'BandA':
        return layout_bandA()
    if tab == 'Plots':
        return layout_plots()

'''


@app.callback(Output('output-data-upload', 'children'),
              [Input('tabs', 'value')])
def render_data(tab):
    if UI_data.get_dataset(tab) is None:
        # TODO, CREATE MAIN PAGE
        return layout_main()
    else:
        return DATA_DIV(tab, UI_data.get_dataset(tab).get_dataframe())


@app.callback(
    Output('memory-output', 'data'),
    [Input('submit_outlier', 'n_clicks'),
     Input('submit_normalize', 'n_clicks'),
     Input('submit_missing', 'n_clicks'),
     Input('data-types', 'n_clicks')],
    [State('outlier_custom_setting', 'value'),
     State('normalize_selection', 'value'),
     State('missing_setting', 'value'),
     State('dropdown-missing', 'value'),
     State('normalize_range', 'value'),
     State('datatable', 'derived_virtual_data'),
     ])
def process_input(outlier_submit, normalize_submit, missing_submit, datatypes_submit, outlier_setting,
                  normalize_selection,
                  missing_setting, missing_navalues, normalize_range, data):
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
    if button_clicked == 'submit_missing' is not None:
        df_updated = handle_missing(df, missing_setting, missing_navalues)
        return df_updated.to_dict("records")
    if button_clicked == 'data-types' is not None:
        print(df.dtypes)
        print(df.infer_objects().dtypes)
        df_updated = discover_types(df)
        print(df_updated)
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
def preset_outliers(value):
    if value == 'a':
        return [0, 1, 2, 3]
    if value == 'b':
        return [0, 1, 2, 3, 4, 5, 6, 7]
    if value == 'c':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@app.callback(
    Output('dropdown-missing', 'options'),
    [Input('add-missing', 'n_clicks')],
    [State('input-missing', 'value'),
     State('dropdown-missing', 'options')],
)
def add_missing_character(click, new_value, current_options):
    ctx = dash.callback_context
    button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_clicked == 'add-missing':
        current_options.append({'label': new_value, 'value': new_value})
        return current_options


@app.callback(Output('plotstab', 'children'),
              [Input('boxplot', 'n_clicks'),
               Input('distribution', 'n_clicks')],
              [State('datatable', 'derived_virtual_data'),
               State('boxplot-setting', 'value')]
              )
def plots(boxplot_click, distri_click, data, setting):
    ctx = dash.callback_context
    button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_clicked == 'boxplot':
        df_ = pd.DataFrame(data)
        df_ = df_.select_dtypes(include=[np.number])

        if setting == 'scaled':
            ss = StandardScaler()
            df_ = ss.fit_transform(df_)
            df_ = pd.DataFrame(df_, columns=df_.columns)
        data = []
        for i in df_.columns:
            data.append(go.Box(
                y=df_[i],
                name=i
            ))

        return layout_boxplot(data)

    if button_clicked == 'distribution':
        df_ = pd.DataFrame(data)
        # df_ = df_.select_dtypes(include=['category'])
        # print(df_)
        df_ = df_.apply(pd.value_counts)
        data = []
        for i in range(df_.shape[0]):
            trace_temp = go.Bar(
                x=np.asarray(df_.columns),
                y=df_.values[i],
                name=df_.index[i]
            )
            data.append(trace_temp)

        return layout_distriplot(data)


if __name__ == '__main__':
    app.run_server(debug=True)
