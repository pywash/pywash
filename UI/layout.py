import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import math
from app import app


def layout_main(datasets):
    # TODO Move logos to top of screen
    scaling_factor_images = 0.3
    return html.Div([
        html.H2("Welcome to the Pywash browser interface"),
        html.H4("This is a work in progress bachelor final project"),
        html.H4("You can start working by selecting or drag and drop a file below"),
        dcc.Upload(
            id='upload-data',
            multiple=True,
            children=[html.Button('Upload File')]),
        merge_component(datasets),
        html.Img(src=app.get_asset_url('TUe.png'),
                 style={'width': 1173 * scaling_factor_images,
                        'height': 320 * scaling_factor_images}),
        html.Img(src=app.get_asset_url('JADS.jpg'),
                 style={'width': 850 * scaling_factor_images,
                        'height': 280 * scaling_factor_images}),
        html.Img(src=app.get_asset_url('TiU.png'),
                 style={'width': 1312 * scaling_factor_images,
                        'height': 333 * scaling_factor_images}),
        html.H6('Powered by: Technical University Eindhoven, '
                'Jheronimus Academy of Data Science and Tilburg University')
    ])


def layout_bandC(datasets):
    return html.Div([
        upload_component(),
        merge_component(datasets),
    ])


def upload_component():
    return html.Div([
        html.H5("Upload a new dataset"),
        dcc.Markdown("You can **click and select** a dataset locally\n"
                     "or **drag and drop** the file in the box underneath"),
        dcc.Upload(
            id='upload-data',
            multiple=True,
            children=[html.Button('Upload File', title='Load a dataset from your local drive')]),
        html.P("You can also load an online dataset by copying the link below"),
        dcc.Textarea(id='upload-url',
                     placeholder='Paste an online url...',
                     style={'width': '100%'}),
        html.Button('Submit url',
                    id='upload-url-submit',
                    title='Load the dataset from the internet'),
    ])


def merge_component(datasets):
    dataset_names = [dataset['props']['value'] for dataset in datasets
                     if dataset['props']['value'] != 'main']
    return html.Div([
        html.H5('Dataset merger'),
        html.Div([
            html.Div([
                dcc.Dropdown(id='dropdown-merger-1',
                             options=[{'label': x, 'value': x} for x in dataset_names],
                             multi=False,
                             placeholder='Select a dataset to merge'),
                dcc.Checklist(id='checklist-merger-1',
                              options=[],
                              values=[])
            ],
                style={'width': '40%'},
                className='six columns'
            ),
            html.Div([
                dcc.Dropdown(id='dropdown-merger-2',
                             options=[{'label': x, 'value': x} for x in dataset_names],
                             multi=False,
                             placeholder='Select a dataset to merge'),
                dcc.Checklist(id='checklist-merger-2',
                              options=[],
                              values=[])
            ],
                style={'width': '40%'},
                className='six columns'),
        ],
            className='row'),
        html.Button('Submit', id='button-merge'),
    ])


def band_tabs(df, all_datasets, locked):
    if df is None:
        types = None
    else:
        types = df.dtypes
    return [dcc.Tabs(id="tabs-cleaning", value='BandC', children=[
        dcc.Tab(id='bandC_tab', label='Band C', value='BandC',
                children=layout_bandC(all_datasets),
                style={
                    'backgroundColor': '#ffcece',
                },
                selected_style={
                    'fontWeight': 'bold',
                    'backgroundColor': '#ff8080',
                    'borderTop': '3px solid #e80d0d',
                }
                ),
        dcc.Tab(id='BandB_tab', label='Band B', value='BandB',
                children=layout_bandB(pd.DataFrame(types, columns=['d_type'])),
                disabled=locked, style={
                'backgroundColor': '#FFFFD1',
            },
                selected_style={
                    'fontWeight': 'bold',
                    'backgroundColor': '#fff79a',
                    'borderTop': '3px solid #ffdd00',
                }),
        dcc.Tab(id='BandS_tab', label='Band A', value='BandA',
                children=layout_bandA(), disabled=locked,
                style={
                    'backgroundColor': '#C3FFBA',
                },
                selected_style={
                    'fontWeight': 'bold',
                    'backgroundColor': '#7FFF81',
                    'borderTop': '3px solid #00A900',
                }
                ),
        dcc.Tab(id='plotstab', label='Plots', value='Plots',
                children=layout_plots(), disabled=locked,
                selected_style={
                    'fontWeight': 'bold',
                    'borderTop': '3px solid #1975fa',
                }
                ),
    ])]


def data_table(filename: str, df):
    return html.Div([
        html.H5(filename),
        dcc.Store(id='memory-output'),
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
            style_cell={'textAlign': 'right', "padding": "5px"},
            style_table={'overflowX': 'auto'},
            style_cell_conditional=[{'if': {'row_index': 'odd'},
                                     'backgroundColor': 'rgb(248, 248, 248)'}],
            style_header={'backgroundColor': '#C2DFFF',
                          'font-size': 'large',
                          'text-align': 'center'},
            style_filter={'backgroundColor': '#DCDCDC',
                          'font-size': 'large'},
        ),
    ])


def export_component():
    return html.Div([
        html.A(html.Button('Download current data', id='download-button'), id='download-link',
               download="cleandata.csv",
               href="",
               target="_blank"),
        dcc.Dropdown(id='download-type',
                     placeholder='Select export file type',
                     multi=False,
                     style={'width': '50%'},
                     options=[{'label': 'CSV', 'value': 'csv'},
                              {'label': 'Arff', 'value': 'arff'}]),
        html.Div(id='datatable-interactivity-container')
    ])


def DATA_DIV(filename, df, all_local_datasets):
    """
    Contains the total layout for the tabs
    In order: Info & Quality of dataset, The band tabs, a data table, some export functions
    Note: When the 'add dataset' tab is selected, only the tabs can be shown and only band C

    :param filename: String name of the dataset
    :param df: Pandas dataframe of the selected dataset (None if on 'add dataset' tab)
    :param all_local_datasets: Dataset object containing all loaded datasets
    :return: The total layout for the selected main tab
    """
    if df is None:
        data = None
        info = None
        export = None
    else:
        data = data_table(filename=filename, df=df)
        info = html.Div([
            html.Div('Current Data Quality: {}'.format('B'), style={'color': 'green', 'fontSize': 20}),
            html.Div('Rows: {} Columns: {}'.format(len(df.index), len(df.columns)), id='data-info')
        ], id='data-quality',
            style={'marginBottom': 25, 'marginTop': 25})
        export = export_component()
    return html.Div([
        html.Div([], id='dummy'),
        info,
        html.Div(id='cleaning-tabs-container',
                 children=band_tabs(df, all_local_datasets, filename == 'main')),
        data,
        export
    ], style={'rowCount': 2, 'width': "100%", 'margin-left': 'auto', 'margin-right': 'auto'})


def layout_bandA():
    return html.Div([
        dcc.Markdown('''###### Outlier Detection'''),
        dcc.Markdown('''Contamination'''),
        dcc.Input(
            id='contamination',
            type='number',
            value=0.1,
            min=0.001,
            max=0.5,
            step=0.001,
            placeholder='i.e. 0.1'
        ),
        html.Button('Estimate contamination', id='submit_contamination'),
        dcc.Markdown('''Algorithm'''),
        dcc.Dropdown(
            options=[
                {'label': 'Recommended', 'value': 'a'},
                {'label': 'Full', 'value': 'b'},
            ],
            placeholder="Select a preset",
            id='outlier_preset',
            style={'width': "50%", 'marginBottom': 5, 'marginTop': 5}
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
            placeholder="Select at least 2 to use LSCP",
            id='outlier_custom_setting',
            style={'width': "75%", 'marginBottom': 5, 'marginTop': 5}
        ),
        html.Button('Detect outliers!', id='submit_outlier'),
        dcc.Markdown('''###### Feature Scaling'''),
        dcc.RadioItems(
            options=[
                {'label': 'normalize', 'value': 'normalize'},
                {'label': 'standardize', 'value': 'standardize'},
            ],
            id='scale_setting',
            value='normalize',
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Dropdown(
            multi=True,
            placeholder="Select columns to scale",
            id='scale_selection',
            style={'width': "50%", 'marginBottom': 5, 'marginTop': 5}
        ),
        dcc.Input(
            id='scale_range',
            placeholder='Range (i.e. "0,1")',
            type='text',
            value=''
        ),
        html.Button('Scale!', id='submit_scale'),
    ], style={'width': "50%", 'marginBottom': 10, 'marginTop': 10})


def layout_bandB(columntypes):
    dtypes = [item for sublist in columntypes.values for item in sublist]
    dtypes = [str(i) for i in dtypes]
    columntypes['d_type'] = dtypes
    columntypes = columntypes.transpose()
    pandas_types = ['object', 'float64', 'int64', 'bool', 'category', 'datetime64[ns]']
    return html.Div([
        dcc.Markdown('''###### Data types'''),
        dash_table.DataTable(
            id='table-dropdown',
            data=columntypes.to_dict('records'),
            columns=[
                {"name": i, "id": i, 'clearable': False, 'presentation': 'dropdown'} for i in columntypes.columns
            ],
            editable=True,
            column_static_dropdown=[
                {"id": i, 'dropdown': [{'label': j, 'value': j} for j in pandas_types]} for i in
                columntypes.columns],
            style_cell={'textAlign': 'center'},
        ),
        dcc.Markdown('''###### Missing values'''),
        html.Div(id='missing-status'),
        dcc.Dropdown(
            id='dropdown-missing',
            options=[
                {'label': 'N/A', 'value': 'N/A'},
                {'label': 'NA', 'value': 'NA'},
                {'label': '?', 'value': '?'},
            ],
            multi=True,
            value=['N/A', 'NA', '?'],
            style={'width': "50%", 'marginBottom': 5, 'marginTop': 5}

        ),
        html.Button('Add Option', id='add-missing'),
        dcc.Input(id='input-missing', value='', placeholder="Add extra character"),
        dcc.RadioItems(
            options=[
                {'label': 'mcar', 'value': 'mcar'},
                {'label': 'mar', 'value': 'mar'},
                {'label': 'mnar', 'value': 'mnar'},
                {'label': 'remove', 'value': 'remove'}
            ],
            id='missing_setting',
            value='mar',
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Fix missing values!', id='submit_missing'),
    ], style={'marginBottom': 10, 'marginTop': 10, 'overflowX': 'auto', 'overflowY': 'hidden'})


def layout_plots():
    return html.Div([
        html.Button('Boxplot', id='boxplot'),
        html.Button('Stacked bar chart', id='cat_distribution'),
        dcc.Dropdown(
            placeholder="Select columns to plot",
            id='plot-selection',
            style={'width': "50%", 'marginBottom': 5, 'marginTop': 5, }
        ),
        html.Button('distribution', id='distribution'),
        html.Button('Parallel coordinates', id='par_coords'),
        dcc.Loading(id='loading-1', children=[html.Div([], id='graph')]),
    ])


def layout_boxplot(data):
    return dcc.Graph(
        figure={
            'data': data,
            'layout': go.Layout(
                xaxis={
                    'type': 'category',
                }
            )
        }
        , id='graphic')


def layout_distriplot(data):
    return dcc.Graph(
        figure={
            'data': data,
            'layout': go.Layout(
                barmode='stack',
                xaxis={
                    'type': 'category',
                }
            )
        }, id='graphic')


def layout_histoplot(data, selected_column):
    return dcc.Graph(figure=ff.create_distplot([data], [selected_column], bin_size=[1 + 3.322 * math.log(len(data))]),
                     id='graphic')


def layout_parcoordsplot(data):
    return dcc.Graph(
        figure={
            'data': data,
        }, id='graphic')
