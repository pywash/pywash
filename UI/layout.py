import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd

from UI.main import app


def layout_main(datasets):
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


def pref_merge_component(datasets):
    return html.Div([
        html.H5("Dataset merger:"),
        html.Div([
            dcc.Dropdown(
                id='dropdown-merging',
                options=[{'label': x, 'value': x} for x in datasets.get_names()],
                multi=True,
                placeholder='Select 2 or more datasets to merge',
                style={'width': "50%"}
            ),
            html.Button('Submit', id='button-merge')
        ])
    ])


def merge_component(datasets):
    return html.Div([
        html.H5('Dataset merger:'),
        html.Div([
            html.Div([
                dcc.Dropdown(id='dropdown-merger-1',
                             options=[{'label': x, 'value': x} for x in datasets.get_names()],
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
                             options=[{'label': x, 'value': x} for x in datasets.get_names()],
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


def DATA_DIV(filename, df):
    return html.Div([
        html.Div([
            html.Div('Current Data Quality: {}'.format('B'), style={'color': 'green', 'fontSize': 20}),
            html.Div('Rows: {} Columns: {}'.format(len(df.index), len(df.columns)))
        ], id='data-quality',
            style={'marginBottom': 25, 'marginTop': 25}),
        html.Div(id='cleaning-tabs-container', children=[
            dcc.Tabs(id="tabs-cleaning", value='BandB', children=[
                dcc.Tab(id='BandA_tab', label='BandB', value='BandB',
                        children=layout_bandB(pd.DataFrame(df.dtypes, columns=['d_type']))),
                dcc.Tab(id='BandB_tab', label='BandA', value='BandA', children=layout_bandA()),
                dcc.Tab(id='plotstab', label='Plots', value='Plots', children=layout_plots()),
            ]),
        ]),
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
        ),
        html.A(html.Button('Download current data', id='download-button'), id='download-link',
               download="cleandata.csv",
               href="",
               target="_blank"),
        html.Div(id='datatable-interactivity-container')
    ], style={'rowCount': 2, 'width': "85%", 'margin-left': 'auto', 'margin-right': 'auto'}
    )


def layout_bandA():
    return html.Div([
        dcc.Markdown('''###### Outlier Detection'''),
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
        dcc.Markdown('''###### Normalization'''),
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
    ], style={'width': "50%", 'marginBottom': 10, 'marginTop': 10})


def layout_bandB(columntypes):
    dtypes = [item for sublist in columntypes.values for item in sublist]
    dtypes = [str(i) for i in dtypes]
    columntypes['d_type'] = dtypes
    columntypes = columntypes.transpose()
    pandas_types = ['object', 'float64', 'int64', 'bool', 'category', 'datetime64']
    return html.Div([
        dcc.Markdown('''###### Missing values'''),
        html.Div(id='missing-status'),
        dcc.Dropdown(
            id='dropdown-missing',
            options=[
                {'label': 'n/a', 'value': 'n/a'},
                {'label': 'na', 'value': 'na'},
                {'label': '--', 'value': '--'},
                {'label': '?', 'value': '?'},
            ],
            multi=True,
            value=['n/a', 'na', '--', '?'],
            style={'width': "50%"}

        ),
        dcc.Input(id='input-missing', value='', placeholder="Add extra character"),
        dcc.RadioItems(
            options=[
                {'label': 'mcar', 'value': 'mcar'},
                {'label': 'mar', 'value': 'mar'},
                {'label': 'mnar', 'value': 'mnar'}
            ],
            id='missing_setting',
            value='mar',
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Fix missing values!', id='submit_missing'),
        html.Button('Add Option', id='add-missing'),
        dcc.Markdown('''###### Data types'''),
        dash_table.DataTable(
            id='table-dropdown',
            data=columntypes.to_dict('records'),
            columns=[
                {"name": i, "id": i, 'presentation': 'dropdown'} for i in columntypes.columns
            ],
            editable=True,
            column_static_dropdown=[{"id": i, 'dropdown': [{'label': j, 'value': j} for j in pandas_types]} for i in
                                    columntypes.columns]
        ),
        html.Button('Infer Data Types!', id='data-types'),
    ], style={'width': "50%", 'marginBottom': 10, 'marginTop': 10})


def layout_plots():
    return html.Div([
        html.Button('Boxplot', id='boxplot'),
        html.Button('Categorical distribution', id='cat_distribution'),
        dcc.Dropdown(
            placeholder="Select columns to plot",
            id='plot-selection',
            style={'width': "50%"}
        ),
        html.Button('distribution', id='distribution')
    ])


def layout_boxplot(data):
    return html.Div([
        html.Button('Boxplot', id='boxplot'),
        html.Button('Categorical distribution', id='cat_distribution'),
        dcc.Dropdown(
            placeholder="Select columns to plot",
            id='plot-selection',
            style={'width': "50%"}
        ),
        html.Button('distribution', id='distribution'),
        dcc.Graph(
            figure={
                'data': data,
                'layout': go.Layout(
                    xaxis={
                        'type': 'category',
                    }
                )
            })
    ])


def layout_distriplot(data):
    return html.Div([
        html.Button('Boxplot', id='boxplot'),
        html.Button('Categorical distribution', id='cat_distribution'),
        dcc.Dropdown(
            placeholder="Select columns to plot",
            id='plot-selection',
            style={'width': "50%"}
        ),
        html.Button('distribution', id='distribution'),
        dcc.Graph(
            figure={
                'data': data,
                'layout': go.Layout(
                    barmode='stack',
                    xaxis={
                        'type': 'category',
                    }
                )
            })
    ])

def layout_histoplot(data, selected_column):
    return html.Div([
        html.Button('Boxplot', id='boxplot'),
        html.Button('Categorical distribution', id='cat_distribution'),
        dcc.Dropdown(
            placeholder="Select columns to plot",
            id='plot-selection',
            style={'width': "50%"}
        ),
        html.Button('distribution', id='distribution'),
        dcc.Graph(figure = ff.create_distplot([data[selected_column]], [selected_column]))
    ])

