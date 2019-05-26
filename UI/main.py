import urllib

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import sd_material_ui
from UI.layout import *
from src.BandA.Normalization import normalize
from src.BandB.DataTypes import discover_type_heuristic
from src.BandB.MissingValues import handle_missing
from src.SharedDataFrame import SharedDataFrame
from UI import utils
from UI.storage import DataSets
import numpy as np

UI_data = DataSets()
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
        # Section to store popups in client browser
        # TODO move Dialogs to layout file
        html.Div(children=[sd_material_ui.Dialog(id='welcome',
                                                 open=True,
                                                 modal=False,
                                                 children=[html.H1('Welcome to PyWash'),
                                                           html.P('Pywash is a very intersting tool.'),
                                                           html.P('I don\'t know what to write')],
                                                 ),
                           sd_material_ui.Dialog(id='merge-warning',
                                                 open=False,
                                                 modal=False,
                                                 children=[html.H1('Not enough datasets selected')],
                                                 # actions=[html.H3('OK'), html.H3('Nah man')]
                                                 )]),
        html.Div(id='tabs_container', children=[dcc.Tabs(id='tabs')]),
        html.Div(id='output-data-upload'),
    ])


@app.callback([Output('tabs_container', 'children'),
               Output('merge-warning', 'open')],
              [Input('button-merge', 'n_clicks'),
               Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('tabs', 'children'),
               State('dropdown-merging', 'value')])
def upload_data(n_clicks, contents: list,
                filenames: list, dates: list, current_tabs: list, merging_datasets: list):
    """ Callback to add/remove datasets from memory and update the tabs """

    def create_tab_interface(tabs: list, warning: bool = False):
        """ Creates the tabs object that the main function should return """
        return dcc.Tabs(id='tabs', value='main', children=tabs), warning

    ctx = dash.callback_context
    print(ctx.triggered)
    print(contents)
    last_event = ctx.triggered[0]['prop_id'].split('.')[0]
    if current_tabs is None:
        current_tabs = [dcc.Tab(id='main', label='Main', value='main')]
    if last_event == 'upload-data':
        # A dataset was uploaded
        if filenames is None:
            return create_tab_interface(current_tabs)
        if filenames is not None:
            print("loading datasets: " + str(filenames))
            # Load the datasets into the Dataset object for storage
            for i in range(len(filenames)):
                # Load all datasets one by one
                new_dataset = SharedDataFrame(file_path=filenames[i],
                                              contents=contents.pop(),
                                              verbose=True)
                # If a dataset is already loaded, load with an appended name
                name_appendix = 1
                while new_dataset.name in UI_data.keys():
                    new_dataset.name = new_dataset.name + '_{}'.format(name_appendix)
                    name_appendix += 1
                # Add the dataset to our storage system
                UI_data.add_dataset(new_dataset.name, new_dataset)
                filenames[i] = new_dataset.name
            # Add filenames to the tabs
            created_tabs = [dcc.Tab(label=name, value=name)
                            for name in filenames]
            current_tabs.extend(created_tabs)
            return create_tab_interface(current_tabs)

    elif last_event == 'button-merge':
        # Datasets were submitted to be merged
        # Test for mergeability, merge and remove left-overs
        if merging_datasets is None or len(merging_datasets) < 2:
            # TODO Develop warning a bit more
            return create_tab_interface(current_tabs, True)
        datasets = [UI_data.get_dataset(dataset) for dataset in merging_datasets]
        # Check which datasets can be merged
        # TODO, Datasets will now double merge
        for dataset in datasets:
            for sdf in datasets:
                if sdf == dataset:
                    continue
                if sdf.is_mergeable(dataset):
                    # Datasets can be merged, confirm and merge
                    # TODO, the ask and confirm part
                    # Merge datasets
                    merged_df = sdf.merge(dataset)
                    merged_sdf = SharedDataFrame(name=sdf.name + '+' + dataset.name, df=merged_df)
                    UI_data.add_dataset(merged_sdf.name, merged_sdf)
                    # Remove datasets from the tabs and add the merged dataset
                    # TODO Remove datasets from UI_data
                    current_tabs.remove(
                        {'props': {'children': None, 'label': sdf.name, 'value': sdf.name}, 'type': 'Tab',
                         'namespace': 'dash_core_components'})
                    current_tabs.append(dcc.Tab(label=merged_sdf.name, value=merged_sdf.name))
        return create_tab_interface(current_tabs)


@app.callback(Output('output-data-upload', 'children'),
              [Input('tabs', 'value')])
def render_data(tab):
    if UI_data.get_dataset(tab) is None:
        # TODO, CREATE MAIN PAGE
        return layout_main(UI_data)
    else:
        return DATA_DIV(tab, UI_data.get_dataset(tab).get_dataframe())


@app.callback(
    Output('memory-output', 'data'),
    [Input('submit_outlier', 'n_clicks'),
     Input('submit_normalize', 'n_clicks'),
     Input('submit_missing', 'n_clicks'),],
    [State('outlier_custom_setting', 'value'),
     State('normalize_selection', 'value'),
     State('missing_setting', 'value'),
     State('dropdown-missing', 'value'),
     State('normalize_range', 'value'),
     State('datatable', 'derived_virtual_data'),
     ])
def process_input(outlier_submit, normalize_submit, missing_submit, outlier_setting,
                  normalize_selection,
                  missing_setting, missing_navalues, normalize_range, data):
    print(data)
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

@app.callback(
    Output('table-dropdown', 'data'),
    [Input('data-types', 'n_clicks')],
    [State('datatable', 'derived_virtual_data')])
def infer_datatypes(datatypes_submit, data):
    ctx = dash.callback_context
    button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
    df = pd.DataFrame(data)
    if button_clicked == 'data-types' is not None:
        inferred_types = discover_type_heuristic(df)
        types_dict = {df.columns[i]: inferred_types[i] for i in range(0, len(df.columns))}
        return [types_dict]


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
                'filter': '{probability} > 0.9999995'
            },
            'backgroundColor': '#a3524e',
            'color': 'white',
        },
        {
            'if': {
                'column_id': 'prediction',
                'filter': '{prediction} eq 1'
            },
            'backgroundColor': '#8b0000',
            'color': 'white',
        },
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


@app.callback([Output('normalize_selection', 'options'),
               Output('plot-selection', 'options')],
              [Input('datatable', 'data')])
def on_data_set_table(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    eligible_features = df.select_dtypes(include=[np.number]).columns.tolist()
    eligible_features = [{"label": i, "value": i} for i in eligible_features]
    return eligible_features, eligible_features

@app.callback(Output('missing-status', 'children'),
              [Input('datatable', 'data')])
def missing_status(data):
    df = pd.DataFrame(data)
    if pd.isnull(df).values.any():
        return html.Div('Status: {}'.format('Missing data detected!'),
                        style={'color': 'red', 'fontSize': 15})
    else:
        return html.Div('Status: {}'.format('No missing data detected'),
                        style={'color': 'green', 'fontSize': 15})


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
               State('table-dropdown', 'derived_virtual_data'),
               State('plot-selection', 'value')]
              )
def plots(boxplot_click, distri_click, data, dtypes, selected_column):
    ctx = dash.callback_context
    button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
    df_ = pd.DataFrame(data).astype(dtypes[0])
    if button_clicked == 'boxplot':
        df_ = df_.select_dtypes(include=[np.number])

        data = []
        for i in df_.columns:
            data.append(go.Box(
                y=df_[i],
                name=i
            ))

        return layout_boxplot(data)

    if button_clicked == 'cat_distribution':
        df_ = df_.select_dtypes(include=['category'])
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

    if button_clicked == 'distribution':
        return layout_histoplot(df_, selected_column)



if __name__ == '__main__':
    app.run_server(debug=True)
