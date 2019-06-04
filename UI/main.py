import urllib

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import sd_material_ui
from UI.layout import *
from src.SharedDataFrame import SharedDataFrame
from UI.storage import DataSets
import numpy as np

UI_data = DataSets()

app = dash.Dash(__name__, assets_folder='./assets')
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
        html.Div(children=[sd_material_ui.Dialog(id='pop-up',
                                                 open=False,
                                                 modal=False,
                                                 children=None,
                                                 ),
                           sd_material_ui.Dialog(id='modal-pop-up',
                                                 open=False,
                                                 modal=False,
                                                 children=None,
                                                 # actions=[html.H3('OK'), html.H3('Nah man')]
                                                 )]),
        html.Div(id='tabs_container', children=[dcc.Tabs(id='tabs')]),
        html.Div(id='output-data-upload'),
    ])


@app.callback(Output('pop-up', 'open'),
              [Input('pop-up', 'children')])
def open_popup(text) -> bool:
    """ Opens popup when it's text is updated """
    if text is not None:
        return True
    return False


@app.callback([Output('tabs_container', 'children'),
               Output('pop-up', 'children')],
              [Input('button-merge', 'n_clicks'),
               Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('tabs', 'children'),
               State('dropdown-merging', 'value')])
def upload_data(n_clicks, contents: list,
                filenames: list, dates: list, current_tabs: list, merging_datasets: list):
    """ Callback to add/remove datasets from memory and update the tabs """

    def create_tab_interface(tabs: list, warning=None):
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
                temp_name = new_dataset.name
                while temp_name in UI_data.keys():
                    temp_name = new_dataset.name + '_{}'.format(name_appendix)
                    name_appendix += 1
                new_dataset.name = temp_name
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
            return create_tab_interface(current_tabs, [html.H1('Not enough datasets selected'),
                                                       html.P('You must select at least 2 datasets to merge')])
        datasets = [UI_data.get_dataset(dataset) for dataset in merging_datasets]
        # Check which datasets can be merged
        for i in range(len(datasets)):
            for j in range(i, len(datasets)):
                if datasets[i] == datasets[j]:
                    continue
                if datasets[i].is_mergeable(datasets[j]):
                    # Datasets can be merged, confirm and merge
                    # TODO, the ask and confirm part
                    # Merge datasets
                    merged_df = datasets[i].merge(datasets[j])
                    merged_sdf = SharedDataFrame(name=datasets[i].name + '+' + datasets[j].name, df=merged_df)
                    UI_data.add_dataset(merged_sdf.name, merged_sdf)
                    # Remove datasets from the tabs and add the merged dataset
                    # TODO Remove datasets from UI_data
                    current_tabs.remove(
                        {'props': {'children': None,
                                   'label': datasets[i].name, 'value': datasets[i].name},
                         'type': 'Tab', 'namespace': 'dash_core_components'})
                    current_tabs.remove(
                        {'props': {'children': None,
                                   'label': datasets[j].name, 'value': datasets[j].name},
                         'type': 'Tab', 'namespace': 'dash_core_components'})
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
     Input('submit_missing', 'n_clicks'), ],
    [State('outlier_custom_setting', 'value'),
     State('normalize_selection', 'value'),
     State('missing_setting', 'value'),
     State('dropdown-missing', 'value'),
     State('normalize_range', 'value'),
     State('tabs', 'value')
     ])
def process_input(outlier_submit, normalize_submit, missing_submit, outlier_setting,
                  normalize_selection, missing_setting, missing_navalues, normalize_range, current_tab):
    sdf = UI_data.get_dataset(current_tab)
    ctx = dash.callback_context
    button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_clicked == 'submit_outlier':
        return sdf.outlier(outlier_setting).to_dict("records")
    if button_clicked == 'submit_normalize' and normalize_range is not None and normalize_selection is not None:
        return sdf.normalize(normalize_selection, normalize_range).to_dict("records")
    if button_clicked == 'submit_missing' is not None:
        return sdf.missing(missing_setting, missing_navalues).to_dict("records")


@app.callback(
    Output('table-dropdown', 'data'),
    [Input('table-dropdown', 'derived_virtual_data'),
     Input('datatable', 'data')],
    [State('tabs', 'value')])
def infer_datatypes(dtypes, updated_data, current_tab):
    sdf = UI_data.get_dataset(current_tab)
    current_dtypes = sdf.get_dtypes()
    if dtypes is None or dtypes == [current_dtypes]:
        raise PreventUpdate
    dtypes = dtypes[0]
    ctx = dash.callback_context
    last_callback = ctx.triggered[0]['prop_id'].split('.')[0]
    if last_callback == 'datatable':
        return [current_dtypes]
    if len(dtypes) != len(current_dtypes):
        dtypes = {k: dtypes[k] for k in dtypes.keys() & current_dtypes.keys()}
    sdf.update_dtypes(dtypes)
    return [sdf.get_dtypes()]



@app.callback(Output('datatable', 'data'),
              [Input('memory-output', 'data')])
def update_datatable(data):
    if data is None:
        raise PreventUpdate

    return data


@app.callback(Output('dummy', 'children'),
              [Input('datatable', 'derived_virtual_data')],
              [State('tabs', 'value')])
def update_sdf(data, current_tab):
    if pd.DataFrame(data).empty:
        raise PreventUpdate
    sdf = UI_data.get_dataset(current_tab)
    sdf.set_data(pd.DataFrame(data))
    pass


@app.callback([Output('datatable', 'columns'),
              Output('table-dropdown', 'columns')],
              [Input('datatable', 'data')])
def update_columns(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    columns = [{"name": i, "id": i, "deletable": True, } for i in df.columns]
    columns_dtypes = [{"name": i, "id": i, 'clearable': False, 'presentation': 'dropdown'} for i in df.columns]
    return columns, columns_dtypes  


@app.callback(Output('datatable', 'style_data_conditional'),
              [Input('datatable', 'derived_virtual_data')])
def update_datatable_styling(data):
    datastyle = [
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
        }]
    return datastyle


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
              [Input('datatable', 'columns')],
              [State('tabs', 'value')])
def on_data_set_table(data, current_tab):
    if data is None:
        raise PreventUpdate
    sdf = UI_data.get_dataset(current_tab)
    eligible_features = sdf.get_dataframe().select_dtypes(include=[np.number]).columns.tolist()
    eligible_features = [{"label": i, "value": i} for i in eligible_features]
    return eligible_features, eligible_features


@app.callback(Output('missing-status', 'children'),
              [Input('datatable', 'data')])
def missing_status(data):
    df = pd.DataFrame(data)
    df = df.replace(r'^\s*$', np.nan, regex=True)
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
        return list(range(4))
    if value == 'b':
        return list(range(8))
    if value == 'c':
        return list(range(10))


@app.callback(
    Output('dropdown-missing', 'options'),
    [Input('add-missing', 'n_clicks')],
    [State('input-missing', 'value'),
     State('dropdown-missing', 'options')],
)
def add_missing_character(click, new_value, current_options):
    try:
        ctx = dash.callback_context
        button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
    except IndexError:
        button_clicked = 'None'

    if button_clicked == 'add-missing':
        current_options.append({'label': new_value, 'value': new_value})
        return current_options
    return current_options


@app.callback(Output('graph', 'children'),
              [Input('boxplot', 'n_clicks'),
               Input('distribution', 'n_clicks'),
               Input('cat_distribution', 'n_clicks'),
               Input('par_coords', 'n_clicks')],
              [State('plot-selection', 'value'),
               State('tabs', 'value')])
def plots(boxplot_click, distri_click, cat_distri_click, par_clicks, selected_column, current_tab):
    ctx = dash.callback_context
    try:
        button_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
        sdf = UI_data.get_dataset(current_tab)
        df_ = sdf.get_dataframe()
    except IndexError:
        raise PreventUpdate

    if button_clicked == 'boxplot':
        df_ = df_.select_dtypes(include=[np.number])

        data = []
        for i in df_.columns:
            data.append(go.Box(
                y=df_[i],
                name=str(i),
                boxpoints='outliers',
            ))

        return layout_boxplot(data)

    if button_clicked == 'cat_distribution':
        df_ = df_.select_dtypes(include=['category'])
        try:
            df_ = df_.apply(pd.value_counts)
        except TypeError:
            df_ = df_.applymap(str)
            df_ = df_.apply(pd.value_counts)
        data = []
        for i in range(df_.shape[0]):
            trace_temp = go.Bar(
                x=np.asarray(df_.columns),
                y=df_.values[i],
                name=str(df_.index[i])
            )
            data.append(trace_temp)

        return layout_distriplot(data)

    if button_clicked == 'distribution':
        df_ = df_[selected_column].dropna()
        return layout_histoplot(df_, selected_column)

    if button_clicked == 'par_coords':
        df_ = df_.select_dtypes(include=[np.number])
        data = [go.Parcoords(
            line=dict(color=df_[str(selected_column)], colorscale='Rainbow', showscale=True),
            dimensions=[{'label': str(i), 'values': df_[i]} for i in df_.columns]
        )]
        return layout_parcoordsplot(data)


@app.callback(Output('datatable', 'selected_rows'),
              [Input('graphic', 'selectedData'),
               Input('graphic', 'clickData'), ])
def plots(selected_data, click_data):
    print(selected_data)
    print(click_data)
    try:
        selected_points = [i['pointNumber'] for i in click_data['points']]
        return selected_points
    except TypeError:
        pass
    try:
        selected_points = [i['pointNumber'] for i in selected_data['points']]
        return selected_points
    except TypeError:
        pass
    try:
        selected_points = [i['pointNumbers'] for i in selected_data['points']]
        flat_list = [item for sublist in selected_points for item in sublist]
        return flat_list
    except TypeError:
        pass


@app.callback(Output("loading-1", "value"), [Input('boxplot', 'n_clicks'),
                                             Input('distribution', 'n_clicks'),
                                             Input('cat_distribution', 'n_clicks'),
                                             Input('par_coords', 'n_clicks')])
def loading(click, click2, click3, click4):
    return time.sleep(1)


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
