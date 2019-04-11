# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import flask
from src import PyWash


app = dash.Dash()

app.layout = html.Div([
        dcc.Input(id='input', value='Initial text', type='text'),   # Component for input
        html.Div(id='output_div'),                                  # Div for output
])


"""
Callbacks

The callbacks is what makes the visualization tool dynamic instead of static.
Without these, the website would be a static picture.

Every callback can only have 1 output and 
every 'id' (object from the layout above) can only be an output in 1 callback.
"""
@app.callback(
    Output('output_div', 'children'),
    [Input('input', 'value')]
)
def update_text(new_text):
    return 'You entered "{}"'.format(new_text)


# The host= '0.0.0.0' is used for the tool to be run on the local network.
# With [YOUR_IP]:8050 other people can access the tool if they are on the same network.
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')