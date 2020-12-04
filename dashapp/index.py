import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


from app import app
from layouts import layoutHome,layout1, layout2
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' :
         return layoutHome
    elif pathname == '/page-1':
        return layout1
    elif pathname == '/page-2':
         return layout2
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)