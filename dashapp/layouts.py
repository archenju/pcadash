import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

import dash
import plotly.express as px
import plotly.graph_objects as go
import app

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

timesData = pd.read_csv("./data/timesData.csv")
df2016 = timesData[timesData.year == 2016]
df2016 = df2016.iloc[:50, :]

#Cleaning up the data:
df2016 = df2016.dropna()
df2016.world_rank = [each.replace('=','') for each in df2016.world_rank]
df2016.world_rank = pd.to_numeric(df2016.world_rank, errors='coerce')
df2016.num_students = [each.replace(',','') for each in df2016.num_students]
df2016.num_students = pd.to_numeric(df2016.num_students, errors='coerce')
df2016.international_students = [str(each).replace('%','') for each in df2016.international_students]
df2016['female_male_ratio'] = [str(each).split() for each in df2016.female_male_ratio]
df2016.female_male_ratio = [(float(each[0]) / float(each[2])) for each in df2016.female_male_ratio] 
df2016.female_male_ratio = pd.to_numeric(df2016.female_male_ratio, errors='coerce')

#Creating the PCA object:
features = ['teaching', 'international','research','citations','income',
         'total_score', 'num_students', 'student_staff_ratio', 'international_students', 'female_male_ratio']
xs = df2016.loc[:, features].values
xs = StandardScaler().fit_transform(xs)
n_components=10
pca = PCA(n_components=n_components)
components = pca.fit_transform(xs)
compsDf = pd.DataFrame(data = components, columns = 
                                ['principal component 1', 'principal component 2',
                                 'principal component 3', 'principal component 4',
                                'principal component 5', 'principal component 6',
                                'principal component 7', 'principal component 8',
                                'principal component 9', 'principal component 10'])

# First PCA figure:
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
figPCA1 = px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"})


screePCA0 = pca.explained_variance_ratio_*100
figPCA0 = go.Figure()
figPCA0.add_trace(go.Scatter(x = np.arange(len(screePCA0))+1, 
                            y = screePCA0.cumsum(), 
                            mode = 'lines+markers',
                            name = 'Inertie cumulée',
                            hovertemplate = "<b>Pourcentage :</b> %{y} %<br>"
                            + "<extra></extra>"))           
figPCA0.add_trace(go.Bar(x = np.arange(len(screePCA0))+1, 
                        y = screePCA0,
                        name = 'Inertie par composante',
                        hovertemplate = "<b>Composante :</b> %{x}<br>" 
                            + "<b>Pourcentage :</b> %{y} %<br>"
                            + "<extra></extra>"))
figPCA0.update_layout(title="Eboulis des valeurs propres",
                     xaxis_title="Composantes",
                     yaxis_title="Inertie (%)")


# Second PCA figure:
total_var = pca.explained_variance_ratio_.sum() * 100
labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = ''
figPCA2 = px.scatter_matrix(
    components,
    color=df2016['world_rank'],
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
figPCA2.update_traces(diagonal_visible=False)
#figPCA2.show()

# Third PCA figure:
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
figPCA3 = px.scatter(components, x=0, y=1, color=df2016['world_rank'])
for i, feature in enumerate(features):
    figPCA3.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    figPCA3.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
#figPCA3.show()


fig1 = px.scatter(df2016, 
                 x="world_rank",
                 y="teaching",
                 #size="citations", 
                 color="research", 
                 hover_name="university_name",
                 log_x=False, size_max=99)


xfig2 = df2016.university_name
fig2trace1 = {
  'x': xfig2,
  'y': df2016.citations,
  'name': 'citations',
  'type': 'bar'}
fig2trace2 = {
  'x': xfig2,
  'y': df2016.teaching,
  'name': 'teaching',
  'type': 'bar'}
data = [fig2trace1, fig2trace2];
layout = {
  'xaxis': {'title': ''},
  'barmode': 'relative',
  'title': 'Citations and teaching for the top 50 universities in 2016'}
fig2 = go.Figure(data = data, layout = layout)















layoutHome = html.Div([
    html.H3('HomePage'),
    dcc.Dropdown(id='app-home-dropdown'),  
    html.Div(id='app-home-display-value'),
    dcc.Link('Table des données TimesData', href='/page-1'),
    html.Br(),
    dcc.Link("Cas d'étude.", href='/page-2')
])

layout2 = html.Div([
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    html.H3('', style={'textAlign':'center'}),
    html.Br(),
    html.Div([dcc.Graph(id='figPCA1', figure=figPCA0)]),
    html.Br(),
    html.Br(),
    html.H3('', style={'textAlign':'center'}),
    html.Br(),
    html.Div([dcc.Graph(id='figPCA2', figure=figPCA2)]),
    html.Br(),
    html.Br(),
    html.H3('', style={'textAlign':'center'}),
    html.Br(),
    html.Div([dcc.Graph(id='figPCA3', figure=figPCA3)]),
    
    html.Br(),
    html.Div(html.Img(src=app.app.get_asset_url('img1.jpg'))),
    html.Div(html.Img(src=app.app.get_asset_url('img2.jpg'))),
    html.Div(html.Img(src=app.app.get_asset_url('img3.jpg'))),
    html.Div(id='page-2-content')
    
])


layout1 = html.Div([
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    html.H3('Universities by rank', style={'textAlign':'center'}),
    html.Br(),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df2016.columns],
        data=df2016.to_dict('records'),
        export_format='csv',
        fixed_rows={'headers': True},
        style_table={'overflowX': 'auto','overflowY': 'auto','maxHeight':'900px'},
        style_cell_conditional=[{'height': 'auto',
            'minWidth': '80px', 'width': '120px', 'maxWidth': '180px',
            'whiteSpace': 'normal','textAlign':'center'}
        ]

    ),
    html.Br(),
    html.H3('Universities by teaching and research', style={'textAlign':'center'}),
    html.Br(),
    html.Div([dcc.Graph(id='universities-fig1', figure=fig1)]),
    html.Br(),
    html.H3('', style={'textAlign':'center'}),
    html.Br(),
    html.Div([dcc.Graph(id='universities-fig2', figure=fig2)]),
    
    html.Div(id='page-1-content')
],)