# Load libraries
from dash import Dash, dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
import plotly.graph_objs as go
import re
#from functools import partial
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, viridis, inferno
#from flask_caching import Cache  # caching doesn't work in heroku
import tempfile
import warnings

# Ignore warnings when loading holoview libraries:
#The dash_core_components package is deprecated. Please replace
#`import dash_core_components as dcc` with `from dash import dcc`
#  import dash_core_components as dcc
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import holoviews as hv
    from holoviews.plotting.plotly.dash import to_dash
    from holoviews.operation.datashader import datashade, dynspread

# Init vars
tree_url = 'https://data.cityofnewyork.us/resource/uvpi-gqnh.json'

# Define helpers for DataShader
#background = "black"
#export = partial(export_image, background = background, export_path="export")
#cm = partial(colormap_select, reverse=(background!="black"))
#NewYorkCity = ((913164.0, 1067279.0), (120966.0, 272275.0))
#cvs = ds.Canvas(700, 700, *NewYorkCity)

# Define colour key for bivariate choropleth (see https://observablehq.com/@benjaminadk/bivariate-choropleth-color-generator)
color_key = {
    '1_Poor___0_None': '#e8e8e8',
    '2_Fair___0_None': '#bddede', 
    '3_Good___0_None': '#8ed4d4', 
    '1_Poor___1or2': '#dabdd4', 
    '2_Fair___1or2': '#bdbdd4', 
    '3_Good___1or2': '#8ebdd4', 
    '1_Poor___3or4': '#cc92c1', 
    '2_Fair___3or4': '#bd92c1', 
    '3_Good___3or4': '#8e92c1',
    '1_Poor___4orMore': '#be64ac', 
    '2_Fair___4orMore': '#bd64ac', 
    '3_Good___4orMore': '#8e64ac'
}

# Set dash app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Get path to temp dir
tmpdir = tempfile.gettempdir() + '\\flask'

# Configure flash cache
#CACHE_CONFIG = {
#    'CACHE_TYPE': 'FileSystemCache',
#    'CACHE_DIR': tmpdir,
#    'CACHE_DEFAULT_TIMEOUT': 0
#}
#cache = Cache()
#cache.init_app(app.server, config=CACHE_CONFIG)

# Define a general function to read from the tree database using pagination
def load_trees(url):
    
    # Set offset to 0 to start with, keep limit at default of 1000
    print('Loading trees from server')
    limit = 1000
    offset = 0

    # Init empty dataframe that will be the return value of the function
    dfreturn = pd.DataFrame()

    # Retrieve records until there are none left
    while True:
        
        # Set new url using new offset
        if re.search('\\?', url):
            soql_url = url + '&$limit=' + str(limit) + '&$offset=' + str(offset)
        else:
            soql_url = url + '?$limit=' + str(limit) + '&$offset=' + str(offset)
        print('\toffest:' + str(offset))
        
        # Fetch from server
        dftmp = pd.read_json(soql_url)
        
        # See if there are any more records
        if dftmp.shape[0] == 0:
            break
            
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEMP - TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #if offset > 19000:
        #    break
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEMP - TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # If this is the first iteration, replace the initialized dataframe with this one;
        # otherwise concatenate it
        if offset == 0:
            dfreturn = dftmp
        else:
            dfreturn = pd.concat([dfreturn, dftmp])
            
        # Set new offset
        offset += dftmp.shape[0]

    # Return dataframe
    return dfreturn

# Define a function to load tree health data by borough and species
def load_tree_health(spc_common):
    
    # Build request
    soql_url = tree_url
    if spc_common == '[all species]':
        soql_url += "?$select=health,count(tree_id) as ct"
        soql_url += "&$group=health&$order=health"
    else:
        soql_url += "?$select=boroname,health,count(tree_id) as ct"
        soql_url += "&$where=spc_common='{}'".format(spc_common.replace("'", "''"))  # escape single quote with two single quotes
        soql_url += "&$group=boroname,health&$order=boroname,health"
    soql_url = soql_url.replace(' ', '%20')
    
    # Query server
    dftrees = load_trees(soql_url)
    
    # Manual factoring
    dftrees.loc[dftrees['health'] == 'Poor', 'health'] = '1_Poor'
    dftrees.loc[dftrees['health'] == 'Fair', 'health'] = '2_Fair'
    dftrees.loc[dftrees['health'] == 'Good', 'health'] = '3_Good'
    dftrees.sort_values(by=['health'], inplace=True)
    return dftrees

# Define a function to load tree health and stewardship data by borough, species
def load_tree_stewardship(spc_common):
    
    # Build request
    soql_url = tree_url
    if spc_common == '[all species]':
        soql_url += "?$select=health,steward,count(tree_id) as ct"
        soql_url += "&$group=steward,health&$order=steward,health"
    else:
        soql_url += "?$select=boroname,health,steward,count(tree_id) as ct"
        soql_url += "&$where=spc_common='{}'".format(spc_common.replace("'", "''"))  # escape single quote with two single quotes
        soql_url += "&$group=boroname,steward,health&$order=boroname,steward,health"
    soql_url = soql_url.replace(' ', '%20')

    # Query server
    dftrees = load_trees(soql_url)
    
    # Drop na values (steward field has some)
    dftress = dftrees.dropna()

    # Manual factoring
    dftrees.loc[dftrees['steward'] == 'None', 'steward'] = '0_None'
    dftrees.loc[dftrees['health'] == 'Poor', 'health'] = '1_Poor'
    dftrees.loc[dftrees['health'] == 'Fair', 'health'] = '2_Fair'
    dftrees.loc[dftrees['health'] == 'Good', 'health'] = '3_Good'
    dftrees.sort_values(by=['steward', 'health'], inplace=True)
    return dftrees

# Define a function to load tree geo data (state plane coords) by borough, species
def load_tree_geo(spc_common):
    
    # Build request
    soql_url = (tree_url + \
                "?$select=tree_id,boroname,spc_common,health,steward,x_sp,y_sp&$where=spc_common='{}'"
               ).format(spc_common.replace("'", "''")).replace(' ', '%20')  # escape single quote with two single quotes
                
    # Query server
    dftrees = load_trees(soql_url)

    # Drop na values (steward field has some)
    dftress = dftrees.dropna()
                
    # Manual factoring
    dftrees.loc[dftrees['steward'] == 'None', 'steward'] = '0_None'
    dftrees.loc[dftrees['steward'].isna(), 'steward'] = '0_Null'
    dftrees.loc[dftrees['health'] == 'Poor', 'health'] = '1_Poor'
    dftrees.loc[dftrees['health'] == 'Fair', 'health'] = '2_Fair'
    dftrees.loc[dftrees['health'] == 'Good', 'health'] = '3_Good'
    
    # Convert state-plane coords to ints
    dftrees.x_sp = dftrees.x_sp.astype(int)
    dftrees.y_sp = dftrees.y_sp.astype(int)
    
    # Split into bivariate color categories
    dftrees['bivar'] = dftrees['health'].str.cat(dftrees['steward'], sep='___').astype('category')
      
    return dftrees

# Generate list of unique species to populate drop-down
soql_url = (
    tree_url + \
    "?$select=spc_common,count(tree_id) as ct" + \
    "&$group=spc_common"
).replace(' ', '%20')
dfspecies = load_trees(soql_url)
dfspecies = dfspecies.dropna()
dfspecies = pd.concat([pd.DataFrame(['[all species]'], columns=['spc_common']), dfspecies])

# Set layout
app.layout = html.Div(
    [
        html.Table([
            html.Tr([
                html.Td([
                    dcc.Markdown(
                        '2015 New York City Tree Census', style={'font-weight': 'bold', 'font-size': '18px'}
                    ),
                    dcc.Markdown(
                        '', id='lbl-title', style={'font-weight': 'bold', 'font-size': '18px'}
                    )
                ], style={'font-size': '18px', 'border': 'none', 'padding': '1px', 'text-align': 'center', 'color': '#3030a0'})
            ], style={'border': 'none'}),
            html.Tr([
                html.Td([
                    html.Table([
                        html.Tr([
                            html.Td([
                                html.Label(['Species:'])
                            ], style={'font-weight': 'bold', 'font-size': '12px', 'border': 'none', 'padding': '0px'}),
                            html.Td([
                                html.Label([''])
                            ], style={'font-size': '12px', 'border': 'none', 'padding': '1px'}),
                            html.Td([
                                html.Label([''])
                            ], style={'font-weight': 'bold', 'font-size': '12px', 'border': 'none', 'padding': '0px'})
                        ], style={'border': 'none'}),
                        html.Tr([
                            html.Td([
                                dcc.Dropdown(
                                    dfspecies['spc_common'], 
                                    'American beech', 
                                    placeholder='Select a species', 
                                    id='dd_spc_common', 
                                    style={'width': '240px', 'font-size': '12px'}
                                )
                            ], style={'border': 'none', 'padding': '0px'}),
                            html.Td([
                                html.Label([''])
                            ], style={'font-size': '12px', 'border': 'none', 'padding': '1px'}),
                            html.Td([
                                dcc.Checklist(
                                    options=[{'label': 'Show map', 'value': 1}],
                                    id='chk_map', 
                                    inline=True,
                                    style={'width': '240px', 'font-size': '12px'}
                                )
                            ], style={'border': 'none', 'padding': '0px'})
                        ], style={'border': 'none', 'padding': '0px'})
                    ], style={'border-collapse': 'collapse', 'padding': '0px', 'border': 'none'})
                ], style={'border': 'none', 'padding': '0px'})
            ]),
            html.Tr([
                html.Td([
                    html.Table([
                        html.Tr([
                            html.Td([
                                dcc.Graph(id='graph1', style={'width': '800px'})
                            ], style={'border': 'none', 'padding': '0px'})
                        ], style={'border': 'none', 'padding': '0px'}),
                        html.Tr([
                            html.Td([
                                dcc.Graph(id='graph2', style={'width': '800px'})
                            ], style={'border': 'none', 'padding': '0px'})
                        ], style={'border': 'none', 'padding': '0px'})
                    ], style={'border-collapse': 'collapse', 'padding': '0px', 'border': 'none'})
                ], style={'border': 'none', 'padding': '0px'}),
                html.Td([
                    html.Div(id='graph3')
                ], rowSpan=2, style={'border': 'none', 'padding': '0px', 'vertical-align': 'top'})
            ], style={'border': 'none', 'padding': '0px'}),
        ], style={'border-collapse': 'collapse', 'padding': '0px', 'border': 'none'}),
        #dcc.Store(id='signal'),  # signal value to trigger callbacks
        html.Div(id='dummy-div')  # dummy div to signal app to load data
    ]
)

# Flash cache  # doesn't work on heroku
#@cache.memoize()
def global_store(func, spc_common):
    return func(spc_common)

"""
@app.callback(Output('signal', 'data'), Input('dummy-div', 'children'))
def populate_df(dummy_value):
    # compute value and send a signal when done
    global_store()
    return
"""

# Callback decorator
@app.callback(
    Output('lbl-title', 'children'),
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Input('dd_spc_common', 'value')
)

# Callback function to update figure
def update_small_figures(spc_common):
    
    # Check for empty dropdown values
    if spc_common is None:
        
        return {}
    
    else:
    
        # Load tree health data from cache
        dfplot1 = global_store(load_tree_health, spc_common)
        
        # Calculate proportions
        dfplot1['p'] = (np.round((dfplot1['ct'] / dfplot1['ct'].sum()) * 100, 1)).astype(str) + '%'
        
        # Set default graph title
        graph_title = spc_common

        # Display tree health bar plot
        if spc_common == '[all species]':
            fig1 = px.bar(dfplot1, x='health', y='ct', text='p', width=800, height=400)
            graph_title = 'All Species'
        else:
            fig1 = px.bar(dfplot1, x='health', y='ct', text='p', width=800, height=400, facet_col='boroname', facet_col_wrap=3)
            for lbl in fig1.layout.annotations:
                lbl.text = lbl.text.split("=")[1]  # Remove 'boroname=' from facet labels
        fig1.update_xaxes(mirror=False, showline=True, ticks='outside', linecolor='black', gridcolor='lightgrey', title='Tree Health')
        fig1.update_yaxes(mirror=False, showline=True, ticks='outside', linecolor='black', gridcolor='lightgrey', title='Trees')
        fig1.update_layout(title='', plot_bgcolor='white')

        # Load stewardship data from cache
        dfplot2 = global_store(load_tree_stewardship, spc_common)

        # 2D histogram
        if spc_common == '[all species]':
            fig2 = px.density_heatmap(dfplot2, x='health', y='steward', z='ct', nbinsx=30, width=800, height=400, \
                labels=dict(health='Tree Health', steward='Stewardship'))
        else:
            fig2 = px.density_heatmap(dfplot2, x='health', y='steward', z='ct', nbinsx=30, width=800, height=400, facet_col='boroname', facet_col_wrap=3, \
                labels=dict(health='Tree Health', steward='Stewardship'))
            for lbl in fig2.layout.annotations:
                lbl.text = lbl.text.split("=")[1]  # Remove 'boroname=' from facet labels
        fig2.update_layout(
            title='', 
            plot_bgcolor='white', 
            coloraxis_colorbar=dict(title='Trees'), 
            xaxis={'categoryarray': ['1_Poor', '2_Fair', '3_Good']},
            yaxis={'categoryarray': ['0_None', '1or2', '3or4', '4orMore']}
        )
        
        # Return
        return [graph_title, fig1, fig2]
    
# Callback decorator
@app.callback(
    Output('graph3', 'children'),
    Input('chk_map', 'value'),
    Input('dd_spc_common', 'value')
)

# Callback function to update figure
def update_datashader(chk_map, spc_common):

    # See if the "show map" checkbox is checked
    if not chk_map is None and 1 in chk_map:
        
        if spc_common == '[all species]':
            
            return "Please select a specific species to display the map"
            
        else:
        
            # Retrieve tree data
            dftrees = global_store(load_tree_geo, spc_common)
            #dftrees = load_tree_geo(spc_common)

            # Datashader - not working
            #agg = cvs.points(dftrees, 'x_sp', 'y_sp', ds.count_cat('bivar'))
            #view = tf.shade(agg, color_key=color_key, how='eq_hist')
            #fig3 = export(tf.spread(view, px=2), 'bivar')

            # Holoview scatter
            hvdataset = hv.Dataset(dftrees)
            scatter = datashade(
                hv.Scatter(
                    hvdataset,
                    kdims='x_sp',
                    vdims='y_sp'
                )
            )
            scatter = dynspread(datashade(scatter))
            scatter = scatter.opts(
                title=spc_common + " - all boroughs ({} trees)".format(len(hvdataset)),
                width=800, height=600,
                bgcolor='white', 
                show_grid=False, 
                xaxis=None,      # remove xaxis
                yaxis=None,      # remove yaxis
                labelled=[],     # remove axis labels
                xticks=[0, ''],  # remove tick marks
                yticks=[0, '']   # remove tick marks
            )  # , color=color_key['bivar']

            fig3 = to_dash(app, [scatter], reset_button=True)[0]

            # Return
            return fig3

    else:
        
        # Return empty figure
        return None

# Run dash app
if __name__ == "__main__":
    app.run_server(debug=True, processes=1, threaded=True)

