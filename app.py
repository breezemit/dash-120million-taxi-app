import json
import logging
import os

import dash
from dash.dependencies import Input, Output, State

from dash import dcc
from dash import html
from dash import dash_table

from flask_caching import Cache

import numpy as np

import pandas as pd
from pandas.core.frame import DataFrame
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

import pylab as plt
import types

import vaex



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('vaex-dash')

external_stylesheets = []
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # used by gunicorn in production mode
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
# set negative to disable(useful for testing/benchmarking)
CACHE_TIMEOUT = int(os.environ.get('DASH_CACHE_TIMEOUT', '60'))


# This has to do with layout/styling
fig_layout_defaults = dict(
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
)

# Markdown / descriptions (should be moved where it fits)
about_md = '''
### QAHub test

An example of an interactive dashboard created with [Vaex](https://github.com/vaexio/vaex) and
[Dash](https://plotly.com/dash/). 

Read [this article](link_placeholder) to learn how to create such dashboards with Vaex and Dash.
'''


data_summary_filtered_md = 'Selected trips'

def create_figure_empty():
    layout = go.Layout(plot_bgcolor='white', width=10, height=10,
                       xaxis=go.layout.XAxis(visible=False),
                       yaxis=go.layout.YAxis(visible=False))
    return go.Figure(layout=layout)


# Taken from https://dash.plotly.com/datatable/conditional-formatting
def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if(i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #96dbfa 0%,
                    #96dbfa {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles

#######################################
# file/path function
#######################################
def find_folder(dir_path):
    dir_list = list()
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file) 
            if os.path.isdir(file_path):
                dir_list.append(file)
    return dir_list
            
#######################################
# Figure/plotly function
#######################################

def create_figure_scatter(summary, original_df, xname, yname,):
    if original_df is None or xname is None or yname is None:
        logger.info('Figure: original_df or xname or yname is empty')
        df = px.data.stocks()
        fig_empty = px.line(df, x='GOOG', y="GOOG")
        return fig_empty
    # scatter 
    fig = go.Figure(data=go.Scattergl(
                    x = original_df[xname] ,
                    y = original_df[yname] ,
                    mode='markers'
                    )
        )
    fig.update_layout(
        title = {
            "text" : str(summary),
            "x" : 0.5,
            "y" : 0.9,
            "xanchor" : "center",
            "yanchor" : "top"
        }
    )
    return fig



def create_figure_histogram_percent(summary, source_df, xname):
    if source_df is None or xname is None:
        logger.info('Figure: source_df or xname  is empty')
        df = px.data.tips()
        fig_empty = px.histogram(df, 
                        x = "total_bill",
                        title = f"Histogram of demo",
                        labels = {"demo": "demo"},
                        opacity = 0.8,
                        nbins = 20, # 用于设置分箱，即柱状图数目。
                        # text_auto = True, # plotly 4.14 不支持，需要更新plotly后才可开启。
                        log_y = True
                        ) 
        return fig_empty
    fig_hist = px.histogram(source_df, 
                        x = xname,
                        title = f"Histogram of {xname}",
                        labels = {xname: xname},
                        opacity = 0.8,
                        nbins = 20, # 用于设置分箱，即柱状图数目。
                        # text_auto = True, # plotly 4.14 不支持，需要更新plotly后才可开启。
                        log_y = True
                        )                               
    
    h_xbins = fig_hist.full_figure_for_development().data[0].xbins  
    x_bins_num = round((h_xbins.end - h_xbins.start) / h_xbins.size) + 1
    x_bins = [h_xbins.start + i*h_xbins.size  for i in range(x_bins_num) ]
    segments = pd.cut(source_df[xname], x_bins, right=False)
    count = pd.value_counts(segments, sort=False)
    # heights, bins = np.histogram(source_df[xname], bins = x_bins_num )
    percent_x_start = (h_xbins.start + h_xbins.size/2)
    percent_x = [percent_x_start+i*h_xbins.size  for i in range(x_bins_num - 1) ]
    heights = count.tolist()
    percent = [i/sum(heights)*100 for i in heights]
    percent_dict = { "x" : percent_x, "percent" : percent}
    df_percent = DataFrame(percent_dict)
    fig_hist.add_traces(list(px.line(df_percent,x='x', y='percent').update_traces(mode='lines+markers', line={"dash": "dash", "color":"firebrick"}, yaxis="y3", name="percent").select_traces())).update_layout(yaxis3={"overlaying": "y", "side": "right"}, showlegend=False)
    fig_hist.update_layout(barmode='group', bargap=0.10,bargroupgap=0.0)
    fig_hist.update_layout(
        title = {
            "text" : str(summary),
            "x" : 0.5,
            "y" : 0.9,
            "xanchor" : "center",
            "yanchor" : "top"
        }
    )
    logger.debug(f"Figure: heights {heights}")
    logger.debug(f"Figure: x_bins_num {x_bins_num}")
    logger.debug(f"Figure: h_xbins {h_xbins}")
    logger.debug(f"Figure: fig_hist {fig_hist}")
    logger.debug(f"Figure: percent {percent}")
    return fig_hist


# ######################################
# Compute/dataframe functions
# ######################################

# ######################################
# Dash specific part
# ######################################

scatter_figure_summary_template_md = '''{}-2022.11.SP1-{}-{}-{}-{}'''
zone_summary_md = scatter_figure_summary_template_md.format("QAHub", "Flow", "Process", "Case", "Result")

# The app layout
app.layout = html.Div(className='app-body', children=[
    # Stores
    # dcc.Store(id='map_clicks', data=0),
    # dcc.Store(id='zone', data=zone_initial),
    # dcc.Store(id='trip_start', data=trip_start_initial),
    # dcc.Store(id='trip_end', data=trip_end_initial),
    # dcc.Store(id='heatmap_limits', data=heatmap_limits_initial),
    # About the app + logos
    html.Div(className="row", children=[
        html.Div(className='twelve columns', children=[
            html.Div(style={'float': 'left'}, children=[
                    html.H1('QAHub: Result exposed'),
                    html.H4(f'Exploring Big data in Real Time')
                ]
            ),
            html.Div(style={'float': 'right'}, children=[
                html.A(
                    html.Img(
                        src=app.get_asset_url("vaex-logo.png"),
                        style={'float': 'right', 'height': '35px', 'margin-top': '20px'}
                    ),
                    href="https://vaex.io/"),
                html.A(
                    html.Img(
                        src=app.get_asset_url("dash-logo.png"),
                        style={'float': 'right', 'height': '75px'}
                    ),
                    href="https://dash.plot.ly/")
            ]),
        ]),
    ]),
    # Control panel one
    html.Div(className="row", id='control-panel1', children=[
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select Product Line Name'),
            dcc.Dropdown(id='ProductName',
                         placeholder='Select product line',
                         options=[{'label': 'GloryEX', 'value': "GloryEX"},
                                  {'label': 'GloryBolt', 'value': 'GloryBolt'},
                                  {'label': 'GloryTime', 'value': 'GloryTime'},
                                  {'label': 'PhyBolt', 'value': 'PhyBolt'},
                                  {'label': 'GloryDB', 'value': 'GloryDB'},],
                         value=[],
                        #  multi=True
                         ),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select product type'),
            dcc.Dropdown(id='Type',
                         placeholder='Select a column of data for Y-axis',
                         options=[{'label': 'Release', 'value': 'Release'},
                                  {'label': 'Personal', 'value': 'Personal'},],
                         value=[],
                        #  multi=True
                         ),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select build number'),
            dcc.Dropdown(id='Number',
                         placeholder='Select a column of data for Y-axis',
                         value=[],
                        #  multi=True
                         ),
        ]),
    ]),
    # Control panel two
    html.Div(className="row", id='control-panel2', children=[
        html.Div(className="one-four columns pretty_container", children=[
            html.Label('Select Flow Name'),
            dcc.Dropdown(id='FlowName',
                         placeholder='Select product line',
                         value=[],
                        #  multi=True 用于开启混合选项
                         ),
        ]),
        html.Div(className="one-four columns pretty_container", children=[
            html.Label('Select Process'),
            dcc.Dropdown(id='Process',
                         placeholder='Select Process',
                         value=[],
                        #  multi=True
                         ),
        ]),
        html.Div(className="one-four columns pretty_container", children=[
            html.Label('Select Case Name'),
            dcc.Dropdown(id='CaseName',
                         placeholder='Select Case Name',
                         value=[],
                        #  multi=True
                         ),
        ]),
        html.Div(className="one-four columns pretty_container", children=[
            html.Label('Select Result Name'),
            dcc.Dropdown(id='ResultName',
                         placeholder='Select Result Name',
                         value=[],
                        #  multi=True
                         ),
        ]),
    ]),
    # Control panel one
    # html.Div(className="row", id='control-panel3', children=[
    #     html.Div(className="one-half columns pretty_container", children=[
    #         html.Label('Select X axis'),
    #         dcc.Dropdown(id='Xaxis',
    #                      placeholder='Select a column of data for X-axis',
    #                      options=[{'label': 'gb_em_ratio', 'value': "gb_em_ratio"},
    #                               {'label': 'rh_lines', 'value': 'rh_lines'},],
    #                      value=[],
    #                     #  multi=True
    #                      ),
    #     ]),
    #     html.Div(className="one-half columns pretty_container", children=[
    #         html.Label('Select Y axis'),
    #         dcc.Dropdown(id='Yaxis',
    #                      placeholder='Select a column of data for Y-axis',
    #                      options=[{'label': 'gb_em_ratio', 'value': "gb_em_ratio"},
    #                               {'label': 'rh_lines', 'value': 'rh_lines'},],
    #                      value=[],
    #                     #  multi=True
    #                      ),
    #     ]),
    # ]),


    # The Visuals
    dcc.Tabs(id='tab', children=[
        dcc.Tab(id='figure_divs', label='Result Figure', children=[
            html.Div(className="row", children=[
                html.Div(className="six columns pretty_container", children=[
                    dcc.Markdown(id='figure_summary', children=zone_summary_md),
                    dcc.Graph(id='scatter_figure',
                              figure=create_figure_scatter("QAHub", None, None, None ),
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})
                ]),
                html.Div(className="six columns pretty_container", children=[
                    dcc.Graph(id='flow_sunburst_figure',
                              figure=create_figure_histogram_percent("QAHub", None, None)),
                ])
            ]),
            # html.Div(className="row", children=[
            #     html.Div(className="fix columns pretty_container", children=[
            #         dcc.Graph(id='flow_sankey_figure',
            #                   figure=figure_sankey_initial,
            #                   config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})
            #     ]),
            #     html.Div(className="fix columns pretty_container", children=[
            #         dash_table.DataTable(id='table', columns=[
            #                 {'name': 'Destination Borough', 'id': 'borough'},
            #                 {'name': 'Destination zone', 'id': 'zone'},
            #                 {'name': 'Number of trips', 'id': 'count_trips'},
            #             ],
            #             data=table_records_intitial,
            #             style_data_conditional=table_style_initial,
            #             style_as_list_view=True,
            #         )
            #     ]),
            # ]),
        ]),
        dcc.Tab(label='Result Table', children=[
            html.Div(className="row", children=[
                html.Div(className="seven columns pretty_container", children=[
                    dcc.Markdown(children='_Click on the map to select trip start and destination._'),
                ]),
                html.Div(className="five columns pretty_container", children=[
                            dcc.Markdown(id='trip_summary_md'),
                ])
            ]),
        ]),
    ]),
    html.Hr(),
    dcc.Markdown(children=about_md),

])


# Flow section

@app.callback(Output('Number', 'options'),
    [Input('ProductName', 'value'),
     Input('Type', 'value'),
     ],
    prevent_initial_call=True
)
def update_build_number(select_product, select_type):
    logger.info('Data: update ProductName=%r BuildType=%r', select_product, select_type)
    build_path = os.path.join("/opt/lixile/TestHub", select_product, select_type )
    build_list = find_folder(build_path)
    return [{'label': str(i), 'value': str(i)} for i in build_list]

@app.callback(Output('FlowName', 'options'),
    [Input('Number', 'value'),
     ],
    [State('ProductName', 'value'),
     State('Type', 'value'),
     ],
    prevent_initial_call=True
)
def update_flow_name(select_build_num, select_product, select_type, ):
    logger.info('Data: update ProductName=%r BuildType=%r BuildNum=%r', select_product, select_type, select_build_num)
    flow_path = os.path.join("/opt/lixile/TestHub", select_product, select_type, select_build_num)
    flow_list = find_folder(flow_path)
    return [{'label': str(i), 'value': str(i)} for i in flow_list]

@app.callback(Output('Process', 'options'),
    [Input('FlowName', 'value'),
     ],
    [State('ProductName', 'value'),
     State('Type', 'value'),
     State('Number', 'value'),
     ],
    prevent_initial_call=True
)
def update_flow_name(select_flow_name, select_product, select_type, select_build_num,):
    logger.info('Data: update ProductName=%r BuildType=%r BuildNum=%r FlowName=%r', select_product, select_type, select_build_num, select_flow_name)
    process_path = os.path.join("/opt/lixile/TestHub", select_product, select_type, select_build_num, select_flow_name)
    process_list = find_folder(process_path)
    return [{'label': str(i), 'value': str(i)} for i in process_list]

@app.callback(Output('CaseName', 'options'),
    [Input('Process', 'value'),
     ],
    [State('ProductName', 'value'),
     State('Type', 'value'),
     State('Number', 'value'),
     State('FlowName', 'value'),
     ],
    prevent_initial_call=True
)
def update_flow_name( select_process, select_product, select_type, select_build_num, select_flow_name,):
    logger.info('Data: update ProductName=%r BuildType=%r BuildNum=%r FlowName=%r Process=%r', select_product, select_type, select_build_num, select_flow_name, select_process)
    process_path = os.path.join("/opt/lixile/TestHub", select_product, select_type, select_build_num, select_flow_name, select_process)
    process_list = find_folder(process_path)
    return [{'label': str(i), 'value': str(i)} for i in process_list]

@app.callback(Output('ResultName', 'options'),
    [Input('CaseName', 'value'),
     ],
    [State('ProductName', 'value'),
     State('Type', 'value'),
     State('Number', 'value'),
     State('FlowName', 'value'),
     State('Process', 'value'),
     ],
    prevent_initial_call=True
)
def update_flow_name( select_case, select_product, select_type, select_build_num, select_flow_name, select_process):
    logger.info('Data: update ProductName=%r BuildType=%r BuildNum=%r FlowName=%r Process=%r CaseName=%r', select_product, select_type, select_build_num, select_flow_name, select_process, select_case)
    result_path = os.path.join("/opt/lixile/TestHub", select_product, select_type, select_build_num, select_flow_name, select_process, select_case)
    result_list = list()
    if os.path.exists(result_path):
        for file in os.listdir(result_path):
            file_path = os.path.join(result_path, file) 
            if os.path.isfile(file_path) and "hdf" in file:
                result_list.append(file)
    return [{'label': str(i).split(".")[0], 'value': str(i)} for i in result_list]

@app.callback(Output('figure_summary', 'children'),
              Output('figure_divs', 'children'),
    [Input('ResultName', 'value'),
     ],
    [State('ProductName', 'value'),
     State('Type', 'value'),
     State('Number', 'value'),
     State('FlowName', 'value'),
     State('Process', 'value'),
     State('CaseName', 'value'),
     ],
    prevent_initial_call=True
)
def update_flow_name( select_result, select_product, select_type, select_build_num, select_flow_name, select_process,select_case):
    logger.info('Data: update ResultName=%r', select_result)
    figures_summary_md = scatter_figure_summary_template_md.format(select_product, select_flow_name,select_process, select_case, select_result.split(".")[0])
    df_file_path = "/opt/lixile/TestHub/" + ("/".join([select_product, select_type, select_build_num, select_flow_name, select_process, select_case, select_result]))
    logger.debug('Data: DataFarme file=%r', df_file_path)
    if os.path.islink(df_file_path):
        df_file_path = os.readlink(df_file_path)
    logger.debug('Data: DataFarme real file=%r', df_file_path)
    if os.path.isfile(df_file_path) and ".hdf5" in df_file_path:
        # df_file = vaex.open(df_file_path).to_pandas_df()
        df_file = vaex.open(df_file_path)
        # t = re.findall("^([a-zA-Z]*?)(?=_)", list(df_file.__iter__())[1])
        cloumn_names = df_file.get_column_names()
        logger.debug('Data: DataFarme get column names=%r', cloumn_names)
        split_key = cloumn_names[0].split("_")[0]
        cloumn_list = list()
        for name in cloumn_names:
            if split_key in name:
                cloumn_name = name.split(split_key + "_")[1]
                if cloumn_name not in cloumn_list:
                    cloumn_list.append(cloumn_name)
        cloumn_dict = dict()
        for cloumn_name in cloumn_list:
            cloumn_dict[cloumn_name] = list()
            for name in cloumn_names:
                if cloumn_name in name:
                    cloumn_dict[cloumn_name].append(name)
        result_cloumn_dict = dict()
        for k in cloumn_dict:
            if len(cloumn_dict[k]) >= 3:
                result_cloumn_dict[k] = cloumn_dict[k]
        logger.debug('Data: DataFarme get column to pigure names=%r', result_cloumn_dict)
        result_df_dict = dict()
        figure_divs = list()
        for k in result_cloumn_dict:
            result_df_dict[k]={"x":result_cloumn_dict[k][2], "y":result_cloumn_dict[k][0], "df": df_file}
            if result_df_dict[k]["df"].count() <= 50000:
                df_scatter = html.Div(className="six columns pretty_container", children=[
                        dcc.Graph(id=k + 'scatter_figure',
                                  figure=create_figure_scatter(figures_summary_md + "-" + k, result_df_dict[k]["df"].to_pandas_df(), result_df_dict[k]["x"], result_df_dict[k]["y"] ),
                                  config={"modeBarButtonsToRemove": ['lasso2d', 'select2d']})])
            else:
                image_dir = os.path.dirname(os.path.abspath(df_file_path))
                image_path = os.path.join(image_dir, figures_summary_md+ "-" + k +"-scatter.png")
                if not os.path.isfile(image_path):
                    fig = plt.figure(figsize=(8, 4))
                    fig.suptitle(figures_summary_md + "-" + k)
                    df_file.viz.scatter(result_df_dict[k]["x"], result_df_dict[k]["y"], length_check=False, s=1)
                    # fig.savefig(image_path, dpi=200, bbox_inches=0)
                    fig.savefig(image_path, bbox_inches=0)
                logger.debug('Pigure:  get server local pigure name=%r', image_path)
                import base64
                with open(image_path, 'rb') as f:
                    image = f.read()
                image_decode = 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')
                df_scatter = html.Div(className="six columns pretty_container", children=[html.Img(id=(k + 'scatter_local_figure'), src=image_decode),])
            df_histogram = html.Div(id=k + 'histogram_figure', className="six columns pretty_container", children=[
                        dcc.Graph(id='flow_sunburst_figure',figure=create_figure_histogram_percent(
                            figures_summary_md, result_df_dict[k]["df"].to_pandas_df(), result_df_dict[k]["x"],)),])
            figure_divs.append(html.Div(className="row", children=[df_scatter, df_histogram]))
                    
        
    # return figures_summary_md, cloumn_dict
    return figures_summary_md, figure_divs


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)
    # app.run_server(host="0.0.0.0",)
