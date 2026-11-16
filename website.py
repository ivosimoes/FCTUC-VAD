import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('vader_lexicon')

# Used to avoid raising warnings when replacing columns in dataframes
pd.options.mode.copy_on_write = True

# Listings
df_list = pd.read_csv('./data/listings_complete_final.csv', 
                      float_precision='high')

# Reviews
df_reviews = pd.read_csv('./data/reviews_complete_final_v2.csv', 
                         parse_dates=['date'], 
                         dtype={'comments': str})
df_reviews['month'] = df_reviews['date'].dt.to_period('M')  # Sparkcloud

# Reviews for events
df_reviews2 = pd.merge(df_reviews, df_list.rename(columns={'id': 'listing_id'}), 
                       on='listing_id', how='left')
df_reviews2.dropna(subset=['latitude', 'longitude'], inplace=True)

# Monuments
df_monuments = pd.read_csv('./data/monuments_final.csv')

# Concert halls
df_concert = pd.read_csv('./data/concert_halls_final.csv')

# Universities
df_uni = pd.read_csv('./data/universities.csv', float_precision='high')

# Transport Centers
df_transport = pd.read_csv('./data/transport_centers.csv', float_precision='high')

# Metro line shapes
df_mshapes = pd.read_csv('./data/metro_shapes_final.csv', float_precision='high')

# Metro stops
df_mstops = pd.read_csv('./data/metro_stops_final.csv', float_precision='high')

# Extract unique years and months for sliders
available_years = sorted(df_reviews['date'].dt.year.unique())
available_months = sorted(df_reviews['date'].dt.to_period('M').unique().to_timestamp())

# Define event dates
EVENTS = {
    "Champions League Final 2021": pd.Timestamp("2021-05-29"),
    "UEFA Nations League Final 2019": pd.Timestamp("2019-06-09"),
    "NOS Primavera Sound 2023": pd.Timestamp("2023-06-07"),
    "Porto Marathon 2024": pd.Timestamp("2024-11-01")
}

# Preprocessing function (for sparkcloud)
def process_text(text):
    stop_words = set(stopwords.words('english')).union({
        'airbnb', 'host', 'apartment', 'stay', 'stayed', 'place',
        'home', 'house', 'would', 'also', 'us', 'could', 'est', 'que', 'para', 'und', 'porto', 'nous', 'très', 'muy', 'muito',
        'die', 'ist', 'sehr', 'con', 'dans', 'wir', 'war', 'sin', 'una', 'tout', 'todo', 'nos', 'apartamento', 'com', 'casa', 'appartement',
        'les', 'une', 'des', 'pas', 'propre', 'avec', 'avons', 'logement', 'das', 'der', 'wohnung', 'man', 'mit', 'alles', 'hat', 'für',
        'uns', 'unterkunft', 'por', 'notre', 'ein', 'eine', 'auch', 'haben', 'den', 'las', 'von', 'auf', 'del', 'está', 'aber',
        'waren', 'sind', 'einen', 'dem', 'ich', 'sich', 'kann', 'nur', 'los', 'wie', 'uma', 'como', 'pero', 'foi', 'fue', 'oporto', 'qui', 'mais',
        'sur', 'een', 'het', 'par', 'vous', 'plus', 'était', 'lappartement', 'tiene', 'hay', 'dos', 'das', 'da', 'do', 'van'
    })
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return [w for w in words if w not in stop_words and len(w) > 2]

# Define consistent styling for all controls
control_label_style = {
    'fontWeight': 'bold',
    'marginBottom': '5px',
    'color': '#555'
}

control_container_style = {
    'backgroundColor': '#f8f9fa',
    'padding': '10px',
    'borderRadius': '5px',
    'border': '1px solid #eee',
    'marginBottom': '15px'
}

# Layout for first tab (overview)
layout1 = html.Div([
    # Top Half - Centered Histogram with its controls
    html.Div([
        html.H2("Histogram of Airbnb Reviews", style={"textAlign": "center", "marginBottom": "20px"}),
        # Controls Row - now properly centered with flexbox
        html.Div([
            html.Div([
                html.Label("Year Range", style=control_label_style),
                    dcc.RangeSlider(
                        min=available_years[0],
                        max=available_years[-1],
                        step=None,
                        value=[available_years[-2], available_years[-1]],
                        marks={str(year): str(year) for year in available_years},
                        id='year_slider',
                        className='uniform-slider'
                    ),
            ], style={'flex': '1 1 50%', 'marginRight': '15px', **control_container_style}),
            
            html.Div([
                html.Label("Room Type", style=control_label_style),
                    dcc.Dropdown(
                        options=['All', 'Hotel Rooms', 'Shared Rooms', 'Private Rooms', 'Entire Homes/Apartments'],
                        value='All',
                        id='room_type_dropdown',
                        className='uniform-dropdown'
                    ),
            ], style={'flex': '1 1 50%', 'marginLeft': '15px', **control_container_style}),
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'width': '100%',
            'margin': '0 auto 0px auto',
            'padding': '0px 30px 15px 30px' # Top, Right, Bottom, Left (Controls)
        }),
        
        # Histogram - centered with full width
        html.Div([
            dcc.Loading(
                id="loading-histogram",
                type="dot",  # Options: "default", "circle", "dot", "cube"
                children=dcc.Graph(id='reviews_histogram')
            )
        ], style={
            'width': '100%',
            'margin': '0 auto',
            'padding': '0px 30px 30px 30px'  # Top, Right, Bottom, Left
        }),
    ]),
    
    # Bottom Half - Split between Scattermap (left) and Sparkcloud (right)
    html.Div([
        # Left Side - Scattermap and its controls
        html.Div([
            html.H2("Map of Airbnb Listings", style={"textAlign": "center", "marginBottom": "20px"}),
            
            # Scattermap controls
            html.Div([
                html.Label("Listing Attribute", style=control_label_style),
                dcc.Dropdown(
                    options=['Number of Reviews', 'Reviews per Month', 'Room Type', 'Accommodates', 'Bathrooms', 'Beds', 
                            'Price', 'Rating', 'Cleanliness Rating', 'Checkin Rating', 'Communication Rating', 
                            'Location Rating', 'Value Rating'],
                    value='Number of Reviews',
                    id='listings_dropdown',
                    className='uniform-dropdown'
                ),
            ], style={**control_container_style, 
                      'padding': '12px 10px 12px 10px'}),
            
            html.Div([
                html.Label("Show Nearby Points of Interest", style=control_label_style),
                dcc.Checklist(
                    options=['Monuments', 'Concert Halls', 'University', 'Metro', 'Transport Centers'],
                    value=[],
                    inline=True,
                    id='marker_checklist',
                    className='uniform-checklist',
                    labelStyle={'marginRight': '15px'},
                    inputStyle={'marginRight': '5px'},
                ),
            ], style={**control_container_style, 
                      'padding': '17.5px 10px 17.5px 10px',
                      'marginBottom': '30px'}),

            html.Div(
                id='symbols-legend',
                style={
                    'border': '1px solid #eee',
                    'padding': '10px',
                    'marginBottom': '15px',
                    'backgroundColor': '#f8f9fa',
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '15px',
                    'borderRadius': '5px'
                },
                children=[
                    html.Div([
                        html.Img(src="assets/monument.svg", style={'width': '20px', 'verticalAlign': 'middle'}),
                        html.Span("Monuments", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Img(src="assets/music.svg", style={'width': '20px', 'verticalAlign': 'middle'}),
                        html.Span("Concert Halls", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Img(src="assets/college.svg", style={'width': '20px', 'verticalAlign': 'middle'}),
                        html.Span("University", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Img(src="assets/airport.svg", style={'width': '20px', 'verticalAlign': 'middle'}),
                        html.Span("Airport", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Img(src="assets/rail.svg", style={'width': '20px', 'verticalAlign': 'middle'}),
                        html.Span("Train Stations", style={'marginLeft': '5px'})
                    ]),
                    html.Div([
                        html.Img(src="assets/bus.svg", style={'width': '20px', 'verticalAlign': 'middle'}),
                        html.Span("Bus Station", style={'marginLeft': '5px'})
                    ])
                ]
            ),
            
            # Scattermap
            dcc.Graph(id='listings_scattermap', style={'marginTop': '15px'}),
        ], style={'width': '50%',
                  'padding': '0px 15px 0px 30px',  # Top, Right, Bottom, Left (Map + Options)
                  'display': 'inline-block', 
                  'verticalAlign': 'top',}),
        
        # Right Side - Sparkcloud and its controls
        html.Div([
            html.H2("SparkCloud of Airbnb Reviews", style={"textAlign": "center", "marginBottom": "20px"}),
            
            # Sparkcloud controls
            html.Div([
                html.Label("Time Interval (by month)", style=control_label_style),
                dcc.RangeSlider(
                    id='month-slider',
                    min=0,
                    max=len(available_months) - 1,
                    value=[0, len(available_months) - 1],
                    marks={0: available_months[0].strftime('%Y-%m'), len(available_months) - 1: available_months[-1].strftime('%Y-%m')},
                    step=1,
                    allowCross=False,
                    tooltip={"placement": "bottom", "always_visible": False},
                    className='uniform-slider'
                )
            ], style={**control_container_style}),
            
            html.Div([
                html.Label("Sentiment Range (compound score)", style=control_label_style),
                dcc.RangeSlider(
                    id='sentiment-slider',
                    min=-1,
                    max=1,
                    value=[-1, 1],
                    step=0.01,
                    marks={-1: "-1", 0: "0", 1: "1"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className='uniform-slider'
                )
            ], style={**control_container_style, 'marginBottom': '30px'}),
            
            html.Div(id='selected-dates', style={
                "textAlign": "center", 
                "fontWeight": "bold", 
                "marginBottom": "15px",
                "padding": "11px 10px 11px 10px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "5px"
            }),
            
            # Sparkcloud
            dcc.Loading(
                id="loading-sparkcloud",
                type="dot",  # Options: "default", "circle", "dot", "cube"
                children=dcc.Graph(id='sparkcloud-graph')
            ),
        ], style={'width': '50%',
                  'padding': '0px 30px 0px 15px',  # Top, Right, Bottom, Left (Sparkcloud + Options)
                  'display': 'inline-block',  
                  'verticalAlign': 'top'}),
    ], style={'width': '100%', 
              'display': 'flex'}),
    
    html.Footer(
        dbc.Container([
            html.P(
                "DEI-FCTUC MECD © 2025 Ivo Simões and António Marques · All rights reserved.",
                className="text-muted",
                style={"fontSize": "0.9rem", "marginBottom": "0"}
            )
            ], style={"marginLeft": "5%"}
        ), style={"marginTop": "20px", "padding": "10px 0"}  # Top, Right, Bottom, Left (Sparkcloud + Options)
    )
])

# Layout for tab 2 (events)
layout2 = html.Div([
    html.H2("Review Trends Around Major Events", style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div([
        html.Label("Select an event:"),
        dcc.Dropdown(
            id='event-dropdown',
            options=[{"label": name, "value": name} for name in EVENTS],
            placeholder="Choose an event",
            clearable=False,
            style={"width": "100%"}
        )
    ], style={"textAlign": "center", 
              "marginRight": "30px",
              "marginLeft": "30px",
              **control_container_style}),

    html.Div([
        html.Div([
            html.Label("Min. number of reviews (per listing):  "),
            #dcc.Dropdown(
            #    id='min-reviews-dropdown',
            #    options=[{"label": str(i), "value": i} for i in [1, 3, 5, 10, 15, 25, 50]],
            #   value=3,
            #    clearable=False,
            #    style={"width": "100%"}
            #)
            dcc.Input(
                id='min-reviews-dropdown',
                type='number',
                placeholder='Enter a number...',
                value=0,
                min=0,
                style={"marginLeft": "10px"}
            )
        ], style={'flex': '1 1 50%', 'marginRight': '15px', **control_container_style}),

        html.Div([
            html.Label("Min. number of beds:  "),
            #dcc.Dropdown(
            #    id='beds-dropdown',
            #    options=[{"label": str(i), "value": i} for i in range(1, 11)],
            #    value=1,
            #    clearable=False,
            #    style={"width": "100%"}
            #)
            dcc.Input(
                id='beds-dropdown',
                type='number',
                placeholder='Enter a number...',
                value=0,
                min=0,
                style={"marginLeft": "10px"}
            )
        ], style={'flex': '1 1 50%', 'marginLeft': '15px', **control_container_style}),
    ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'width': '100%',
            'margin': '0 auto 0px auto',
            'padding': '0px 30px 0px 30px' # Top, Right, Bottom, Left (Controls)
    }),

    html.Div([
        html.Label("Minimum Airbnb Rating:"),
        dcc.Slider(
            id='rating-slider',
            min=1,
            max=5,
            step=0.1,
            value=3,
            marks={i: f'{i}' for i in range(1, 6)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        )
    ], style={"textAlign": "center",
              "marginRight": "30px",
              "marginLeft": "30px",
              **control_container_style,
              "marginBottom": "30px"}),

    html.Div([
        dcc.Graph(id='event-review-graph'),
    ], style={
        'width': '100%',
        'margin': '0 auto',
        'padding': '0px 30px 30px 30px'  # Top, Right, Bottom, Left (Hist)
    }),
    
    html.Div([
        html.Label("Map Mode:"),
        dcc.RadioItems(
            id='map-mode-radio',
            options=[
                {'label': 'Bubble Map', 'value': 'both'},
                {'label': 'Heatmap', 'value': 'heat'}
            ],
            value='both',
            labelStyle={'display': 'inline-block', 'marginRight': '15px'},
            inputStyle={'marginRight': '5px'},
        )
    ], style={"textAlign": "center", 
              "marginRight": "30px",
              "marginLeft": "30px",
              **control_container_style}),

    html.Div([
        dcc.Graph(id='event-map'),
    ], style={
        'width': '100%',
        'margin': '0 auto',
        'padding': '0px 30px 0px 30px'  # Top, Right, Bottom, Left (Hist)
    }),
    
    html.Footer(
        dbc.Container([
            html.P(
                "DEI-FCTUC MECD © 2025 Ivo Simões and António Marques · All rights reserved.",
                className="text-muted",
                style={"fontSize": "0.9rem", "marginBottom": "0"}
            )
            ], style={"marginLeft": "5%"}
        ), style={"marginTop": "20px", "padding": "10px 0"}  # Top, Right, Bottom, Left (Sparkcloud + Options)
    )
])

# Initialize the app
app = Dash(name="Airbnb Porto",
           external_stylesheets=[dbc.themes.MORPH])

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/", id="tab1-link")),
            dbc.NavItem(dbc.NavLink("Events", href="/events", id="tab2-link")),
        ],
        brand="Airbnb Porto Analysis",
        color="primary",
        dark=True,
        className="mb-4"  # Margin bottom
    ),
    
    html.Div(id="page-content")  # This will hold layout1 or layout2
])

# Callback to switch layouts based on URL path
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/events":
        return layout2
    else:
        return layout1  # default or homepage

# Callback to update df_reviews histogram based on update function
@app.callback(
    Output('reviews_histogram', 'figure'),
    Input('year_slider', 'value'),
    Input('room_type_dropdown', 'value')
)
def update_reviews_hist(year, room_type):
    attr_dict = {
        "All": "All",
        "Hotel Rooms": "Hotel room",
        "Shared Rooms": "Shared room",
        "Private Rooms": "Private room",
        "Entire Homes/Apartments": "Entire home/apt"
    }
    attr = attr_dict[room_type]
    
    # Get df_reviews for a specific year
    filtered_df = df_reviews[df_reviews['date'].dt.year >= year[0]].copy()
    filtered_df = filtered_df[filtered_df['date'].dt.year <= year[1]]
    if attr != 'All':
        filtered_df = filtered_df[filtered_df['room_type'] == attr]
    
    # Create histogram for specified year
    fig = px.histogram(
        filtered_df,
        x='date',
        nbins=(year[1]-year[0])*12+12,  # 1 bin for each month
        color='room_type',
        hover_data=filtered_df.columns,
        text_auto=True,
        #title=f"Monthly Reviews for {room_type}"
    )
    
    # Improve formatting on the x-axis (date formatting)
    fig.update_xaxes(
        dtick='M1',
        tickformat='%b %Y'
    )
    
    # Improve graph layout
    fig.update_layout(
        transition_duration=500,
        xaxis_title_text='Month', # xaxis label
        yaxis_title_text='Review Count', # yaxis label
        bargap=0.05,
        legend=dict(
            title="Room Type"
        ),
        plot_bgcolor="#f8f9fa",   # very light grey-blue
        paper_bgcolor="#f8f9fa",
    )
    
    return fig

# Callback to update proximity scattermap based on update function
@app.callback(
    Output('listings_scattermap', 'figure'),
    Input('listings_dropdown', 'value'),
    Input('marker_checklist', 'value')
)
def update_listings(plot_var, plot_markers):
    attr_dict = {  # Decodes dropdown options to actual dataframe attributes
        "Number of Reviews": "number_of_reviews",
        "Reviews per Month": "reviews_per_month",
        "Room Type": "room_type",
        "Accommodates": "accommodates",
        "Bathrooms": "bathrooms",
        "Beds": "beds",
        "Price": "price",
        "Rating": "review_scores_rating",
        "Cleanliness Rating": "review_scores_cleanliness",
        "Checkin Rating": "review_scores_checkin", 
        "Communication Rating": "review_scores_communication",
        "Location Rating": "review_scores_location",
        "Value Rating": "review_scores_value"
    }
    
    # ['neighbourhood', 'number_of_reviews', 'price', 'rating', 'accomodates', 'beds', 'bathrooms']
    attr = attr_dict[plot_var]  # Get dataframe attribute from dropdown option
    
    # Filter out nan values and sort values for clearer plotting
    filtered_df = df_list.copy()
    filtered_df = filtered_df.dropna(subset=[attr]).sort_values(attr, ascending=False)
    na_df = df_list[df_list[attr].isna()].copy()
    
    # Data to be displayed by each data point
    hover_data = ['host_response_time', 'room_type', 'accommodates', 'bathrooms', 'beds', 
                  'price', 'number_of_reviews', 'review_scores_rating']
    
    # Get the highlighted attribute to the first position to be displayed
    try:
        hover_data.remove(attr)
    except ValueError:
        pass
    hover_data.insert(0, attr)
    
    # Different settings for the many plot variables
    fig = None
    if attr == 'price':
        # Define a new column containing intervals of prices
        # Note: The ranges are approximations based on quartile ranges
        bins = [0, 50, 75, 100, 1000, 10000]
        labels = ['$0-50', '$50-75', '$75-100', '$100-1K', '$1K-10K']
        filtered_df['price_interval'] = pd.cut(filtered_df['price'], bins=bins, labels=labels)
        
        color_prices_map = {  # Define colors for each price interval
            '$0-50': 'blue',
            '$50-75': 'green',
            '$75-100': 'yellow',
            '$100-1K': 'orange',
            '$1K-10K': 'red'
        }
        
        size_prices_map = {  # Define sizes for each price interval
            '$0-50': 0.1,
            '$50-75': 0.2,
            '$75-100': 0.4,
            '$100-1K': 0.7,
            '$1K-10K': 1
        }
        filtered_df['plot_size'] = filtered_df['price_interval'].map(size_prices_map)
        
        fig = px.scatter_map(
            data_frame=filtered_df,
            lat='latitude',
            lon='longitude',
            color='price_interval',
            color_discrete_map=color_prices_map,
            size='plot_size',
            hover_data=hover_data,
            #title=f"Airbnb Listings by {plot_var}",
            zoom=10
        )
    elif attr.startswith("review_scores"):
        # Define a new column containing intervals of review scores
        # Note: The ranges are approximations based on quartile ranges
        bins = [0, 4.6, 4.8, 4.9, 5.0]
        labels = ['Low (<4.6)', 'Mid-Low (4.6-4.8)', 'Mid-High (4.8-4.9)', 'High (≥4.9)']
        filtered_df['review_score_interval'] = pd.cut(filtered_df[attr], bins=bins, labels=labels)
        
        color_reviews_map = {  # Define colors for each review score interval
            'Low (<4.6)': 'red',
            'Mid-Low (4.6-4.8)': 'orange',
            'Mid-High (4.8-4.9)': 'yellow',
            'High (≥4.9)': 'green'
        }
        
        size_reviews_map = {  # Define sizes for each review score interval
            'Low (<4.6)': 0.1,
            'Mid-Low (4.6-4.8)': 0.2,
            'Mid-High (4.8-4.9)': 0.5,
            'High (≥4.9)': 0.8
        }
        filtered_df['plot_size'] = filtered_df['review_score_interval'].map(size_reviews_map)
        
        fig = px.scatter_map(
            data_frame=filtered_df,
            lat='latitude',
            lon='longitude',
            color='review_score_interval',
            color_discrete_map=color_reviews_map,
            size='plot_size',
            hover_data=hover_data,
            #title=f"Airbnb Listings by {plot_var}",
            zoom=10
        )
    else:
        fig = px.scatter_map(
            data_frame=filtered_df,
            lat='latitude',
            lon='longitude',
            color=attr,
            color_continuous_scale=None if attr=='room_type' else 'Turbo',
            color_discrete_map={
                "Private room": "#636efa",
                "Entire home/apt": "#ef553b",
                "Shared room": "#00cc96",
                "Hotel room": "#ab63fa"
            },
            size=None if attr=='room_type' else attr,
            hover_data=hover_data,
            #title=f"Airbnb Listings by {plot_var}",
            size_max=40 if plot_var != "Accommodates" else 28,
            zoom=10
        )
    
    fig.update_traces(
        marker=dict(
            opacity=0.3 if plot_var != "Room Type" else 0.6
        )
    )
    
    # Plot missing data points of the plot variable
    if len(na_df) > 0:
        fig.add_trace(go.Scattermap(
            mode='markers',
            lat=na_df['latitude'],
            lon=na_df['longitude'],
            marker=dict(
                color='black',
                size=5,
            ),
            opacity=1,
            name='Missing Data',
            hovertext=f'No Data Available for {plot_var}',
            legend='legend2',
            legendgroup='Missing'
        ))
        fig.data = (fig.data[-1],) + fig.data[:-1]  # Overlap initial points with missing data ones
    
    # Plot the various markers/POIs
    if "Monuments" in plot_markers:
        fig.add_trace(go.Scattermap(
            lat=df_monuments["latitude"],
            lon=df_monuments["longitude"],
            mode="markers",
            marker=dict(
                symbol="monument",
                color="gold",
                size=10
            ),
            name="Monuments",
            hovertext=df_monuments["label"],
            legend="legend2",
            legendgroup="POIs",
            legendgrouptitle_text="Points of Interest",
            showlegend=False
        ))
        
    if "Concert Halls" in plot_markers:
        fig.add_trace(go.Scattermap(
            lat=df_concert["latitude"],
            lon=df_concert["longitude"],
            mode="markers",
            marker=dict(
                symbol="music",
                color="purple",
                size=10
            ),
            name="Concert Halls",
            hovertext=df_concert["label"],
            legend="legend2",
            legendgroup="POIs",
            legendgrouptitle_text="Points of Interest",
            showlegend=False
        ))
        
    if "University" in plot_markers:
        fig.add_trace(go.Scattermap(
            lat=df_uni["latitude"],
            lon=df_uni["longitude"],
            mode="markers",
            marker=dict(
                symbol="college",
                color="steelblue",
                size=10
            ),
            name="University",
            hovertext=df_uni["label"],
            legend="legend2",
            legendgroup="POIs",
            legendgrouptitle_text="Points of Interest",
            showlegend=False
        ))
        
    if "Transport Centers" in plot_markers:
        fig.add_trace(go.Scattermap(
            lat=df_transport["latitude"],
            lon=df_transport["longitude"],
            mode="markers",
            marker=dict(
                symbol=["airport", "rail", "rail", "bus"],  # "airport" for the airport
                color="darkgreen",
                size=10
            ),
            name="Transport Centers",
            hovertext=df_transport["label"],
            legend="legend2",
            legendgroup="POIs",
            legendgrouptitle_text="Points of Interest",
            showlegend=False
        ))
        
    if "Metro" in plot_markers:
        line_color = {
            'A': 'blue',
            'B': 'red',
            'C': 'green',
            'D': 'yellow',
            'E': 'purple',
            'F': 'orange'
        }
        
        # Add metro lines
        for shape_id, shape in df_mshapes.groupby('shape_id'):
            fig.add_trace(go.Scattermap(
                lat=shape["shape_pt_lat"],
                lon=shape["shape_pt_lon"],
                mode="lines",
                line=dict(
                    color=line_color[shape_id[0]],
                    width=5
                ),
                name=f"Line {shape_id[0]}",
                hoverinfo="none",
                legend="legend2",
                legendgrouptitle_text="Metro",
                legendgroup="metro"
            ))
            
        # Add station names
        fig.add_trace(go.Scattermap(
            lat=df_mstops["stop_lat"],
            lon=df_mstops["stop_lon"],
            mode="markers",
            marker=dict(
                color="white",
                size=5
            ),
            name="Metro Stations",
            hovertext=df_mstops.apply(
                lambda row: f"Station: {row['stop_name']}<br>Lines: {row['lines']}",
                axis=1
            ),
            legend="legend2",
            legendgroup="metro"
        ))
    
    fig.update_layout(transition_duration=500,
                      height=550,  # Match sparkcloud
                      coloraxis_colorbar=dict(
                          title=plot_var,
                      ),
                      legend=dict(
                          title=plot_var    
                      ),
                      legend2=dict(  # POIs and metro legends
                          x=0.02,  # Left side
                          y=0.98,  # Top
                          xanchor='auto',
                          yanchor='auto',
                          bgcolor='lavender',
                          itemsizing='constant'
                      ),
                      plot_bgcolor="#f8f9fa",   # very light grey-blue
                      paper_bgcolor="#f8f9fa")
    
    
    return fig

@app.callback(
    Output('sparkcloud-graph', 'figure'),
    Output('selected-dates', 'children'),
    Input('month-slider', 'value'),
    Input('sentiment-slider', 'value')
)
def update_sparkcloud(slider_range, sentiment_range):
    start_month = available_months[slider_range[0]]
    end_month = available_months[slider_range[1]]

    filtered = df_reviews[
        (df_reviews['date'] >= start_month) &
        (df_reviews['date'] <= end_month) &
        (df_reviews['sentiment'] >= sentiment_range[0]) &
        (df_reviews['sentiment'] <= sentiment_range[1])
    ]

    if filtered.empty:
        return go.Figure(), f"No reviews found for the selected range."

    sampled = filtered.sample(min(10000, len(filtered)), random_state=42)
    word_time_freq = defaultdict(lambda: defaultdict(int))

    for _, row in sampled.iterrows():
        month = row['month']
        words = process_text(row['comments'])
        for word in words:
            word_time_freq[word][month] += 1
            
    total_freq = {word: sum(freqs.values()) for word, freqs in word_time_freq.items()}
    top_words = sorted(total_freq.items(), key=lambda x: x[1], reverse=True)[:12]

    all_months = pd.period_range(start_month.to_period('M'), end_month.to_period('M'), freq='M')
    word_data = []
    for word, total in top_words:
        counts = [word_time_freq[word].get(m, 0) for m in all_months]
        word_data.append({"word": word, "counts": counts, "total": total})

    WORDS_PER_ROW = 4
    X_SPACING = 20
    Y_SPACING = 12
    SPARK_HEIGHT = 2.5
    SPARK_OFFSET = 2.5
    WORD_HEIGHT = 3

    fig = go.Figure()

    for idx, item in enumerate(word_data):
        row = idx // WORDS_PER_ROW
        col = idx % WORDS_PER_ROW
        x_start = col * X_SPACING
        y_start = -row * (WORD_HEIGHT + SPARK_HEIGHT + Y_SPACING)
        font_size = np.log(item['total'] + 1) * 4.5 - 6

        fig.add_trace(go.Scatter(
            x=[x_start + X_SPACING / 2],
            y=[y_start],
            text=[item['word']],
            mode='text',
            textfont=dict(size=font_size, color='teal'),
            hoverinfo="text",
            hovertext=f"<b>{item['word']}</b><br>Total: {item['total']}",
            showlegend=False
        ))

        norm = np.array(item['counts'], dtype=float)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-5)
        spark_x = np.linspace(x_start + 1, x_start + X_SPACING - 4, len(norm))
        spark_y = y_start - SPARK_OFFSET - norm * SPARK_HEIGHT

        fig.add_trace(go.Scatter(
            x=spark_x,
            y=spark_y,
            mode="lines",
            line=dict(color="lightblue", width=1),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        title="<b>SparkCloud of Airbnb Reviews - Porto</b><br><sub>Words sized by frequency, with sparklines showing monthly usage</sub>",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=550,
        margin=dict(t=80, l=20, r=20, b=40),
        plot_bgcolor="#f8f9fa",   # very light grey-blue
        paper_bgcolor="#f8f9fa"
    )

    selected_range_text = (
        f"Selected date range: {start_month.strftime('%Y-%m')} to {end_month.strftime('%Y-%m')} | "
        f"Sentiment range: {sentiment_range[0]:.2f} to {sentiment_range[1]:.2f}"
    )

    return fig, selected_range_text

@app.callback(
    Output('event-review-graph', 'figure'),
    Output('event-map', 'figure'),
    Input('event-dropdown', 'value'),
    Input('min-reviews-dropdown', 'value'),
    Input('map-mode-radio', 'value'),
    Input('rating-slider', 'value'),
    Input('beds-dropdown', 'value')
)
def update_graphs(selected_event, min_reviews, map_mode, min_rating, min_beds):
    if not selected_event:
        return {}, {}

    event_date = EVENTS[selected_event]
    start_date = event_date - pd.Timedelta(days=30)
    end_date = event_date + pd.Timedelta(days=30)

    # Filter for line graph
    filtered_graph = df_reviews2[(df_reviews2['date'] >= start_date) & (df_reviews2['date'] <= end_date)]
    review_counts = filtered_graph.groupby('listing_id').size().reset_index(name='review_count')
    filtered_graph = pd.merge(filtered_graph, review_counts, on='listing_id', how='left')
    filtered_graph = filtered_graph[filtered_graph['review_count'] >= min_reviews]
    filtered_graph = filtered_graph[
        (filtered_graph['review_scores_rating'] >= min_rating) &
        (filtered_graph['beds'] >= min_beds)
    ]

    # Line chart
    daily_counts = filtered_graph.groupby(filtered_graph['date'].dt.date).size().reset_index(name='review_count')
    fig_line = px.line(
        daily_counts, x='date', y='review_count',
        title=f"Review Activity Around {selected_event}",
        labels={'date': 'Date', 'review_count': 'Number of Reviews'}
    )
    fig_line.update_traces(mode='lines+markers')
    fig_line.update_layout(height=500, margin=dict(t=60, l=40, r=40, b=40))

    # Map range
    map_start = event_date - pd.Timedelta(days=14)
    map_end = event_date + pd.Timedelta(days=14)
    filtered_map = df_reviews2[(df_reviews2['date'] >= map_start) & (df_reviews2['date'] <= map_end)]

    review_counts_map = filtered_map.groupby('listing_id').size().reset_index(name='review_count')
    filtered_map = pd.merge(filtered_map, review_counts_map, on='listing_id', how='left')
    filtered_map = filtered_map[filtered_map['review_count'] >= min_reviews]
    filtered_map = filtered_map[
        (filtered_map['review_scores_rating'] >= min_rating) &
        (filtered_map['beds'] >= min_beds)
    ]

    # === Generate map depending on mode ===
    if map_mode == 'heat':
        filtered_map['review_density'] = (
            filtered_map['review_count'] - filtered_map['review_count'].min()
        ) / (filtered_map['review_count'].max() - filtered_map['review_count'].min() + 1e-5)

        fig_map = px.density_map(
            filtered_map,
            lat='latitude',
            lon='longitude',
            z='review_density',
            radius=20,
            center=dict(lat=41.15, lon=-8.61),
            zoom=12.5,
            map_style="carto-positron",
            opacity=0.8,
            color_continuous_scale="Blues",
            labels={'review_density': 'Review Density'}
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(title="Review Density"),
            hovermode="closest"
        )

    elif map_mode == 'both':
        # Normalize bubble size for visibility
        filtered_map['scaled_size'] = (
            (filtered_map['review_count'] - filtered_map['review_count'].min()) /
            (filtered_map['review_count'].max() - filtered_map['review_count'].min() + 1e-5)
        ) * 15 + 3

        fig_map = px.scatter_map(
            filtered_map,
            lat='latitude',
            lon='longitude',
            size='scaled_size',
            color='review_count',
            hover_data={
                'listing_id': True,
                'beds': True,
                'review_scores_rating': True,
                'review_count': True,
                'scaled_size': False,
                'latitude': False,
                'longitude': False
            },
            size_max=18,
            opacity=0.65,
            color_continuous_scale="Blues_r",
            labels={'review_count': 'Number of Reviews'}
        )
        fig_map.update_layout(
            coloraxis_colorbar=dict(title="Number of Reviews"),
            hovermode="closest"
        )

    fig_map.update_layout(
        mapbox=dict(
            style="carto-positron",
            zoom=12.5,
            center=dict(lat=41.15, lon=-8.61)
        ),
        title=f"Map of Reviews Around {selected_event}",
        height=550,
        plot_bgcolor="#f8f9fa",   # very light grey-blue
        paper_bgcolor="#f8f9fa"
    )

    return fig_line, fig_map

if __name__ == '__main__':
    app.run(debug=False)
