#Import libraries
import os
import pandas as pd
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from prophet import Prophet
import plotly.express as px

##1.Data Loading and Understanding:
#Load data
data = pd.read_csv('Crime_Data_from_2020_to_Present_20241215.csv',delimiter=',')
print(data.head()) #display the first rows

print(data.info())
print(data.columns)
data.head(10)

##2. Data Cleaning:
#drop the column which all its values are NAN
df = data.dropna(axis=1, how='all')

#print the number of nan values in each attribute
print(df.isnull().sum())

#Replace the NAN values with "unknown" word
columns_to_replace = ['Mocodes','TIME OCC','Vict Sex','Vict Descent','Premis Cd','Premis Desc','Weapon Desc','Weapon Used Cd','Crm Cd 1','Crm Cd 2','Crm Cd 3','Crm Cd 4','Cross Street','Status']
df[columns_to_replace] = df[columns_to_replace].fillna("unknown")

#check NAN values
print(df.isnull().sum())

#Find the duplicate rows
df.duplicated().value_counts()

##3.Correlation:
#Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

#Calculate the correlation matrix
corr_matrix = numeric_df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, vmin=1 , vmax=1)

##4.Create Our dataframe:
dataframe=df[['DR_NO','DATE OCC','AREA NAME','Crm Cd Desc', 'Mocodes','Premis Desc','Weapon Desc','LAT','LON']].copy()
# dataframe.head(10)

# Remove rows where 'Premis Desc' or 'Weapon Desc' contain 'unknown'
dataframe = dataframe[~dataframe['Premis Desc'].str.contains('unknown', case=False, na=False)]
dataframe = dataframe[~dataframe['Weapon Desc'].str.contains('unknown', case=False, na=False)]

# Convert LAT and LON to numeric
dataframe["LAT"] = pd.to_numeric(dataframe["LAT"], errors="coerce")
dataframe["LON"] = pd.to_numeric(dataframe["LON"], errors="coerce")

# Drop rows where LAT or LON could not be converted to numeric
dataframe = dataframe.dropna(subset=["LAT", "LON"]).reset_index(drop=True)

# Define the possible formats to test
formats = ['%m/%d/%Y %H:%M', '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y', '%Y-%m-%d']

# Function used to detect the format of a date string
def detect_format(date_str):
    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return fmt
        except ValueError:
            continue
    return "Unknown Format"

# Apply the previous function 
dataframe['date_format'] = dataframe['DATE OCC'].apply(lambda x: detect_format(x) if pd.notna(x) else "NaT")

# Check the results
print(dataframe[['DATE OCC', 'date_format']].head())


# Define function to parse and standardize dates
def parse_and_standardize(date_str):
    formats = ['%m/%d/%Y %H:%M', '%m/%d/%Y %I:%M:%S %p']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    # Return NaT if no format matches
    return pd.NaT  

# Apply the function to the 'DATE OCC' column
dataframe['DATE OCC'] = dataframe['DATE OCC'].apply(parse_and_standardize)


# Extract date features
dataframe['day_of_week'] = dataframe['DATE OCC'].dt.day_name()  
dataframe['hour'] = dataframe['DATE OCC'].dt.hour            
dataframe['month'] = dataframe['DATE OCC'].dt.month          

# Print the result
print(dataframe[['DATE OCC', 'day_of_week', 'hour', 'month']])

# Try to automatically infer the datetime format
dataframe['DATE OCC'] = pd.to_datetime(dataframe['DATE OCC'], errors='coerce', infer_datetime_format=True)

# Calculate total number of crimes
total_crimes = len(dataframe)

# Group the data for Crime Frequency by Premises Type 
top_10_premises = dataframe.groupby('Premis Desc').size().reset_index(name='Count').sort_values(by='Count', ascending=False).head(10)

# Group the data for Premises Safety Rating 
premises_safety = df.groupby('Premis Desc').size().reset_index(name='Count')
premises_safety['Safety Rating'] = 100 - premises_safety['Count'] * 2
top_premises_safety = premises_safety.sort_values(by='Safety Rating', ascending=False).head(20)

# Aggregate crime counts by date and Premis Desc and Weapon Desc
dataframe_premis = dataframe.groupby(['DATE OCC', 'Premis Desc']).size().reset_index(name='Crime Count')
dataframe_weapon = dataframe.groupby(['DATE OCC', 'Weapon Desc']).size().reset_index(name='Weapon Count')
grouped = dataframe.groupby(['DATE OCC', 'Premis Desc', 'Weapon Desc']).size().reset_index(name='count')

# Create a pie chart for crimes by day of the week
crime_day = dataframe.groupby('day_of_week').size()
crime_day = crime_day[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
crime_day = crime_day.reset_index(name='Count')


weapons_df = dataframe[dataframe['Weapon Desc'].str.lower() != 'unknown']

#  Count Crimes by Weapon Type
weapon_counts = weapons_df['Weapon Desc'].value_counts().reset_index()

# In order to indicate that one contains weapon type and the other contains crime count we are going to rename the columns
weapon_counts.columns = ['Weapon Type', 'Crime Count']

# Print the top 10 weapons involved in crimes
print(weapon_counts.head(10))

# Convert the 'DATE OCC' column to a datetime format 
dataframe['DATE OCC'] = pd.to_datetime(dataframe['DATE OCC'])

# Group the data by the 'DATE OCC' and 'AREA NAME'
dataframe_grouped = dataframe.groupby(['DATE OCC', 'AREA NAME']).size().reset_index(name='Crime Count')


app = dash.Dash(__name__)
app.layout = html.Div([

    # Title of the Dashboard
    html.Div([
        html.H1("Crime Prediction Dashboard of Los Angeles", style={'textAlign': 'center', 'fontFamily': 'Arial', 'color': '#FFFFFF'}),
    ], style={'padding': '20px', 'backgroundColor': '#636EFA', 'color': 'white','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'marginBottom': '30px'}),

    # Display Total Number of Crimes
    html.Div([ 
        html.H2(f"Total Number of Crimes: {total_crimes}", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    ], style={'width': '200px', 'height': '100px','padding':'20px','boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'margin': 'auto', 'marginBottom': '30px'}),

    # Create a two-column grid for graphs
    html.Div([

        # Crime Frequency by Premises Type (Top 10)
        html.Div([
            dcc.Graph(
                figure=px.bar(top_10_premises, x='Premis Desc', y='Count', title="Top 10 Crime Frequency by Premises Type")
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'marginBottom': '30px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

        # Premises Safety Ratings (Top 20)
        html.Div([
            dcc.Graph(
                figure=px.bar(top_premises_safety, x='Premis Desc', y='Safety Rating', title="Top 20 Safety Ratings by Premises")
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '30px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

    # Create a two-column grid for other graphs
    html.Div([

        # Top 10 Weapons Involved in Crimes
        html.Div([
            dcc.Graph(
                figure=px.bar(
                    weapon_counts.head(10),
                    x='Crime Count',
                    y='Weapon Type',
                    orientation='h',
                    title="Top 10 Weapons Involved in Crimes",
                    labels={"Crime Count": "Number of Crimes", "Weapon Type": "Weapon"}
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'marginBottom': '30px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

        # Crimes by Day of the Week - Pie Chart
        html.Div([
            dcc.Graph(
                figure=px.pie(
                    crime_day,
                    names='day_of_week',
                    values='Count',
                    title="Crimes by Day of the Week",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginBottom': '30px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

    # Crime Prediction with Prophet for Premises
    html.Div([
        html.H3("Crimes Prediction for Premises", style={'fontFamily': 'Arial', 'color': '#333','font-size':'25px','text-align':'center','padding-top':'20px'}),
        dcc.Dropdown(
            id='premis-desc-dropdown',
            options=[{'label': premis, 'value': premis} for premis in dataframe['Premis Desc'].unique()],
            placeholder="Select a Premis Desc"
        ),
        dcc.RadioItems(
            id='prediction-range',
            options=[
                {'label': '30 Days', 'value': 30},
                {'label': '6 Months', 'value': 180},
                {'label': '1 Year', 'value': 365}
            ],
            value=30, 
            inline=True
        ),
        dcc.Graph(id='prediction-graph')
    ], style={'marginBottom': '50px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

    # Weapon Usage Prediction with Prophet
    html.Div([
        html.H3("Weapon Usage Prediction", style={'fontFamily': 'Arial', 'color': '#333','font-size':'25px','text-align':'center','padding-top':'20px'}),
        dcc.Dropdown(
            id='weapon-desc-dropdown',
            options=[{'label': weapon, 'value': weapon} for weapon in dataframe['Weapon Desc'].unique()],
            placeholder="Select a Weapon Desc"
        ),
        dcc.RadioItems(
            id='weapon-prediction-range',
            options=[
                {'label': '30 Days', 'value': 30},
                {'label': '6 Months', 'value': 180},
                {'label': '1 Year', 'value': 365}
            ],
            value=30,  
            inline=True
        ),
        dcc.Graph(id='weapon-prediction-graph')
    ], style={'marginBottom': '50px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

    # Weapon Usage Prediction by Premises with Dropdown for Forecast
    html.Div([
        html.H3("Weapon Usage Prediction for Selected Premises", style={'fontFamily': 'Arial', 'color': '#333','font-size':'25px','text-align':'center','padding-top':'20px'}),
        dcc.Dropdown(
            id='premis-dropdown',
            options=[{'label': premis, 'value': premis} for premis in grouped['Premis Desc'].unique()],
            value='MULTI-UNIT DWELLING', 
            placeholder='Select a premise...'
        ),
        dcc.RadioItems(
            id='forecast-period-dropdown',
            options=[
                {'label': '30 Days', 'value': 30},
                {'label': '6 Months', 'value': 180},
                {'label': '1 Year', 'value': 365}
            ],
            value=30,  
            inline=True
        ),
        dcc.Graph(id='prediction-graph-premis')
    ], style={'marginBottom': '50px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px','background':'white'}),

    # Crime Prediction Area Section
    html.Div([
        html.H3("Crime Prediction by Area", style={'fontFamily': 'Arial', 'color': '#333','font-size':'25px','text-align':'center','padding-top':'20px'}),
        dcc.Dropdown(
            id='area-name-dropdown',
            options=[{'label': area, 'value': area} for area in dataframe['AREA NAME'].unique()],
            placeholder="Select an AREA NAME",
            value=None
        ),
        dcc.RadioItems(
            id='prediction-graph-area-choise',
            options=[
                {'label': '30 Days', 'value': 30},
                {'label': '6 Months', 'value': 180},
                {'label': '1 Year', 'value': 365}
            ],
            value=30, 
            inline=True
        ),
        dcc.Graph(id='prediction-graph-area')
    ], style={'marginBottom': '50px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)', 'borderRadius': '10px'}),

])


@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('premis-desc-dropdown', 'value'),
     Input('prediction-range', 'value')]
)
def update_premis_graph(selected_premis_desc, prediction_range):
    if not selected_premis_desc:
        return go.Figure()

    filteredData = dataframe_premis[dataframe_premis['Premis Desc'] == selected_premis_desc]
    if filteredData.empty or len(filteredData) < 2:
        return go.Figure(layout={'title': f"Not enough data for '{selected_premis_desc}'"})

    prophet_df = filteredData.rename(columns={"DATE OCC": "ds", "Crime Count": "y"})
    prophet_df = prophet_df.sort_values('ds')

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=prediction_range)
    forecast = model.predict(future)

    # Round the predicted values to integers
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    forecast['yhat_upper'] = forecast['yhat_upper'].round().astype(int)
    forecast['yhat_lower'] = forecast['yhat_lower'].round().astype(int)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines+markers', name='Historical Data'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))

    fig.update_layout(title=f"Crime Prediction for {selected_premis_desc}", xaxis_title="Date", yaxis_title="Crime Count",xaxis=dict(
            tickformat="%Y-%m-%d", 
            tickangle=45  
        ),)
    return fig


@app.callback(
    Output('weapon-prediction-graph', 'figure'),
    [Input('weapon-desc-dropdown', 'value'),
     Input('weapon-prediction-range', 'value')]
)
def update_weapon_graph(selected_weapon_desc, prediction_range):
    if not selected_weapon_desc:
        return go.Figure()

    filteredData = dataframe_weapon[dataframe_weapon['Weapon Desc'] == selected_weapon_desc]
    if filteredData.empty or len(filteredData) < 2:
        return go.Figure(layout={'title': f"Not enough data for '{selected_weapon_desc}'"})

    prophet_df = filteredData.rename(columns={"DATE OCC": "ds", "Weapon Count": "y"})
    prophet_df = prophet_df.sort_values('ds')

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=prediction_range)
    forecast = model.predict(future)

    # Round the predicted values to integers
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    forecast['yhat_upper'] = forecast['yhat_upper'].round().astype(int)
    forecast['yhat_lower'] = forecast['yhat_lower'].round().astype(int)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines+markers', name='Historical Data'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))

    fig.update_layout(title=f"Weapon Usage Prediction for {selected_weapon_desc}", xaxis_title="Date", yaxis_title="Weapon Count",xaxis=dict(
            tickformat="%Y-%m-%d",  
            tickangle=45  
        ),)
    return fig


@app.callback(
    Output('prediction-graph-premis', 'figure'),
    [Input('premis-dropdown', 'value'),
     Input('forecast-period-dropdown', 'value')]
)
def update_graph(selected_premis, forecast_period):
    # Filter data for the selected premise
    premisData = grouped[grouped['Premis Desc'] == selected_premis]

    # Find the top 3 weapons used in the selected premise
    top_weapons = (
        premisData.groupby('Weapon Desc')['count'].sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    # Create a figure to hold all predictions
    fig = go.Figure()

    # Loop through each top weapon and forecast usage
    for weapon in top_weapons:
        weaponData = premisData[premisData['Weapon Desc'] == weapon][['DATE OCC', 'count']]
        weaponData.rename(columns={'DATE OCC': 'ds', 'count': 'y'}, inplace=True)

        # Train Prophet model
        model = Prophet()
        model.fit(weaponData)

        # Create future dataframe for the selected forecast period 
        future = model.make_future_dataframe(periods=forecast_period)

        # Generate forecast
        forecast = model.predict(future)

        # convert the predicted values to integers
        forecast['yhat'] = forecast['yhat'].round().astype(int)

        # Add the forecasted data to the graph
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name=f'Predicted Usage: {weapon}',
            line=dict(width=2)
        ))

        # Convert the actual data to the nearest integer and add to the graph
        weaponData['y'] = weaponData['y'].astype(int)
        fig.add_trace(go.Scatter(
            x=weaponData['ds'], y=weaponData['y'],
            mode='markers', name=f'Actual Usage: {weapon}',
            marker=dict(size=6)
        ))

    # Update layout and show exact day on x-axis
    fig.update_layout(
        title=f"Top Weapon Predictions for {selected_premis}",
        xaxis_title="Date",
        yaxis_title="Count",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=45  
        ),
    )


    return fig


@app.callback(
    Output('prediction-graph-area', 'figure'),
    [Input('area-name-dropdown', 'value'),
    Input('prediction-graph-area-choise','value')]
)


def update_graph(selected_area_name,period_area):
    if not selected_area_name:
        return go.Figure()

    # Filter data for the selected AREA NAME
    filteredData = dataframe_grouped[dataframe_grouped['AREA NAME'] == selected_area_name]

    # Prepare data for Prophet
    prophet_df = filteredData.rename(columns={"DATE OCC": "ds", "Crime Count": "y"})

    # Fit Prophet model
    model = Prophet()
    model.fit(prophet_df)

    # Create future dataframe and make predictions
    future = model.make_future_dataframe(periods=period_area)  # Predict 30 days into the future
    forecast = model.predict(future)

    # Convert forecast values to integers
    forecast['yhat'] = forecast['yhat'].astype(int)
    forecast['yhat_upper'] = forecast['yhat_upper'].astype(int)
    forecast['yhat_lower'] = forecast['yhat_lower'].astype(int)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    # Plot results
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines+markers', name='Historical Data'))

    # Add forecast
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Add uncertainty intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')
    ))

    # Customize layout
    fig.update_layout(
        title=f"Crime Prediction for {selected_area_name}",
        xaxis_title="Date",
        yaxis_title="Crime Count",
        legend_title="Legend",
        xaxis=dict(
            tickformat="%Y-%m-%d",  
            tickangle=45 
        ),
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)





