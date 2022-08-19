from datetime import datetime, timedelta
import requests
import pandas as pd
try:
    from .util import *     #when route.py is being imported elsewhere
except:
    from util import *      #when route.py is being run directlyimport pandas as pd

key = ''
with open(dir + '/key.txt', 'r') as keyfile:
    split = keyfile.read().split('\n')
    key = split[2]  # visual crossing key (free)

def get_weather_hour(latitude, longitude, time: datetime, doPrint=False):
    '''Gets solar, cloud, wind, precip, and temp from VisualCrossing for a particular hour, using 1 record cost'''
    forecast_sec = round(time.timestamp())
    requests_text = ('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline' 
        + '/' + str(latitude) + ',' + str(longitude)
        + '/' + str(forecast_sec)
        + '?key=' + key
        + '&include=current'
        + '&elements=datetimeEpoch,solarradiation,cloudcover,windspeed,winddir,precip,temp'
    )
    if(doPrint): print(requests_text)
    return requests.get(requests_text).json()


def get_weather_range(latitude, longitude, start_day: datetime, start_hour=7, end_hour=20, num_days=1, save=None, doPrint=False):
    '''Gets solar, cloud, wind, precip, and temp from VisualCrossing in the time range, for multiple days. 
    The number of record costs used is the total number of forecasted hours. Optionally save as csv.'''

    timestamps = []
    solars = []
    cloudcovers = []
    windspeeds = []
    winddirs = []
    precips = []
    temps = []

    for day_index in range(num_days):
        forecast_day = datetime(start_day.year, start_day.month, start_day.day) + timedelta(days=day_index)

        for hour in range(start_hour, end_hour + 1):

            weather = get_weather_hour(latitude, longitude, forecast_day + timedelta(hours=hour), doPrint=doPrint)

            timestamps.append(weather['currentConditions']['datetimeEpoch'])
            solars.append(weather['currentConditions']['solarradiation'])
            cloudcovers.append(weather['currentConditions']['cloudcover'])
            windspeeds.append(weather['currentConditions']['windspeed'])
            winddirs.append(weather['currentConditions']['winddir'])
            precips.append(weather['currentConditions']['precip'])
            temps.append(weather['currentConditions']['temp'])

    weather_df = pd.DataFrame.from_dict({
        'date': [datetime.fromtimestamp(timestamp) for timestamp in timestamps],
        'timestamp': timestamps,
        'solar': solars,
        'cloudcover': cloudcovers,
        'windspeed': windspeeds,
        'winddir': winddirs,
        'precip': precips,
        'temp': temps,
    })

    if save is not None:
        weather_df.to_csv(save, index=False)
            
    return weather_df