
import requests
import json
from datetime import date, datetime
from datetime import timedelta
import os
import numpy as np
import pandas as pd

dir = os.path.dirname(__file__)

to_dates = np.vectorize(datetime.fromtimestamp)

key = ''
with open(dir + '/key.txt', 'r') as keyfile:
    split = keyfile.read().split('\n')
    key = split[2]  # visual crossing key (free)

#format: https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/London,UK/1601510400/1609372800?key=YOUR_API_KEY 
def request_weather(latitude, longitude, day_start: datetime, days_ahead=1, save=None):
    day_start = datetime(day_start.year, day_start.month, day_start.day)
    startsec = round(day_start.timestamp())
    endsec = round((day_start + timedelta(days=days_ahead)).timestamp())

    requests_text = ('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline' 
        + '/' + str(latitude) + ',' + str(longitude)
        + '/' + str(startsec)
        + '/' + str(endsec)
        + '?key=' + key 
    )

    print(requests_text)

    weather = requests.get(requests_text).json()

    timestamps = []
    solars = []
    cloudcovers = []
    windspeeds = []
    winddirs = []

    try:
        for day in weather['days']:
            for hour in day['hours']:
                timestamps.append(hour['datetimeEpoch'])
                solars.append(hour['solarradiation'])
                cloudcovers.append(hour['cloudcover'])
                windspeeds.append(hour['windspeed'])
                winddirs.append(hour['winddir'])
    except Exception as e:
        print('error' + e)

    weather_df = pd.DataFrame.from_dict({
        'index': to_dates(timestamps),
        'timestamp': timestamps,
        'solar': solars,
        'cloudcover': cloudcovers,
        'windspeed': windspeeds,
        'winddir': winddirs
    })

    if save is not None:
        weather_df.to_csv(save, index=False)
            
    return weather_df


def mpersec(mph):
    return mph * 0.44704

def mph(mpersec):
    return mpersec * 2.23694

def print_dict(d, indent=0):
   for key, value in d.items():
        print('\t' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
