import requests
from datetime import datetime
from timezonefinder import TimezoneFinder

tzfinder = TimezoneFinder()

def get_openmeteo(latlong, start:datetime, stop:datetime):

    vars = [
        'shortwave_radiation',      #total sun for horizontal array 
        'direct_normal_irradiance', #used to calculate total sun for tilted array
        'diffuse_radiation',        #used to calculate total sun for tilited array (add to direct)
        'windspeed_10m',
        'winddirection_10m',
    ]

    #get timezone at coordinate so timestamps make sense
    timezone = tzfinder.timezone_at(lat=latlong[0], lng=latlong[1])

    forecast = {}
    for var in vars:

        response = requests.request(
            method = "GET", 
            url = "https://api.open-meteo.com/v1/forecast?", 
            params = {
                "latitude": latlong[0],
                "longitude": latlong[1],
                "hourly": var,
                "windspeed_unit": "ms",
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": stop.strftime("%Y-%m-%d"),
                "timeformat": "unixtime",
                "timezone": timezone
            }
        ).json()
        
        try:
            values = response['hourly'][var]
            timestamps = response['hourly']['time']
        except Exception as e:
            print(f"Error: {e} \n {response}")

        while values and values is None:    #remove trailing Nones
            values[var].pop()

        if('dates' not in forecast):
            forecast['timestamps'] = timestamps[:len(values)] #truncate timestamps to match values
        forecast[var] = values

    if('direct_normal_irradiance' in forecast and 'diffuse_radiation' in forecast):
        forecast['sun_flat'] = forecast['shortwave_radiation']
        forecast['sun_tilt'] = [sum(x) for x in zip(forecast['direct_normal_irradiance'], forecast['diffuse_radiation'])]

        del forecast['shortwave_radiation']
        del forecast['direct_normal_irradiance']
        del forecast['diffuse_radiation']
        
    return forecast