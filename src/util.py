
from datetime import datetime
import os
import numpy as np
import math

dir = os.path.dirname(__file__)

to_dates = np.vectorize(datetime.fromtimestamp)

def meters_to_miles(meters):
    return meters * 0.0006214

def miles_to_meters(miles):
    return miles * 1609.34

def feet_to_meters(feet):
    return feet * 0.3048

def meters_to_feet(meters):
    return meters * 3.28

def mpersec(mph):
    return mph * 0.44704

def mph(mpersec):
    return mpersec * 2.23694

def latlong_dist(origin, destination):
    '''haversine formula for getting earth surface distance (km) between 2 lat/long pairs'''
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def print_dict(d, indent=0):
   for key, value in d.items():
        print('\t' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + repr(value))


# if __name__ == "__main__":
    # get_weather_days(44.9691314, -93.530618, datetime(2022, 8, 19), start_hour=7, end_hour=20, num_days=2, save='wayzata_8-19')