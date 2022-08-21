
from datetime import datetime
import os
import numpy as np
from numpy import sin, cos, pi
import math

dir = os.path.dirname(__file__)

to_dates = np.vectorize(datetime.fromtimestamp)

def meters2miles(meters):
    return meters * 0.0006214

def miles2meters(miles):
    return miles * 1609.34

def feet2meters(feet):
    return feet * 0.3048

def meters2feet(meters):
    return meters * 3.28

def mph2mpersec(mph):
    return mph * 0.44704

def mpersec2mph(mpersec):
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

def solar_altitude_angle(time_obj:datetime, latitude, longitude, tz_offset):
    day_of_year = time_obj.tm_yday
    Latitude = latitude * (2 * np.pi / 360)

    local_solar_time_meridian = 15*tz_offset

    B = (day_of_year - 81)*360./365. * (2 * np.pi / 360.)
    E = 9.87*sin(2*B) - 7.53*cos(B) - 1.58*sin(B)
    time_correction_factor = 4 * \
        (longitude - local_solar_time_meridian) + E  # in minutes
    local_solar_time = time_obj.tm_hour + \
        (time_obj.tm_min+time_correction_factor)/60.

    Solar_Hour_Angle = (12 - local_solar_time) * 15 * (2 * np.pi / 360)
    Solar_Declination = (-23.45 * cos((day_of_year+10)
                                      * 2*pi/365)) * (2 * np.pi / 360)

    Solar_Altitude_Angle = np.arcsin(cos(Latitude) * cos(Solar_Declination) * cos(
        Solar_Hour_Angle) + sin(Latitude) * sin(Solar_Declination))

    return Solar_Altitude_Angle

def print_dict(d, indent=0):
   for key, value in d.items():
        print('\t' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + repr(value))


# if __name__ == "__main__":
    # get_weather_days(44.9691314, -93.530618, datetime(2022, 8, 19), start_hour=7, end_hour=20, num_days=2, save='wayzata_8-19')