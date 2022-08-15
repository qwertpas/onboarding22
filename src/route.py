from datetime import date
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
import pickle
import os
import json
from util import * 

dir = os.path.dirname(__file__)

class Route():
    def __init__(self):
        #list of dictionaries, each representing a leg
        self.leg_list = []


    def add_base_leg(self, leg_csv, start):
        df = pd.read_csv(leg_csv)

        name = df['name'].iat[0] #get name from first row

        df.fillna(method='bfill', inplace=True) #fill leading NaNs with the first valid value

        #convert distance to meters
        if('distance (mi)' in df.columns):
            df['distance (m)'] = df['distance (mi)'] * 1609.34
        else:
            assert 'distance (km)' in df.columns
            df['distance (m)'] = df['distance (km)'] * 1000            

        #convert to radians
        df['slope (rad)'] = np.arctan(df['slope (%)'].values * 0.01)

        dists_raw = df['distance (m)'].values
        longitude_interp = interp1d(dists_raw, df['longitude'].values, fill_value="extrapolate")
        latitude_interp = interp1d(dists_raw, df['latitude'].values, fill_value="extrapolate")
        slope_interp = interp1d(dists_raw, df['slope (%)'].values, fill_value="extrapolate")
        headings_interp = interp1d(dists_raw, df['longitude'].values, fill_value="extrapolate")

        dists = np.arange(0, dists_raw[-1]+100, 100)
        longitudes = longitude_interp(dists)
        latitudes = latitude_interp(dists)
        slopes = slope_interp(dists)
        headings = headings_interp(dists)
        
        self.leg_list.append({
            'name': name,
            'dists': dists,
            'longitudes': longitudes,
            'latitudes': latitudes,
            'slopes': slopes,
            'headings': headings,
        })

    #if date already exists in 
    def query_weather(self, day: date, override=False):
        for leg in self.leg_list:
            if('day' not in leg)
            date.year

    def add_loop(self, loop_dists, loop_slopes, loop_headings):
        return

    def save_as(self, name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "wb") as f:
            pickle.dump(self, f)

    def open(name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "rb") as f:
            return pickle.load(f)


def main():
    route = Route()
    route.add_base_leg(dir + '/route_data/gps/asc2022/stage1_ckpt1.csv')

    route.save_as("test")
    new_route = Route.open('test')

    print_dict(new_route.leg_list[0])

if __name__ == "__main__":
    main()