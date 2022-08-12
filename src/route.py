import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import pickle
import os
import json
from util import * 

dir = os.path.dirname(__file__)

class Route():
    def __init__(self):
        #list of dictionaries, each representing a leg
        self.leg_list = []


    def addBaseLeg(self, leg_csv):
        df = pd.read_csv(leg_csv)

        name = df['name'].iat[0] #get name from first row

        #if in US units, convert to metric
        if('altitude (ft)' in df.columns):
            df['altitude (m)'] = df['altitude (ft)'] * 0.3048
        if('distance (mi)' in df.columns):
            df['distance (km)'] = df['distance (mi)'] * 1.60934
        if('distance_interval (ft)' in df.columns):
            df['distance_interval (m)'] = df['distance_interval (ft)'] * 0.3048

        df.fillna(method='bfill', inplace=True)
        
        self.leg_list.append({
            'name': name,
            'longitudes': df['longitude'].values,
            'latitudes': df['latitude'].values,
            'dists_m': df['distance (km)'].values * 1000,
            'dist_intervals_m': df['distance_interval (m)'].values,
            'slopes_rad': np.arctan(df['slope (%)'].values * 0.01),
            'headings_rad': np.deg2rad(df['course'].values),
        })

    def addLoop(self, loop_dists, loop_slopes, loop_headings):
        return

    def saveAs(self, name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "wb") as f:
            pickle.dump(self, f)

    def open(name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "rb") as f:
            return pickle.load(f)


def main():
    route = Route()
    route.addBaseLeg(dir + '/route_data/gps/asc2022/stage1_ckpt1.csv')

    route.saveAs("test")
    new_route = Route.open('test')

    print_dict(new_route.leg_list[0])

if __name__ == "__main__":
    main()