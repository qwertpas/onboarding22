from datetime import date
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
import pickle
import os
import visualcrossing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm import tqdm
try:
    from .util import *     #when route.py is being imported elsewhere
except:
    from util import *      #when route.py is being run directly

dir = os.path.dirname(__file__)


class Route():
    def __init__(self):
        #list of dictionaries, each representing a leg
        self.leg_list = []
        self.total_length = 0

    def get_geography(csv_path:str):
        df = pd.read_csv(csv_path)

        name = df['name'].iat[0] #get name from first row

        df.fillna(method='bfill', inplace=True) #fill leading NaNs with the first valid value

        #convert distance to meters
        if('distance (mi)' in df.columns):
            df['distance (m)'] = df['distance (mi)'] * miles2meters(1)
        else:
            assert 'distance (km)' in df.columns
            df['distance (m)'] = df['distance (km)'] * 1000

        if('altitude (ft)' in df.columns):
            df['altitude (m)'] = df['altitude (ft)'] * feet2meters(1)   

        #convert to radians
        df['slope (rad)'] = np.arctan(df['slope (%)'].values * 0.01)

        dists = df['distance (m)'].values
        longitude_interp = interp1d(dists, df['longitude'].values, fill_value="extrapolate")
        latitude_interp = interp1d(dists, df['latitude'].values, fill_value="extrapolate")
        slope_interp = interp1d(dists, df['slope (%)'].values, fill_value="extrapolate")
        altitude_interp = interp1d(dists, df['altitude (m)'].values, fill_value="extrapolate")
        headings_interp = interp1d(dists, df['longitude'].values, fill_value="extrapolate")

        geo = {
            'name': name,
            'length': dists[-1],
            'longitude': longitude_interp,
            'latitude': latitude_interp,
            'slope': slope_interp,
            'altitude': altitude_interp,
            'heading': headings_interp,
        }
        return geo


    def add_leg(self, csv_path:str, start:datetime, open:datetime, close:datetime): 
        '''
            add a dict containing the geographic info of a base leg, in interpolant form.
            To get the slope at a distance d, use leg_list['slope'](d)
        '''        
        geo = Route.get_geography(csv_path)
        self.total_length += geo['length']
        self.leg_list.append({
            'name': geo['name'],
            'length': geo['length'],
            'type': 'base',
            'start': start,
            'open': open,
            'close': close,
            'longitude': geo['longitude'],
            'latitude': geo['latitude'],
            'slope': geo['slope'],
            'altitude': geo['altitude'],
            'heading': geo['heading'],
        })

    def add_loop_leg(self, csv_path:str, start:datetime, open:datetime, close:datetime):   
        '''
            add a dict containing the geographic info of a loop, in interpolant form.
            To get the slope at a distance d, use leg_list\['slope'](d)
        '''   
        geo = Route.get_geography(csv_path)
        self.total_length += geo['length']
        self.leg_list.append({
            'name': geo['name'],
            'length': geo['length'],
            'type': 'loop',
            'start': start,
            'open': open,
            'close': close,
            'longitude': geo['longitude'],
            'latitude': geo['latitude'],
            'slope': geo['slope'],
            'altitude': geo['altitude'],
            'heading': geo['heading'],
        })

    def gen_weather(self, start_leg=0, stop_leg=-1, dist_step=miles2meters(15), start_time=9):
        
        if(stop_leg == None or stop_leg == -1):
            stop_leg = len(self.leg_list)

        print(f"\nGenerating weather for legs {start_leg} to {stop_leg-1}")

        records_used = 0

        for i in range(start_leg, stop_leg):
            leg = self.leg_list[i]

            #skip this leg if weather is already there
            if('solar' in leg):
                print(f"Weather exists for \"{leg['name']}\" ")
                continue
            print(f"Generating weather for \"{leg['name']}\" ")

            #(dist, time) points where weather is evaluated
            weather_pts = []

            #values of weather elements at weather_pts
            weather_vals = {
                'solarradiation': [],
                'cloudcover': [],
                'windspeed': [],
                'winddir': [],
                'precip': [],
                'temp': [],
            }

            #get weather at points spaced dist_step meters apart, use tqdm loading bar
            dists = np.arange(0, leg['length']+dist_step, dist_step)
            for dist in tqdm(dists):
                print(f"Getting weather at {round(dist)} m")

                latitude = leg['latitude'](dist)
                longitude = leg['longitude'](dist)

                start_date = leg['open']
                year = start_date.year
                month = start_date.month
                start_day = start_date.day
                num_days = (leg['close'].day - start_day) + 1

                df, records_used_pt = visualcrossing.get_weather_range(
                    latitude, longitude,
                    start_day=datetime(year, month, start_day),
                    num_days=num_days,
                    doPrint=False
                )
                records_used += records_used_pt

                for index, row in df.iterrows():    #iterate through each hour of forecast
                    weather_pts.append((dist, row['datetimeEpoch']))

                    for key in weather_vals:         #iterate through solar, wind, etc.
                        weather_vals[key].append(row[key]) 

            print("Done getting weather at points, start interpolating")
            for key in weather_vals:
                interp = LinearNDInterpolator(points=weather_pts, values=weather_vals[key])
                leg[key] = interp #add interp to leg dict
        
        print(f"Finished adding weather data to legs {start_leg} to {stop_leg-1}")
        print(f"Used {records_used} records \n")
        return records_used


    def save_as(self, name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "wb") as f:
            pickle.dump(self, f)

    def open(name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "rb") as f:
            return pickle.load(f)


def main():

    # Generate route: 
    route = Route()
    route.add_base_leg(
        dir + '/route_data/gps/asc2022/stage1_ckpt1.csv', 
        open=   datetime(2022, 7, 9, 11, 15),
        release=datetime(2022, 7, 9, 13, 45),
        close=  datetime(2022, 7, 9, 13, 45),
    )
    route.add_loop_leg(
        dir + '/route_data/gps/asc2022/stage1_ckpt1_loop.csv', 
        open=   datetime(2022, 7, 9, 11, 15),
        close=  datetime(2022, 7, 9, 14, 00),
    )
    route.add_base_leg(
        dir + '/route_data/gps/asc2022/stage1_ckpt2.csv', 
        open=   datetime(2022, 7, 10, 9, 00),
        release=datetime(2022, 7, 11, 18, 00),
        close=  datetime(2022, 7, 10, 18, 00),
    )
    route.add_loop_leg(
        dir + '/route_data/gps/asc2022/stage1_ckpt2_loop.csv', 
        open=   datetime(2022, 7, 10, 11, 15),
        close=  datetime(2022, 7, 10, 14, 00),
    )

    # route.gen_weather(dist_step=10000)
    route.save_as("ind-gra_2022,7,9-10_10km")


    new_route = Route.open("ind-gra_2022,7,9-10_10km")

    # print(len(new_route.leg_list))

    # print(new_route)
    print(new_route.total_length)
    for leg in new_route.leg_list:
        print_dict(leg)
        print('\n')

        # dist_min = 0
        # dist_max = leg['length']
        # dist_res = 1000     #1 km
        # time_min = datetime(2022, 8, 2, 7, 0).timestamp()
        # time_max = datetime(2022, 8, 2, 20, 0).timestamp()
        # time_res = 30*60    #30 minutes
        
        # Dists, Times = np.mgrid[dist_min:dist_max:dist_res, time_min:time_max:time_res]

        # fig = plt.figure()
        # plt.scatter(to_dates(Times.flatten()), meters2miles(Dists.flatten()), c=leg['solarradiation'](Dists, Times).flatten(), cmap='inferno')


        # test_pt = (10000, datetime(2022, 8, 2, 12, 30).timestamp())


    # plt.show()

if __name__ == "__main__":
    main()