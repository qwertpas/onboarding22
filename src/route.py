from math import dist
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
import pickle
import os
import visualcrossing
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


    def get_geography(csv_path:str):
        df = pd.read_csv(csv_path)

        name = df['name'].iat[0] #get name from first row

        df.fillna(method='bfill', inplace=True) #fill leading NaNs with the first valid value

        #convert distance to meters
        if('distance (mi)' in df.columns):
            df['distance (m)'] = df['distance (mi)'] * miles_to_meters(1)
        else:
            assert 'distance (km)' in df.columns
            df['distance (m)'] = df['distance (km)'] * 1000

        if('altitude (ft)' in df.columns):
            df['altitude (m)'] = df['altitude (ft)'] * feet_to_meters(1)   

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


    def add_base_leg(self, csv_path:str, open: datetime, release: datetime, close: datetime): 
        '''
            add a dict containing the geographic info of a base leg, in interpolant form.
            To get the slope at a distance d, use leg_list['slope'](d)
        '''        
        geo = Route.get_geography(csv_path)
        self.leg_list.append({
            'name': geo['name'],
            'length': geo['length'],
            'type': 'base',
            'open': open,
            'release': release,
            'close': close,
            'longitude': geo['longitude'],
            'latitude': geo['latitude'],
            'slope': geo['slope'],
            'altitude': geo['altitude'],
            'heading': geo['heading'],
        })

    def add_loop_leg(self, csv_path:str, open: datetime, close: datetime):   
        '''
            add a dict containing the geographic info of a loop, in interpolant form.
            To get the slope at a distance d, use leg_list\['slope'](d)
        '''   
        geo = Route.get_geography(csv_path)
        self.leg_list.append({
            'name': geo['name'],
            'length': geo['length'],
            'type': 'loop',
            'open': open,
            'close': close,
            'longitude': geo['longitude'],
            'latitude': geo['latitude'],
            'slope': geo['slope'],
            'altitude': geo['altitude'],
            'heading': geo['heading'],
        })

    def gen_weather(self, start_leg=0, stop_leg=-1, dist_step=miles_to_meters(15), start_time=9):
        
        if(stop_leg == None or stop_leg == -1):
            stop_leg = len(self.leg_list)

        print(f"Generating weather for legs {start_leg} to {stop_leg-1}")

        for i in range(start_leg, stop_leg):
            leg = self.leg_list[i]

            if('solar' in leg):         #skip this leg if weather is already there
                print(f"Skipping, weather exists for {leg['name']}")
                continue
            print(f"Generating weather for {leg['name']}")

            #create dict of weather data types
            weather_pts = dict.fromkeys([
                'solar', 
                'cloudcover',
                'windspeed',
                'winddir',
                'precip',
                'temp'
            ])
            for key in weather_pts:
                weather_pts[key] = {
                    'points': [],       #array of tuples for (distance, time)
                    'values': [],       #array of scalars for value of weather data
                }

            dists = np.arange(0, leg['length']+dist_step, dist_step)
            for dist in tqdm(dists):    #get weather at points spaced dist_step meters apart, use tqdm loading bar
                print(f"Getting weather at {round(dist)} m")

                latitude = leg['latitude'](dist)
                longitude = leg['longitude'](dist)

                open_date = leg['open']
                year = open_date.year
                month = open_date.month
                start_day = open_date.day
                num_days = (leg['close'].day - start_day) + 1

                df = visualcrossing.get_weather_range(
                    latitude, longitude,
                    start_day=datetime(year, month, start_day),
                    num_days=num_days,
                    doPrint=False
                )

                # print(df)

                for index, row in df.iterrows():    #iterate through each hour of forecast
                    for key in weather_pts:         #iterate through solar, wind, etc.
                        weather_pts[key]['points'].append((dist, row['timestamp']))
                        weather_pts[key]['values'].append(row[key]) 

            print("Done generating weather points, start interpolating")
            for key in weather_pts:

                print(weather_pts[key])
                interp = LinearNDInterpolator(points=weather_pts[key]['points'], values=weather_pts[key]['values'])
                leg[key] = interp


    def add_loop(self, loop_csv):
        return

    def save_as(self, name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "wb") as f:
            pickle.dump(self, f)

    def open(name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "rb") as f:
            return pickle.load(f)


def main():
    # route = Route()
    # route.add_base_leg(
    #     dir + '/route_data/gps/asc2022/stage1_ckpt1.csv', 
    #     open=datetime(2022, 8, 2, 11, 15),
    #     release=datetime(2022, 8, 2, 13, 45),
    #     close=datetime(2022, 8, 2, 13, 45),
    # )
    # route.add_loop_leg(
    #     dir + '/route_data/gps/asc2022/stage1_ckpt1_loop.csv', 
    #     open=datetime(2022, 8, 2, 11, 15),
    #     close=datetime(2022, 8, 2, 14, 00),
    # )
    # route.gen_weather(dist_step=60000)
    # route.save_as("ind-top_2022-8-2")



    new_route = Route.open('ind-top_2022-8-2')
    new_route.gen_weather(dist_step=60000)

    # print(len(new_route.leg_list))

    # print(new_route)
    for leg in new_route.leg_list:
        print_dict(leg)
        print('\n')

    # print(new_route.leg_list[0]['latitude'](160000))

if __name__ == "__main__":
    main()