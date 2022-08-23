from datetime import date, timedelta
from math import radians
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
import pickle
import os
import forecast.openmeteo

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm import tqdm
try:
    from .util import *     #when route.py is being imported elsewhere
except:
    from util import *      #when route.py is being run directly

dir = os.path.dirname(__file__)

CHARGE_START_HOUR = 7   #battery taken out of impound
DRIVE_START_HOUR = 9    #solar starts driving
DRIVE_STOP_HOUR = 18    #solar stops driving
CHARGE_STOP_HOUR = 20   #battery put into impound

MORNING_CHARGE_HOURS = DRIVE_START_HOUR - CHARGE_START_HOUR
EVENING_CHARGE_HOURS = CHARGE_STOP_HOUR - DRIVE_STOP_HOUR
HOURS_NOT_CHARGING = (DRIVE_START_HOUR + 24) - DRIVE_STOP_HOUR

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

        dists = df['distance (m)'].values
        longitude_interp = interp1d(dists, df['longitude'].values, fill_value="extrapolate")
        latitude_interp = interp1d(dists, df['latitude'].values, fill_value="extrapolate")
        slope_interp = interp1d(dists, df['slope (%)'].values, fill_value="extrapolate")
        altitude_interp = interp1d(dists, df['altitude (m)'].values, fill_value="extrapolate")
        headings_interp = interp1d(dists, df['course'].values, fill_value="extrapolate", kind='nearest') #use nearest because interpolating the angle wraparound at 0-360 sweeps through angles in between

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


    def add_leg(self, type:str, end:str, csv_path:str, start:datetime, open:datetime, close:datetime): 
        '''
            Add a dict to the route containing info of a base leg or loop. 
            Set type to 'base' or 'loop'. Set start to the first possible time that one can drive the leg,
            open to when the checkpoint/stagestop at the end of the leg opens, and close when one must 
            finish the leg. 
            Geographic data are interp1d objects. To get the slope at a distance d: leg_list\['slope'](d)
        '''
        assert type=='base' or type=='loop'
        assert end=='checkpoint' or end=='stagestop'

        geo = Route.get_geography(csv_path)
        self.total_length += geo['length']

        num_days = close.day - start.day + 1      #number of days that the leg can span
        max_time = (close - start).total_seconds()/3600. - HOURS_NOT_CHARGING*(num_days-1)
        min_time = (open - start).total_seconds()/3600. - HOURS_NOT_CHARGING*(num_days-1)

        leg = ({
            'name': geo['name'],
            'length': geo['length'],
            'type': type,
            'end': end,
            'start': start,
            'open': open,
            'close': close,
            'max_time': max_time,
            'min_time': min_time,
            'longitude': geo['longitude'],
            'latitude': geo['latitude'],
            'slope': geo['slope'],
            'altitude': geo['altitude'],
            'heading': geo['heading'],
        })
        self.leg_list.append(leg)


    def gen_weather(self, start_leg=0, stop_leg=-1, dist_step=miles2meters(15)):
        '''
        Generate 
        Weather data are 2D linear interpolants. To get the irradiance at a distance d and time t: leg_list\['solarradiance'](d, t)
        '''
        
        if(stop_leg == None or stop_leg == -1):
            stop_leg = len(self.leg_list)

        print(f"\nGenerating weather for legs {start_leg} to {stop_leg-1}")

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
            weather_vals = {}
            weather_vals['headwind'] = []

            #get weather at points spaced dist_step meters apart, use tqdm loading bar
            dists = np.arange(0, leg['length']+dist_step, dist_step)
            for dist in tqdm(dists):

                latitude = leg['latitude'](dist).item()
                longitude = leg['longitude'](dist).item()

                print(f"\n Getting weather at {round(dist)}m: ({round(latitude, 5)}, {round(longitude, 5)})")

                #fill weather_pts and weather_vals
                timestamps, wind_solars = forecast.openmeteo.get_wind_solar(latitude, longitude, leg['start'], leg['close'])
                roaddir = leg['heading'](dist).item()

                for i in range(len(timestamps)):
                    weather_pts.append((dist, timestamps[i]))
                    for val_name in wind_solars:
                        if(val_name not in weather_vals):
                            weather_vals[val_name] = [wind_solars[val_name][i]]
                        else:
                            weather_vals[val_name].append(wind_solars[val_name][i])

                    speed = wind_solars['windspeed_10m'][i]
                    winddir = wind_solars['winddirection_10m'][i]
                    headwind = speed * cos(radians(winddir - roaddir))
                    weather_vals['headwind'].append(headwind)
                    
            del weather_vals['windspeed_10m']
            del weather_vals['winddirection_10m']

            for key in weather_vals:
                interp = LinearNDInterpolator(points=weather_pts, values=weather_vals[key])
                leg[key] = interp #add interp to leg dict
        
            print(f"Finished adding weather data to leg {leg['name']}")

    def save_as(self, name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "wb") as f:
            pickle.dump(self, f)

    def open(name):
        with open(dir + '/route_data/saved_routes/' + name + '.route', "rb") as f:
            return pickle.load(f)


def main():

    # Generate route: 
    route = Route()
    route.add_leg(
        type =      'base',
        end =       'checkpoint',
        csv_path =  dir + '/route_data/gps/asc2022/stage1_ckpt1.csv', 
        start =     datetime(2022, 7, 9, 9, 00),
        open =      datetime(2022, 7, 9, 11, 15),
        close =     datetime(2022, 7, 9, 13, 45),
    )
    route.add_leg(
        type =      'loop',
        end =       'checkpoint',
        csv_path =  dir + '/route_data/gps/asc2022/stage1_ckpt1_loop.csv', 
        start =     datetime(2022, 7, 9, 12, 00),   #add 45min to ckpt open for hold time
        open =      datetime(2022, 7, 9, 11, 15),
        close =     datetime(2022, 7, 9, 14, 00),
    )
    route.add_leg(
        type =      'base',
        end =       'stagestop',
        csv_path =  dir + '/route_data/gps/asc2022/stage1_ckpt2.csv', 
        start =     datetime(2022, 7, 9, 13, 45),   #ckpt1 earliest release time
        open =      datetime(2022, 7, 10, 9, 00),
        close =     datetime(2022, 7, 10, 18, 00),
    )
    route.add_leg(
        type =      'loop',
        end =       'stagestop',
        csv_path =  dir + '/route_data/gps/asc2022/stage1_ckpt2_loop.csv', 
        start =     datetime(2022, 7, 10, 9, 45),   #add 45min to stage open for hold time
        open =      datetime(2022, 7, 10, 9, 00),
        close =     datetime(2022, 7, 10, 18, 00),
    )
    ## UNCOMMENT BELOW TO GENERATE ROUTE FILE
    # route.gen_weather(dist_step=5000)
    # route.save_as("ind-gra_2022,7,9-10_5km_openmeteo")


    new_route = Route.open("ind-gra_2022,7,9-10_5km_openmeteo")
    print(new_route.total_length)

    for leg in new_route.leg_list:

        # if leg['name'] != 'BL. Grand Island Loop': continue

        print_dict(leg)
        print('\n')

        dist_min = 0
        dist_max = leg['length']
        dist_res = 1000     #1 km
        time_min = leg['start'].timestamp()
        time_max = leg['close'].timestamp()
        time_res = 10*60    #30 minutes
        
        Dists, Times = np.mgrid[dist_min:dist_max:dist_res, time_min:time_max:time_res]

        # plt.figure()
        # plt.title(f"{leg['name']} headwind")
        # plt.scatter(to_dates(Times.flatten()), meters2miles(Dists.flatten()), c=leg['headwind'](Dists, Times).flatten(), cmap='inferno')
        # plt.colorbar()

        # plt.figure()
        # plt.title(f"{leg['name']} sun_tilt")
        # plt.scatter(to_dates(Times.flatten()), meters2miles(Dists.flatten()), c=leg['sun_tilt'](Dists, Times).flatten(), cmap='inferno')
        # plt.colorbar()

        # plt.figure()
        # plt.title(f"{leg['name']} winddir")
        # plt.scatter(to_dates(Times.flatten()), meters2miles(Dists.flatten()), c=leg['winddirection_10m'](Dists, Times).flatten(), cmap='inferno')
        # plt.colorbar()

        plt.figure()
        var = 'slope'
        plt.title(f"{leg['name']} {var}")
        plt.plot(meters2miles(Dists.flatten()), leg[var](Dists.flatten()), 'o-')


    plt.show()

if __name__ == "__main__":
    main()