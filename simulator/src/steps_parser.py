import pandas as pd
import os
from util import *
from bisect import bisect_left

dir = os.path.dirname(__file__)

class StepsParser():    

    stopsigns = np.array([])

    def __init__(self, csv, keywords = ['SL ', 'Stop Sign', 'TURN']):
        df = pd.read_csv(csv, skiprows=2) #first two rows of csv exported from Excel is weird

        #magic that gets all the rows that contain the keywords
        stop_steps = df[df.stack().str.contains('|'.join(self.keywords)).groupby(level=0).any()]
        self.stopsigns = stop_steps['Trip'].to_numpy() * miles2meters(1)

        speedlimits = df[['Trip', 'Spd']].dropna(subset='Spd')

        self.dists = speedlimits['Trip'].to_numpy()
        self.limits = speedlimits['Spd'].to_numpy()

        self.speedlimit = lambda 

    def __call__(self, dist:float) -> float:
        speedlimit = self.limits[bisect_left(self.dists, dist)-1]
        return speedlimit


if __name__ == '__main__':
    steps = Speedlimits(dir + '/route_data/asc2022/steps/steps_stage1_ckpt1.csv')

    # print(steps.stops)
    print(steps(100))
