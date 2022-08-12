import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import pickle
import os

dir = os.path.dirname(__file__)

class Route():
    def __init__(self):
        # weather = LinearNDInterpolator()
        # stopsigns = []
        # slopes = pd.read_csv()

        self.leg_list = [] #list that is going to be filled with dictionaries, each 

        self.dists_list = []
        self.slopes_list = []
        self.latitudes_list = []
        self.longitudes_list = []

    def addBaseLeg(self, leg_csv):
        df = pd.read_csv(leg_csv)
        self.leg_list.append({
            'dists': df['distance (mi)'].values,
            'slopes': df['slope (%)'].values,
        })

        return

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

    print(new_route.leg_list)


if __name__ == "__main__":
    main()