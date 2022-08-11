from ast import main
import json
import pandas as pd

class Car:

    def __init__(self, car_name):
        self.car_name = car_name

        with open(f"./lib/cars/{car_name}.json", 'r') as props_json:
            self.props = json.load(props_json)
        

    def get_drag_coeff(self):
        return self.props['drag_coeff']




def main():
    car = Car('brizo')
    print(car.get_drag_coeff())

if __name__ == "__main__":
    main()