from datetime import timedelta
import gym
from gym import spaces
import numpy as np
import json


# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import sys, os


dir = os.path.dirname(__file__)
sys.path.insert(0, dir+'/../')   #allow imports from parent directory "onboarding22"

from simulator.blit import BlitManager
from route.route import *
from util import *

class RaceEnv(gym.Env):

    def __init__(self, render=True, car="brizo_fsgp22", route="ind-gra_2022,7,9-10_5km_openmeteo"):

        cars_dir = os.path.dirname(__file__) + '/../cars'
        with open(f"{cars_dir}/{car}.json", 'r') as props_json:
            self.car_props = json.load(props_json)
        
        route_obj = Route.open(route)
        self.legs = route_obj.leg_list

        self.legs_completed = 0
        self.current_leg = self.legs[0]
        self.leg_index = 0
        self.leg_progress = 0
        self.speed = 0
        self.energy = self.car_props['max_watthours']*3600  #joules left in battery
        self.brake_energy = 0                               #joules dissipated in mechanical brakes
        self.time = self.legs[0]['start'] #datetime object
        self.miles_driven = 0
        self.miles_earned = 0
        self.motor_power = 0
        self.array_power = 0
        
        self.try_loop = False

        self.done = False

        self.next_stop_dist = 0
        self.next_stop_index = 0
        self.limit = None
        self.next_limit_dist = 0
        self.next_limit_index = 0

        self.timestep = 5 #5 second intervals

        self.observation_spaces= spaces.Dict({
            "dist_traveled": spaces.Box(0, float('inf')),
            "slope": spaces.Box(-10, 10)
        })

        #action is setting the target speed and choosing whether to try loops
        self.action_space = spaces.Dict({
            "target_mph": spaces.Box(mph2mpersec(self.car_props['min_mph']), mph2mpersec(self.car_props['max_mph'])),
            "acceleration": spaces.Box(0, self.car_props['max_accel']),
            "deceleration": spaces.Box(self.car_props['max_decel'], 0),
            "try_loop": spaces.Discrete(2),
        })

        self.action_keyboard = {
            "target_mph": 5,
            "acceleration": self.car_props['max_accel'],
            "deceleration": self.car_props['max_decel'],
            "try_loop": False,
        }

        self.log = {
            "leg_names": [],
            "times": [[]],
            "dists": [[]],
            "speeds": [[]],
            "energies": [[]],
            "motor_powers": [[]],
            "array_powers": [[]],
        }

        self.do_render = render
        if(self.do_render):
            self.render_init()
            

        print(f"Start race at {self.time}")

    

    def reset_leg(self):
        self.leg_progress = 0
        self.speed = 0
        self.next_stop_dist = 0
        self.next_stop_index = 0
        self.limit = None
        self.next_limit_dist = 0
        self.next_limit_index = 0
        self.distwindow_l = 0
        self.distwindow_r = miles2meters(10)
        self.dists_window = np.arange(self.distwindow_l, self.distwindow_r, step=10)
        limit_dist_pts, limit_pts = self.current_leg['speedlimit']
        self.limit_dist_pts, self.limit_pts = ffill(limit_dist_pts, limit_pts)
        self.render_init()
        

    def reset(self):
        # We need the following line to seed self.np_random
        super().reset()

        self.leg_index = 0
        self.legs_completed = 0
        self.time = self.legs[0]['start']
        self.energy = self.car_props['max_watthours']*3600

        self.reset_leg()
        
        self.action_keyboard = {
            "target_mph": 5,
            "acceleration": self.car_props['max_accel'],
            "deceleration": self.car_props['max_decel'],
            "try_loop": False,
        }



    def charge(self, time_length:timedelta):
        '''
        Updates energy and time, simulating the car sitting in place tilting its array optimally towards the sun for a period of time
        '''
        time_length = max(timedelta(0), time_length)

        leg = self.legs[self.leg_index]
        end_time = self.time + time_length

        timestep = 60
        times = np.arange(self.time.timestamp(), end_time.timestamp()+timestep, step=timestep)
        irradiances = np.array([leg['sun_tilt'](self.leg_progress, time) for time in times])
        powers = irradiances * self.car_props['array_multiplier']

        self.energy += powers.sum()
        self.energy = min(self.energy, self.car_props['max_watthours']*3600)
        self.time = self.time + time_length


    def process_leg_finish(self):
        '''
        An absolute mess of logic that processes loops, holdtimes, charging hours.
        Assumes the race always ends in a stage stop, and there are never 2 loops in a row.
        '''
        leg = self.legs[self.leg_index]

        if(self.time > leg['close'] and (leg['type']=='loop' or leg['end']=='stagestop')):
            print("Earned 0 miles")
        else:
            print(f"Earned {round(meters2miles(leg['length']))} miles")
            self.miles_earned += leg['length'] #earn miles if completed on time

        is_last_leg = self.leg_index == (len(self.legs) - 1)
        if(is_last_leg and leg['type']=='base'):
            self.done = True
            print("Ended on a base leg. Completed entire route!")
            return

        holdtime = timedelta(minutes=15) if (leg['type']=='loop') else timedelta(minutes=45)

        if(self.time < leg['open']):    #if arrive too early, wait for checkpoint/stagestop to open
            self.charge(leg['open'] - self.time)

        if(leg['end'] == 'checkpoint'):

            self.charge(min(leg['close'] - self.time, holdtime)) #stay at checkpoint/stagestop for the required holdtime, or it closes

            next_leg = self.legs[self.leg_index+1] #ended at a checkpoint not stagestop, so there must be another leg

            if(self.time < leg['close']): #there's still time left after serving required hold time

                if(self.try_loop and (leg['type']=='loop' or next_leg['type']=='loop')): #there's a loop and user wants to do it
                    if(leg['type']=='loop'):
                        print(f"Redo loop: {leg['name']} at {self.time}")
                        return
                    else:
                        print(f"Try the upcoming loop: {next_leg['name']} at {self.time}")
                        self.leg_index += 1
                        return
                else:
                    
                    while next_leg['type']=='loop':
                        self.leg_index += 1
                        next_leg = self.legs[self.leg_index]

                    self.charge(next_leg['start'] - self.time) #wait for release time                        

                    print(f"End at checkpoint, move onto next base leg: {next_leg['name']} at {self.time}")
                    return

            else: #checkpoint closed, need to move onto next base leg. To get to this point self.time is the close time.
                if(leg['type']=='loop'):
                    print(f"Did not arrive before close: {leg['name']}")

                if(next_leg['type']=='loop'):
                    print('Next leg is a loop, skipping to the base leg after that')
                    self.leg_index += 2
                    return
                else:
                    while next_leg['type']!='base': #pick the next base leg
                        self.leg_index += 1
                        next_leg = self.legs[self.leg_index]
                    self.leg_index += 1
                    print(f"Start the upcoming base leg: {next_leg['name']} at {self.time}")
                    # print(self.legs[self.leg_index])
                    return

        else:                   #leg ends at a stage stop.
                
            if(self.time < leg['close']): #arrived before stage close.

                self.charge(min(leg['close'] - self.time, holdtime)) #stay at stagestop for the required holdtime, or it closes


                if(self.time < leg['close']): #stage hasn't closed yet after serving hold time

                    if(self.try_loop and leg['type']=='loop'):
                        print(f"Completed the loop and trying it again: {leg['name']} at {self.time}")
                        self.charge(timedelta(minutes=15))
                        return

                    if(is_last_leg): #for final leg to get to this point, must be a loop and try_loop==False
                        print('Completed the loop at the end of the race, and will not be attemping more.')
                        self.charge(leg['close'] - self.time)                        #charge until stage close
                        self.charge(timedelta(hours=EVENING_CHARGE_HOURS))    #evening charging
                        self.done = True
                        return

                    next_leg = self.legs[self.leg_index+1]
                    if(self.try_loop and next_leg['type']=='loop'):
                        print(f"Completed the base leg and trying the next loop: {next_leg['name']}")
                        self.charge(timedelta(minutes=15))
                        self.leg_index += 1
                        return
                    
                    #could be base route, or a loop that user doesn't want to try again.
                    self.charge(leg['close'] - self.time)                        #charge until stage close
                    
                #at this point stage must be closed. 
                if(is_last_leg):
                    print('Completed the last loop, not enough time to complete another. Race done.')
                    self.done = True
                    return

                next_leg = self.legs[self.leg_index+1]
                if(self.try_loop and next_leg['type']=='loop'):
                    self.leg_index += 1
                    print(f"Wait for next loop tomorrow: {next_leg['name']}")

                else:
                    while next_leg['type']!='base': #if don't want to try loop, pick the next base leg
                        self.leg_index += 1

                        if(not self.leg_index < len(self.legs)):
                            print(f"Completed last base leg: {leg['name']}")
                            self.done = True
                            return

                        next_leg = self.legs[self.leg_index]
                    print(f"Wait for next base leg tomorrow: {next_leg['name']}")
                
                self.charge(timedelta(hours=EVENING_CHARGE_HOURS))    #evening charging
                self.time = datetime(self.time.year, self.time.month, self.time.day+1, CHARGE_START_HOUR) #time skip to beginning of morning charging
                self.charge(timedelta(hours=MORNING_CHARGE_HOURS))    #morning charging
                # self.leg_index += 1
                return

            else:
                if(leg['type']=='base'):
                    print('Did not make stagestop on time, considered trailered')        
                    self.done = True
                    self.miles_earned = 0
                    return
                else:

                    if(is_last_leg):
                        print('Loop arrived after stage close, does not count. Race done.')
                        self.done = True
                        return

                    print('Loop arrived after stage close, does not count. Moving onto next leg.')
                    self.charge(leg['close'] - self.time + timedelta(hours=EVENING_CHARGE_HOURS))      #charge until end of evening charging
                    self.time = next_leg['start'] - timedelta(MORNING_CHARGE_HOURS) #time skip to beginning of morning charging
                    self.charge(timedelta(hours=MORNING_CHARGE_HOURS))    #morning charging
                    self.leg_index += 1
                    return



    def get_motor_power(self, accel, speed, headwind, dist_change, alt_change):
        '''
        Motor power loss in W, positive meaning power is used
        '''
        P_drag = self.car_props['P_drag']
        P_fric = self.car_props['P_fric']
        P_accel = self.car_props['P_accel']
        mg = self.car_props['mass'] * 9.81

        #can probably be made into a matrix
        power_ff = speed * (P_drag*(speed - headwind)**2 + P_fric + mg*(alt_change / dist_change))      #power used to keep the avg speed
        power_acc = P_accel*accel*speed                                                                 #power used to accelerate (or decelerate)
        return power_ff + power_acc


    def step(self, action):
        leg = self.legs[self.leg_index]
        self.current_leg = leg

        v_0 = self.speed
        dt = self.timestep
        d_0 = self.leg_progress     #meters completed of the current leg
        w = leg['headwind'](d_0, self.time.timestamp())

        self.log['times'][-1].append(self.time.timestamp())
        self.log['dists'][-1].append(self.leg_progress)
        self.log['speeds'][-1].append(self.speed)
        self.log['energies'][-1].append(self.energy)
        self.log['motor_powers'][-1].append(self.motor_power)
        self.log['array_powers'][-1].append(self.array_power)

        P_max_out = self.car_props['max_motor_output_power'] #max motor drive power (positive)
        P_max_in = self.car_props['max_motor_input_power'] #max regen power (positive)
        min_mph = self.car_props['min_mph']
        max_mph = self.car_props['max_mph']

        assert action['acceleration'] > 0, "Acceleration must be positive"
        assert action['deceleration'] < 0, "Deceleration must be negative"
        assert action['target_mph'] > min_mph and action['target_mph'] < max_mph, f"Target speed must be between {min_mph} mph and {max_mph} mph"

        a_acc = float(action['acceleration'])
        a_dec = float(action['deceleration'])
        v_t = float(action['target_mph']) * mph2mpersec()

        # SPEEDLIMIT
        if(d_0 >= self.next_limit_dist):     #update speed limit if passed next sign
            self.limit = leg['speedlimit'][1][self.next_limit_index]

            if(self.next_limit_index+1 < len(leg['speedlimit'][0])):
                self.next_limit_index += 1
                self.next_limit_dist = leg['speedlimit'][0][self.next_limit_index]
            else:
                self.next_limit_dist = float('inf')


        # STOPPING
        if(d_0 > self.next_stop_dist - 1000):       #check if within a reasonable stopping distance (1km)

            stopping_dist = -v_0*v_0 / (2*a_dec)    #calculate distance it would take to decel to 0 at current speed

            if(d_0 > self.next_stop_dist - stopping_dist):  #within distance to be able to decel to 0 at a constant decel
                a = a_dec
                v_avg = v_0/2.
                alt_change = leg['altitude'](self.next_stop_dist) - leg['altitude'](d_0)
                stopping_time = -v_0/a

                motor_power = self.get_motor_power(a, v_avg, w, stopping_dist, alt_change)
                array_power = leg['sun_flat'](d_0, self.time.timestamp()) * self.car_props['array_multiplier']
                self.energy += (array_power - motor_power) * stopping_time
                self.energy = min(self.energy, self.car_props['max_watthours']*3600)

                self.time += timedelta(seconds=stopping_time)
                self.leg_progress = self.next_stop_dist
                self.speed = 0

                if(self.next_stop_index+1 < len(leg['stop_dists'])):
                    self.next_stop_index += 1
                    self.next_stop_dist = leg['stop_dists'][self.next_stop_index]  #completed the stop
                else:
                    self.next_stop_dist = float('inf')

                return self.done


        # CALCULATE ACTUAL ACCELERATION
        d_f_est = d_0 + v_t*dt      #estimate dist at end of step for now by assuming actualspeed=targetspeed
        sinslope = (leg['altitude'](d_f_est) - leg['altitude'](d_0)) / (d_f_est - d_0)      #approximate slope

        P_drag = self.car_props['P_drag']
        P_fric = self.car_props['P_fric']
        P_accel = self.car_props['P_accel']
        mg = self.car_props['mass'] * 9.81

        v_t = min(v_t, self.limit) #apply speed limit to target speed
        v_error = v_t - v_0
        if(v_error > 0):        #need to speed up, a > 0
            if(np.abs(v_0) > 1): #avoid divide by 0
                motor_accel_limit = 1/P_accel * (P_max_out/v_0 - P_drag*(v_0-w)**2 - P_fric - mg*sinslope) #max achieveable accel for motor
                a = min(a_acc, motor_accel_limit)
            else:
                a = a_acc
        elif(v_error < 0):                   #need to slow down, a < 0
            if(np.abs(v_0) > 0.1):
                motor_decel_limit = 1/P_accel * (P_max_in/v_0 - P_drag*(v_0-w)**2 - P_fric - mg*sinslope) #max achieveable decel for motor (negative)
                brake_power = motor_decel_limit - a_dec             #power that mechanical brakes dissipate
                self.brake_energy += brake_power * dt
            a = a_dec           #assume accel can always reach the amount needed because of mechanical brakes
        else:
            a = 0

        if(np.abs(v_error) < a * dt): #adjust dt to not overshoot target speed
            dt = np.abs(v_error / a)

        # CALCULATE DISTANCE, SPEED, AND POWER NEEDED
        v_f = v_0 + a*dt                #get speed after accelerating
        v_avg = 0.5 * (v_0 + v_f)       #speed increases linearly so v_avg can be used in integration with no accuracy loss
        d_f = d_0 + v_avg * dt               #integrate velocity to get distance at end of step
        self.leg_progress = d_f
        self.speed = v_f

        alt_change = leg['altitude'](d_f) - leg['altitude'](d_0)
        self.motor_power = self.get_motor_power(a, v_avg, w, d_f-d_0, alt_change)
        self.array_power = leg['sun_flat'](d_0, self.time.timestamp()) * self.car_props['array_multiplier']

        self.energy += (self.array_power - self.motor_power) * dt
        self.energy = min(self.energy, self.car_props['max_watthours']*3600)
        self.time += timedelta(seconds=dt)

        # CHECK IF COMPLETED CURRENT LEG
        if(d_f >= leg['length']):
            print(f"Completed leg: {leg['name']} at {self.time}")
            self.legs_completed += 1
            self.try_loop = action['try_loop']
            self.process_leg_finish() #will update leg and self.done if needed


            #just do the first leg for now
            if(self.leg_index != 0):
                self.done = True
                return self.done

            self.current_leg = self.legs[self.leg_index]
            self.reset_leg()
            
            self.log['leg_names'].append(leg['name'])
            for item in self.log:
                if(item != 'leg_names'):
                    self.log[item].append([])

        # CHECK IF END OF DAY
        if(self.time > datetime(self.time.year, self.time.month, self.time.day, DRIVE_STOP_HOUR)):
            self.charge(timedelta(hours=CHARGE_STOP_HOUR - DRIVE_STOP_HOUR))
            self.time = datetime(self.time.year, self.time.month, self.time.day+1, CHARGE_START_HOUR)
            self.charge(timedelta(hours=DRIVE_START_HOUR - CHARGE_START_HOUR))
            print("End of day")


        if(self.do_render):
            self.render()

        return self.done




    def render_init(self):
        plt.close('all')

        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        plt.figure(figsize=(15, 13 * screen_height/screen_width))

        ax_elev = plt.subplot2grid((3, 8), (0, 0), colspan=8)
        ax_speed = plt.subplot2grid((3, 8), (1, 0), colspan=7, rowspan=2)
        ax_power = plt.subplot2grid((3, 8), (1, 7), rowspan=1)
        ax_battery = plt.subplot2grid((3, 8), (2, 7), rowspan=1)

        #in meters
        self.distwindow_l = 0
        self.distwindow_r = miles2meters(10)

        #elevation axes
        ax_elev.set_ylabel("Elevation (meters)")
        dists_leg = np.arange(0, self.current_leg['length'], step=30)
        elevs = self.current_leg['altitude'](dists_leg)
        self.min_elev = min(elevs)
        self.max_elev = max(elevs)
        (self.ln_elev,) = ax_elev.plot(dists_leg * meters2miles(), elevs, '-', label="elevation")
        (self.ln_distwindow_l,) = ax_elev.plot((meters2miles(self.distwindow_l), meters2miles(self.distwindow_l)), (self.min_elev, self.max_elev), 'y-')
        (self.ln_distwindow_r,) = ax_elev.plot((meters2miles(self.distwindow_r), meters2miles(self.distwindow_r)), (self.min_elev, self.max_elev), 'y-')
        (self.pt_elev,) = ax_elev.plot(0, self.current_leg['altitude'](0), 'ko', markersize=5)
        ax_elev.legend(loc='lower left')


        #speed axes
        ax_speed.set_ylabel("Speed (mph)")
        ax_speed.set_xlabel("Distance (miles)")
        ax_speed.set_xlim(0, meters2miles(self.distwindow_r))
        ax_speed.set_ylim(0, self.car_props['max_mph'])

        self.dists_window = np.arange(self.distwindow_l, self.distwindow_r, step=10)
        limit_dist_pts, limit_pts = self.current_leg['speedlimit']
        self.limit_dist_pts, self.limit_pts = ffill(limit_dist_pts, limit_pts)

        limit_dist_pts, limit_pts = trim_to_range(self.limit_dist_pts, self.limit_pts, self.dists_window[0], self.dists_window[-1])

        (self.ln_limit,) = ax_speed.plot(limit_dist_pts*meters2miles(), limit_pts*mpersec2mph(), label='Speed limit', c='gray')
        (self.ln_speed,) = ax_speed.plot(0, 0, label='Car speed', c='orange')
        (self.pt_speed,) = ax_speed.plot(0, 0, 'ko', markersize=5)

        ax_speed.legend(loc='upper left')


        #power axes
        ax_power.set_title("Array power (W)")
        ax_power.set_xlim(0, 3600)
        ax_power.set_ylim(0, 1000)
        ax_power.get_xaxis().set_visible(False)
        (self.ln_arraypower,) = ax_power.plot(0, 0, label='Array power', c='green')
        (self.ln_motorpower,) = ax_power.plot(0, 0, label='Motor power', c='red')


        plt.tight_layout()
        self.fig = plt.gcf()

        self.bm = BlitManager(self.fig, (
            self.pt_elev, self.ln_distwindow_l, self.ln_distwindow_r,
            self.ln_limit, self.ln_speed, self.pt_speed,
            self.ln_arraypower
        ))

        plt.show(block=False)
        plt.pause(.01) #wait a bit for things to be drawn and cached
        self.bm.update()

        #closing the first window deletes the second window
        def on_close(event):
            sys.exit()
        self.fig.canvas.mpl_connect('close_event', on_close)

        def press(event):
            print('press', event.key)
            if(event.key == 'up'):
                self.action_keyboard['target_mph'] += 50
                self.pt_elev.set_xdata(self.action_keyboard['target_mph'] * meters2miles())
                self.pt_elev.set_ydata(self.current_leg['altitude'](self.action_keyboard['target_mph']))
        self.fig.canvas.mpl_connect('key_press_event', press)





        self.tx = ax_speed.text(5, self.car_props['max_mph']-5, f"{self.current_leg['name']}", fontsize=20, ha='center')
        plt.pause(1)
        for i in [3, 2, 1]:
            self.tx.set_text(i)
            plt.pause(1)
        self.tx.set_text(f"{self.time.strftime('%m/%d/%Y, %H:%M')}")

    def render(self):
        self.pt_elev.set_xdata(meters2miles(self.leg_progress))
        self.pt_elev.set_ydata(self.current_leg['altitude'](self.leg_progress))

        if(self.leg_progress > miles2meters(7)):
            self.distwindow_l = self.leg_progress - miles2meters(7)
            self.distwindow_r = self.leg_progress + miles2meters(3)
            self.dists_window = np.arange(self.distwindow_l, self.distwindow_r, step=10)
        

        dists_so_far = np.array(self.log['dists'][self.legs_completed])
        speeds_so_far = np.array(self.log['speeds'][self.legs_completed])
        speeds_dists_window, speeds_window = trim_to_range(dists_so_far, speeds_so_far, self.dists_window[0], self.dists_window[-1])
        
        try:
            dist_shift = speeds_dists_window[0]
        except:
            dist_shift = 0

        self.ln_speed.set_xdata(meters2miles(speeds_dists_window-dist_shift))
        self.ln_speed.set_ydata(mpersec2mph(speeds_window))

        limit_dist_pts, limit_pts = trim_to_range(self.limit_dist_pts, self.limit_pts, self.dists_window[0] - miles2meters(1), self.dists_window[-1] + miles2meters(3))
        self.ln_distwindow_l.set_xdata((meters2miles(self.distwindow_l), meters2miles(self.distwindow_l)))
        self.ln_distwindow_r.set_xdata((meters2miles(self.distwindow_r), meters2miles(self.distwindow_r)))

        self.ln_limit.set_xdata(meters2miles(limit_dist_pts - dist_shift))
        self.ln_limit.set_ydata(mpersec2mph(limit_pts))
        
        self.pt_speed.set_xdata(meters2miles(self.leg_progress-dist_shift))
        self.pt_speed.set_ydata(mpersec2mph(self.speed))

        times_so_far = np.array(self.log['times'][self.legs_completed])
        arraypowers_so_far = np.array(self.log['array_powers'][self.legs_completed])
        times_window, arraypowers_window = trim_to_range(times_so_far, arraypowers_so_far, self.time.timestamp()-3600, self.time.timestamp())
        try:
            time_shift = times_window[0]
        except:
            time_shift = 0
        self.ln_arraypower.set_xdata((times_window-time_shift))
        self.ln_arraypower.set_ydata(arraypowers_window)

        self.tx.set_text(f"{self.time.strftime('%m/%d/%Y, %H:%M')}")


        self.bm.update()
        plt.pause(1e-6)







def main():


    # import tkinter as tk

    # root = tk.Tk()

    # screen_width = root.winfo_screenwidth()
    # screen_height = root.winfo_screenheight()

    env = RaceEnv(render=True)

    # print(env.legs)

    action = {
        "target_mph": 54,
        "acceleration": 0.5,
        "deceleration": -0.5,
        "try_loop": False,
    }


    while True:
        
        # action['target_mph'] = np.random.randint(6, 35)
        done = env.step(action)
        
        if done == True:
            break
    
    print(f"Total earned {env.miles_earned * meters2miles()} miles")

    # doPlot = True
    # if(not doPlot): return

    # for i in range(len(env.log['leg_names'])):
    #     times = env.log['times'][i]
    #     dists = np.array(env.log['dists'][i]) * meters2miles()
    #     speeds = np.array(env.log['speeds'][i]) * mpersec2mph()
    #     energies = np.array(env.log['energies'][i]) / 3600.
    #     motor_powers = np.array(env.log['motor_powers'][i])
    #     array_powers = np.array(env.log['array_powers'][i])
        
    #     for test_leg in env.legs:
    #         if test_leg['name'] == env.log['leg_names'][i]: leg = test_leg

    #     fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 13*screen_height/screen_width))
    #     axs[0].plot(times, dists, label='motor_power')
    #     axs[1].plot(times, speeds, label='mph')
    #     axs[2].plot(times, energies, label='watthours in battery')

    #     # axs[1].vlines(leg['stop_dists']*meters2miles(), ymin=0, ymax=55, colors='red', linewidth=0.5, label='stops')

    #     fig.suptitle(env.log['leg_names'][i])
    #     fig.legend()
    
    # plt.show()



if __name__ == "__main__":
    main()
