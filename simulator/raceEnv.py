from datetime import timedelta
from tkinter import E
import gym
from gym import spaces
from gym.utils.renderer import Renderer
import pygame
import numpy as np
from numpy import sin, cos
import json
import matplotlib.pyplot as plt

import sys, os
dir = os.path.dirname(__file__)
sys.path.insert(0, dir+'/../')   #allow imports from parent directory "onboarding22"

from route.route import MORNING_CHARGE_HOURS, EVENING_CHARGE_HOURS, Route
from util import *



class RaceEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"], 
        "render_fps": 4
    }

    def __init__(self, render_mode="human", car="brizo_fsgp22", route="ind-gra_2022,7,9-10_5km_openmeteo"):

        cars_dir = os.path.dirname(__file__) + '/../cars'
        with open(f"{cars_dir}/{car}.json", 'r') as props_json:
            self.car_props = json.load(props_json)
        
        route_obj = Route.open(route)
        self.legs = route_obj.leg_list

        self.leg_index = 0
        self.leg_progress = 0
        self.speed = 0
        self.energy = self.car_props['max_watthours']*3600  #joules left in battery
        self.brake_energy = 0                               #joules dissipated in mechanical brakes
        self.time = self.legs[0]['start'] #datetime object
        self.miles_driven = 0
        self.miles_earned = 0
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

        self.log = {
            "times": [],
            "dists": [],
            "speeds": [],
            "energies": []
        }


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = Renderer(self.render_mode, self._render_frame)

        self.window = None
        self.window_size = 512  # The size of the PyGame window
        self.clock = None

    def _get_obs(self):
        return {
            'speed': self.speed,
            'energy': self.energy,
        }

    def reset_leg(self):
        self.leg_progress = 0
        self.speed = 0
        self.next_stop_dist = 0
        self.next_stop_index = 0
        self.limit = None
        self.next_limit_dist = 0
        self.next_limit_index = 0
        

    def reset(self):
        # We need the following line to seed self.np_random
        super().reset()

        self.leg_index = 0
        self.time = self.legs[0]['start']
        self.energy = self.car_props['max_watthours']*3600

        self.reset_leg()
        
        observation = self._get_obs()
        self._renderer.reset()
        self._renderer.render_step()

        return observation


    def charge(self, time_length:timedelta):
        '''
        Updates energy and time, simulating the car sitting in place tilting its array optimally towards the sun for a period of time
        '''
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

        if(self.time < leg['close']): self.miles_earned += leg['length'] #earn miles if completed on time

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
                        print(f"Redo loop: {leg['name']}")
                        return
                    else:
                        print(f"Do the upcoming loop: {next_leg['name']}")
                        self.leg_index += 1
                        return
                else:
                    
                    while next_leg['type']=='loop':
                        self.leg_index += 1
                        next_leg = self.legs[self.leg_index]

                    self.charge(next_leg['start'] - self.time) #wait for release time                        

                    print(f"End at checkpoint, move onto next base leg: {next_leg['name']}")
                    return

            else: #checkpoint closed, need to move onto next base leg. To get to this point self.time is the close time.
                if(next_leg['type']=='loop'):
                    print('Next leg is a loop, skipping to the base leg after that')
                    self.leg_index += 2
                    return
                else:
                    print('Do the upcoming base leg')
                    self.leg_index += 1
                    return

        else:                   #leg ends at a stage stop.
                
            if(self.time < leg['close']): #arrived before stage close.

                self.charge(min(leg['close'] - self.time, holdtime)) #stay at stagestop for the required holdtime, or it closes

                if(self.time < leg['close']): #stage hasn't closed yet after serving hold time

                    if(self.try_loop and leg['type']=='loop'):
                        print(f"Completed the loop and trying it again: {leg['name']}")
                        self.charge(timedelta(minutes=15))
                        return

                    if(is_last_leg): #for final leg to get to this point, must be a loop and try_loop==False
                        print('Completed the loop at the end of the race, and will not be attemping more.')
                        self.charge(leg['close'] - self.time)                        #charge until stage close
                        self.charge(timedelta(hours=EVENING_CHARGE_HOURS))    #evening charging
                        self.done = True
                        return
                    
                    #could be base route, or a loop that user doesn't want to try again.
                    self.charge(leg['close'] - self.time)                        #charge until stage close
                    
                #at this point stage must be closed. 
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
                self.time = next_leg['start'] - timedelta(MORNING_CHARGE_HOURS) #time skip to beginning of morning charging
                self.charge(timedelta(hours=MORNING_CHARGE_HOURS))    #morning charging
                # self.leg_index += 1
                return

            else:
                if(leg['type']=='base'):
                    print('did not make stagestop on time, considered trailered')        
                    self.done = True
                    return
                else:
                    print('loop arrived after stage close, does not count.')
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

        v_0 = self.speed
        dt = self.timestep
        d_0 = self.leg_progress     #meters completed of the current leg
        w = leg['headwind'](d_0, self.time.timestamp())

        self.log['times'].append(self.time)
        self.log['dists'].append(self.leg_progress)
        self.log['speeds'].append(self.speed)
        self.log['energies'].append(self.energy)

        P_max_out = self.car_props['max_motor_output_power'] #max motor drive power (positive)
        P_max_in = self.car_props['max_motor_input_power'] #max regen power (positive)
        min_mph = self.car_props['min_mph']
        max_mph = self.car_props['max_mph']

        assert action['acceleration'] > 0, "Acceleration must be positive"
        assert action['deceleration'] < 0, "Deceleration must be negative"
        assert action['target_mph'] > min_mph and action['target_mph'] < max_mph, f"Target speed must be between {min_mph} mph and {max_mph} mph"

        a_acc = float(action['acceleration'])
        a_dec = float(action['deceleration'])
        v_t = float(action['target_mph'])

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

                if(self.next_stop_index+1 < len(leg['stop_dists'])):
                    self.next_stop_index += 1
                    self.next_stop_dist = leg['stop_dists'][self.next_stop_index]  #completed the stop
                else:
                    self.next_stop_dist = leg['length']

                observation = self._get_obs()
                return self._get_obs, self.miles_earned, self.done


        # CALCULATE ACTUAL ACCELERATION
        d_f = d_0 + v_t*dt      #estimate dist at end of step for now by assuming actualspeed=targetspeed
        sinslope = (leg['altitude'](d_f) - leg['altitude'](d_0)) / (d_f - d_0)      #approximate slope

        P_drag = self.car_props['P_drag']
        P_fric = self.car_props['P_fric']
        P_accel = self.car_props['P_accel']
        mg = self.car_props['mass'] * 9.81

        v_t = min(v_t, self.limit) #apply speed limit to target speed
        v_error = v_t - v_0
        if(v_error > 0):        #need to speed up, a > 0
            if(np.abs(v_0) > 0.1):
                motor_accel_limit = 1/P_accel * (P_max_out/v_0 - P_drag*(v_0-w)**2 - P_fric - mg*sinslope) #max achieveable accel for motor
                a = min(a_acc, motor_accel_limit)
            else:
                a = a_acc
        else:                   #need to slow down, a < 0
            if(np.abs(v_0) > 0.1):
                motor_decel_limit = 1/P_accel * (P_max_in/v_0 - P_drag*(v_0-w)**2 - P_fric - mg*sinslope) #max achieveable decel for motor (negative)
                brake_power = motor_decel_limit - a_dec             #power that mechanical brakes dissipate
                self.brake_energy += brake_power * dt
            a = a_dec           #assume accel can always reach the amount needed because of mechanical brakes

                

        # CALCULATE DISTANCE, SPEED, AND POWER NEEDED
        v_f = v_0 + a*dt                #get speed after accelerating
        v_avg = 0.5 * (v_0 + v_f)       #speed increases linearly so v_avg can be used in integration with no accuracy loss
        d_f += v_avg * dt               #integrate velocity to get distance at end of step
        self.leg_progress = d_f
        self.speed = v_f    

        alt_change = leg['altitude'](d_f) - leg['altitude'](d_0)
        self.motor_power = self.get_motor_power(a, v_avg, w, d_f-d_0, alt_change)
        self.array_power = leg['sun_flat'](d_0, self.time.timestamp()) * self.car_props['array_multiplier']

        self.energy += (self.motor_power - self.array_power) * dt
        self.energy = min(self.energy, self.car_props['max_watthours']*3600)


        # CHECK IF COMPLETED CURRENT LEG
        if(d_f >= leg['length']):
            print(f"Completed leg: {leg['name']}")
            self.try_loop = action['try_loop']
            self.process_leg_finish() #will update leg and self.done if needed
            self.reset_leg()

        # else:
            # print(d_f, leg['length'], self.speed)


        # self._renderer.render_step()

        observation = self._get_obs()
        reward = self.miles_earned
        return observation, reward, self.done










    def render(self):
        return self._renderer.get_renders()

    def _render_frame(self, mode):
        assert mode is not None

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size
        )  # The size of a single grid square in pixels

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def main():
    env = RaceEnv(render_mode='human')

    obs = env.reset()

    action = {
        "target_mph": 35,
        "acceleration": 1,
        "deceleration": -1,
        "try_loop": False,
    }

    observation, reward, done = env.step(action)

    while not done:
        # Take a random action
        # action = env.action_space.sample()
        print(env.leg_progress, env.leg_index)

        observation, reward, done = env.step(action)
        
        # Render the game
        # env.render()
        
        if done == True:
            break

    times = env.log['times']
    dists = np.array(env.log['dists']) * meters2miles()
    speeds = np.array(env.log['speeds']) * mpersec2mph()
    energies = np.array(env.log['energies']) / 3600.

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(times, dists, label='miles')
    axs[1].plot(times, speeds, label='mph')
    axs[2].plot(times, energies, label='watthours in battery')
    fig.legend()
    plt.show()

    env.close()



if __name__ == "__main__":
    main()