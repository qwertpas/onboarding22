from datetime import timedelta
from tkinter import E
import gym
from gym import spaces
from gym.utils.renderer import Renderer
import pygame
import numpy as np
from numpy import sin, cos
import json
from route import MORNING_CHARGE_HOURS, EVENING_CHARGE_HOURS, Route
from util import *

dir = os.path.dirname(__file__)

class RaceEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"], 
        "render_fps": 4
    }

    def __init__(self, render_mode="human", car="brizo_fsgp22", route="ind-gra_2022,7,9-10_5km_openmeteo"):

        with open(f"{dir}/cars/{car}.json", 'r') as props_json:
            self.car_props = json.load(props_json)
        
        route_obj = Route.open(route)
        self.legs = route_obj.leg_list
        self.total_length = route_obj.total_length

        self.leg_index = 0
        self.leg_progress = 0
        self.speed = 0
        self.energy = 0
        self.time = self.legs[0]['start']
        self.miles_earned = 0
        self.target_speed = 0
        self.try_loop = False
        self.done = False
        self.trailered = False

        self.timestep = 5 #5 second intervals
        self.speed_tolerance = 0.447  #1mph

        self.observation_spaces= spaces.Dict({
            "dist_traveled": spaces.Box(0, self.total_length),
            "slope": spaces.Box(-10, 10)
        })

        #action is setting the target speed and choosing whether to try loops
        self.action_space = spaces.Dict({
            "target_speed": spaces.Box(0, mpersec2mph(self.car_props['max_mph'])),
            "try_loop": spaces.Discrete(2),
        })


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

    def reset(self, energy_budget=5400):
        # We need the following line to seed self.np_random
        super().reset()

        self.leg_index = 0
        self.time = self.legs[0]['start']
        self.energy = 0

        
        observation = self._get_obs()
        self._renderer.reset()
        self._renderer.render_step()

        return observation


    def charge(self, time_length:timedelta, tilted=True, updateTime=True):

        leg = self.legs[self.leg_index]



        end_time = self.time + time_length

        #array of charging times in seconds, spaced a minute apart
        np.arange(self.time.timestamp(), end_time.timestamp()+60, step=60) 

        # leg['solar'](dist, )


        if updateTime: self.time += time_length

        # self.energy += solar_func(self.leg_progress, self.time) * time_length


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
                        print('redo loop')
                        return
                    else:
                        print('do the upcoming loop')
                        self.leg_index += 1
                        return
                else:
                    self.charge(next_leg['start'] - self.time) #wait for release time
                    print('move onto next base leg')
                    self.leg_index += 1
                    return

            else: #checkpoint closed, need to move onto next base leg. To get to this point self.time is the close time.
                if(next_leg['type']=='loop'):
                    print('next leg is a loop, skipping to the base leg after that')
                    self.leg_index += 2
                    return
                else:
                    print('do the upcoming base leg')
                    self.leg_index += 1
                    return

        else:                   #leg ends at a stage stop.
                
            if(self.time < leg['close']): #arrived before stage close.

                self.charge(min(leg['close'] - self.time, holdtime)) #stay at stagestop for the required holdtime, or it closes

                if(self.time < leg['close']): #stage hasn't closed yet

                    if(self.try_loop and leg['type']=='loop'):
                        print('redo loop')
                        return

                    if(is_last_leg): #for final leg to get to this point, must be a loop and try_loop==False
                        print('completed last loop')
                        self.charge(leg['close'] - self.time)                        #charge until stage close
                        self.charge(timedelta(hours=EVENING_CHARGE_HOURS))    #evening charging
                        self.done = True
                        return
                    
                    #could be base route, or a loop that user doesn't want to try again.
                    self.charge(leg['close'] - self.time)                        #charge until stage close
                    
                #at this point stage must be closed and there's more tomorrow
                print('wait for next leg tomorrow')
                self.charge(timedelta(hours=EVENING_CHARGE_HOURS))    #evening charging
                self.time = next_leg['start'] - timedelta(MORNING_CHARGE_HOURS) #time skip to beginning of morning charging
                self.charge(timedelta(hours=MORNING_CHARGE_HOURS))    #morning charging
                self.leg_index += 1
                return

            else:
                if(leg['type']=='base'):
                    print('did not make stagestop on time, considered trailered')        
                    self.trailered = True
                else:
                    print('loop arrived after stage close, does not count. ')

                self.charge(leg['close'] - self.time + timedelta(hours=EVENING_CHARGE_HOURS))      #charge until end of evening charging
                self.time = next_leg['start'] - timedelta(MORNING_CHARGE_HOURS) #time skip to beginning of morning charging
                self.charge(timedelta(hours=MORNING_CHARGE_HOURS))    #morning charging
                self.leg_index += 1
                return





    def step(self, action):
        leg = self.legs[self.leg_index]

        a = action['target_accel']
        v_t = action['target_speed']
        v_0 = self.speed
        dt = self.timestep
        d_0 = self.leg_progress

        K_m = self.car_props['K_m'] #motor
        K_d = self.car_props['K_d'] #drag
        K_f = self.car_props['K_f'] #friction
        K_g = self.car_props['K_g'] #gravity
        P_max = self.car_props['P_max'] #max motor power
        v_max = mph2mpersec(self.car_props['max_mph']) 

        w = leg['headwind'](d_0)
        d_f = d_0 + v_t*dt      #estimate dist at end of step for now by assuming actualspeed=targetspeed
        sinslope = (leg['altitude'](d_f) - leg['altitude'](d_0)) / (d_f - d_0)

        if(np.abs(v_t - v_0) < 0.5):    #speed within target, no accel needed
            a = 0
        elif(v_0 > v_max):              #at max speed, can only decelerate
            a = min(a, 0)
        else:                           #limit accel based on max power
            a_max = 1/v_0 * (P_max*K_m - v_avg*K_d*(v_0-w)**2 - K_m*K_f - K_m*K_g*sinslope)
            a = min(a, a_max)

        v_f = v_0 + a*dt                #calculate v_f for real with updated accel
        v_avg = 0.5 * (v_0 + v_f)
        d_f += v_avg * dt               #trapezoidal integration is exact bc accel is constant throughout timestep


        power_ff = v_avg/K_m * (K_d*(v_avg - w)**2) + K_f + K_g*sinslope #probably possible to isolate fric, integrate after knowing time
        power_ext = a * v_avg / K_m
        self.motor_power = power_ff + power_ext

        self.energy -= self.motor_power

        if(d_f > leg['length']):
            self.try_loop = action['try_loop']
            self.process_leg_finish()


        self.leg_progress = d_f
        self.speed = v_f
        observation = self._get_obs()

        # self._renderer.render_step()

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

    action = env.action_space.sample()

    observation, reward, done = env.step(action)

    while not done:
        # Take a random action
        action = env.action_space.sample()
        observation, reward, done = env.step(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break

    env.close()



if __name__ == "__main__":
    main()