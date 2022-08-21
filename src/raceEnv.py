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
import route
from util import *

dir = os.path.dirname(__file__)

class RaceEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"], 
        "render_fps": 4
    }

    def __init__(self, render_mode="human", car="brizo_22fsgp", route="ind-gra_2022,7,9-10_10km"):

        with open(f"{dir}/cars/{car}.json", 'r') as props_json:
            self.car_props = json.load(props_json)
        
        route_obj = Route.open(route)
        self.legs = route_obj.leg_list
        self.total_length = route_obj.total_length

        self.leg_index = 0
        self.leg_progress = 0
        self.energy = 0
        self.time = self.legs[0]['start']
        self.miles_earned = 0
        self.target_speed = 0
        self.try_loop = False
        self.done = False
        self.trailered = False

        self.observation_spaces= spaces.Dict({
            "dist_traveled": spaces.Box(0, self.route.total_length),
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
            "agent": self._agent_location, 
            "target": self._target_location,
        }

    def reset(self, energy_budget=5400):
        # We need the following line to seed self.np_random
        super().reset()

        self.leg_index = 0
        self.time = self.legs[0]['start']
        self.energy = 0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()

        self._renderer.reset()
        self._renderer.render_step()

        return observation


    def charge(self, time_length:timedelta, tilted=True, updateTime=True):

        leg = self.legs[self.leg_index]



        end_time = self.time + time_length

        #array of charging times in seconds, spaced a minute apart
        np.arange(self.time.timestamp(), end_time.timestamp()+60, step=60) 

        leg['solar'](dist, )


        if updateTime: self.time += time_length

        self.energy += solar_func(self.leg_progress, self.time) * time_length

    def start_next_base_leg(self):
        while(not self.legs[self.leg_index]['type'] == 'base'):
            self.leg_index += 1


    def timing(self):

        '''
        An absolute mess of logic that processes loops, holdtimes, charging hours.
        Assumes the race always ends in a stage stop, and there are never 2 loops in a row.
        '''
        leg = self.legs[self.leg_index]

        if(self.leg_progress >= leg['length']):     #leg finished

            if(self.time < leg['close']): miles_earned += leg['length'] #earn miles if completed on time

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
                        self.charge(next_leg['start']) #wait for release time
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

        
        action['target_m/s']

        power_ff = v/K_m * (K_d*(v - w)^2) + K_f + K_g*sin(slope)
        accel = K_m * power_ext / v

        power_ext = accel * v / K_m

        power_total = power_ff + power_ext

        energy -= power_total

        


        done = np.array_equal(self._agent_location, self._target_location)

        reward = 1 if done else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        self._renderer.render_step()

        return observation, reward, done, info

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
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

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
    # env = GridWorldEnv()''

    env = RaceEnv(render_mode='human')

    obs = env.reset()


    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break

    env.close()



if __name__ == "__main__":
    main()