import gym
import pygame
from gym.utils.play import play, PlayPlot

mapping = {
    (pygame.K_LEFT,): [-2], 
    (pygame.K_RIGHT,): [2],
}


def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]
plotter = PlayPlot(callback, 30 * 5, ["reward"])
env = gym.make("Pendulum-v1")
play(env, keys_to_action=mapping, noop=[0])