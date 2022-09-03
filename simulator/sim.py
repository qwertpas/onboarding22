from raceEnv import RaceEnv



def main():
    env = RaceEnv(render_mode='human')

    obs = env.reset()

    action = env.action_space.sample()
    print(action)

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