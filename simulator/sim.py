from raceEnv import RaceEnv
from simulator.blit import Display



def main():
    env = RaceEnv(render=False)
    disp = Display(env)
    while(True):
        disp.update()





if __name__ == "__main__":
    main()