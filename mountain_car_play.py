from CustomMountainCar import CustomMountainCar

env = CustomMountainCar(extra_curve=False, extra_const=3.0)
env.reset()
env.play_with_pygame()
#env.plot()