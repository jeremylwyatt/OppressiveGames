
# this file runs the simulation of a Nash demand game with many rounds
# the key parameters are set in the config file config.py
# the main program file is in nash-demand-game-simulation.py

# first import the config file
from config import *

# then import the main program file
from nash_demand_game_simulation import *

# create an instance of the key parameters class
params = Parameters()

# create the population of all agents (red and blue)
agents = create_population(params)

# call simulation
# this returns a reward and a move history for each agent population (blue, red)
# it requires a population of blue and red agents, a reward function, and a set of moves (bids for resources)
r_hist_blue, r_hist_red, ag_move_hist_blue, ag_move_hist_red = simulate(agents, reward_function, moves, params)

# print the reward histories
print(r_hist_blue)
print(r_hist_red)

# turn the lists into numpy.arrays
time_series1 = numpy.array(r_hist_blue)
time_series2 = numpy.array(r_hist_red)

# analyse the resulting reward time series for blue and red
# 1000 is the window size for averaging
analyse_simulation(time_series1, time_series2, params.window)

print('done')
