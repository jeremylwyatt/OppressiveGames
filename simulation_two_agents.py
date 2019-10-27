import random

# print(random.randint(1,100))

# This script runs rounds of simulation of a Nash demand game

# define population sizes for each group
population1_size = 100
population2_size = 100

# define the payoffs
# define the possible bids (these can be payoffs)
L = 4  # Low bid
M = 5  # Medium bid
H = 6  # High bid

# define the disagreement points
# these are payoffs if agents bid for more resource than exists
D = 0.5
d = 0

# define the reward (payoff) function for a single round of play
# between two agents using the above defined payoffs
# we index this reward 'matrix' with a 'row' index and a 'column' index
reward_function = [[[L, L], [L, M], [L, H]],
                   [[M, L], [M, M], [D, d]],
                   [[H, L], [D, d], [D, d]]]

# define the moves as indices into the reward function
L_play = 0  # this indexes the first 'row' or the first 'column' above
M_play = 1  # this indexes the second 'row' or 'column'
H_play = 2  # this indexes the third 'row' or 'column'

moves = [L_play, M_play, H_play]

# this is a memory we will use later on to determine agent moves automatically
# agent1_memory is agent1's memory of the moves of agent2
# and vice versa
agent1_memory = [M_play]  # [H_play, M_play, H_play, H_play]
agent2_memory = [M_play]  # [L_play, M_play, L_play, L_play]

# the list of agents
# each agent is completely defined by its memory
agents = [agent1_memory, agent2_memory]

max_memory_length = 4

# we define moves for both agents
agent1_move = L_play
agent2_move = M_play

# this returns the result of the moves of each agent
def play(agent1_move, agent2_move, reward_function):
    return reward_function[agent1_move][agent2_move]

# this runs multiple rounds of agents playing each other
def simulate(agents, reward_function, moves, number_of_rounds = 100, max_memory_length = 4):

    # how many rounds
    rounds = range(1, number_of_rounds)

    # for each round
    for round in rounds:

        # run a single round of the game and print the result
        print()
        print('round ' + str(round))

        # choose a pair of agents at random
        agent_i = 0  # rand(1,100)
        agent_j = 1  # rand(1,100)

        # each agent chooses what to do
        # based on its memory of the plays of the other agent
        agent_i_play = agent_choose(agent_i, agents[agent_i], reward_function, moves)
        agent_j_play = agent_choose(agent_j, agents[agent_j], reward_function, moves)

        # add agent_j's move to agent_i's memory
        agents[agent_i].append(agent_j_play)

        if len(agents[agent_i]) > max_memory_length:
            # pop the oldest (first) item on agent1's memory
            agents[agent_i].pop(0)

        # add agent_i's move to agent_j's memory
        agents[agent_j].append(agent_i_play)

        if len(agents[agent_j]) > max_memory_length:
            # pop the oldest (first) item on agent_j's memory
            agents[agent_j].pop(0)

        print('rewards ' + str(play(agent_i_play, agent_j_play, reward_function)))



def agent_choose(agent_number, agent_memory, reward_function, moves):

    # random move chosen by agent
    agent_move = random.randint(0, 2)

    if agent_number == 0:
        policy_other_agent = prob_dist(agent_memory, moves)

        maximum = 0
        for action in moves:
            running_total = 0
            for opponent_action in moves:
                running_total += policy_other_agent[opponent_action] * reward_function[action][opponent_action][agent_number]
            if running_total > maximum:
                maximum = running_total
                best_response = action

    elif agent_number == 1:
        policy_other_agent = prob_dist(agent_memory, moves)
        maximum = 0
        for action in moves:
            running_total = 0
            for opponent_action in moves:
                running_total += policy_other_agent[opponent_action] * reward_function[opponent_action][action][agent_number]
            if running_total > maximum:
                maximum = running_total
                best_response = action

    print('agent number ' + str(agent_number + 1) + ' memory ' + str(convert_agent(agent_memory)) + ' move ' + str(
            convert_agent([best_response])))

    return best_response




# this utility function converts agent actions from numbers into letters
# this is only to be used for printing to enable readability of the output
def convert_agent(agent_moves):

    agent_moves_letters = []

    for move in agent_moves:
        if move == 0:
            agent_moves_letters.append('L')
        elif move == 1:
            agent_moves_letters.append('M')
        elif move == 2:
            agent_moves_letters.append('H')

    return agent_moves_letters



def prob_dist(agent_memory, moves):

    # create a list of zeros to hold my probability distribution
    probability_distribution = []
    for move in moves:
        probability_distribution.append(0)

    # the probability increments are 1/length of the memory
    # (data the agent has to create the probability distribution)
    # if agent has memory length 4 this will 1/4
    increment = 1/len(agent_memory)

    # now build up the probability distribution from the memory
    for move in agent_memory:
        probability_distribution[move] += increment

    return probability_distribution




simulate(agents, reward_function, moves, 100)

# 0 is agent_number, agents[0] is memory, reward function, moves
#print(agent_choose(0, agents[0], reward_function, moves))


