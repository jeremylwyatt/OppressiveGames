import random
import numpy
import matplotlib.pyplot as plt

# This script runs rounds of simulation of a Nash demand game
# there are the following functions
# play(agent_i_move, agent_j_move, reward_func)
# simulate(agents, reward_func, moves, number_of_rounds = 100, max_mem_length = 4, pop1size = 100, pop2size = 100)
# record_reward_history(agent_i_pop_code, agent_j_pop_code, outcome, reward_history_blue, reward_history_yellow)
# agent_choose(agent_i, agent_i_number, agent_j, agent_j_number, reward_func, moves)
# select_reward_matrix(agent_i_pop_code, agent_j_pop_code, reward_func)
# best_response(agent, policy_other_agent, reward_matrix, moves, agent_order)
# expected_payoff(action, policy_other_agent, reward_matrix, moves, agent_order)
# convert_moves(agent_moves)
# prob_dist(agent_memory, moves)
# analyse_simulation(ts1, ts2, j)


# define population sizes for each group
population1_size = 100
population2_size = 100

# define the payoffs
# define the possible bids (these can be payoffs)
L = 4  # Low bid
M = 5  # Medium bid
H = 6  # High bid

B = 0  # Powerful population code
Y = 1  # Weaker population code

# define the disagreement points
# these are payoffs if agents bid for more resource than exists
B_D = 4  # Strong (blue) population disagreement point
Y_d = 0  # Weak (yellow) population disagreement point

# define the reward (payoff) function for a single round of play
# between two agents using the above defined payoffs
# we index this reward 'matrix' with a 'row' index and a 'column' index
# first matrix is for B,B
# second matrix is for B,Y
# third matrix is for Y,B
# fourth matrix is for Y,Y

reward_function = [
# Blue vs Blue
                    [[[L, L], [L, M], [L, H]],
                     [[M, L], [M, M], [B_D, B_D]],
                     [[H, L], [B_D, B_D], [B_D, B_D]]],
# Blue vs Yellow
                    [[[L, L], [L, M], [L, H]],
                     [[M, L], [M, M], [B_D, Y_d]],
                     [[H, L], [B_D, Y_d], [B_D, Y_d]]],
# Yellow vs Blue
                    [[[L, L], [L, M], [L, H]],
                     [[M, L], [M, M], [Y_d, B_D]],
                     [[H, L], [Y_d, B_D], [Y_d, B_D]]],
# Yellow vs Yellow
                    [[[L, L], [L, M], [L, H]],
                     [[M, L], [M, M], [Y_d, Y_d]],
                     [[H, L], [Y_d, Y_d], [Y_d, Y_d]]]]

# define the moves as indices into the reward function
L_move = 0  # this indexes the first 'row' or the first 'column' above
M_move = 1  # this indexes the second 'row' or 'column'
H_move = 2  # this indexes the third 'row' or 'column'

moves = [L_move, M_move, H_move]

# the list of agents from population 1
# each agent comprises
# code indicating membership of population 1 (blue population)
# a memory of interactions with blue agents (always come first)
# a memory of interactions with yellow agents (always come second)
# all memories are empty
agents = []
for x in range(0, population1_size):
    if random.randint(1, 100) > 95:
        a = 1
    else:
        a = random.betavariate(1, 4)
    agents.append([[random.randint(0, 2)], [random.randint(0, 2)], a, B])

# append the list of agents from population 2
for y in range(1, population2_size):
    agents.append([[random.randint(0, 2)], [random.randint(0, 2)], 0, Y])


#print(len(agents))

# each agent is completely defined by its memory
# agents = [agent1_memory, agent2_memory, ...agent_i_memory, agent_j_memory...]
max_memory_length = 4

# this returns the result of the moves of each agent
def play(agent_i_move, agent_j_move, reward_func):

    # this line adds together the population codes of the two agents
    # to determine which of the three reward matrices should be used
    # reward_matrix_index = agent_i_pop_code + agent_j_pop_code

    # return the reward pair for the moves of agent_i and agent_j
    return reward_func[agent_i_move][agent_j_move]


# this runs multiple rounds of agents playing each other
def simulate(agents, reward_func, moves, number_of_rounds=100, max_mem_length=4, pop1size=100, pop2size=100, interaction='both'):

    # how many rounds
    rounds = range(1, number_of_rounds)

    # reward history
    reward_history_blue = []
    reward_history_orange = []
    agent_move_history_blue = []
    agent_move_history_orange = []

    # for each round
    for turn in rounds:

        # run a single round of the game and print the result
        print()
        print('round ' + str(turn))

        if interaction == 'inter':
            # choose a pair of agents at random
            # one from each group
            agent_i = random.randint(0, pop1size - 1)
            agent_j = random.randint(pop1size, pop1size + pop2size - 2)
        elif interaction == 'both':
            # choose a pair of agents at random
            # could be any group combination
            agent_i = random.randint(0, pop1size + pop2size - 2)
            agent_j = random.randint(0, pop1size + pop2size - 2)

        print('agent_i ' + str(agent_i))
        print('agent_j ' + str(agent_j))

        # recover the population type of each agent
        # it's always the first element in the agent
        agent_i_pop_code = agents[agent_i][-1]
        agent_j_pop_code = agents[agent_j][-1]

        # each agent chooses what to do
        # based on their memory of the moves of the other agent
        agent_i_move, agent_j_move, reward_matrix = agent_choose(agents[agent_i], agent_i, agents[agent_j], agent_j, reward_func, moves)

        # if either agent is from the dominant group, they might be racist
        if agent_i_pop_code == B or agent_j_pop_code == B:
            # find out first if one makes a racist utterance
            utterance_i, utterance_j = agent_utterance(agents[agent_i], agents[agent_j], agent_i_pop_code, agent_j_pop_code)
            # if so update the beliefs of some other agents in the blue group
            if utterance_i == 'slur':
                agents = update_beliefs(agents, pop1size+pop2size-2)
            elif utterance_j == 'slur':
                agents = update_beliefs(agents, pop1size+pop2size-2)

        # add agent_j's move to agent_i's memory
        agents[agent_i][agent_j_pop_code].append(agent_j_move)

        if len(agents[agent_i][agent_j_pop_code]) > max_mem_length:
            # delete the oldest (first) item on agent1's memory
            agents[agent_i][agent_j_pop_code].pop(0)

        # add agent_i's move to agent_j's memory
        agents[agent_j][agent_i_pop_code].append(agent_i_move)

        if len(agents[agent_j][agent_i_pop_code]) > max_mem_length:
            # delete the oldest (first) item on agent_j's memory
            agents[agent_j][agent_i_pop_code].pop(0)

        outcome = play(agent_i_move, agent_j_move, reward_matrix)
        print('rewards ' + str(outcome))

        reward_history_blue, reward_history_orange = record_reward_history(agent_i_pop_code, agent_j_pop_code, outcome, reward_history_blue, reward_history_orange)

        agent_move_history_blue, agent_move_history_orange = record_move_history(turn, agent_i_pop_code, agent_i_move, agent_j_pop_code, agent_j_move, agent_move_history_blue, agent_move_history_orange)


    return reward_history_blue, reward_history_orange, agent_move_history_blue, agent_move_history_orange


def agent_utterance(agent_i, agent_j, agent_i_pop_code, agent_j_pop_code):

    agent_racism_indice = 2

    p_u_given_r = 0.5
    utterance_i = 'null'
    utterance_j = 'null'

    #
    if agent_i_pop_code == 0 and agent_j_pop_code == 1:
        print('agent racism ' + str(agent_i[agent_racism_indice]))
        print('p of slur ' + str(p_u_given_r))
        if random.betavariate(1,1) < (p_u_given_r * agent_i[agent_racism_indice]):
            utterance_i = 'slur'

    if agent_j_pop_code == 0 and agent_i_pop_code == 1:
        if random.betavariate(1, 1) < (p_u_given_r * agent_j[agent_racism_indice]):
            utterance_j = 'slur'


    return utterance_i, utterance_j


# this updates the racism level of audience members exposed to a racist utterance
def update_beliefs(agents, popsize):

    p_u_given_r = 0.5
    p_u_given_not_r = 0.03
    agent_racism_indice = 2

    # Pick N the number of agents in audience
    # up to 10
    N = random.randint(1, 3)
    # select that number of audience members randomly
    audience = [random.randint(0, popsize) for i in range(1, N)]

    # now for each individual
    for individual in audience:
        # find out how racist it was to begin with
        p_r = agents[individual][agent_racism_indice]
        # update its racism using Bayes' rule
        post_r_given_u = (p_u_given_r * p_r)/(p_u_given_r * p_r + p_u_given_not_r * (1-p_r))
        agents[individual][agent_racism_indice] = post_r_given_u

    return agents



# appends the reward histories with the latest results
def record_reward_history(agent_i_pop_code, agent_j_pop_code, outcome, reward_history_blue, reward_history_yellow):

    if agent_i_pop_code == 0:
        reward_history_blue.append(outcome[0])
    else:
        reward_history_yellow.append(outcome[0])

    if agent_j_pop_code == 0:
        reward_history_blue.append(outcome[1])
    else:
        reward_history_yellow.append(outcome[1])

    return reward_history_blue, reward_history_yellow



# records the move history, tagged by the turn number
# grouped by agent colour
def record_move_history(turn, agent_i_pop_code, agent_i_move, agent_j_pop_code, agent_j_move, agent_move_history_blue, agent_move_history_orange):

    if agent_i_pop_code == 0:
        agent_move_history_blue.append([agent_i_move, turn])
    elif agent_i_pop_code == 1:
        agent_move_history_orange.append([agent_i_move, turn])

    if agent_j_pop_code == 0:
        agent_move_history_blue.append([agent_j_move, turn])
    elif agent_j_pop_code == 1:
        agent_move_history_orange.append([agent_j_move, turn])

    return agent_move_history_blue, agent_move_history_orange



# this function chooses the best response to the memory of the opponent's move
def agent_choose(agent_i, agent_i_number, agent_j, agent_j_number, reward_func, moves):

    # list position where racism score is stored
    agent_racism_indice = 2

    # recover the population type of each agent
    # it's always the last element in the agent
    agent_i_pop_code = agent_i[-1]
    agent_j_pop_code = agent_j[-1]

    print('Agent i type ' + str(agent_i_pop_code))
    print('Agent j type ' + str(agent_j_pop_code))

    # first obtain the policy of the other agent you are playing against
    # from your memory [what you remember they played in the last rounds]
    agent_i_memory_of_policy_other_agent = prob_dist(agent_i[agent_j_pop_code], moves)
    agent_j_memory_of_policy_other_agent = prob_dist(agent_j[agent_i_pop_code], moves)

    print('Agent i memory of agent j type policy ' + str(agent_i_memory_of_policy_other_agent))
    print('Agent j memory of agent i type policy ' + str(agent_j_memory_of_policy_other_agent))

    # then obtain policies of agents of your own type (if they are different)
    agent_i_memory_of_policy_own_group = prob_dist(agent_i[agent_i_pop_code], moves)
    agent_j_memory_of_policy_own_group = prob_dist(agent_j[agent_j_pop_code], moves)

    # choose reward matrix to use for this agent pair
    reward_matrix = select_reward_matrix(agent_i_pop_code, agent_j_pop_code, reward_func)
    # choose best responses for each agent against the other
    agent_i_best_response, payoffs_i_to_j = best_response(agent_i, agent_i_memory_of_policy_other_agent, reward_matrix, moves, 0)
    agent_j_best_response, payoffs_j_to_i = best_response(agent_j, agent_j_memory_of_policy_other_agent, reward_matrix, moves, 1)

    # now repeat for agent_i against vs its own group member
    # choose reward matrix to use for this agent pair
    reward_matrix_own_i = select_reward_matrix(agent_i_pop_code, agent_i_pop_code, reward_func)
    # choose best responses for each agent against the other
    agent_i_best_response_to_i, payoffs_i_to_i = best_response(agent_i, agent_i_memory_of_policy_own_group, reward_matrix_own_i, moves, 0)

    # now repeat for agent_j against vs its own group member
    # choose reward matrix to use for this agent pair
    reward_matrix_own_j = select_reward_matrix(agent_j_pop_code, agent_j_pop_code, reward_func)
    # choose best responses for each agent against the other
    agent_j_best_response_to_j, payoffs_j_to_j = best_response(agent_j, agent_j_memory_of_policy_own_group,
                                                              reward_matrix_own_j, moves, 1)
    # print('payoffs_i_to_j' + str(payoffs_i_to_j))
    # print('payoffs_i_to_i' + str(payoffs_i_to_i))
    # print('payoffs_j_to_i' + str(payoffs_j_to_i))
    # print('payoffs_j_to_j' + str(payoffs_j_to_j))

    # treat out group members no worse than you would in group members
    if max(payoffs_i_to_i) < max(payoffs_i_to_j):
        # agent will not follow equality maxim if they are racist
        # but this is randomised
        if random.betavariate(1, 1) > agent_i[agent_racism_indice]:
            agent_i_best_response = agent_i_best_response_to_i

    if max(payoffs_j_to_j) < max(payoffs_j_to_i):
        # agent will not follow equality maxim if they are racist
        # but this is randomised
        if random.betavariate(1, 1) > agent_j[agent_racism_indice]:
            agent_j_best_response = agent_j_best_response_to_j

    # print the agent memory and the best response for each agent
    print('agent number ' + str(agent_i_number + 1) + ' memory ' + str(convert_moves(agent_i[agent_j_pop_code])) + ' move ' + str(
            convert_moves([agent_i_best_response])))
    print('agent number ' + str(agent_j_number + 1) + ' memory ' + str(convert_moves(agent_j[agent_i_pop_code])) + ' move ' + str(
            convert_moves([agent_j_best_response])))

    return agent_i_best_response, agent_j_best_response, reward_matrix


# this function returns the correct reward matrix
def select_reward_matrix(agent_i_pop_code, agent_j_pop_code, reward_func):

    # agent_population_codes
    B = 0
    Y = 1

    # pick matrix
    if agent_i_pop_code == B and agent_j_pop_code == B:
        reward_matrix_index = 0
    elif agent_i_pop_code == B and agent_j_pop_code == Y:
        reward_matrix_index = 1
    elif agent_i_pop_code == Y and agent_j_pop_code == B:
        reward_matrix_index = 2
    elif agent_i_pop_code == Y and agent_j_pop_code == Y:
        reward_matrix_index = 3

    # return it
    return reward_func[reward_matrix_index]


# this computes the best response as expectimax for the reward (payoff) received this round
# agent_order is the specification of whether we are optimising for
# the first or second agent as listed in each reward pair
# in the reward matrix
def best_response(agent, policy_other_agent, reward_matrix, moves, agent_order):

    # record the value of the best response so far
    # start with 0 as a lower bound
    maximum = 0

    # set up list to store payoffs
    payoffs = []

    # for each possible action
    for action in moves:

        #print('action ' + str(action))

        # calculate expected payoff for action
        payoff_for_action = expected_payoff(action, policy_other_agent, reward_matrix, moves, agent_order)
        #print('value of action ' + str(payoff_for_action))

        payoffs.append(payoff_for_action)

        # if action has highest expected payoff so far
        # then record both payoff and action as best_response
        if payoff_for_action > maximum:
            maximum = payoff_for_action
            best_response = action

    return best_response, payoffs


# calculates the expected payoff for a given action
def expected_payoff(action, policy_other_agent, reward_matrix, moves, agent_order):

    # print which agent you are (row or column)
    #if agent_order == 0:
    #    print('agent order == 0')
    #else:
    #    print('agent order == 1')

    # create a running total for the expected value of the opponent's move
    running_total = 0
    # for each possible opponent response
    for opponent_action in moves:
        # calculate the probability of that response times the value of the outcome to you
        # use this to increment the running total
        if agent_order == 0:
            # whether you are row agent
            running_total += policy_other_agent[opponent_action] * reward_matrix[action][opponent_action][agent_order]
        else:
            # or column agent
            running_total += policy_other_agent[opponent_action] * reward_matrix[opponent_action][action][agent_order]

    return running_total


# this utility function converts agent actions from numbers into letters
# this is only to be used for printing to enable readability of the output
def convert_moves(agent_moves):

    agent_moves_letters = []

    for move in agent_moves:
        if move == 0:
            agent_moves_letters.append('L')
        elif move == 1:
            agent_moves_letters.append('M')
        elif move == 2:
            agent_moves_letters.append('H')

    return agent_moves_letters


# this returns a probability distribution over actions from an agent's memory
def prob_dist(agent_memory, moves):

    if agent_memory == []:
        probability_distribution = [1/len(moves) for x in range(0, len(moves))]

    else:
        # create an empty list to hold my probability distribution
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


def analyse_simulation(ts1, ts2, j):

    ts1_sum = numpy.cumsum(ts1)
    ts1_index = numpy.arange(1, len(ts1)+1, 1)
    ts1_av = ts1_sum / ts1_index

    ts2_sum = numpy.cumsum(ts2)
    ts2_index = numpy.arange(1, len(ts2) + 1, 1)
    ts2_av = ts2_sum / ts2_index

    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim(0, 6.1)
    plt.plot(ts1_av)
    plt.plot(ts2_av)
    plt.show()

    fig1 = plt.figure()
    ax = plt.gca()
    ax.set_ylim(0, 6.1)

    ts1_sum = numpy.zeros(len(ts1)-j)
    for i in range(j, len(ts1), 1):
        ts1_sum[i-j] = numpy.sum(ts1[i-j:i])/j

    ts2_sum = numpy.zeros(len(ts2)-j)
    for i in range(j, len(ts2), 1):
        ts2_sum[i-j] = numpy.sum(ts2[i-j:i])/j

    plt.plot(ts1_sum)
    plt.plot(ts2_sum)
    plt.show()



reward_history_blue, reward_history_orange, agent_move_history_blue, agent_move_history_orange = simulate(agents, reward_function, moves, number_of_rounds=10000, interaction='both')

print(reward_history_blue)
print(reward_history_orange)
# 0 is agent_number, agents[0] is memory, reward function, moves
#print(agent_choose(0, agents[0], reward_function, moves))

time_series1 = numpy.array(reward_history_blue)
time_series2 = numpy.array(reward_history_orange)

analyse_simulation(time_series1, time_series2, 1000)

