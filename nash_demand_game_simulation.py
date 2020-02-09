# import the plotting library
import matplotlib.pyplot as plt
import random
import numpy

# This script runs rounds of simulation of a Nash demand game
# there are the following functions

# create_population(agents, population1_size, population2_size)
# play(agent_i_move, agent_j_move, reward_func)
# simulate(agents, reward_func, moves, number_of_rounds = 100, max_mem_length = 4, pop1size = 100, pop2size = 100)
# agent_utterance(agent_i, agent_j, agent_i_pop_code, agent_j_pop_code)
# update_beliefs(agents, popsize)
# record_reward_history(agent_i_pop_code, agent_j_pop_code, outcome, reward_history_blue, reward_history_red)
# agent_choose(agent_i, agent_i_number, agent_j, agent_j_number, reward_func, moves)
# select_reward_matrix(agent_i_pop_code, agent_j_pop_code, reward_func)
# best_response(agent, policy_other_agent, reward_matrix, moves, agent_order)
# expected_payoff(action, policy_other_agent, reward_matrix, moves, agent_order)
# convert_moves(agent_moves)
# prob_dist(agent_memory, moves)
# analyse_simulation(ts1, ts2, j)

# this function creates the agents one by one
# each agent comprises
# a memory of interactions with blue agents (first)
# a memory of interactions with red agents (second)
# racism index (third)
# code indicating membership of population (B=0 for population1; R=1 for population2) (last)
# all memories are initialised with a single imagined random play


def create_population(p):
    # initialise the agents to be an empty list
    agents = []
    # append the list of agents from population 1 (Blue, dominant)
    for x in range(0, p.pop1size):
        # with a probability of 0.05
        if random.randint(1, 100) <= p.racist_likelihood:
            # set the racism parameter to be 1
            # corresponds to believing any racist claim with probability 1
            a = 1
        else:
            # draw the racism parameter from a beta density
            # this is a density over the interval 0,1
            # the parameters 1,4 mean that the density is weighted towards 0
            a = random.betavariate(p.betaA, p.betaB)
        # append the next agent to the list of agents
        agents.append([[random.randint(0, 2)], [random.randint(0, 2)], a, p.B])

    # create and append, one by one, agents from population 2 (red, oppressed)
    for y in range(1, p.pop2size):
        # these agents always have a racism parameter of 0
        agents.append([[random.randint(0, 2)], [random.randint(0, 2)], 0, p.R])

    return agents


# this returns the result of the moves of each agent
def play(agent_i_move, agent_j_move, reward_func):

    # return the reward pair for the moves of agent_i and agent_j
    return reward_func[agent_i_move][agent_j_move]


# this runs multiple rounds of agents playing each other
def simulate(agents, reward_func, moves, p):

    # how many rounds
    rounds = range(1, p.number_of_rounds)

    # reward history
    reward_history_blue = []
    reward_history_red = []
    agent_move_history_blue = []
    agent_move_history_red = []

    # for each round
    for turn in rounds:

        # run a single round of the game and print the result
        print()
        print('round ' + str(turn))

        if p.interaction == 'inter':
            # choose a pair of agents at random
            # one from each group
            agent_i = random.randint(0, p.pop1size - 1)
            agent_j = random.randint(p.pop1size, p.pop1size + p.pop2size - 2)
        else:
            # p.interaction == 'both':
            # choose a pair of agents at random
            # could be any group combination
            agent_i = random.randint(0, p.pop1size + p.pop2size - 2)
            agent_j = random.randint(0, p.pop1size + p.pop2size - 2)

        print('agent_i ' + str(agent_i))
        print('agent_j ' + str(agent_j))

        # recover the population type of each agent
        # it's always the first element in the agent
        agent_i_pop_code = agents[agent_i][-1]
        agent_j_pop_code = agents[agent_j][-1]

        # each agent chooses what to do
        # based on their memory of the moves of the other agent
        agent_i_move, agent_j_move, reward_matrix = agent_choose(agents[agent_i], agent_i, agents[agent_j], agent_j, reward_func, moves, p)

        if p.slurring:
            # if either agent is from the dominant group, they might be racist
            if agent_i_pop_code == p.B or agent_j_pop_code == p.B:
                # find out first if one makes a racist utterance
                utterance_i, utterance_j = agent_utterance(agents[agent_i], agents[agent_j], agent_i_pop_code, agent_j_pop_code, p)
                # if so update the beliefs of some other agents in the blue group
                if utterance_i == 'slur':
                    agents = update_beliefs(agents, p.pop1size + p.pop2size - 2, p)
                elif utterance_j == 'slur':
                    agents = update_beliefs(agents, p.pop1size + p.pop2size - 2, p)

        # add agent_j's move to agent_i's memory
        agents[agent_i][agent_j_pop_code].append(agent_j_move)

        if len(agents[agent_i][agent_j_pop_code]) > p.max_mem_length:
            # delete the oldest (first) item on agent_i's memory
            agents[agent_i][agent_j_pop_code].pop(0)

        # add agent_i's move to agent_j's memory
        agents[agent_j][agent_i_pop_code].append(agent_i_move)

        if len(agents[agent_j][agent_i_pop_code]) > p.max_mem_length:
            # delete the oldest (first) item on agent_j's memory
            agents[agent_j][agent_i_pop_code].pop(0)

        # return the reward for the bids
        outcome = play(agent_i_move, agent_j_move, reward_matrix)
        print('rewards ' + str(outcome))

        # record the rewards in the reward histories
        reward_history_blue, reward_history_red = record_reward_history(agent_i_pop_code, agent_j_pop_code, outcome, reward_history_blue, reward_history_red, p)

        # record the moves
        agent_move_history_blue, agent_move_history_red = record_move_history(turn, agent_i_pop_code, agent_i_move, agent_j_pop_code, agent_j_move, agent_move_history_blue, agent_move_history_red, p)

    return reward_history_blue, reward_history_red, agent_move_history_blue, agent_move_history_red


# This tests whether a slurring utterance will be generated
# The utterance is generated with a probability dependent on
# r - accepted a situationally relevant racist belief
# v - decided to violate a moral norm given a situationally relevant racist belief
def agent_utterance(agent_i, agent_j, agent_i_pop_code, agent_j_pop_code, p):
    utterance_i = 'null'
    utterance_j = 'null'

    # if either agent is racist there is a chance they say something racist
    if agent_i_pop_code == p.B and agent_j_pop_code == p.R:
        print('agent racism ' + str(agent_i[p.agent_racism_index]))
        print('p of slur ' + str(p.p_u_given_r))
        # probability something racist is uttered
        if random.betavariate(1, 1) < (p.p_u_given_r * agent_i[p.agent_racism_index]):
            utterance_i = 'slur'
            print(utterance_i)

    if agent_j_pop_code == p.B and agent_i_pop_code == p.R:
        if random.betavariate(1, 1) < (p.p_u_given_r * agent_j[p.agent_racism_index]):
            utterance_j = 'slur'
            print(utterance_j)

    return utterance_i, utterance_j


# this updates the racism level of audience members exposed to a racist utterance
def update_beliefs(agents, popsize, p):

    # Pick N the number of agents in audience
    # up to 10
    N = random.randint(1, p.max_audience_size)

    print("audience size " + str(N))

    # select that number of audience members randomly
    audience = [random.randint(0, popsize) for i in range(0, N)]

    # now for each individual
    for individual in audience:
        # find out how racist it was to begin with
        p_r = agents[individual][p.agent_racism_index]
        # update its racism using Bayes' rule
        post_r_given_u = (p.p_u_given_r * p_r)/(p.p_u_given_r * p_r + p.p_u_given_not_r * (1-p_r))

        print("agent " + str(individual) + " had P(r)=" + str(p_r) + " now has P(r)=" +str(post_r_given_u))
        agents[individual][p.agent_racism_index] = post_r_given_u

    return agents


# appends the reward histories with the latest results
def record_reward_history(agent_i_pop_code, agent_j_pop_code, outcome, reward_history_blue, reward_history_red, p):

    if agent_i_pop_code == p.B:
        reward_history_blue.append(outcome[0])
    else:
        reward_history_red.append(outcome[0])

    if agent_j_pop_code == p.B:
        reward_history_blue.append(outcome[1])
    else:
        reward_history_red.append(outcome[1])

    return reward_history_blue, reward_history_red


# records the move history, tagged by the turn number
# grouped by agent colour
def record_move_history(turn, agent_i_pop_code, agent_i_move, agent_j_pop_code, agent_j_move, agent_move_history_blue, agent_move_history_red, p):

    if agent_i_pop_code == p.B:
        agent_move_history_blue.append([agent_i_move, turn])
    elif agent_i_pop_code == p.R:
        agent_move_history_red.append([agent_i_move, turn])

    if agent_j_pop_code == p.B:
        agent_move_history_blue.append([agent_j_move, turn])
    elif agent_j_pop_code == p.R:
        agent_move_history_red.append([agent_j_move, turn])

    return agent_move_history_blue, agent_move_history_red


# this function chooses the best response to the memory of the opponent's move
def agent_choose(agent_i, agent_i_number, agent_j, agent_j_number, reward_func, moves, p):

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
    reward_matrix = select_reward_matrix(agent_i_pop_code, agent_j_pop_code, reward_func, p)

    # choose best responses for each agent against the other
    agent_i_best_response, payoffs_i_to_j = best_response(agent_i, agent_i_memory_of_policy_other_agent, reward_matrix, moves, 0)
    agent_j_best_response, payoffs_j_to_i = best_response(agent_j, agent_j_memory_of_policy_other_agent, reward_matrix, moves, 1)

    # now repeat for agent_i against vs its own group member
    # choose reward matrix to use for this agent pair
    reward_matrix_own_i = select_reward_matrix(agent_i_pop_code, agent_i_pop_code, reward_func, p)
    # choose best responses for each agent against the other
    agent_i_best_response_to_i, payoffs_i_to_i = best_response(agent_i, agent_i_memory_of_policy_own_group, reward_matrix_own_i, moves, 0)

    # now repeat for agent_j against vs its own group member
    # choose reward matrix to use for this agent pair
    reward_matrix_own_j = select_reward_matrix(agent_j_pop_code, agent_j_pop_code, reward_func, p)
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
        if random.betavariate(1, 1) > agent_i[p.agent_racism_index]:
            agent_i_best_response = agent_i_best_response_to_i

    if max(payoffs_j_to_j) < max(payoffs_j_to_i):
        # agent will not follow equality maxim if they are racist
        # but this is randomised
        if random.betavariate(1, 1) > agent_j[p.agent_racism_index]:
            agent_j_best_response = agent_j_best_response_to_j

    # print the agent memory and the best response for each agent
    print('agent number ' + str(agent_i_number) + ' memory ' + str(convert_moves(agent_i[agent_j_pop_code])) + ' move ' + str(
            convert_moves([agent_i_best_response])))
    print('agent number ' + str(agent_j_number) + ' memory ' + str(convert_moves(agent_j[agent_i_pop_code])) + ' move ' + str(
            convert_moves([agent_j_best_response])))

    return agent_i_best_response, agent_j_best_response, reward_matrix


# this function returns the correct reward matrix
def select_reward_matrix(agent_i_pop_code, agent_j_pop_code, reward_func, p):

    # pick matrix
    if agent_i_pop_code == p.B and agent_j_pop_code == p.B:
        reward_matrix_index = 0
    elif agent_i_pop_code == p.B and agent_j_pop_code == p.R:
        reward_matrix_index = 1
    elif agent_i_pop_code == p.R and agent_j_pop_code == p.B:
        reward_matrix_index = 2
    else:
        # agent_i_pop_code == p.R and agent_j_pop_code == p.R:
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

        # print('action ' + str(action))

        # calculate expected payoff for action
        payoff_for_action = expected_payoff(action, policy_other_agent, reward_matrix, moves, agent_order)
        # print('value of action ' + str(payoff_for_action))

        payoffs.append(payoff_for_action)

        # if action has highest expected payoff so far
        # then record both payoff and action as best_response
        if payoff_for_action > maximum:
            maximum = payoff_for_action
            best_response = action

    return best_response, payoffs


# calculates the expected payoff for a given action
def expected_payoff(action, policy_other_agent, reward_matrix, moves, agent_order):

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

