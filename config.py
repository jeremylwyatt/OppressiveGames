class Parameters:

    racist_likelihood = 0.05        # proportion of agents in the Blue pop that are strongly racist
    B = 0                           # Blue agent code (used to access memory)
    R = 1                           # Red agent code (used to access memory)
    max_mem_length = 4              # each agent has a maximum memory length
    number_of_rounds = 10000        # number of rounds of the Nash demand game
    window = 1000                   # window used to smooth time series of rewards for graph
    pop1size = 100                  # blue population size
    pop2size = 100                  # red population size
    interaction = 'both'            # 'both' means you can draw R,R and B,B as well as B,R and R,B
    betaA = 1                       # parameters controlling Beta density for strength of racist belief distribution
    betaB = 4
    agent_racism_index = 2          # index for where the parameter a (racism) is stored
    p_u_given_r = 0.5               # prob of racist utterance given racist beliefs
    p_u_given_not_r = 0.03          # prob of racist utterance given no racist beliefs
    max_audience_size = 1           # maximum size of an audience for an utterance
    slurring = True                 # either slurring is switched on or not


# define the payoffs
# define the possible bids (these can be payoffs)
L = 4  # Low bid
M = 5  # Medium bid
H = 6  # High bid

# define the disagreement points
# these are payoffs if agents bid for more resource than exists
B_D = 4  # Strong (blue) population disagreement point
R_d = 0  # Weak (red) population disagreement point

# define the reward (payoff) function for a single round of play
# between two agents using the above defined payoffs
# we index this reward 'matrix' with a 'row' index and a 'column' index
# first matrix is for B,B
# second matrix is for B,R
# third matrix is for R,B
# fourth matrix is for R,R

reward_function = [
    # Blue vs Blue
    [[[L, L], [L, M], [L, H]],
     [[M, L], [M, M], [B_D, B_D]],
     [[H, L], [B_D, B_D], [B_D, B_D]]],
    # Blue vs Red
    [[[L, L], [L, M], [L, H]],
     [[M, L], [M, M], [B_D, R_d]],
     [[H, L], [B_D, R_d], [B_D, R_d]]],
    # Red vs Blue
    [[[L, L], [L, M], [L, H]],
     [[M, L], [M, M], [R_d, B_D]],
     [[H, L], [R_d, B_D], [R_d, B_D]]],
    # Red vs Red
    [[[L, L], [L, M], [L, H]],
     [[M, L], [M, M], [R_d, R_d]],
     [[H, L], [R_d, R_d], [R_d, R_d]]]]

# define the moves as indices into the reward function
L_move = 0  # this indexes the first 'row' or the first 'column' above
M_move = 1  # this indexes the second 'row' or 'column'
H_move = 2  # this indexes the third 'row' or 'column'

moves = [L_move, M_move, H_move]
