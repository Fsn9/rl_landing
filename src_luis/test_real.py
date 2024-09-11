#!/usr/bin/python

from ros2_msg import *
from environment_simple_test import *
from dqn import *
from agent import *
import matplotlib.pyplot as plt
import math 

 
def main(args=None):

    """
    Test the trained model by running several episodes and evaluating its performance.
    """
    # Load trained model and set parameters
    input_dictionary = torch.load(open("model", 'rb'))
    dict_keys = np.array(list(input_dictionary.keys())).astype(int)
    max_index = np.max(dict_keys)
    input_dictionary = input_dictionary[max_index]
    parameters = input_dictionary['parameters']

    # Initialize agent and load parameters
    agent = dqn()
    agent.set_parameters(parameters)
    agent.load_state(state=input_dictionary)

    # Initialize environment
    env = CustomEnv()
    env.reset()

    durations = []
    returns = []
    status_string = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
            "return over all episodes so far = {3:<6.1f}            ")
   
    N = 10
    verbose = True

    for i in range(N):

        state, info = env.reset()
        s = state.copy()
        print(s)
        episode_return = 0.
        
        for n in itertools.count():

            action = agent.act(state)            
            #
            state, step_reward, terminated, truncated, info = env.step(action)
            s = state.copy()
            #
            done = terminated or truncated
            episode_return += step_reward
            #
            if done:
                #
                durations.append(n+1)
                returns.append(episode_return)
                #
                if verbose:
                    if i < N-1:
                        end ='\r'
                    else:
                        end = '\n'
                    print(status_string.format(i+1,N,episode_return,
                                        np.mean(np.array(returns))),
                                    end=end)
                break
        
    env.close()

if __name__ == '__main__':
    main()
