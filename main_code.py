#!/usr/bin/python

from ros2_msg import *
#from environment import *
from environment_simple import *
from rl_landing.controller import dqn
from agent import *
import matplotlib.pyplot as plt
import argparse

def save_q_values_to_file(agent, filename="dict_S_Qv.txt"):
    """
    Save the Q-values for different grid positions to a file.
    """
    with open(filename, 'a') as file:
        for i in np.arange(-6, 6, GRID_RESOL):
            for j in np.arange(-6, 6, GRID_RESOL):
                for k in np.arange(0, 8, GRID_RESOL):
                    policy_net = agent.neural_networks['policy_net'].to(device)
                    action_scores = policy_net(torch.tensor([i, j, k], dtype=torch.float).to(device))
                    file.write(f"({i}, {j}, {k}) -> {action_scores}\n")


def evaluate_model(agent, env, N=5, verbose=True):
    """
    Evaluate the trained agent over N episodes and return the durations and returns.
    """
    durations, returns = [], []
    status_string = ("Run {0} of {1} completed with return {2:<5.1f}. "
                     "Mean return over all episodes so far = {3:<6.1f}            ")

    for i in range(N):
        state, info = env.reset()
        episode_return = 0.0
        time.sleep(1)

        for n in itertools.count():
            action = agent.act(state)
            state, step_reward, terminated, truncated, info = env.step(action)

            episode_return += step_reward
            done = terminated or truncated

            if done:
                durations.append(n + 1)
                returns.append(episode_return)

                if verbose:
                    end = '\r' if i < N - 1 else '\n'
                    print(status_string.format(i + 1, N, episode_return, np.mean(np.array(returns))), end=end)
                break
        time.sleep(1)

    return durations, returns


def running_mean(x, N=20):
    """
    Compute the running mean of a list or array.
    """
    x_out = np.zeros(len(x) - N, dtype=float)
    for i in range(len(x) - N):
        x_out[i] = np.mean(x[i:i + N + 1])
    return x_out


def plot_returns_and_durations(training_results, filename=None, N=20):
    """
    Plot both the return per episode and the duration per episode during training.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot return per episode
    x = training_results['episode_returns']
    t = np.arange(len(x)) + 1
    ax.plot(t, x, label='training', color='dodgerblue')

    # Add running mean
    x_running_mean = running_mean(x=x, N=N)
    t_running = np.arange(len(x_running_mean)) + N
    ax.plot(t_running, x_running_mean, color='black', label='running mean')

    # Customize plot
    ax.axhline(200, ls='--', color='red')
    ax.set_ylim(-499, 350)
    ax.set_xlim(0, len(t_running) + 100)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')

    ax.legend(loc='lower right')
    plt.show()

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def train():
    """
    Train the agent in the environment.
    """
    # Initialize the agent and environment
    agent = dqn()
    env = CustomEnv()
    env.reset()

    # Train the agent
    training_results = agent.train(environment=env, verbose=True, model_filename='model', training_filename='train')

    # Flush and close the writer (assuming a TensorBoard writer or similar)
    writer.flush()
    writer.close()
    env.close()

    # Save Q-values to file after training
    save_q_values_to_file(agent)

    # Plot returns and durations
    plot_returns_and_durations(training_results=training_results)


def test():
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

    # Evaluate the trained model
    durations, returns = evaluate_model(agent, env, N=5, verbose=True)

    env.close()


def main():
    """
    Main function that allows the user to choose between 'train' and 'test' modes.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or test the DQN agent.")
    parser.add_argument("mode", choices=['train', 'test'], help="Choose whether to train or test the agent.")

    # Parse the arguments
    args = parser.parse_args()

    # Run the appropriate mode
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()


if __name__ == '__main__':
    main()
