import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
import warnings
import time
from torch.utils.tensorboard import SummaryWriter
from rl_landing.ros2_msg import *

from rl_landing.networks import neural_network

from rl_landing.replay_memory import Transition, memory

writer = SummaryWriter(log_dir=f"Reward_size_{str(GRID_RESOL)}")

device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class agent_base():

    def __init__(self):
        """
        Initializes the agent class
        
        """
        n_state = 3
        n_actions = 9

        default_parameters = {     
            'neural_networks':
                {
                'policy_net':{
                    'layers':[n_state,256,256,128,n_actions],
                            },
                'target_net':{
                    'layers':[n_state,256,256,128,n_actions],
                            }            
                },
            'optimizers':
                {
                'policy_net':{
                    'optimizer':'RMSprop',
                        'optimizer_args':{'lr':1e-3}, # learning rate
                            }
                },
            'losses':
                {
                'policy_net':{
                    'loss':'MSELoss',
                }
                },
            #
            'target_net_update_stride': 1,
            'target_net_update_tau': 1e-2,
            #
            'epsilon':1.0, # initial value for epsilon
            'epsilon_1':0.1, # final value for epsilon
            'd_epsilon':0.005, # decrease of epsilon
            #
            'n_memory':20000,
            'training_stride':5,
            'batch_size':32,
            'saving_stride':100,
            #
            'n_episodes_max': 2000,
            'n_solving_episodes': 100,
            'solving_threshold_min':2500,
            'solving_threshold_mean':3500,
            #
            'discount_factor':0.99,
            #
            'n_state':n_state, 
            'n_actions':n_actions,
            #
            'doubledqn': False,
        }

        #set parameters
        self.set_parameters(parameters=default_parameters)
        self.parameters = copy.deepcopy(default_parameters)

        #initialize neural_network
        self.initialize_neural_networks(neural_networks=\
                                            default_parameters['neural_networks'])
        
        # initialize the optimizer and loss function used for training
        self.initialize_optimizers(optimizers=default_parameters['optimizers'])
        self.initialize_losses(losses=default_parameters['losses'])
        
        self.in_training = False

  
    def set_parameters(self,parameters):
        """Set training parameters"""
        
        ########################################
        # Discount factor for Bellman equation #
        ########################################
        try: 
            self.discount_factor = parameters['discount_factor']
        except KeyError:
            pass
        
        #################################
        # Experience replay memory size #
        #################################
        try: 
            self.n_memory = int(parameters['n_memory'])
            self.memory = memory(self.n_memory)
        except KeyError:
            pass
        
        ###############################
        # Parameters for optimization #
        ###############################
        try: # number of simulation timesteps between optimization steps
            self.training_stride = parameters['training_stride']
        except KeyError:
            pass
        
        try: # size of mini-batch for each optimization step
            self.batch_size = int(parameters['batch_size'])
        except KeyError:
            pass
        
        try: # IO during training: every saving_stride episodes, the
            # current status of the training is saved to disk
            self.saving_stride = parameters['saving_stride']
        except KeyError:
            pass
        
        ##############################################
        # Parameters for training stopping criterion #
        ##############################################
        try: # maximal number of episodes until the training is stopped
            # (if stopping criterion is not met before)
            self.n_episodes_max = parameters['n_episodes_max']
        except KeyError:
            pass
        
        try: # # of the last N_solving episodes that need to fulfill the
            # stopping criterion for minimal and mean episode return
            self.n_solving_episodes = parameters['n_solving_episodes']
        except KeyError:
            pass
        
        try: # minimal return over last N_solving_episodes
            self.solving_threshold_min = parameters['solving_threshold_min']
        except KeyError:
            pass
        
        try: # mean return over last N_solving_episodes
            self.solving_threshold_mean = parameters['solving_threshold_mean']
        except KeyError:
            pass
        
        try: # set parameter N_state
            self.n_state = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state (= # of input"\
                         +" nodes for neural network) needs to be supplied.")
        
        try: # set parameter N_actions
            self.n_actions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions (= # of output"\
                         +" nodes for neural network) needs to be supplied.")

    def get_parameters(self):
        """Return dictionary with parameters of the current agent instance"""

        return self.parameters

    def initialize_neural_networks(self,neural_networks):
        """Initialize all neural networks"""

        self.neural_networks = {}
        for key, value in neural_networks.items():
            self.neural_networks[key] = neural_network(value['layers']).to(device)

    def initialize_optimizers(self,optimizers):
        """Initialize optimizers"""

        self.optimizers = {}
        for key, value in optimizers.items():
            self.optimizers[key] = torch.optim.RMSprop(
                        self.neural_networks[key].parameters(),
                            **value['optimizer_args'])

    def initialize_losses(self,losses):
        """Instantiate loss functions"""

        self.losses = {}
        for key, value in losses.items():
            self.losses[key] = nn.MSELoss()

    def get_number_of_model_parameters(self,name='policy_net'):
        """Return the number of trainable neural network parameters"""
        # from https://stackoverflow.com/a/49201237
        return sum(p.numel() for p in self.neural_networks[name].parameters() \
                                    if p.requires_grad)


    def get_state(self):
        '''Return dictionary with current state of neural net and optimizer'''
        #
        state = {'parameters':self.get_parameters()}
        #
        for name,neural_network in self.neural_networks.items():
            state[name] = copy.deepcopy(neural_network.state_dict())
        #
        for name,optimizer in (self.optimizers).items():
            #
            state[name+'_optimizer'] = copy.deepcopy(optimizer.state_dict())
        #
        return state


    def load_state(self,state):
        '''
        Load given states for neural networks and optimizer

        The argument "state" has to be a dictionary with the following
        (key, value) pairs:

        1. state['parameters'] = dictionary with the agents parameters
        2. For every neural network, there should be a state dictionary:
            state['$name'] = state dictionary of neural_network['$name']
        3. For every optimizer, there should be a state dictionary:
            state['$name_optimizer'] = state dictionary of optimizers['$name']
        '''
        #
        parameters=state['parameters']
        #
        self.check_parameter_dictionary_compatibility(parameters=parameters)
        #
        self.__init__()
        #
        #
        for name,state_dict in (state).items():
            if name == 'parameters':
                continue
            elif 'optimizer' in name:
                name = name.replace('_optimizer','')
                self.optimizers[name].load_state_dict(state_dict)
            else:
                self.neural_networks[name].load_state_dict(state_dict)
        #


    def check_parameter_dictionary_compatibility(self,parameters):
        """Check compatibility of provided parameter dictionary with class"""

        error_string = ("Error loading state. Provided parameter {0} = {1} ",
                    "is inconsistent with agent class parameter {0} = {2}. ",
                    "Please instantiate a new agent class with parameters",
                    " matching those of the model you would like to load.")
        try:
            n_state =  parameters['n_state']
            if n_state != self.n_state:
                raise RuntimeError(error_string.format('n_state',n_state,
                                                self.n_state))
        except KeyError:
            pass
        #
        try:
            n_actions =  parameters['n_actions']
            if n_actions != self.n_actions:
                raise RuntimeError(error_string.format('n_actions',n_actions,
                                                self.n_actions))
        except KeyError:
            pass


    def evaluate_stopping_criterion(self,list_of_returns):
        """ Evaluate stopping criterion """
        # if we have run at least self.N_solving_episodes, check
        # whether the stopping criterion is met
        if len(list_of_returns) < self.n_solving_episodes:
            return False, 0., 0.
        #
        # get numpy array with recent returns
        recent_returns = np.array(list_of_returns)
        recent_returns = recent_returns[-self.n_solving_episodes:]
        #
        # calculate minimal and mean return over the last
        # self.n_solving_episodes epsiodes
        minimal_return = np.min(recent_returns)
        mean_return = np.mean(recent_returns)
        #
        # check whether stopping criterion is met
        if minimal_return > self.solving_threshold_min:
            if mean_return > self.solving_threshold_mean:
                return True, minimal_return, mean_return
        # if stopping crtierion is not met:
        return False, minimal_return, mean_return


    def act(self,state):
        """
        Select an action for the current state
        """
        #
        # This typically uses the policy net. See the child classes below
        # for examples:
        # - dqn: makes decisions using an epsilon-greedy algorithm
        # - actor_critic: draws a stochastic decision with probabilities given
        #                 by the current stochastic policy
        #
        # As an example, we here draw a fully random action:
        return np.random.randint(self.n_actions)


    def add_memory(self,memory):
        """Add current experience tuple to the memory"""
        self.memory.push(*memory)

    def get_samples_from_memory(self):
        '''
        Get a tuple (states, actions, next_states, rewards, episode_end? )
        from the memory, as appopriate for experience replay
        '''
        #
        # get random sample of transitions from memory
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        #
        # convert list of Transition elements to Transition element with lists
        # (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*current_transitions))
        #
        # convert lists of current transitions to torch tensors
        state_batch = torch.cat( [s.unsqueeze(0) for s in batch.state],
                                        dim=0).to(device)
        # state_batch.shape = [batch_size, N_states]
        next_state_batch = torch.cat(
                         [s.unsqueeze(0) for s in batch.next_state],dim=0)
        action_batch = torch.cat(batch.action).to(device)
        # action_batch.shape = [batch_size]
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.tensor(batch.done).float().to(device)
        #
        return state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch


    def run_optimization_step(self, epoch):
        """Run one optimization step

        Keyword argument:
        epoch (int) -- number of current training epoch
        """
        #
        # Here is where the actual optimization happens.
        #
        # This method MUST be implemented in any child class, and might look
        # very different depending on the learning algorithm.
        # Note that any implementation must contain the argument "epoch", as
        # this method is called as run_optimization_step(epoch=epoch) in the
        # method self.train() below.
        #

    def train(self,environment,
                    verbose=True,
                    model_filename=None,
                    training_filename=None,
                ):
        """
        Train the agent on a provided environment

        Keyword arguments:
        environment -- environment used by the agent to train. This should be
                       an instance of a class with methods "reset" and "step".
                       - environment.reset() should reset the environment to
                         an initial state and return a tuple,
                            current_state, info = environment.reset(),
                         such current_state is an initial state of the with
                         np.shape(current_state) = (self.N_state,)
                       - environment.set(action) should take an integer in
                         {0, ..., self.N_action-1} and return a tuple,
                            s, r, te, tr, info = environment.step(action),
                         where s is the next state with shape (self.N_state,),
                         r is the current reward (a float), and where te and
                         tr are two Booleans that tell us whether the episode
                         has terminated (te == True) or has been truncated
                         (tr == True)
        verbose (Bool) -- Print progress of training to terminal. Defaults to
                          True
        model_filename (string) -- Output filename for final trained model and
                                   periodic snapshots of the model during
                                   training. Defaults to None, in which case
                                   nothing is not written to disk
        training_filename (string) -- Output filename for training data,
                                      namely lists of episode durations,
                                      episode returns, number of training
                                      epochs, and total number of steps
                                      simulated. Defaults to None, in which
                                      case no training data is written to disk
        """
        self.in_training = True
        #
        training_complete = False
        step_counter = 0 # total number of simulated environment steps
        epoch_counter = 0 # number of training epochs
        #
        # lists for documenting the training
        episode_durations = [] # duration of each training episodes
        episode_returns = [] # return of each training episode
        steps_simulated = [] # total number of steps simulated at the end of
                             # each training episode
        training_epochs = [] # total number of training epochs at the end of
                             # each training episode
        #
        output_state_dicts = {} # dictionary in which we will save the status
                                # of the neural networks and optimizer
                                # every self.saving_stride steps epochs during
                                # training.
                                # We also store the final neural network
                                # resulting from our training in this
                                # dictionary
        #
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            #
            status_progress_string = ( # for outputting status during training
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        #
        for n_episode in range(self.n_episodes_max):
            with open('output.txt', 'w') as file:
                # Write text into the file
                file.write(str(n_episode))
                file.write("\n")
            #
            # reset environment and reward of current episode
            state, info = environment.reset()
            current_total_reward = 0.
            #
            for i in itertools.count(): # timesteps of environment
                #time.sleep(0.25)
                #
                # select action using policy net
                action = self.act(state=state)
                #
                # perform action
                print("Episode: ", n_episode, " step: ",  i)
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)
                #
                step_counter += 1 # increase total steps simulated
                done = terminated or truncated # did the episode end?
                current_total_reward += reward # add current reward to total
                #
                # store the transition in memory
                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                # print("s",state)
                # print("a",action)
                # print("n",next_state)
                # print("r",reward)
                # print("d",done)
                self.add_memory([torch.tensor(state),
                            action,
                            torch.tensor(next_state),
                            reward,
                            done])
                #
                state = next_state
                #
                if step_counter % self.training_stride == 0:
                    # train model
                    print("Otimization step")
                    self.run_optimization_step(epoch=epoch_counter) # optimize
                    epoch_counter += 1 # increase count of optimization steps
                #
                if done: # if current episode ended
                    #
                    # update training statistics
                    episode_durations.append(i + 1)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)
                    #
                    # check whether the stopping criterion is met
                    training_complete, min_ret, mean_ret = \
                            self.evaluate_stopping_criterion(\
                                list_of_returns=episode_returns)
                    if verbose:
                            # print training stats
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if min_ret > self.solving_threshold_min:
                                if mean_ret > self.solving_threshold_mean:
                                    end='\n'
                            #
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   min_ret,mean_ret),
                                        end=end)
                    break
                
            writer.add_scalar("Reward/episode", current_total_reward, n_episode)
            #
            # Save model and training stats to disk
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.n_episodes_max-1:
                #
                if model_filename != None:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)
                #
                training_results = {'episode_durations':episode_durations,
                            'epsiode_returns':episode_returns,
                            'n_training_epochs':training_epochs,
                            'n_steps_simulated':steps_simulated,
                            'training_completed':False,
                            }
                if training_filename != None:
                    self.save_dictionary(dictionary=training_results,
                                        filename=training_filename)
            #
            if training_complete:
                # we stop if the stopping criterion was met at the end of
                # the current episode
                training_results['training_completed'] = True
                break
        #
        if not training_complete:
            # if we stopped the training because the max number of
            # episodes was reached, we throw a warning
            warning_string = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warning_string.format(self.n_episodes_max))
        #
        self.in_training = False
        #
        return training_results

    def save_dictionary(self,dictionary,filename):
        """Save a dictionary in hdf5 format"""

        with h5py.File(filename, 'w') as hf:
            self.save_dictionary_recursively(h5file=hf,
                                            path='/',
                                            dictionary=dictionary)

    def save_dictionary_recursively(self,h5file,path,dictionary):
        #
        """
        slightly adapted from https://codereview.stackexchange.com/a/121308
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.save_dictionary_recursively(h5file,
                                                path + str(key) + '/',
                                                value)
            else:
                h5file[path + str(key)] = value

    def load_dictionary(self,filename):
        with h5py.File(filename, 'r') as hf:
            return self.load_dictionary_recursively(h5file=hf,
                                                    path='/')

    def load_dictionary_recursively(self,h5file, path):
        """
        From https://codereview.stackexchange.com/a/121308
        """
        return_dict = {}
        for key, value in h5file[path].items():
            if isinstance(value, h5py._hl.dataset.Dataset):
                return_dict[key] = value.value
            elif isinstance(value, h5py._hl.group.Group):
                return_dict[key] = self.load_dictionary_recursively(\
                                            h5file=h5file,
                                            path=path + key + '/')
        return return_dict
