from agent import *
from illegal_actions import IllegalActions

action_opposition_matrix = np.array([[0,1,0,0,0,0,0,1,1],
                                     [1,0,0,0,0,1,1,0,0],
                                     [0,0,0,1,0,1,0,1,0],
                                     [0,0,1,0,0,0,1,0,1],
                                     [0,0,0,0,0,0,0,0,0],
                                     [0,1,1,0,0,0,0,0,1],
                                     [0,1,0,1,0,0,0,1,0],
                                     [1,0,1,0,0,0,1,0,0],
                                     [1,0,0,1,0,1,0,0,0]]) *1

illegalActions = IllegalActions(action_opposition_matrix)


class dqn(agent_base):

    def __init__(self):
        super().__init__()
        parameters = self.get_default_parameters()
        self.set_parameters(parameters)
        self.in_training = False

    def get_default_parameters(self):
        '''
        Create and return dictionary with the default parameters of the dqn
        algorithm
        '''
        #
        default_parameters = super().get_parameters()
        #
        # add default parameters specific to the dqn algorithm
        default_parameters['neural_networks']['target_net'] = {}
        default_parameters['neural_networks']['target_net']['layers'] = \
        copy.deepcopy(\
                default_parameters['neural_networks']['policy_net']['layers'])
        #
        #
        # soft update stride for target net:
        default_parameters['target_net_update_stride'] = 1
        # soft update parameter for target net:
        default_parameters['target_net_update_tau'] = 1e-2
        #
        # Parameters for epsilon-greedy policy with epoch-dependent epsilon
        default_parameters['epsilon'] = 1.0 # initial value for epsilon
        default_parameters['epsilon_1'] = 0.1 # final value for epsilon
        default_parameters['d_epsilon'] = 0.00005 # decrease of epsilon
            # after each training epoch
        #
        default_parameters['doubledqn'] = False
        #
        return default_parameters


    def set_parameters(self,parameters):
        #
        super().set_parameters(parameters=parameters)
        #
        ##################################################
        # Use deep Q-learning or double deep Q-learning? #
        ##################################################
        try: # False -> use DQN; True -> use double DQN
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        #
        ##########################################
        # Parameters for updating the target net #
        ##########################################
        try: # after how many training epochs do we update the target net?
            self.target_net_update_stride = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass
        #
        try: # tau for soft update of target net (value 1 means hard update)
            self.target_net_update_tau = parameters['target_net_update_tau']
            # check if provided parameter is within bounds
            error_msg = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            error_msg = error_msg.format(self.target_net_update_tau)
            if self.target_net_update_tau < 0:
                raise RuntimeError(error_msg)
            elif self.target_net_update_tau > 1:
                raise RuntimeError(error_msg)
        except KeyError:
            pass
        #
        #
        ########################################
        # Parameters for epsilon-greedy policy #
        ########################################
        try: # probability for random action for epsilon-greedy policy
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        #
        try: # final probability for random action during training
            #  for epsilon-greedy policy
            self.epsilon_1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass
        #
        try: # amount by which epsilon decreases during each training epoch
            #  until the final value self.epsilon_1 is reached
            self.d_epsilon = \
                    parameters['d_epsilon']
        except KeyError:
            pass

    def act(self,state,epsilon=0.):
        """
        Use policy net to select an action for the current state

        We use an epsilon-greedy algorithm:
        - With probability epsilon we take a random action (uniformly drawn
          from the finite number of available actions)
        - With probability 1-epsilon we take the optimal action (as predicted
          by the policy net)

        By default epsilon = 0, which means that we actually use the greedy
        algorithm for action selection
        """
        #
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            #
            policy_net = self.neural_networks['policy_net'].to(device)
            #
            with torch.no_grad():
                policy_net.eval()
                #action = policy_net(torch.tensor(state, dtype=torch.float).to(device)).argmax(0).item()
                
                action_scores = policy_net(torch.tensor(state, dtype=torch.float).to(device))
                #print("Q values:", action_scores)
                action_scores = illegalActions.apply(method='policy',action_scores=action_scores)
                sampled_action = action_scores.argmax(0).item()

                policy_net.train()

                illegalActions.update(sampled_action)
                return sampled_action
        else:
            # perform random action
            #return torch.randint(low=0,high=self.n_actions,size=(1,)).item()

            sampled_action = illegalActions.apply(method='random')
            illegalActions.update(sampled_action)

            return sampled_action

    def update_epsilon(self):
        """
        Update epsilon for epsilon-greedy algorithm

        For training we assume that
        epsilon(n) = max{ epsilon_0 - d_epsilon * n ,  epsilon_1 },
        where n is the number of training epochs.

        For epsilon_0 > epsilon_1 the function epsilon(n) is piecewise linear.
        It first decreases from epsilon_0 to epsilon_1 with a slope d_epsilon,
        and then becomes constant at the value epsilon_1.

        This ensures that during the initial phase of training the neural
        network explores more randomly, and in later stages of the training
        follows more the policy learned by the neural net.
        """
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self,epoch):
        """Run one optimization step for the policy net"""
        #
        # if we have less sample transitions than we would draw in an
        # optimization step, we do nothing
        if len(self.memory) < self.batch_size:
            return
        #
        state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch = self.get_samples_from_memory()
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        reward_batch = reward_batch.to(device)
        done_batch = done_batch.to(device)
        #
        policy_net = self.neural_networks['policy_net'].to(device)
        target_net = self.neural_networks['target_net'].to(device)
        #
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net'].to(device)
        #
        policy_net.train() # turn on training mode
        #
        # Evaluate left-hand side of the Bellman equation using policy net
        LHS = policy_net(state_batch.to(device=device, dtype=torch.float)).gather(dim=1, index=action_batch.unsqueeze(1))



        # LHS.shape = [batch_size, 1]
        #
        # Evaluate right-hand side of Bellman equation
        if self.doubleDQN:
            # double deep-Q learning paper: https://arxiv.org/abs/1509.06461
            #
            # in double deep Q-learning, we use the policy net for choosing
            # the action on the right-hand side of the Bellman equation. We
            # then use the target net to evaluate the Q-function on the
            # chosen action
            argmax_next_state = policy_net(next_state_batch).argmax(
                                                                    dim=1)
            # argmax_next_state.shape = [batch_size]
            #
            Q_next_state = target_net(next_state_batch).gather(
                dim=1,index=argmax_next_state.unsqueeze(1)).squeeze(1)
            # shapes of the various tensor appearing in the previous line:
            # self.target_net(next_state_batch).shape = [batch_size,N_actions]
            # self.target_net(next_state_batch).gather(dim=1,
            #   index=argmax_next_state.unsqueeze(1)).shape = [batch_size, 1]
            # Q_next_state.shape = [batch_size]
        else:
            # in deep Q-learning, we use the target net both for choosing
            # the action on the right-hand side of the Bellman equation, and
            # for evaluating the Q-function on that action
            Q_next_state = target_net(next_state_batch.to(device=device, dtype=torch.float))
            
            #Q learning normalization for this specific case, with unique reward function
            #Q_next_state = (Q_next_state - (-200*np.sqrt(72))) / (400 - (-200*np.sqrt(72))) 
            
            Q_next_state = Q_next_state.max(1)[0].detach()

            #Q_next_state.shape = [batch_size]
            
        #print("Done batch: ",done_batch)
        #print("Reward batch: ",reward_batch)
        #print("Q next state: ",Q_next_state)
        
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        #print("RHS: ",RHS)
        RHS = RHS.unsqueeze(1) # RHS.shape = [batch_size, 1]
        #
        # optimize the model
        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        #
        policy_net.eval() # turn off training mode
        #
        self.update_epsilon() # for epsilon-greedy algorithm
        #
        if epoch % self.target_net_update_stride == 0:
            print("Updating target net")
            self.soft_update_target_net() # soft update target net
        #

    def soft_update_target_net(self):
        """Soft update parameters of target net"""
        #
        # the following code is from https://stackoverflow.com/q/48560227
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.neural_networks['target_net'].load_state_dict(dict_params2)
