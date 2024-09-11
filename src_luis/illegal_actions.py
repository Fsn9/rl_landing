#
#
# @file      illegal_actions.py
# @version   0.1
# @date      2024-04-01
# @authors   Maria Inês Pereira <maria.i.pereira@inesctec.pt>
#
# @brief     Illegal Actions package
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# @copyright Copyright (c) 2024, INESC TEC - CRAS, All rights reserved.
#

#!/usr/bin/env python


import numpy as np
import torch
import random
from agent import *


class IllegalActions:

    def __init__(self, opposition_matrix) -> None:

        """
        Create a new illegal actions object.

        :param opposition_matrix: a numpy array of dimensions NxN, where N is the dimension of the Action space.
                                  each matrix cell contains the opposition level between the two corresponding actions.

        """

        # Fetch Action space dimension -> N
        self.action_dim = opposition_matrix.shape[0]

        # Create array to store which actions are illegal at each iteration
        self.currently_illegal = np.zeros(self.action_dim)

        # Store opposition info
        self.opposition_matrix = opposition_matrix


    def update(self, latest_action):

        """
        Updates the illegal actions object. 

        :param latest_action: integer representing the latest action taken by the agent.

        """

        # Decrease illegal actions from previous iteration by one timestep
        self.currently_illegal = np.where(self.currently_illegal > 0, self.currently_illegal - 1, self.currently_illegal)

        # Add new illegal actions according to opposition levels of latest action
        self.currently_illegal = np.where(self.currently_illegal > 0, self.currently_illegal, self.opposition_matrix[latest_action])


    def apply(self, method='random', action_scores=None):

        """
        Applies the illegal actions object. 
        

        :param method: either 'random' or 'policy'. 
                       If 'random', function runs a random.choice excluding the illegal actions and returns the sampled action.
                       If 'policy', function updates the action scores, by setting to 1e-6 the score of any illegal action.

        :param action_scores (optional): torch tensor with the scores of each action


        :return : If method is 'random', function returns a sampled action.
                  If method is 'policy', function returns the updated action scores.

        """

        if method == 'random':

            to_exclude = [i for i, x in enumerate(self.currently_illegal) if x > 0]
            sampled_action = random.choice(list(set([x for x in range(0, self.action_dim)]) - set(to_exclude)))

            return sampled_action

        elif method == 'policy':

            epsilon = 1e-6
            action_scores = torch.where(torch.tensor(self.currently_illegal > 0).to(device), torch.tensor(epsilon), action_scores).to(device)

            return action_scores



# For testing
# if __name__ == '__main__':

#     print('Heyo') 

#     # Initialize IllegalActions object with action oppostion matrix
#     illegalActions = IllegalActions(np.array([[0,1],[1,0]]))

#     # Update IllegalActions object with latest action
#     latest_action = 0
#     illegalActions.update(latest_action)

#     # Apply IllegalActions object according to action sampling method ('random' or 'policy')
#     #action_scores = illegalActions.apply(method='policy',action_scores=torch.tensor([0.3,0.7]))
#     sampled_action = illegalActions.apply(method='random')