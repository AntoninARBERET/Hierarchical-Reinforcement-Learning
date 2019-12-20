from agents import AbstractAgent
import numpy as np

class RandomAgent(AbstractAgent):
    
    def __init__(self, env, M=50, N=20, K=20, epsilon=0.1, experiment_id = None, data_path=None):
        AbstractAgent.__init__(self, env, M, N, K, epsilon, experiment_id, data_path, "Random")
    
    #return a random action
    def selectAction(self, state):
        return(self.actions[np.random.randint(len(self.actions))+1])