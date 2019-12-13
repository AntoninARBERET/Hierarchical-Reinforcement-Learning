from agents import AbstractAgent
import numpy as np

class RandomAgent(AbstractAgent):
    def selectAction(self, state):
        return(self.actions[np.random.randint(len(self.actions))+1])