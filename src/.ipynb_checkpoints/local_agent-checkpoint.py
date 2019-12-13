from agents import AbstractAgent
import numpy as np

class LocalAgent(AbstractAgent):
    
    def selectAction(self, state):
        max_G = 0
        max_act = None
        for act in self.FMDP:
            dbn = self.FMDP[act]
            G = 0
            for i_cpt in dbn["cpts"]:
                G+= dbn["cpts"][i_cpt].tree_entropy_gain(state)
            if(max_act == None or G>max_G):
                max_G = G
                max_act = self.actions[act]
        return(max_act)