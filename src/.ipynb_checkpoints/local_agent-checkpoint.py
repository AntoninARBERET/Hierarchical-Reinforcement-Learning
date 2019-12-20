from agents import AbstractAgent
import numpy as np
import math

class LocalAgent(AbstractAgent):
    
    def __init__(self, env, M=50, N=20, K=20, epsilon=0.1, experiment_id = None, data_path=None):
        AbstractAgent.__init__(self, env, M, N, K, epsilon, experiment_id, data_path, "Local")
    
    #return the best action in the state : with the best entropy gain
    def selectAction(self, state):
        if(np.random.random()<self.epsilon):
            return(self.actions[np.random.randint(len(self.actions))+1])
        max_G = 0
        max_act = None
        max_act_list = []
        
        for act in self.FMDP:
            dbn = self.FMDP[act]
            G = 0
            for i_cpt in dbn["cpts"]:
                G+= dbn["cpts"][i_cpt].tree_entropy_gain(state)
            #print("G for {} = {}".format(act, G))
            if(G==max_G or (math.isnan(G) and math.isnan(max_G))):
                max_act_list.append(self.actions[act])
                max_act= -1
            elif(max_act == None or math.isnan(G) or G>max_G):
                max_act_list = []
                max_G = G
                max_act = self.actions[act]
        if(len(max_act_list)>0):
            max_act=max_act_list[np.random.randint(len(max_act_list))]
        
        #if no increase possible 
        if(not(math.isnan(max_G)) and max_G <0):
            max_act=(self.actions[np.random.randint(len(self.actions))+1])
        #print("Selected act :\t {} \t \t \t G : {} \t \t \t State : {}".format(max_act, max_G, state))
        return(max_act)