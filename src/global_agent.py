from agents import AbstractAgent
import numpy as np
import math
from itertools import product
class GlobalAgent(AbstractAgent):
    
    def __init__(self, env, M=50, N=20, K=20, epsilon=0.1, experiment_id = None, data_path=None):
        AbstractAgent.__init__(self, env, M, N, K, epsilon, experiment_id, data_path, "Global")
    
    #return the next action to do in order to reach a state and execute an interesting action there
    def selectAction(self, state):
        #an option is running
        if not self.current_option == None:
            
            #if option done
            if self.current_option.done:
                self.current_option.update_sig(1)
                self.current_option.reset()
                self.current_option=None
                self.plan_ptr+=1
                return(self.selectAction(state))
            
            #less than M step, continue
            if self.current_option.step<self.M:
                return self.current_option.next_step(state)
            
            #More than M step, terminate
            else:
                self.current_option.update_sig(0)
                self.current_option.reset()
                self.current_option=None
                self.current_plan=None
                self.plan_prt=-1
                #TODO Remove option if sig too low
                return(self.selectAction(state))
            
        #No option running
        else:
            
            #last option succed, follow plan
            if not self.current_plan == None:
                
                #plan is over
                if(self.plan_ptr>=len(self.current_plan)):
                    self.current_plan=None
                    self.plan_ptr=-1
                    return(self.selectAction(state))
                
                #next step is an action, may fail
                if self.current_plan[self.plan_ptr].is_action():
                    self.plan_ptr+=1
                    return self.current_plan[self.plan_ptr-1]
                
                #option
                else:
                    self.current_option=self.current_plan[self.plan_ptr]
                    return self.current_option.next_step(state)
            
            #no plan
            else:
                self.compute_plan(state)
                self.plan_ptr=0
                #next step is an action, may fail
                if self.current_plan[self.plan_ptr].is_action():
                    self.plan_ptr+=1
                    return self.current_plan[self.plan_ptr-1]
                
                #option
                else:
                    self.current_option=self.current_plan[self.plan_ptr]
                    return self.current_option.next_step(state)
                
                
                
        return(max_act)

    
    #generate every reachable state with controllable variable from given state
    def generate_rechable_states(self, state):
        reachable = []
        prod = list(product([False,True],repeat=len(self.C)))
        
        for p in prod:
            tmp_state = state.copy()
            i=0
            for c in self.C:
                tmp_state[c-1]=p[i]
                
                i+=1
            reachable.append(tmp_state)
            
        return(reachable)
    
    #return the best action in the state, which is the one with the best entropy gain
    def best_state_action(self, state):
                
        reachable_states = self.generate_rechable_states(state)
        
        max_G = 0
        max_act = None
        max_act_list = []
        max_state = None
        max_state_list = []
        for s in reachable_states:
            for act in self.FMDP:
                dbn = self.FMDP[act]
                G = 0
                for i_cpt in dbn["cpts"]:
                    G+= dbn["cpts"][i_cpt].tree_entropy_gain(s)
                if(self.print_entropies):
                    print("G for {} in {} = {}".format(act,s, G))
                if(G==max_G or (math.isnan(G) and math.isnan(max_G))):
                    max_act_list.append(self.actions[act])
                    max_state_list.append(s)
                    max_act= -1
                    max_state = -1
                elif(max_act == None or math.isnan(G) or G>max_G):
                    max_act_list = []
                    max_state_list=[]
                    max_G = G
                    max_act = self.actions[act]
                    max_state=s
        if(len(max_act_list)>0):
            i = np.random.randint(len(max_act_list))
            max_act=max_act_list[i]
            max_state=max_state_list[i]
            
        if(not(math.isnan(max_G)) and max_G <0):
            max_act=(self.actions[np.random.randint(len(self.actions))+1])
            max_state=reachable_states[np.random.randint(len(reachable_states))]
        if(self.print_entropies):    
            print(max_G)    
        return((max_state, max_act))
        
    
    #create a plan to reach the state and execute the action
    def compute_plan(self, state):
        (s_target,a) = self.best_state_action(state)
        to_set =[]
        plan = []
        for i in range(len(state)):
            if(not state[i]==s_target[i]):
                plan.append(self.get_solution(i+1, s_target[i]))
        plan.append(a)
        self.current_plan=plan
        self.plan_ptr=0
        
        if(self.print_entropies):
            print("Current state \t : {}\nTarget state : {}\n\t Action : {}".format(state,s_target, a))
            for s in plan:
                print(s)
            
        return plan
    
    