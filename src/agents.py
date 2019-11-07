import lightbox
import numpy as np
from anytree import Node
from anytree.node.nodemixin import NodeMixin

#Parent class for local, global and random agents
class AbstractAgent :
    
    #Constructor
    def __init__(self, env):
        #lighbox env
        self.env = env
        
        #agent type
        self.type = "Abstract"
        
        #FMDP : dict of DBNs which are dict of CPTs initialized to single leafs and dict of parents
        self.FMDP = dict()
        for act in range(1, env.get_nb_light()+1):
            self.FMDP[act] = {"cpts" : dict(), "parents" : dict()}
            for var in range(1, env.get_nb_light()+1):
                self.FMDP[act]["cpts"][var] = self.CPTNode(self.FMDP[act], env.get_nb_light())
                self.FMDP[act]["parents"][var] = []
                
                
        #Actions
        self.actions = dict()
        for act in range(1, env.get_nb_light()+1):
            self.actions[act]=self.Action(act, self.env)
            
        #Options
        self.options = dict()
        
        #For each option, a list (or tree?) of skills called by the option
        self.options_hierarchies = dict() 
        
        #current state of the env
        self.state_t = env.get_state()
        
        #current option
        self.current_option = None
        
        #iteration
        self.t = 1
    
    #string description
    def __str__(self):
        return("{} agent, currently at iteration {} and learned {} options".format(self.type, self.t, len(self.options)))
    
    #launch the agent
    def start(self):
        while(not self.stop_condition()):
            self.behaviour()
    
    #condition to stop the agent
    def stop_condition(self):
        #temporary, for tests
        if(self.t>10):
            return True
        #maybe changed
        return False
        
        
    
    #bahaviour af the agent, the main loop
    def behaviour(self):
        act = self.selectAction( self.state_t)
        act.execute()
        state_t1 = self.env.get_state()
        self.update(self.state_t, act, state_t1)
        self.state_t = state_t1
        self.t+=1
        
    #Action selection, different for each type of agent
    def selectAction(self, state):
        print("Not implemented in abstract")
        return(self.actions[1])

    #Update the FMDP
    def update(self, state_t, act, state_t1):
        print("To implement")

        
#--------INTERNAL CLASSES------------------
        
#--------internal class for actions--------
    class Action():
        
        #constructor
        def __init__(self, light_id, env):
            self.light_id = light_id
            self.env = env
            
        #string description
        def __str__(self):
            return("A{}".fortmat(self.light_id))
        
        #execution
        def execute(self):
            self.env.turn_on(self.light_id)
            
#--------Internal class for option--------
    class Option():
        
        #Constructor : variable associated, sigma_0, list of parent variables
        def __init__(self, my_agent, variable, parents, sig_0):
            self.my_agent = my_agent
            self.varaible = variable
            self.sig_0 = sig_0
            self.sig = sig_0
            print("TODO finish option")
            
        class OptionTree(NodeMixin):
            ##Befor next step, chech if accessible for BIC...
            print("TODO BIC")
            
        
#--------internal class for CPT nodes-----
    class CPTNode(NodeMixin):
        
        #constructor
        def __init__(self, DBN, nb_var, parents = []):
            self.DBN = DBN
            self.var = -1
            self.leaf_distib = 0
            self.BIC = 0
            self.dataset = []
            self.nb_var = nb_var
            #
            self.distrib_vect = dict() 
            for i in range(1, nb_var+1):
                if not i in parents:
                    self.distrib_vect[i] =[0,0]
        
            #print("TODO CPT node")
            
        #string description
        def __str__(self):
            return(super().__str__() + " leaf : {}, datapoints : {}, distrib_vect {}".format(self.is_leaf(), len(self.dataset), self.distrib_vect))
            
        def is_leaf(self):
                return(len(self.children)==0)
            
        def add_datapoint(self, s_0, act, s_1):
            val_on_s_0 = s_0[self.var-1]

                
            #if leaf, add to dataset and updat distrib_vect   
            if(self.is_leaf()):
                self.dataset.append({"s_0" : s_0, "a" : act, "s_1" : s_1})
                for j in self.distrib_vect:
                    j_val = 0
                    if s_0[j-1]:
                        j_val=1
                        
                    self.distrib_vect[j][j_val]+=1
                #TODO compute BIC 
                
            #if not leaf
            else:
                if(val_on_s_0==False):
                    self.children[0].add_datapoint( s_0, act, s_1)
                else:
                    self.children[1].add_datapoint( s_0, act, s_1)
        
        #BIC computation          
        def compute_BIC(self):
            if(len(self.dataset)==0):
                return 0
            #Likelihood
            L=0
            #create N_ijk
            N=[]
            for i in range(1, self.nb_var+1):
                N.append(dict())
            #fill N_ijk
            for d in self.dataset:
                s_0 = d["s_0"]
                s_1 = d["s_1"]
                for i in range(1, self.nb_var+1):
                    #get parents(i) state
                    j_state_0=""
                    for j in self.DBN["parents"][i]:
                        if(s_0[j-1]==False):
                            j_state_0+="0"
                        else:
                            j_state_0+="1"
                    #get k
                    k=0
                    if(s_1[i-1]==True):
                        k=1                        
                    #check presence of the j key        
                    if not j_state_0 in N[i-1]:
                        N[i-1][j_state_0]=[0,0]
                    #increment    
                    N[i-1][j_state_0][k]+=1
                
                #get the sum
                for i in range(1, self.nb_var+1):
                    for j in N[i-1]:
                        #get both N_ijk values
                        N_ij0 = N[i-1][j][0]
                        N_ij1 = N[i-1][j][1]
                        #teta_ijk = N_ijk/sum_on_k(N_ijk) : probability estimated bu counting
                        teta_ij0 = N_ij0/(N_ij0+N_ij1)
                        teta_ij1 = N_ij1/(N_ij0+N_ij1)
                        #Add to L in tetas are not 0
                        if(teta_ij0>0 and teta_ij1 >0):
                            L+=N_ij0 * np.log(teta_ij0) + N_ij1 * np.log(teta_ij1)
                                    
            #finally compute BIC            
            BIC = L - ((self.nb_var)/2 * np.log(len(self.dataset)))
            self.BIC = BIC
            return BIC
            