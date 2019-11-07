import lightbox
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
        
        #FMDP : dict of DBNs which are dict of CPTs initialized to single leafs
        self.FMDP = dict()
        for act in range(1, env.get_nb_light()+1):
            self.FMDP[act] = dict()
            for var in range(1, env.get_nb_light()+1):
                self.FMDP[act][var] = self.CPTNode()
                
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
        
    #Execute the action
    def execute(self, act):
        print("To implement")
    
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
            
        class OptionTree(NodeMixin):
            ##Befor next step, chech if accessible for BIC...
            
        
#--------internal class for CPT nodes-----
    class CPTNode(NodeMixin):
        def __init__(self):
            self.var = 1
            print("TODO CPT node")