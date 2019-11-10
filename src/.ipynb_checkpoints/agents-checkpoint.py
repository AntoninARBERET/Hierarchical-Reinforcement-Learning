import lightbox
import numpy as np
from anytree import Node, RenderTree
from anytree.node.nodemixin import NodeMixin

#==========================================================================================================
#============================================ ABSTRACT AGENT ==============================================
#==========================================================================================================

#Parent of all agents

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
                self.FMDP[act]["cpts"][var] = self.CPTNode(self.FMDP[act], var,self)
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
        current_DBN = self.FMDP[act]
        for ind in current_DBN["cpts"]:
            #print("add point in DBN {}, CPT{}".format(act, ind))
            current_DBN["cpts"][ind].add_datapoint(state_t, act, state_t1)
        print("To implement")
        
    #Return solution to set the var to 1 : an option if exists, an action if not
    def get_solution(self, var):
        print("Implement get solution")
        return("Solution for {}".format(var))
        
    def create_option(self, var, parents, sig_0):
        return self.Option(self, var, parents, sig_0)
        
        
#--------INTERNAL CLASSES------------------
        
#==========================================================================================================
#===================================== ACTION, internal class==============================================
#==========================================================================================================

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
            


#==========================================================================================================
#====================================== OPTION, internal class ============================================
#==========================================================================================================


    class Option():
        
        #Constructor : variable associated, sigma_0, list of parent variables
        def __init__(self, my_agent, variable, parents, sig_0):
            self.my_agent = my_agent
            self.variable = variable
            self.sig_0 = sig_0
            self.sig = sig_0
            self.step = 0
            previousNode = None
            #for each parent
            for i in range(len(parents)):
                #create a Node
                node = self.OptionTreeNode(parents[i], 1)
                #set the 0 child
                child_0 = self.OptionTreeNode(-1, 0, solution = my_agent.get_solution(parents[i]))
                child_0.parent = node
                #check if root
                if(i==0):
                    self.root=node
                    node.child_01 = None
                #if not, link to the parent
                else:
                    node.parent=previousNode
                #if last, set the 1 child
                if(i==len(parents)-1):
                    child_1 = self.OptionTreeNode(-1, 1, my_agent.get_solution(self.variable))
                    child_1.parent=node
                previousNode = node
                
            self.exec_pointer = self.root
            
        
            
        def step(self, state):
            print("TODO")
            
        def print_tree(self):
            print("Option {}".format(self.variable))
            self.root.print_tree()
            
            
        class OptionTreeNode(NodeMixin):
            def __init__(self, var, child_01, solution=None):
                self.var = var
                self.solution = solution
                self.child_01 = child_01
                
            def print_tree(self):
                for pre, _, node in RenderTree(self):
                    treestr = u"%s%s" % (pre, node.var)
                    if(node.is_root):
                        treestr = u"%s%s" % (pre,node.var)
                        print(treestr.ljust(8))
                    else:
                        if(node.var == -1):
                            #change when real solution
                            treestr = u"%s%s" % (pre, "{} [{}]".format(node.solution,node.child_01))
                        else:    
                            treestr = u"%s%s" % (pre, "{} [{}]".format(node.var,node.child_01))
                        print(treestr.ljust(8))

    
    
    
#==========================================================================================================
#======================================= CPT NODES, internal class ========================================
#==========================================================================================================



    class CPTNode(NodeMixin):
        
        #constructor
        def __init__(self, DBN, tree_var, my_agent, parents_list = None, dataset = None, child_01= None):
            self.name="X"
            self.tree_var = tree_var
            self.DBN = DBN
            self.my_agent = my_agent
            self.var = -1
            self.leaf_distib = 0
            self.BIC = 0
            if(dataset == None):
                self.dataset = []
            else:
                self.dataset = dataset 
            if(parents_list == None):
                self.parents_list = []
            else:
                self.parents_list = parents_list  
            self.child_01=child_01
            self.nb_var = my_agent.env.get_nb_light()
            #
            self.distrib_vect = dict() 
            for i in range(1, self.nb_var+1):
                if not i in self.parents_list:
                    self.distrib_vect[i] =[0,0]
        
            #print("TODO CPT node")
            
        #string description
        def __str__(self):
            return(" leaf : {}, datapoints : {}, distrib_vect {}".format(self.is_leaf(), len(self.dataset), self.distrib_vect))
        
        def print_tree(self):
            for pre, _, node in RenderTree(self):
                #treestr = u"%s%s" % (pre, node.var)
                if(node.is_root):
                    treestr = u"%s%s" % (pre, node.var)
                    print(treestr.ljust(8))
                else:
                    treestr = u"%s%s" % (pre, "-{}-> {}".format(node.child_01,node.var))
                    print(treestr.ljust(8))

            
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
                            #print("i = {} j = {} - >\n\tk = 0 Nijk = {} teta = {}\n\tk = 1 Nijk = {} teta = {}\n\tincrement L of {}".format(i, j, N_ij0, teta_ij0, N_ij1, teta_ij1, N_ij0 * np.log(teta_ij0) + N_ij1 * np.log(teta_ij1)))
                            L+= (N_ij0 * np.log(teta_ij0) + N_ij1 * np.log(teta_ij1))
                                    
            #finally compute BIC            
            BIC = L - ((self.nb_var)/2 * np.log(len(self.dataset)))
            self.BIC = BIC
            return BIC
        
        #BIC Whithout sum on i
        def compute_BIC_Mono(self):
            if(len(self.dataset)==0):
                return 0
            #Likelihood
            L=0
            #create N_ijk
            N=dict()
            i=self.tree_var
            
            #fill N_ijk
            for d in self.dataset:
                s_0 = d["s_0"]
                s_1 = d["s_1"]
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
                if not j_state_0 in N:
                    N[j_state_0]=[0,0]
                #increment    
                N[j_state_0][k]+=1
            
            #get the sum
                for j in N:
                    #get both N_ijk values
                    N_ij0 = N[j][0]
                    N_ij1 = N[j][1]
                    #teta_ijk = N_ijk/sum_on_k(N_ijk) : probability estimated bu counting
                    teta_ij0 = N_ij0/(N_ij0+N_ij1)
                    teta_ij1 = N_ij1/(N_ij0+N_ij1)
                    #Add to L in tetas are not 0
                    if(teta_ij0>0 and teta_ij1 >0):
                        #print("i = {} j = {} - >\n\tk = 0 Nijk = {} teta = {}\n\tk = 1 Nijk = {} teta = {}\n\tincrement L of {}".format(i, j, N_ij0, teta_ij0, N_ij1, teta_ij1, N_ij0 * np.log(teta_ij0) + N_ij1 * np.log(teta_ij1)))
                        L+= (N_ij0 * np.log(teta_ij0) + N_ij1 * np.log(teta_ij1))
                                    
            #finally compute BIC            
            BIC = L - ((self.nb_var)/2 * np.log(len(self.dataset)))
            self.BIC = BIC
            return BIC
        
        
        def try_refinement(self, var):
            #split dataset on the refinement var
            dataset_0 = []
            dataset_1 = []
            for d in self.dataset:
                if(d["s_0"][var-1]==0):
                    dataset_0.append(d)
                else:
                    dataset_1.append(d)
            children_parents = self.parents_list + [var]        
            child_0 = type(self)(self.DBN,self.tree_var, self.my_agent, parents_list = children_parents, dataset =dataset_0, child_01 = 0) 
            child_1 = type(self)(self.DBN, self.tree_var, self.my_agent, parents_list = children_parents, dataset =dataset_1, child_01 = 1)
            #if children BICs are better
            if(child_0.compute_BIC_Mono() + child_1.compute_BIC_Mono() > self.compute_BIC_Mono()):
                #create refin
                child_0.parent = self
                child_1.parent = self
                self.var=var
                self.dataset=[]
                #TODO Create Option
                return True
            #no refinement
            return False
            
