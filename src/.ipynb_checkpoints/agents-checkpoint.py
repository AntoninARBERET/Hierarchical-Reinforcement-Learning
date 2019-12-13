import lightbox
import numpy as np
from anytree import Node, RenderTree, LevelOrderIter
from anytree.node.nodemixin import NodeMixin
from scipy.stats import chi2_contingency

#==========================================================================================================
#============================================ ABSTRACT AGENT ==============================================
#==========================================================================================================

#Parent of all agents

class AbstractAgent :
    
    #Constructor
    def __init__(self, env, M=50, N=20, K=20):
        #lighbox env
        self.env = env
        
        #agent type
        self.type = "Abstract"
        
        #FMDP : dict of DBNs which are dict of CPTs initialized to 1 split tree on var and dict of parents
        self.FMDP = dict()
        for act in range(1, env.get_nb_light()+1):
            self.FMDP[act] = {"cpts" : dict(), "parents" : dict()}
            for var in range(1, env.get_nb_light()+1):
                self.FMDP[act]["cpts"][var] = self.CPTNode(self.FMDP[act], var,self)
                self.FMDP[act]["parents"][var] = []
                child_0 = self.CPTNode(self.FMDP[act], var,self, parents_list = [var], child_01= 0)
                child_1 = self.CPTNode(self.FMDP[act], var,self, parents_list = [var], child_01= 1)
                child_0.parent = self.FMDP[act]["cpts"][var]
                child_1.parent = self.FMDP[act]["cpts"][var]
                self.FMDP[act]["cpts"][var].var=var
                
                
        #Actions
        self.actions = dict()
        for act in range(1, env.get_nb_light()+1):
            self.actions[act]=self.Action(act, self.env)
            
        #
        self.action_to_set = dict()
            
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
        
        #Controllable set
        self.C = set()
        
        #waiting to get in C
        self.queue_for_C = set()
        
        #M step max in an option
        self.M = M
        
        #N execution before considering removing a refinement
        self.N = N
        
        #K minimum nb of datapoint to condider a refinement
        self.K = K
        
        #To be removed later, artificially associate lights to actions
        self.artificial_setup()
        
        #Inde treshold for chi square TODO pass as parameter
        self.inde_treshold = 0.995
        
        #self.artificial_setup()
        
    #used to give basicaction of first level to the agent, temporarly
    def artificial_setup(self):
        for i in range(1, 21):
            self.action_to_set[i]=self.actions[i]
        for i in range(1, 10):
            self.C.add(i)
    
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
        maxit = 3000
        perc=self.t/maxit*100
        if((perc-np.floor(perc))==0 and np.floor(perc)%5==0):
            print("{}%".format(perc))
        if(self.t>maxit):
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
        #print("Not implemented in abstract")
        #return(self.actions[np.random.randint(20)+1])
        if(np.random.rand()<0.9):
            return(self.actions[np.random.randint(9)+1])
        return(self.actions[np.random.randint(6)+10])

    #Update the FMDP
    def update(self, state_t, act, state_t1):
        if(isinstance(act,int)):
            act = self.actions[act]
        current_DBN = self.FMDP[act.light_id]
        for ind in current_DBN["cpts"]:
            #print("add point in DBN {}, CPT{}".format(act, ind))
            leaf = current_DBN["cpts"][ind].add_datapoint(state_t, act, state_t1)
            leaf.compute_BIC_Mono()
            leaf.try_every_refinements()
            current_DBN["cpts"][ind].check_refinements(state_t)
        
    #Return solution to set the var to 1 : an option if exists, an action if not
    def get_solution(self, var, target_value):
        #print("solution for {} asked".format(var))
        #print("Implement get solution")
        if(var in self.C):
            if((var, target_value) in self.options):
                return self.options[(var, target_value)]
        if(var in self.action_to_set):
            return self.action_to_set[var]
        print(self.action_to_set)
        print("No soultion for {}".format(var))
        return(-1)
    
     #Return Action to set the var to 1 : an option if exists, an action if not
    def get_action(self, var):
        #print("solution for {} asked".format(var))
        #print("Implement get solution")
        if(var in self.action_to_set):
            return self.action_to_set[var]
        print("No action for {}".format(var))
        return(-1)
    
    def try_option(self, var,target_value, parents, sig_0, opt_root):
        if ((var, target_value) in self.options):
            if self.options[(var, target_value)].sig_0 > sig_0:
                return False
            self.options[(var, target_value)].opt_root.used=False
            self.remove(var, target_value)
        o=self.create_option(var,target_value, parents, sig_0, opt_root)
        #print("Create: ")
        #o.print_tree()
        #o.opt_root.parent.print_tree()
        opt_root.used = True
        return True
        
        
    def create_option(self, var, target_value, parents, sig_0, opt_root):
        
        o = self.Option(self, var, target_value, parents, sig_0, opt_root)
        #check if controllable, add to C or queue_for_C accordingly
        if self.is_controllable(o.variable):
            #TODO Check if an option wait for me to be controllable
                self.C.add(o.variable)
                self.check_update_C()
                
        else:
            self.queue_for_C.add(o.variable) 
        return o
    
    def remove(self, var, target_value):
        self.options[(var, target_value)].opt_root.used=False
        #if((var, target_value) in self.options):
            #print("remove : ")
            #self.options[(var, target_value)].print_tree()
        if(not (var, target_value) in self.options):
            print("No option for {} to remove".format((var, target_value)))
        removed_o = self.options.pop((var, target_value), None)
        
        #remove option from other options hierarchie 
        for node in LevelOrderIter(removed_o.root):
            if(isinstance(node.solution, AbstractAgent.Option)):
                #print("In remove {} -> {} \nTry to remove O{} [{}] of O{} [{}] hierarchie ".format(var,target_value, removed_o.variable,hex(id(removed_o)), node.solution.variable, hex(id(node.solution))))
                
                self.options_hierarchies[node.solution].remove(removed_o)
        
        #iterating over hierarchy even if it changes size
        tmp_hierarchy_set = self.options_hierarchies[removed_o].copy()
        for calling_opt in tmp_hierarchy_set:
            if(calling_opt in self.options_hierarchies[removed_o]):
                self.remove(calling_opt.variable, calling_opt.target_value)
        
        self.options_hierarchies.pop(removed_o,None)
        if(var in self.C):
            self.C.remove(var)
        if(var in self.queue_for_C):
            self.queue_for_C.remove(var)
            
        #To handle always controllable var, should be removed later
        if(self.is_controllable(var)):
            self.C.add(var)

    #remove an option and check if an other one can replace it
    def remove_and_recreate_option(self, var, target_value):
        
        self.remove(var, target_value)
        
        for key in self.FMDP:
            bn=self.FMDP[key]
            root = bn["cpts"][var].children[1-target_value]
            tree=root
            if not tree.is_leaf():
                opt_parents = []
                while(not tree.is_leaf()):
                    opt_parents.append(tree.var)
                    tree=tree.children[target_value]
                    
                sig=0
                for d in tree.dataset:
                    if d["s_1"][var] == target_value :
                        sig+=1
                sig = sig/len(tree.dataset)
                self.try_option(var, target_value, opt_parents, sig, root)
                
        
    def set_running_option(self, option):
        self.current_option(option)
        
    def is_controllable(self, var):
        #should be removed later
        if var in [1,2,3,4,5,6,7,8,9]:
            return True
        
        #if(not (var, 0) in self.options):
        #    return False
        #O_off = self.options[(var, 0)]
        
        if(not (var, 1) in self.options):
            return False
        O_on = self.options[(var, 1)]
        
        for p in O_on.parents:
            if p not in self.C:
                return False
            
        #for p in O_off.parents:
        #    if p not in self.C:
        #        return False
        return True
    
    
    #Check if a waiting var can be add to C    
    def check_update_C(self):
        updated = True
        while(updated == True):
            updated=False
            for var in self.queue_for_C:
                if(self.is_controllable(var)):
                    self.queue_for_C.remove(var)
                    self.C.add(var)
                    updated=True
                
        
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
            return("A{}".format(self.light_id))
        
        #execution
        def execute(self):
            self.env.turn_on(self.light_id)
        
        def is_action(self):
            return True
            


#==========================================================================================================
#====================================== OPTION, internal class ============================================
#==========================================================================================================


    class Option():
        
        #Constructor : variable associated, sigma_0, list of parent variables
        def __init__(self, my_agent, variable, target_value, parents, sig_0, opt_root):
            #agent
            self.my_agent = my_agent
            #variable on which the option should have an effect
            self.variable = variable
            #value in which the variable is supposed to be set after the option execution
            self.target_value = target_value
            #parents
            self.parents = parents
            #sigma and sigma_0
            self.sig_0 = sig_0
            self.sig = sig_0
            #nb of execution, used to update sigma
            self.nb_exec=0
            #step of the current execution
            self.step = 0
            #used when called as a nested option
            self.previous_option = None
            #True after terminal action
            self.done = False
            
            self.opt_root = opt_root
            
            previousNode = None
            options_called=[]
            #for each parent
            for i in range(len(parents)):
                #create a Node
                node = self.OptionTreeNode(parents[i], 1)
                #set the 0 child
                tmp_solution = my_agent.get_solution(parents[i], target_value)
                if(isinstance(tmp_solution, AbstractAgent.Option)):
                    #if(tmp_solution.check_if_call(self, [])):#wtf
                        
                        #tmp_solution = get_action(self.variable)
                    #else:
                    my_agent.options_hierarchies[tmp_solution].add(self)
                if(target_value == 1):
                    child_0 = self.OptionTreeNode(-1, 0, solution = tmp_solution)
                    child_0.parent = node
                else:
                    child_1 = self.OptionTreeNode(-1, 1, solution = tmp_solution)
                #if solution is option, add to option called
                #print(tmp_solution)
                if(not tmp_solution.is_action):
                    options_called.append(tmp_solution)
                #check if root
                if(i==0):
                    self.root=node
                    node.child_01 = None
                #if not, link to the parent
                else:
                    node.parent=previousNode
                #if last, set the 1 child
                if(i==len(parents)-1):
                    tmp_solution = my_agent.get_action(self.variable)
                    if(target_value == 1):
                        child_1 = self.OptionTreeNode(-1, 1, tmp_solution, is_terminal=True)
                        child_1.parent=node
                    else:
                        child_0 = self.OptionTreeNode(-1, 0, tmp_solution, is_terminal=True)
                        child_0.parent=node
                    #if solution is option, add to option called
                    if(not tmp_solution.is_action):
                        options_called.append(tmp_solution)
                if(target_value == 0):
                    child_1.parent = node
                previousNode = node
                
            #Will probably be removed    
            self.exec_pointer = self.root
            self.my_agent.options[(self.variable, target_value)] = self
            
            
                
            #register myself as caller of nested options    
            for o in options_called:
                self.my_agent.options_hierarchies[o].append(self)
            #init my own caller list    
            self.my_agent.options_hierarchies[self]=set()
            
        def __str__(self):
            val="Off"
            if(self.target_value == 1):
                val = "On"
            return("O{} -> {}".format(self.variable, val))
            
        def next_step(self, state):
            next_move = self.root
            self.step+=1
            #go down in the tree according to the state until the node contain an action or an option
            while(next_move.solution==None):
                var = next_move.var
                if(state[var-1]):
                    next_move=next_move.children[1]
                else:
                    next_move=next_move.children[0]
                    
            #find an action, return it    
            if(next_move.solution.is_action()):
                if(next_move.is_terminal):
                    self.done = True
                return(next_move.solution)
            #find an option   
            else:
                #set the nested option as current one and return the first action
                next_move.solution.reset()
                next_move.solution.set_previous(self)
                self.my_agent.set_running_option(next_move.solution)
                return(next_move.solution.next_step())
            
        #update sig and go back to the previous option if needed 
        def update_sig(self, delta):
            self.nb_exec+=1
            self.sig=self.sig+((delta-self.sig)/self.nb_exec)
            if(not (self.previous_option==None)):
                my_agent.set_running_option(self.previous_option)
                self.previous_option = None

        #Check if this option call opt
        def check_if_call(self, opt, visited):
            if(self in self.my_agent.options_hierarchies[opt]):
                return True
            new_visited = visited + [self]
            for node in LevelOrderIter(self.root):
                if(isinstance(node.solution, AbstractAgent.Option) and not (node.solution in visited)):
                    if(node.solution.check_if_call(opt, new_visited)):
                        return True
            return False
        
        #when called as nested option, stock the calling option in previous
        def set_previous(self, previous_option):
            self.previous_option = previous_option
        
        #reset before a new execution
        def reset(self):
            self.step = 0
            self.exec_pointer = self.root
            self.done=False
        
        #amazing display
        def print_tree(self):
            val="Off"
            if(self.target_value == 1):
                val = "On"
            print("O{} -> {}".format(self.variable, val))
            self.root.print_tree()   
        
        #used to check if a node contain an option or an action
        def is_action(self):
            return False
        
        #Internal class for nodes of the policy tree of the option
        class OptionTreeNode(NodeMixin):
            #Constructor : var of the split (-1 if option/action node), child_01 = 0 or 1, solution = action/option if needed, is_terminal = True for the terminal action node 
            def __init__(self, var, child_01, solution=None, is_terminal=False):
                self.var = var
                self.solution = solution
                self.child_01 = child_01
                self.is_terminal = is_terminal
                
            #another amazing display   
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
            self.BIC = 0
            self.used=False
            if(dataset == None):
                self.dataset = []
            else:
                self.dataset = dataset
            self.leaf_distrib = 0
            if(len(self.dataset)>0):
                for d in self.dataset:
                    if(d["s_1"][self.tree_var-1]==1):
                        self.leaf_distrib+=1
                self.leaf_distrib = self.leaf_distrib / (len(self.dataset))
            
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
                #treestr = u"%s%s" % (pre, node.var)$
                if(node.is_root and node.is_leaf()):
                    treestr = u"%s%s" % (pre, "{} {}".format(node.var,node.leaf_distrib))
                    print(treestr.ljust(8))
                elif(node.is_root):
                    treestr = u"%s%s" % (pre, node.var)
                    print(treestr.ljust(8))
                elif(node.is_leaf()):
                    treestr = u"%s%s" % (pre, "-{}-> {}".format(node.child_01,node.leaf_distrib))
                    print(treestr.ljust(8))
                
                else:
                    treestr = u"%s%s" % (pre, "-{}-> {}".format(node.child_01,node.var))
                    print(treestr.ljust(8))

            
        def is_leaf(self):
                return(len(self.children)==0)
            
        #return the leaf where the point has been added   
        def add_datapoint(self, s_0, act, s_1):
            val_on_s_0 = s_0[self.var-1]

                
            #if leaf, add to dataset and updat distrib_vect   
            if(self.is_leaf()):
                self.leaf_distrib = (self.leaf_distrib*len(self.dataset) + s_1[self.tree_var-1]) / (len(self.dataset) + 1)
                self.dataset.append({"s_0" : s_0, "a" : act, "s_1" : s_1})    
                for j in self.distrib_vect:
                    j_val = 0
                    if s_0[j-1]:
                        j_val=1
                        
                    self.distrib_vect[j][j_val]+=1
                return(self)
                
            #if not leaf
            else:
                if(val_on_s_0==False):
                    return(self.children[0].add_datapoint( s_0, act, s_1))
                else:
                    return(self.children[1].add_datapoint( s_0, act, s_1))
        
        #recreate distribution vectore, used on splits and prunes
        def recalculate_distrib_vect(self):
            self.distrib_vect = dict() 
            for i in range(1, self.nb_var+1):
                if not i in self.parents_list:
                    self.distrib_vect[i] =[0,0]
            for i in range(1, self.nb_var+1):
                if not i in self.parents_list:
                    for d in self.dataset:
                        i_val = 0
                        if d["s_0"][i-1]:
                            i_val=1
                        self.distrib_vect[i][i_val]+=1
                        
        #Shannon's entropy over a distribution vector            
        def shannon(self, vect):
            tot = 0
            h = 0
            
            for x in vect:
                tot+=x
            for x in vect:
                if(x>0):
                    h-= (x/tot) * np.log(x/tot)
            return h
        
        def entropy(self):
            h=0
            for v in self.distrib_vect:
                h+=self.shannon(self.distrib_vect[v])
            return h
        
        #calculate entropy gain in this node if the action is executed in the given state and
        #the datapoint goes in this dataset
        def entropy_gain(self, state):
            h0 = self.entropy()
            
            h1=0
            for v in self.distrib_vect:
                tmp_v = self.distrib_vect[v].copy()
                if(state[v-1]==True):
                    tmp_v[1]+=1
                else:
                    tmp_v[0]+=1
                h1+=self.shannon(tmp_v)
            return h1 - h0 
        
        #go down in the tree to find the dataset in which state goes and return the entropie gain
        def tree_entropy_gain(self, state):
            val_on_s_0 = state[self.var-1]

                
            #if leaf, return gain 
            if(self.is_leaf()):
                return(self.entropy_gain(state))
                
            #if not leaf
            else:
                if(val_on_s_0==False):
                    return(self.children[0].tree_entropy_gain(state))
                else:
                    return(self.children[1].tree_entropy_gain(state))
        
        
        #BIC computation          
        def compute_BIC_not_used(self):
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
                #for j in self.DBN["parents"][i]:
                for j in self.parents_list:
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
            #print(L)                        
            #finally compute BIC            
            BIC = L - ((self.nb_var)/2 * np.log(len(self.dataset)))
            self.BIC = BIC
            return BIC
        
        def try_every_refinements(self):
            if(len(self.dataset)<self.my_agent.K):
                return -1
            refinements = []
            #for every non-parent variable
            for var in range(1, self.nb_var+1):
                if(not(var in self.parents_list or var == self.tree_var)):
                    #if refinement, stock it
                    tmp_refin =self.try_refinement( var)
                    if(tmp_refin[0]):
                        refinements.append(tmp_refin)
                        
            #if no refinment return -1
            if(len(refinements)==0):
                return(-1)
            
            max_refin=None
            max_delta_refin = 0
            #choose the refinement with the best delta Childrend BICs - BIC
            for tmp_refin in refinements:
                if(tmp_refin[2]>max_delta_refin):
                    max_refin = tmp_refin
                    max_delta_refin = tmp_refin[2]
            #create refin
            (_, var, delta, child_0, child_1) = max_refin
            child_0.parent = self
            child_1.parent = self
            #to set tree var to the other value
            target_var = 1 - self.dataset[0]["s_0"][self.tree_var-1]
            self.var=var
            self.dataset=[]
            children_parents = self.parents_list + [var] 
            #TODO Sigma estimation
            sig=0
            for d in self.children[target_var].dataset:
                if d["s_1"][self.tree_var-1] == target_var :
                    sig+=1
            sig = sig/len(child_1.dataset)
            opt_parents = children_parents.copy()
            opt_parents.remove(self.tree_var)
            #print("var : {}, delta : {}".format(var, delta))
            #self.DBN["cpts"][self.tree_var].print_tree()
            child_0.recalculate_distrib_vect()
            child_1.recalculate_distrib_vect()
            self.my_agent.try_option(self.tree_var, target_var, opt_parents, sig, self.DBN["cpts"][self.tree_var].children[1-target_var])
        
        #return a tuple (bool, var, delta, child_0, child_1)
        def try_refinement(self, var):
            #split dataset on the refinement var
            dataset_0 = []
            dataset_1 = []
            for d in self.dataset:
                if(d["s_0"][var-1]==0):
                    dataset_0.append(d)
                else:
                    dataset_1.append(d)
            #avoid size 1 dataset because the BIC is 0 (log(1) = 0) and refine every time
            if(len(dataset_0) == 1 or len(dataset_1)==1):
                return (False, None, None, None ,None)
            children_parents = self.parents_list + [var]        
            child_0 = type(self)(self.DBN,self.tree_var, self.my_agent, parents_list = children_parents, dataset =dataset_0, child_01 = 0) 
            child_1 = type(self)(self.DBN, self.tree_var, self.my_agent, parents_list = children_parents, dataset =dataset_1, child_01 = 1)
            #if children BICs are better
            BIC0 = child_0.compute_BIC_Mono()
            BIC1 = child_1.compute_BIC_Mono()
            if(BIC0 + BIC1 > self.compute_BIC_Mono()):
                #print("BIC0 = {}, BIC1 = {}, my BIC = {}, var = {}, tree_var = {}".format(BIC0, BIC1, self.compute_BIC_Mono(),var, self.tree_var))
                #if(False):
                #    print("Dataset parent")
                #    tmpind = 0
                #    for d in self.dataset:
                #        tmpind += 1
                #        print("{} : {} ; {}".format(tmpind,d["s_0"],d["s_1"]))
                #    tmpind = 0
                #    print("Dataset child0")
                #    for d in child_0.dataset:
                #        tmpind += 1
                #        print("{} : {} ; {}".format(tmpind,d["s_0"],d["s_1"]))
                #    tmpind = 0
                ##    print("Dataset child1")
                 #   for d in child_1.dataset:
                 #       tmpind += 1
                 #       print("{} : {} ; {}".format(tmpind,d["s_0"],d["s_1"]))
                return (True, var, BIC0+BIC1-self.BIC, child_0, child_1)
            #no refinement
            return (False, None, None, None ,None)
            
        #return true if the var of the node is independent from the tree var => refinement not consistent    
        def chi_2(self):
            refined_var = self.var
            desc_dataset = []
            for desc in self.descendants:
                desc_dataset = desc_dataset + desc.dataset
            n_tot=len(desc_dataset)
            if(n_tot==0):
                return False
            #n_ij
            n=[[0,0],[0,0]]
            
            for d in desc_dataset:
                i=d["s_0"][refined_var-1]
                j=d["s_1"][self.tree_var-1]
                n[i][j]+=1
            if((n[0][0]==0 or n[1][1]==0) and (n[0][1]==0 or n[1][0]==0)):
                return True
            chi2 = chi2_contingency(n)
            independency = chi2[1] > (1-self.my_agent.inde_treshold)
            #if(independency):
                #print(chi2)
            return(independency)
        
        def prune(self):
            desc_dataset=[]
            for desc in self.descendants:
                desc_dataset = desc_dataset + desc.dataset
                #just to be sure
                del desc
            
            self.dataset=desc_dataset
            if(len(self.dataset)>0):
                for d in self.dataset:
                    if(d["s_1"][self.tree_var-1]==1):
                        self.leaf_distrib+=1
                self.leaf_distrib = self.leaf_distrib / (len(self.dataset))
            self.children=[]
            self.recalculate_distrib_vect()
            
        def check_refinements(self, s_0):
            val_on_s_0 = s_0[self.var-1] 
            #if leaf, no refinment 
            if(self.is_leaf()):
                return False
            
            elif(( self.var != self.tree_var ) and self.chi_2()):
                #print("Prune on DBN : {}, CPT : {}, Node of var : {}".format(self.DBN, self.tree_var, self.var))
                
                tmp_target = 1-s_0[self.tree_var-1]
                opt_root = self.DBN["cpts"][self.tree_var].children[1-tmp_target]
                #print("Prune on : {} in : ".format(self.var))
                #opt_root.parent.print_tree()
                #opt_root.print_tree()
                #print(opt_root.used)
                self.prune()
                
                if opt_root.used:
                    self.my_agent.remove_and_recreate_option(self.tree_var, tmp_target)
                return True
                #TODO Option and things
                
            #if not leaf
            else:
                if(val_on_s_0==False):
                    return(self.children[0].check_refinements( s_0))
                else:
                    return(self.children[1].check_refinements( s_0))