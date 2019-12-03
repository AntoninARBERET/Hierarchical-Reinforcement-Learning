import numpy as np
import pygraphviz as pgv
from graphviz import Digraph

class LightBox:
    

    #constructor, create a light box with specified lights and dependecies if BOTH are given. If not, construc the default one
    def __init__(self, light_by_lvl=None, dependencies=None):
        #attributes
        self.nb_level = 0
        self.list_ligth = []
        
        if(light_by_lvl is None and dependencies is None):
            light_by_lvl = [9,6,4,1]
            dependencies = [[],[],[],[],[],[],[],[],[],[1,4,7],[2,5],[3,6],[2,3],[5,6],[7,8,9],[10,11],[11,12],[13,14],[14,15],[17,18]]
        elif(light_by_lvl is None and dependencies is None):
            print("You must give both light_by_lvl and dependencies argument, or no one but you can't give one. Creating default lightbox.")
            light_by_lvl = [9,6,4,1]
            dependencies = [[],[],[],[],[],[],[],[],[],[1,4,7],[2,5],[3,6],[2,3],[5,6],[7,8],[10,11],[11,12],[13,14],[14,15],[17,18]]
            
        self.nb_level = len(light_by_lvl)
        self.list_light = []
        currentLvl = 1
        indNextLevel = light_by_lvl[0]
        for i in range(len(dependencies)):
            if(i==indNextLevel):
                indNextLevel+=light_by_lvl[currentLvl]
                currentLvl+=1
            light = dict()
            light["id"]=i+1
            light["level"]=currentLvl
            light["dependencies"]=dependencies[i]
            light["state"]=False
            self.list_light.append(light)
            
    def __str__(self):
        currentLvl=1
        s=""
        s=s+"-----------lvl {}----------\n".format(currentLvl)
        for i in range(len(self.list_light)):
            l = self.list_light[i]
            if(l["level"]>currentLvl):
                currentLvl+=1
                s=s+"-----------lvl {}----------\n".format(currentLvl)
            s=s+"Light {} \tvalue = {}".format(l["id"],l["state"])
            if len(l["dependencies"])>0:
                s=s+"\tDepend on : "
                for k in range(len(l["dependencies"])):
                    s=s+"{} ".format(l["dependencies"][k])
            s=s+"\n"
        return s
    
    #turn a light on
    def turn_on(self, light_id):
        #10% of fail on the action
        if np.random.random() < 0.1 :
            return
        l = self.list_light[light_id - 1]
        ##TODO remove later
        switch_off=True
        if(l["state"] and switch_off):
            l["state"]=False
            return
        #for each dependencies
        for dep in l["dependencies"]:
            #if it's off, shutdown the whole box
            if self.list_light[dep-1]["state"] == False:
                self.shut_down()
                return
        l["state"]=True
    
    #shutdown every lights
    def shut_down(self):
        for l in self.list_light:
            l["state"] = False
    
    #return state as a list of boolean
    def get_state(self):
        state = []
        for l in self.list_light:
            state.append(l["state"])
        return state
    
    #return number of lights
    def get_nb_light(self):
        return len(self.list_light)
    
    def show(self):
        graph = Digraph()
        for l in self.list_light:
            col="lightgrey"
            if(l["state"]):
                col="yellow"
            graph.node("{}".format(l["id"]),style='filled', color=col)
            #graph.get_node(attr['fillcolor']="#CCCCFF"
        for l in self.list_light:
            for dep in l["dependencies"]:
                e="{}{}".format(l["id"],dep)
                graph.edge("{}".format(dep), "{}".format(l["id"]))
        return graph