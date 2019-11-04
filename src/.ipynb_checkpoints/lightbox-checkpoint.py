class LightBox:

        
    def __init__(self, light_by_lvl=None, dependencies=None):
        if(light_by_lvl is None and dependencies is None):
            light_by_lvl = [9,6,4,1]
            dependencies = [[],[],[],[],[],[],[],[],[],[1,4,7],[2,5],[3,6],[2,3],[5,6],[7,8],[10,11],[11,12],[13,14],[14,15],[17,18]]
        elif(light_by_lvl is None and dependencies is None):
            print("You must give both light_by_lvl and dependencies argument, or no one but you can't give one")
            return
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