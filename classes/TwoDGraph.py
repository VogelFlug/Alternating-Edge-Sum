import numpy as np
#from . import Graphutil as util

class TwoDGraph:
    vertices: np.ndarray
    faces: tuple[int,int,int]
    vertexnumber: int
    facenumber: int

    neighbourhood: tuple[set[int]]
    edgecounter: np.ndarray

    def __init__(self, *, vgl: str = "nuh uh", vertices = [], faces = []):
        if(vgl == "nuh uh"):
            self.vertices = vertices
            self.faces = faces
            self.vertexnumber = vertices.shape[1]
            self.facenumber = len(faces)
        else:
            self.get2dgraphfromvgl(vgl)
            self.initneighandedgecounter()

    def get2dgraphfromvgl(self, data: str):
        lines = data.splitlines()
        nroflines = len(lines)
        linecounter = 0

        #get rid of header
        while(lines[linecounter][0] != "#"):
            linecounter += 1
        linecounter += 1
        
        #get number of vertices and faces
        self.vertexnumber = int(lines[linecounter][0])
        self.facenumber = int(lines[linecounter][2])
        linecounter += 1

        #get vertexpositions in the most scuffed way possible
        vertices = np.zeros((2,self.vertexnumber))
        vertexcounter = 0
        while(lines[linecounter][0] != "#"):
            vertex = []
            line = lines[linecounter]
            chacount = 0
            linened = False
            while(True):
                number =""

                while(line[chacount] != " "):
                    number = number + line[chacount]
                    if(chacount == len(line)-1):
                        vertex.append(float(number))
                        linened = True
                        break
                    chacount += 1
                if(linened == True):
                    break
                chacount += 1
                vertex.append(float(number))
            vertices[:,vertexcounter] = vertex
            linecounter += 1
            vertexcounter += 1
        self.vertices = vertices
        linecounter += 1

        #get facelist in the same scuffed way
        faces = []
        while(linecounter < nroflines):
            face = []
            line = lines[linecounter]
            chacount = 0
            linened = False
            while(True):
                number =""

                while(line[chacount] != " "):
                    number = number + line[chacount]
                    if(chacount == len(line)-1):
                        face.append(int(number))
                        linened = True
                        break
                    chacount += 1
                if(linened == True):
                    break
                chacount += 1
                face.append(int(number))
            faces.append(face)
            linecounter += 1
        self.faces = tuple(faces)

        return
    
    


    def initneighandedgecounter(self):
        neighbourhood = []
        edgecounter = np.zeros((self.vertexnumber, self.vertexnumber))
        for i in range(0,self.vertexnumber): 
            neighbourhood.append(set())
        for i,j,k in self.faces:
            #get all neighbours for each vertex
            neighbourhood[i].add(j)
            neighbourhood[i].add(k)
            neighbourhood[j].add(i)
            neighbourhood[j].add(k)
            neighbourhood[k].add(i)
            neighbourhood[k].add(j)

            #create a matrix to keep track of how many times each edge appears in our list of faces
            edgecounter[i,j] += 1
            edgecounter[j,i] += 1
            edgecounter[i,k] += 1
            edgecounter[k,i] += 1
            edgecounter[j,k] += 1
            edgecounter[k,j] += 1

        self.neighbourhood = tuple(neighbourhood)
        self.edgecounter = edgecounter
