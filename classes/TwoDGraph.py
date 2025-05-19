class TwoDGraph:
    vertices: tuple[int,int]
    faces: tuple[int,int,int]
    vertexnumber: int
    facenumber: int

    def __init__(self, data: str):
        self.get2dgraphfromvgl(data)

    def get2dgraphfromvgl(self, data: str):
        lines = data.splitlines()
        nroflines = len(lines)
        linecounter = 0

        #get rid of header
        while(lines[linecounter][0] != "#"):
            linecounter += 1
        linecounter += 1
        
        #get number of vertices and faces
        self.vertexnumber = lines[linecounter][0]
        self.facenumber = lines[linecounter][2]
        linecounter += 1

        #get vertexpositions in the most scuffed way possible
        vertices = []
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
                        vertex.append(int(number))
                        linened = True
                        break
                    chacount += 1
                if(linened == True):
                    break
                chacount += 1
                vertex.append(int(number))
            vertices.append(vertex)
            linecounter += 1
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
        self.faces = faces

        return
    
    
