# This is for 2 dimensional data that has a full Graph already given at the start

import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')

from main import main
from classes.TwoDGraph import TwoDGraph

filepath = "data/2dfolder/fulldata/notquitemb.txt"
attempts = 1
stepsize = 30000

if __name__ == '__main__':
    # if(len(sys.argv)<2):
    #     raise Exception("You need to provide Graph Data")
    #filepath = sys.argv[1]

    data = ""
    with open(filepath , "r") as f:
        data = f.read()
    Graph = TwoDGraph(vgl = data)
    main(Graph, "output" + filepath[4:-4], attempts, stepsize)


