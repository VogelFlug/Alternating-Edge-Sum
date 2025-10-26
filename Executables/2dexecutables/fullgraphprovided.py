# This is for 2 dimensional data that has a full Graph already given at the start

import sys
sys.path.insert(0, 'C:/Users/ich/Desktop/Uni/Alternating-Edge-Sum')
sys.path.insert(0, 'C:/Users/daveb/OneDrive/Desktop/Uni/Alternating-Edge-Sum/')

from main import main
from util.TwoDGraph import TwoDGraph

filepath = "data/2dfolder/fulldata/megabasic.txt"
outputfolder = "output/2dfolder/optimizesvg/"
attempts = 1
stepsize = 5000

if __name__ == '__main__':
    # if(len(sys.argv)<2):
    #     raise Exception("You need to provide Graph Data")
    #filepath = sys.argv[1]

    data = ""
    with open(filepath , "r") as f:
        data = f.read()
    Graph = TwoDGraph(vgl = data)
    main(Graph, outputfolder + filepath[-13:-4], attempts, stepsize)


