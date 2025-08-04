# This is for data where only a hull is provided. The points inside the hull will be randomized and then connected via Delauney Triangulation. TODO

import sys

from main import main
from classes.TwoDGraph import TwoDGraph

if __name__ == '__main__':
    if(len(sys.argv)<2):
        raise Exception("You need to provide Graph Data")
    
    filepath = sys.argv[1]

    data = ""
    with open(filepath , "r") as f:
        data = f.read()
    Graph = TwoDGraph(vgl = data)
    main(Graph, filepath)


