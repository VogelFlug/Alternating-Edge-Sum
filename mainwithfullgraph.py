# This is for 2 dimensional data that has a full Graph already given at the start

import sys

from main import main
from classes.TwoDGraph import TwoDGraph

if __name__ == '__main__':
    x = sys.argv[1]
    #print(x)
    data = ""
    with open(x , "r") as f:
        data = f.read()
    Graph = TwoDGraph(vgl = data)
    main(Graph)