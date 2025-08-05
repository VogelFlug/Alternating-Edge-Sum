# This is for data where only a hull is provided. The points inside the hull will be randomized and then connected via Delauney Triangulation. TODO

import sys

from main import main
from classes.TwoDGraph import TwoDGraph
from scipy.spatial import Delaunay

if __name__ == '__main__':
    if(len(sys.argv)<2):
        raise Exception("You need to provide Graph Data")
    
    filepath = sys.argv[1]

    #get hull vertices, TODO: very scuffed solution, will improve at later date
    data = ""
    with open(filepath , "r") as f:
        data = f.read()
    vertices = TwoDGraph(vgl = data).vertices





