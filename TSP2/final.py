import sys
from pprint import pprint
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QPushButton
from PyQt5.QtGui import QIcon
import pickle
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPlainTextEdit
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QSize
import sys, random
import pickle
import random
import numpy as np
import scipy.spatial
from PyQt5.QtCore import QPoint
import collections
import math
import time
import heapq
import tkinter as tk

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

global tri

open_sets = dict()
closed_sets = dict()
paths = dict()

iteration = 0
counter = 0

closed_set = set()
open_set = set()

points = []
dict_index_coord = {}
dict_coord_index = {}
edge_list = []
adjacency_list = []
node_colour = []
result_path = []

PRINT_RESULTS = False

class Edge:
	def __init__(self,v1,v2,edge_cost):
		self.v1=v1
		self.v2=v2
		self.edge_cost=edge_cost

	def __lt__(self,other):
		return self.edge_cost<other.edge_cost
class Edge1:
	def __init__(self,v1,v2,savings):
		self.v1=v1
		self.v2=v2
		self.savings=savings

	def __lt__(self,other):
		return self.savings>other.savings

## function to generate the next move
def movegen(adj, n_ind):
    return adj[n_ind]

#This is where you have to write you solution, according to the assignment whihc was given to you
#Points is list of structure which has x_coord, y_coord, isMax(if True then the node is Max node), level(Root at level 0)
#Edge List is the list of edges connecting the points mentioned above, you will need this when you have to color the edge for showing step by step representation
#Paths and open sets and closed_sets is a dictionary, after each iteration in Paths insert the edge from the edge_list which you want to colour
#open_sets and closed_sets is also a dictionary which will contain the index of the Point from list points

#Counter is the number of iterations it took, For eg: if counter is 5, you have to add your step by step solution in paths and visited_set as
#paths[0].....path[4] and open_sets[0]...open_sets[4] and closed_sets[0]...closed_sets[4]. This can be viewed as an animation in every mouse press event.

## If you are given a TSP algorithm, Please edit this function below to write your answer
def nearestAllowedNeighour(currentCity,visited,n_node):
    min=sys.float_info.max
    i=1
    nearestNeighbour=0
    while i<=n_node:
        if visited[i]==False and math.sqrt((dict_index_coord[currentCity][0]-dict_index_coord[i][0])**2+(dict_index_coord[currentCity][1]-dict_index_coord[i][1])**2)<min:
            min=math.sqrt((dict_index_coord[currentCity][0]-dict_index_coord[i][0])**2+(dict_index_coord[currentCity][1]-dict_index_coord[i][1])**2)
            nearestNeighbour=i

        i=i+1

    return nearestNeighbour
def cost(result_path):
  sum=0
  for i in range(len(result_path)-1):
    sum=sum+math.sqrt(result_path[i+1]**2-result[i]**2)
  return sum  

def Algorithm_TSP1(start_node, n_node):
    # Write your TSP Code here
    currentCity=start_node
	global result_path
	result_path=[start_node]
	visited=[False for i in range(n_node+1)]
	visited[start_node]=True
	while len(result_path)<n_node:
		neighbour=nearestAllowedNeighour(currentCity,visited,n_node);
		visited[neighbour]=True
		result_path.append(neighbour)
		currentCity=neighbour

	result_path.append(start_node)
	print(result_path)
	total_cost=0.0
	i=0
	while i<len(result_path)-1:
		total_cost=total_cost+math.sqrt((dict_index_coord[result_path[i]][0]-dict_index_coord[result_path[i+1]][0])**2+(dict_index_coord[result_path[i]][1]-dict_index_coord[result_path[i+1]][1])**2)
		i=i+1
	print("Total cost="+str(total_cost))

def find(parent,i):
	j=i
	while parent[i]!=i:
		i=parent[i]

	while parent[j]!=i:
		temp=parent[j]
		parent[j]=i
		j=temp
	return i

def dfs(start_node,next_node,visited,result_path):
	stack=[]
	stack.append(start_node)
	
	while len(stack)>0:
		v=stack.pop()
		visited[v]=True
		result_path.append(v)
		for vertex in next_node[v]:
			if visited[vertex]==False:
				stack.append(vertex)

	


def Algorithm_TSP2(start_node, n_node):
	# Write your TSP Code here
	
	global result_path
	result_path.clear()
	edges=[]
	i=1
	while i<n_node:
		j=i+1
		while j<=n_node:
			edges.append(Edge(i,j,math.sqrt((dict_index_coord[i][0]-dict_index_coord[j][0])**2+(dict_index_coord[i][1]-dict_index_coord[j][1])**2)))
			j=j+1
		
		i=i+1

	edges=sorted(edges)
	
	rank=[0 for i in range(n_node+1)]
	parent=[i for i in range(n_node+1)]
	#print(parent)
	degree=[0 for i in range(n_node+1)]
	next_node=[[] for i in range(n_node+1)]
	visited=[False for i in range(n_node+1)]
	edge_count=0
	for edge in edges:
		a=find(parent,edge.v1)
		b=find(parent,edge.v2)
		if a==b:
			continue
		elif degree[edge.v1]>=2 or degree[edge.v2]>=2:
			continue
		else:
			edge_count=edge_count+1
			if rank[a]<rank[b]:
				parent[a]=b
			elif rank[a]>rank[b]:
				parent[b]=a
			else:
				parent[a]=b
				rank[b]=rank[b]+1
			next_node[edge.v1].append(edge.v2)
			next_node[edge.v2].append(edge.v1)
			degree[edge.v1]=degree[edge.v1]+1
			degree[edge.v2]=degree[edge.v2]+1
			edge_count=edge_count+1
			if edge_count==n_node-1:
				break

	oneDegree=[]
	i=1
	while i<=n_node:
		if degree[i]==1:
			oneDegree.append(i)
		i=i+1

	degree[oneDegree[0]]=degree[oneDegree[0]]+1
	degree[oneDegree[1]]=degree[oneDegree[1]]+1
	next_node[oneDegree[0]].append(oneDegree[1])
	next_node[oneDegree[1]].append(oneDegree[0])
	dfs(start_node,next_node,visited,result_path)
	result_path.append(start_node)
	
	print(result_path)
	#print("len="+str(len(result_path)))
	#print(degree)
	total_cost=0.0
	i=0
	while i<len(result_path)-1:
		total_cost=total_cost+math.sqrt((dict_index_coord[result_path[i]][0]-dict_index_coord[result_path[i+1]][0])**2+(dict_index_coord[result_path[i]][1]-dict_index_coord[result_path[i+1]][1])**2)
		i=i+1
	print("Total cost="+str(total_cost))

def Algorithm_TSP3(start_node, n_node):
	# Write your TSP Code here
	
	global result_path
	result_path.clear()
	edges=[]
	i=1
	while i<n_node:
		j=i+1
		while j<=n_node:
			edges.append(Edge1(i,j,math.sqrt((dict_index_coord[start_node][0]-dict_index_coord[i][0])**2+(dict_index_coord[start_node][1]-dict_index_coord[i][1])**2)+math.sqrt((dict_index_coord[start_node][0]-dict_index_coord[j][0])**2+(dict_index_coord[start_node][1]-dict_index_coord[j][1])**2)-math.sqrt((dict_index_coord[i][0]-dict_index_coord[j][0])**2+(dict_index_coord[i][1]-dict_index_coord[j][1])**2)))
			j=j+1		
		i=i+1
	edges=sorted(edges)
	accessible=[True for i in range(n_node+1)]
	accessible[start_node]=False
	next_node=[[] for i in range(n_node+1)]
	degree=[0 for i in range(n_node+1)]
	visited=[False for i in range(n_node+1)]
	partner=[i for i in range(n_node+1)]
	count=0
	for edg in edges:
		if accessible[edg.v1]==True and accessible[edg.v2]==True and partner[edg.v1]!=edg.v2:
			degree[edg.v1]=degree[edg.v1]+1
			degree[edg.v2]=degree[edg.v2]+1
			next_node[edg.v1].append(edg.v2)
			next_node[edg.v2].append(edg.v1)
			if partner[edg.v1]!=edg.v1:
				accessible[edg.v1]=False
			if partner[edg.v2]!=edg.v2:
				accessible[edg.v2]=False
			temp=partner[edg.v1];
			partner[partner[edg.v1]]=partner[edg.v2]
			partner[partner[edg.v2]]=temp
			count=count+1
			if count==n_node-2:
				break
	
	i=1
	while i<=n_node:
		if degree[i]==1:
			next_node[start_node].append(i)
			next_node[i].append(start_node)
		i=i+1

	dfs(start_node,next_node,visited,result_path)
	result_path.append(start_node)
	
	print(result_path)
	total_cost=0.0
	i=0
	while i<len(result_path)-1:
		total_cost=total_cost+math.sqrt((dict_index_coord[result_path[i]][0]-dict_index_coord[result_path[i+1]][0])**2+(dict_index_coord[result_path[i]][1]-dict_index_coord[result_path[i+1]][1])**2)
		i=i+1
	print("Total cost="+str(total_cost))



## If you are given a Search algorithm, Please edit this function below to write your answer
def Algorithm(adjacency_list, start_node, goal_node, n_node):
    global closed_set
    global open_set
    global node_colour
    global result_path

    global open_sets
    global closed_sets
    global paths
    global counter

    open_set.clear()
    closed_set.clear()

    # This is where you have to write you solution, according to the assignment which is given to you
    return []


# generate points randomly for a graph
def generate_points(xl, yl, number=50):
    x_coordinates = np.random.randint(xl, size=number)
    y_coordinates = np.random.randint(yl, size=number)
    for i, j in zip(list(x_coordinates), list(y_coordinates)):
        points.append([i, j + 70])

    if PRINT_RESULTS:
        print("GENERATE POINTS \n", points)

    return points


## function to make adjacency list of given neighbors
def make_adj_list(a, d, n):
    t = [[] for i in range(n + 1)]
    tt = []
    for i in a:
        t[d[i[0]]].append(d[i[1]])
        t[d[i[1]]].append(d[i[0]])
    for i in t:
        tt.append(list(set(i)))
    return tt


## function to find neighbor used in triangulation
def find_neighbors(pindex, triang):
    if PRINT_RESULTS:
        print("Find Neigh", triang.vertex_neighbor_vertices[1][
                            triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]])
    return triang.vertex_neighbor_vertices[1][
           triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]]


def make_edge_list(points, tri, bf):
    global edge_list
    for pindex in range(len(points)):
        neighbor_indices = find_neighbors(pindex, tri)
        for i in range(len(neighbor_indices)):
            if i % bf != 0:
                edge_list.append([(points[pindex][0], points[pindex][1]),
                                  (points[neighbor_indices[i]][0], points[neighbor_indices[i]][1])])

    if PRINT_RESULTS:
        print("Edge List", edge_list)
    return edge_list


## function to make dictionary mapping indices to coordinate points
def make_dict_index_coord(points):
    global dict_index_coord
    idx = 1
    for point in points:
        dict_index_coord[idx] = (point[0], point[1])
        idx += 1

    if PRINT_RESULTS:
        print("Dict_index\n", dict_index_coord)
    return dict_index_coord


## function to make dictionary mapping coordinate points to indices
def make_dict_coord_index(points):
    global dict_coord_index
    cnt = 1
    for i in points:
        dict_coord_index[(i[0], i[1])] = cnt
        cnt += 1
    return dict_coord_index


## Below are class init definitions
class Graph(QMainWindow):

    def init_graph_paramaters(self):
        global points
        global dict_index_coord
        global edge_list
        points = []
        dict_index_coord = {}
        edge_list = []
        self.nodes = int(self.nodeText.toPlainText())
        self.op=int(self.optionText.toPlainText())
        self.bf = int(self.bfText.toPlainText())
        self.graph_node_colour = [0 for i in range(self.nodes + 1)]

    def generateTSP(self):
        global node_colour
        self.event = "Generate_TSP"
        self.init_graph_paramaters()
        x_dim = screen_width - 100
        y_dim = screen_height - 150
        generate_points(x_dim, y_dim, self.nodes)
        make_dict_index_coord(points)
        make_dict_coord_index(points)
        node_colour = [0 for i in range(self.nodes + 1)]
        self.update()

    def generateGraph(self):
        global tri
        global node_colour
        self.event = "Generate_Graph"
        self.init_graph_paramaters()
        x_dim = screen_width - 100
        y_dim = screen_height - 150
        generate_points(x_dim, y_dim, self.nodes)
        tri = scipy.spatial.Delaunay(np.array(points))
        make_edge_list(points, tri, self.bf)
        make_dict_index_coord(points)
        make_dict_coord_index(points)
        node_colour = [0 for i in range(self.nodes + 1)]
        self.update()

    def reset_screen(self):
        self.event = "Reset_Screen"
        self.delAct.setEnabled(False)
        self.startAct.setEnabled(False)
        self.goalAct.setEnabled(False)
        self.update()

    ## function to save the graph data structure as pickle files
    def saveGraph(self):
        global points
        global dict_index_coord
        global edge_list

        with open('points.pkl', 'wb') as f:
            pickle.dump(points, f)

        with open('dict_index_coord.pkl', 'wb') as f:
            pickle.dump(dict_index_coord, f)

        with open('edge_list.pkl', 'wb') as f:
            pickle.dump(edge_list, f)

    ## function to load the saved graph data structure from pickle files
    def loadGraph(self):
        global points
        global dict_index_coord
        global tri
        global edge_list
        global result_path

        with open('points.pkl', 'rb') as f:
            points = pickle.load(f)

        with open('dict_index_coord.pkl', 'rb') as f:
            dict_index_coord = pickle.load(f)

        with open('edge_list.pkl', 'rb') as f:
            edge_list = pickle.load(f)

        tri = scipy.spatial.Delaunay(np.array(points))
        self.nodes = len(points)
        result_path = []

        self.event = "Generate_Graph"
        self.delAct.setEnabled(True)
        self.startAct.setEnabled(False)
        self.goalAct.setEnabled(False)

    ## code for GUI
    def initUI(self):

        self.saveAct = QAction('&Save Graph', self)
        self.saveAct.setShortcut('Ctrl+S')
        self.saveAct.setStatusTip('Save Graph')
        self.saveAct.triggered.connect(self.saveGraph)

        self.loadAct = QAction('&Load Graph', self)
        self.loadAct.setShortcut('Ctrl+O')
        self.loadAct.setStatusTip('Load Graph')
        self.loadAct.triggered.connect(self.loadGraph)

        self.exitAct = QAction('&Exit', self)
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(qApp.quit)

        # Action of generating graphs, TSP
        self.genAct = QAction('&Generate Graph', self)
        self.genAct.triggered.connect(self.generateGraph)

        self.genTSPAct = QAction('&Generate TSP', self)
        self.genTSPAct.triggered.connect(self.generateTSP)

        # Text Mode on Screen


        self.optionLabel = QLabel('Option:')
        self.optionText = QPlainTextEdit('enter(1:nn, 2:grd 3:sav)')
        self.optionText.setFixedSize(180, 28)



        self.nodeLabel = QLabel('Number of nodes:')
        self.nodeText = QPlainTextEdit('100')
        self.nodeText.setFixedSize(80, 28)



        self.bfLabel = QLabel('Branching Factor:')
        self.bfText = QPlainTextEdit('2')
        self.bfText.setFixedSize(80, 28)

        self.delAct = QAction('&Delete Node', self)
        self.startAct = QAction('&Start Node', self)
        self.goalAct = QAction('&Goal Node', self)

        self.resetAct = QAction('&Reset Screen', self)
        self.resetAct.triggered.connect(self.reset_screen)

        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('File')
        self.fileMenu.addAction(self.saveAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addAction(self.exitAct)

        self.toolbar = self.addToolBar('')
        self.toolbar.addWidget(self.optionLabel)
        self.toolbar.addWidget(self.optionText)
        self.toolbar.addWidget(self.nodeLabel)
        self.toolbar.addWidget(self.nodeText)
        self.toolbar.addWidget(self.bfLabel)
        self.toolbar.addWidget(self.bfText)

        self.toolbar.addAction(self.genAct)
        self.toolbar.addAction(self.genTSPAct)
        self.toolbar.addAction(self.delAct)
        self.toolbar.addAction(self.startAct)
        self.toolbar.addAction(self.goalAct)

        self.toolbar.addAction(self.resetAct)
        self.setMouseTracking(True)
        self.delAct.setEnabled(False)
        self.startAct.setEnabled(False)
        self.goalAct.setEnabled(False)

    def __init__(self):
        super().__init__()

        self.nodes = -1
        self.bf = -1
        self.dict_index_coord = {}
        self.open_list = []
        self.closed_list = []
        self.init_phase = -1

        self.start_x = 0
        self.start_y = 0

        self.del_x = 0
        self.del_y = 0

        self.goal_x = 0
        self.goal_y = 0

        self.initUI()
        self.setMinimumSize(QSize(screen_width, screen_height))
        self.setWindowTitle("Alviz v1.0")

    ## function to detect closest node from your mouse click
    def findClosestCoordinate(self, min_x, min_y, x, y):
        min_dist = 999999999
        global result_path
        global iteration
        global open_sets
        global closed_sets
        global open_set
        global closed_set
        global paths

        for p in points:
            dist = (x - p[0]) ** 2 + (y - p[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                min_x = p[0]
                min_y = p[1]

        if self.event == "Generate_Graph":
            self.del_x = min_x
            self.del_y = min_y
            self.event = "Del_Node"

        elif self.event == "Generate_TSP":
            self.start_x = min_x
            self.start_y = min_y
            start_node = dict_coord_index[(self.start_x, self.start_y)]
            
            if self.op==1:
                Algorithm_TSP1(start_node, self.nodes)
            elif self.op==2:
                Algorithm_TSP2(start_node, self.nodes)    
            elif self.op==3:  
                Algorithm_TSP3(start_node, self.nodes)
            else :
              Algorithm_TSP1(start_node, self.nodes)
              Algorithm_TSP2(start_node, self.nodes)
              Algorithm_TSP3(start_node, self.nodes)
              
                     
           
            self.event = "Display_Path"
            iteration = 0           


        elif self.event == "Del_Node":
            self.start_x = min_x
            self.start_y = min_y
            self.event = "Start_Node"

        elif self.event == "Start_Node":
            self.goal_x = min_x
            self.goal_y = min_y
            self.event = "Goal_Node"

        elif self.event == "Goal_Node":
            print("Goal Node selected")
            start_node = dict_coord_index[(self.start_x, self.start_y)]
            goal_node = dict_coord_index[(self.goal_x, self.goal_y)]
            Algorithm(adjacency_list, start_node, goal_node, self.nodes)    
            self.event = "Display_Path"
            iteration = 0

        elif self.event == "Display_Path":
            iteration += 1
            if iteration < counter:
				#If you need more data structures that has to be coloured add it here
                open_set = open_sets[iteration]
                closed_set = closed_sets[iteration]
                result_path = paths[iteration]
                for j in open_set:
                    node_colour[j] = 1  ##Magenta
                for j in closed_set:
                    node_colour[j] = 2  ##Blue
            else:
                print("Program Completed")

    # After every mouse press event on screen this function is called
    def mousePressEvent(self, canvas):
        x = canvas.x()
        y = canvas.y()

        min_x = 99999
        min_y = 99999
        self.findClosestCoordinate(min_x, min_y, x, y)

    # This function is called continously after some time intervals by python program
    # Understand this function for better understanding of your assignment and what is expected.
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawPoints(qp)
        qp.end()

    def drawPoints(self, qp):
        global tri
        global adjacency_list
        global node_colour
        global edge_list

        qp.setPen(Qt.red)
        if self.event == "Generate_Graph" or self.event == "Generate_TSP":
            for point in points:
                center = QPoint(point[0], point[1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center, 5, 5)
            if self.event == "Generate_Graph":
                self.drawLine(qp)

        elif self.event == "Del_Node":
            if [self.del_x, self.del_y] in points:
                node_deleted = (self.del_x, self.del_y)
                temp = []
                for edge in edge_list:
                    if node_deleted in edge:
                        continue
                    temp.append(edge)

                edge_list = temp
                points.remove([self.del_x, self.del_y])
                self.nodes -= 1
                make_dict_index_coord(points)
                make_dict_coord_index(points)
                adjacency_list = make_adj_list(edge_list, dict_coord_index, self.nodes)
                node_colour = [0 for i in range(self.nodes + 1)]

            for point in points:
                center = QPoint(point[0], point[1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center, 5, 5)

            self.drawLine(qp)
            self.delAct.setEnabled(False)
            self.startAct.setEnabled(True)
            self.goalAct.setEnabled(False)

        elif self.event == "Start_Node":
            for point in points:
                center = QPoint(point[0], point[1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center, 5, 5)
            center = QPoint(self.start_x, self.start_y)
            qp.setBrush(Qt.green)
            qp.drawEllipse(center, 10, 10)
            self.drawLine(qp)
            self.startAct.setEnabled(False)
            self.delAct.setEnabled(False)
            self.goalAct.setEnabled(True)


        elif self.event == "Goal_Node":
            for point in points:
                center = QPoint(point[0], point[1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center, 5, 5)

            center = QPoint(self.start_x, self.start_y)
            qp.setBrush(Qt.green)
            qp.drawEllipse(center, 10, 10)
            self.drawLine(qp)

            center = QPoint(self.goal_x, self.goal_y)
            qp.setBrush(Qt.red)
            qp.drawEllipse(center, 10, 10)
            self.drawLine(qp)
            self.goalAct.setEnabled(False)

        elif self.event == "Display_Path":
            for i in range(1, len(node_colour)):
                point = dict_index_coord[i]
                e = node_colour[i]
                center = QPoint(point[0], point[1])
                if e == 0:
                    qp.setBrush(Qt.yellow)
                    qp.drawEllipse(center, 5, 5)
                elif e == 1:
                    qp.setBrush(Qt.magenta)
                    qp.drawEllipse(center, 5, 5)
                elif e == 2:
                    qp.setBrush(Qt.blue)
                    qp.drawEllipse(center, 5, 5)
                elif e == 3:
                    qp.setBrush(Qt.cyan)
                    qp.drawEllipse(center, 5, 5)
                elif e == 4:
                    qp.setBrush(Qt.red)
                    qp.drawEllipse(center, 5, 5)

            center = QPoint(self.start_x, self.start_y)
            qp.setBrush(Qt.green)
            qp.drawEllipse(center, 10, 10)
            self.drawLine(qp)
            center = QPoint(self.goal_x, self.goal_y)
            qp.setBrush(Qt.red)
            qp.drawEllipse(center, 10, 10)
            self.drawLine(qp)

        self.update()

    def drawLine(self, qp):
        global result_path

        pen = QPen(Qt.black, 1, Qt.DashDotDotLine)
        qp.setPen(pen)
        for e in edge_list:
            qp.drawLine(e[0][0], e[0][1], e[1][0], e[1][1])

        if len(result_path) != 0:
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            qp.setPen(pen)
            for i in range(0, len(result_path) - 1, 1):
                prev_point = dict_index_coord[result_path[i]]
                next_point = dict_index_coord[result_path[i + 1]]
                qp.drawLine(prev_point[0], prev_point[1], next_point[0], next_point[1])

        if self.event == "Generate_Graph":
            self.delAct.setEnabled(True)
            self.startAct.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    graph = Graph()
    graph.show()
    sys.exit(app.exec_())
