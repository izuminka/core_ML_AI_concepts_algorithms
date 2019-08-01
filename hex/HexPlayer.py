import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import time
import string
import argparse

class Hex:
    def __init__(self, dim, color) :
        self._dim = dim
        self._board = np.ones([dim,dim])
        self._myColor = color
        self._myVal = 0
        self._enemyColor = self.getEnemyColor()
        self._enemyVal = np.inf
        self._playDir = self.getPlayDir()
        self._graphMyConnected = {}
        self._graphEnemyConnected = {}
        self._result = ''
        self._time = time.time()
        self._depth = 3



    def getEnemyColor(self) :
        if self._myColor == "RED" :
            return "BLUE"
        return "RED"

    def getPlayDir(self):
        if self._myColor == "RED" :
            return "y"
        return "x"
    
    def getPlayerVal(self, player) :
        if player == "RED" :
            return 0
        return np.inf
    
    

    def getOppColor(self, player_color) :
        if player_color == "RED":
            return "BLUE"
        return "RED"
    


    def getPossibleMoves(self) :
#        pos = np.argwhere(self._board == 1)
#        np.random.shuffle(pos)
        return np.argwhere(self._board == 1)

    def makeMove(self, position, player_val):
        x,y = position
        self._board[x][y] = player_val


    def complete(self) :
        if len(np.argwhere(self._board == 1)) == 0:
            return True

    def neighbors(self):
        neighbors_vectors = [[0,1],[1,0],[1,-1],[0,-1],[-1,0],[-1,1]]
        neighbors_vectors = [np.array(n) for n in neighbors_vectors]
        return np.array(neighbors_vectors)
    
    def get_graph_connected_nodes(self, player_val):
        nn_trans_vect =  self.neighbors()
        ones = np.argwhere(self._board == player_val)
        ones_tup = [tuple(v) for v in ones]
        
        graph = {t:[] for t in ones_tup}
        for p in ones:
            for nn_v in nn_trans_vect:
                neighbor = tuple(p+nn_v)
                if neighbor in ones_tup:
                    graph[tuple(p)].append(neighbor)
        if player_val==0:
            self._graphMyConnected = graph
        elif player_val==np.inf:
            self._graphEnemyConnected = graph
            
    
    def bfs(self, start_node, player_val):
        graph = player_val
        if player_val==0:
            graph = self._graphMyConnected
        else:
            graph = self._graphEnemyConnected
        visited= [start_node]
        levels_visited = {start_node:0}  
        
        queue = [start_node]
        explored_nodes = []
       
        while queue: # reverse oredr to speed up
            node = queue.pop(0)
            explored_nodes.append(node)
            neighbours = graph[node]
            for neigh in neighbours:
                if neigh not in visited:
                    queue.append(neigh)
                    visited.append(neigh)
                    levels_visited[neigh]= levels_visited[node]+1
        return explored_nodes
    
    def is_connected(self, player_val):
        self.get_graph_connected_nodes(player_val)
        if player_val == 0:
            player_col = self._myColor
        else:
            player_col = self._enemyColor
        
        if player_col == "RED":
            starting_nodes = np.argwhere(self._board[:,0] == player_val)
            for start in starting_nodes:
                start_node = tuple([start[0],0])
                nodes_connected = self.bfs(start_node,player_val)
                for node in nodes_connected:
                    if node[1]==dim-1:
                        end_node = node
                        return start_node, end_node #'red path is connected'
            return None, None
        
        elif player_col == "BLUE":
            starting_nodes = np.argwhere(self._board[0,:] == player_val)
            for start in starting_nodes:
                start_node = tuple([0,start[0]])
                nodes_connected = self.bfs(start_node,player_val)
                for node in nodes_connected:
                    if node[0]==dim-1:
                        end_node = node
                        return start_node, end_node #'red path is connected'
            return None, None
        
    def getWinner(self):        
        start_node, finish_node = self.is_connected(self._myVal)
        if start_node != None and finish_node != None:
            self._result = 'WIN'
            return self._myVal
        else:
            self._result = 'LOSS'
            return self._enemyVal
    
    def is_over(self):        
        start_node, finish_node = self.is_connected(self._myVal)
        if start_node != None and finish_node != None:
            return 'WIN'
        
        start_node, finish_node = self.is_connected(self._enemyVal)
        if start_node != None and finish_node != None:
            return 'Loss'
        
        return 'in_progress'


    def evalFun(self):
        if self._myColor=="RED":
            red_hex = np.argwhere(self._board == 0)
            blue_hex = np.argwhere(self._board == np.inf)
        else:
            red_hex = np.argwhere(self._board == np.inf)
            blue_hex = np.argwhere(self._board == 0)
        red_hex = [tuple(v) for v in red_hex]
        blue_hex = [tuple(v) for v in blue_hex]
        
        points = 0
        for p in red_hex:
            up = tuple(p+np.array([0,1]))
            sqew_up  = tuple(p+np.array([-1,1]))
            upright = tuple(p+np.array([0,1])+np.array([1,0]))

            down = tuple(p+np.array([0,-1]))
            sqew_down  = tuple(p+np.array([1,-1]))
            downleft = tuple(p+np.array([-1,-1]))

            if up in red_hex:
                points+=5
            if down in red_hex:
                points+=5
              
            if sqew_up in red_hex:
                points+=2
            if sqew_down in red_hex:
                points+=2
                
            
            if up in red_hex and down in red_hex:
                points+=10
                
            if up in red_hex and upright in red_hex:
                points+=5
            if down in red_hex and downleft in red_hex:
                points+=5
                
            if up in blue_hex:
                points-=15
            if down in blue_hex:
                points-=15
            if sqew_up in blue_hex:
                points-=10
            if sqew_down in blue_hex:
                points-=10

        for p in blue_hex:
            right = tuple(p+np.array([1,0]))
            sqew_right = tuple(p+np.array([1,-1]))
            rightdown = tuple(p+np.array([2,0])+np.array([0,-1]))
            
            left = tuple(p+np.array([-1,0]))
            sqew_left = tuple(p+np.array([-1,1]))
            leftup = tuple(p+np.array([-2,0])+np.array([0,1]))
            
            if right in blue_hex:
                points-=5
            if left in blue_hex:
                points-=5
                
            if sqew_left in red_hex:
                points+=2
            if sqew_right in red_hex:
                points+=2
                
            if left in blue_hex and right in blue_hex:
                points-=10
                
            if right in blue_hex and rightdown in blue_hex:
                points-=5
            if left in blue_hex and leftup in blue_hex:
                points-=5
            
            if right in red_hex:
                points+=15
            if left in red_hex:
                points+=15
            if sqew_right in red_hex:
                points+=10
            if sqew_left in red_hex:
                points+=10

        return points
    

    def alphabeta(self, player_color, depth = 0, alpha=-np.inf, beta=np.inf) :
        if player_color == "RED": # I am playing
            best = -np.inf
        else:
            best = np.inf
        
        if depth == self._depth:
            return self.evalFun(),  None
        
#        if self.complete():
#            if self.getWinner() == self._myVal:
#                return 100, None
#            else:
#                return -100, None
            
        for move in self.getPossibleMoves():
            if player_color == self._myColor:
                player_val = 0
            else:
                player_val = np.inf
            self.makeMove(move, player_val)
            val, _ = self.alphabeta(self.getOppColor(player_color), depth+1,alpha,beta)
            self.makeMove(move, 1)
            if player_color == "RED":
                if val > best :
                    best, bestMove = val, move
#                    print(player_color,best,bestMove ,depth)
                    alpha = max([alpha, best])
                    if beta <= alpha:
                        break
                    if time.time() - self._time > 27:
                        break
            else :
                if val < best :
                    best, bestMove = val, move
                    print(player_color,best,bestMove ,depth)
                    beta = min([beta, best])
                    if beta <= alpha:
                        break
                    if time.time() - self._time > 27:
                        break
        return best, bestMove
    
    

    
    #plotting bs
    def transform_coords_hex(self, coords):
        r = 0.57735
        y_scale = r*(1+np.sin(np.pi/6))
        coord_transformed = [ [c[0]+0.5*c[1], c[1]*y_scale] for c in coords]
        return coord_transformed

    def plot_hex_coords(self,coords_class_trans):
        r = 0.57735
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        
        for color_class in coords_class_trans:
            coords = coords_class_trans[color_class]
            for c in coords:
                x,y = c
                hex_test = RegularPolygon((x, y), numVertices=6, radius=r,
                                          edgecolor='black', facecolor=color_class)
                ax.add_patch(hex_test)
                ax.scatter(x, y)
        plt.plot(np.arange(0,3), np.ones(3)*-0.7,color='red')
        plt.show()
    
    def get_g(self, opp=False):
        """color_coords_board"""
        color_coords = {}
        color_coords[self._myColor.lower()] = np.argwhere(self._board == 0)
        color_coords['white'] = np.argwhere(self._board == 1)
        if opp==True:
                color_coords[self._enemyColor.lower()] = np.argwhere(self._board == np.inf)
        return color_coords
    
    def plot_board(self):
        color_coords = self.get_g(opp=True)
        color_coords_transf = {}
        for c_class in color_coords:
            color_coords_transf[c_class] = self.transform_coords_hex(color_coords[c_class])
        self.plot_hex_coords(color_coords_transf)


#    def get_graph_connected_empty_nodes(self):
#        nn_trans_vect =  self.neighbors()
#        ones = np.argwhere(self._board == 1)
#        ones_tup = [tuple(v) for v in ones]
#                
#        graph = {t:[] for t in ones_tup}
#        for p in ones:
#            for nn_v in nn_trans_vect:
#                neighbor = tuple(p+nn_v)
#                if neighbor in ones_tup:
#                    graph[tuple(p)].append(neighbor)
#        return graph
#    
#    def bfs_paths_huristic(self, start_node, player_val):
#        paths_cap = 10
#        my_p_paths = []
#        opp_o_paths = []
#        
#        graph = self.get_graph_connected_empty_nodes()
#        visited= [start_node]
#        levels_visited = {start_node:0}  
#        
#        queue = [start_node]
#        explored_nodes = []
#       
#        while queue: # reverse oredr to speed up
#            node = queue.pop(0)
#            explored_nodes.append(node)
#            neighbours = graph[node]
#            for neigh in neighbours:
#                if neigh not in visited:
#                    queue.append(neigh)
#                    visited.append(neigh)
#                    levels_visited[neigh]= levels_visited[node]+1
#        return explored_nodes



def user_str_to_tuple(letter_to_x, move_str):
    x = letter_to_x[move_str[0]]
    y = int(move_str[1])
    return (x,y)

def ai_tuple_to_str(x_to_letter, move):
    letter = x_to_letter[move[0]]
    num = str(move[1])
    return letter+num



#parser = argparse.ArgumentParser()
#parser.add_argument('-p','--ai_color', type=str)
#parser.add_argument('-s','--board_size', type=str)
#args = parser.parse_args()
#
#color = args.ai_color
#dim = int(args.board_size)
dim = 7
color = "BLUE"

#INIT
letter_to_x = {string.ascii_uppercase[i]:i for i in range(dim)}
x_to_letter = {i:string.ascii_uppercase[i] for i in range(dim)}
  
game = Hex(dim, color)
open_move = 1
if color=='RED':
    while(True):
        #AI MOVE
#        t1 = time.time()
        if open_move==1:
            pos = int(dim/2)
            game.makeMove([pos,pos], game._myVal)
            open_move = 0
            game.plot_board()
        else:
            game._time = time.time()    #rescet time
            val, bestMove = game.alphabeta(game._myVal)
            game.makeMove(bestMove, game._myVal)
            
            print(ai_tuple_to_str(x_to_letter, bestMove))
    #        print(time.time() -t1)
            game.plot_board()
            game_status = game.is_over()
            if game_status != 'in_progress':
                print('AI '+game_status)
                break
    
        #USER MOVE
        move = user_str_to_tuple(letter_to_x, input())
        while(game._board[move[0]][move[1]] != 1):
            print('invalid')
            move = user_str_to_tuple(letter_to_x, input())
#        game._time = time.time()
        game.makeMove(move, game._enemyVal)
        
        game.plot_board()
        game_status = game.is_over()
        if game_status != 'in_progress':
            print('AI '+game_status)
            break
        

elif color=='BLUE':
    while(True):
        
        #USER MOVE
        move = user_str_to_tuple(letter_to_x, input())
        while(game._board[move[0]][move[1]] != 1):
            print('invalid')
            move = user_str_to_tuple(letter_to_x, input())
        game.makeMove(move, game._enemyVal)
        
        game.plot_board()
        game_status = game.is_over()
        if game_status != 'in_progress':
            print('AI '+game_status)
            break
        
        #AI MOVE
        if open_move==1:
            pos = int(dim/2)
            move = [pos,pos]
            if game._board[pos][pos]==1:
                game.makeMove([pos,pos], game._myVal)
            else:
                game.makeMove([pos-1,pos], game._myVal)
            open_move = 0
            game.plot_board()
        else:
            game._time = time.time()    #rescet time
            val, bestMove = game.alphabeta(game._myColor)
            game.makeMove(bestMove, game._myVal)
        
        print(ai_tuple_to_str(x_to_letter, bestMove))

        game.plot_board()
        t1 = time.time()
        game_status = game.is_over()
        if game_status != 'in_progress':
            print('AI '+game_status)
            break
        print(time.time() -t1)