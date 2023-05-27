from Tetris.tetris import Tetris
from Tetris.global_variables import ROTATE_KEY, RIGHT_KEY, LEFT_KEY, DOWN_KEY
import utils
from copy import deepcopy

class DFS:
    def __init__(self, t: Tetris):
        self.T = t
    
    def dfs(self, s):
        visited = [False for i in range(self.V)]
        stack = []
        stack.append(s)
 
        while (len(stack)):
            s = stack[-1]
            stack.pop()
 
            if (not visited[s]):
                print(s,end=' ')
                visited[s] = True
 
            for node in self.adj[s]:
                if (not visited[node]):
                    stack.append(node)

    def possible_moves(self, p):
        grid_copy = deepcopy(self.T.grid)
        # por cada rotacion de la piza
        for i in self.T.current_piece.shape:
            pass

    
    def best_move(serlf, m):
        pass
        # calcular puntaje por cada jugada

        # ordenarlos de menor a mayor

        # retornar jugada con mejor puntaje


