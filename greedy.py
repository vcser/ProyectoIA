from Tetris.tetris import Tetris
from Tetris.global_variables import ROTATE_KEY, RIGHT_KEY, LEFT_KEY, DOWN_KEY
import utils
from copy import deepcopy
from Tetris.global_variables import *
from Tetris.shapes import *

class GREEDY:
    def __init__(self, t: Tetris):
        self.T = t
    
    def greedy(self, s):    # En greedy solo importa que posicion es la mejor, nada mas
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

    def possible_moves(self, pm):
        grid_copy = deepcopy(self.T.grid)
        # por cada rotacion de la pieza
        if self.T.current_piece == SHAPES_LIST[0]:  # PIEZA S
            for i in self.T.current_piece.shape:
                # hay que evaluar cada posible posicion en el tablero
                # p.e. pieza Z tiene 8 posiciones a evaluar con rotacion normal y 9 rotada
                pass
        
        elif self.T.current_piece == SHAPES_LIST[1]:  # PIEZA Z
            pass

        elif self.T.current_piece == SHAPES_LIST[2]:  # PIEZA I
            pass

        elif self.T.current_piece == SHAPES_LIST[3]:  # PIEZA O
            pass

        elif self.T.current_piece == SHAPES_LIST[4]:  # PIEZA J
            pass

        elif self.T.current_piece == SHAPES_LIST[5]:  # PIEZA L
            pass

        elif self.T.current_piece == SHAPES_LIST[6]:  # PIEZA T
            pass



    def move_score(self, ms):
        pass
    
    def best_move(self, bm):
        pass
        # calcular puntaje por cada jugada
        
        # Criterios:
        # - Espacio utilizado
        # - Huecos creados
        # - Filas eliminadas

        # retornar jugada con mejor puntaje


