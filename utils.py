from Tetris.tetris import Tetris
from copy import deepcopy
from Tetris.global_variables import LEFT_KEY, RIGHT_KEY, DOWN_KEY, PULL_DOWN_KEY, ROTATE_KEY
import random
import time

def clear_rows(t: Tetris):
    # set initial number of cleared lines as 0
    num_cleared_rows = 0
    # check for each row
    for i in range(len(t.grid)):
        # if any block is empty skip row
        if (0, 0, 0) in t.grid[i]:
            continue

        # if row is complete then increment number of cleared lines
        num_cleared_rows += 1
    return num_cleared_rows

def calculate_Tfilas():
    pass

def calculate_Tcolumnas():
    pass


def calculate_possible_moves(t: Tetris):
    piece = deepcopy(t.current_piece)
    shape = piece.shape
    moves = {}
    prevscore = t.score
    
    # por cada rotacion de la pieza
    for sh in range(0, len(shape)):
        print("rotacion =", sh)
        # obtener ancho de la pieza
        X = [i for (i, _) in piece.get_formatted_shape()]
        width = max(X) - min(X)
        # obtener offset de la rotacion actual
        piece.rotation = sh
        left = 2 - piece.get_left_bound()
        # iterar desde izquierda a derecha
        for i in range(0, 10):
            piece.rotation = 0
            actions = [ROTATE_KEY] * sh
            # crear una copia del juego para hacer la simulacion
            tetris_copy = Tetris()
            tetris_copy.locked_pos = deepcopy(t.locked_pos)
            tetris_copy.grid = deepcopy(t.grid)
            tetris_copy.current_piece = piece
            tetris_copy.next_piece = piece
            
            piece.y = 0
            piece.x = 5

            if (not piece.in_valid_space(tetris_copy.grid)):
                continue
            # calcular movimientos necesarios (action = left || right)
            if i < 5:
                actions += [LEFT_KEY] * (5-i)
            else:
                actions += [RIGHT_KEY] * (i-5)
            # hacer pull down (action = space)
            actions += [PULL_DOWN_KEY]
            for action in actions:
                tetris_copy.play_game(action)
                # tetris_copy.current_piece = piece
                # tetris_copy.next_piece = piece

            # calcular puntaje generado por las acciones anteriores
            # score = tetris_copy.score
            # grid = tetris_copy.set_grid()
            # calcular altura
            altura = 20 - piece.y
            altura = 0 if altura < 0 else altura
            print("altura = ", altura)
            # altura = altura if altura > 0 else 0
            filas = clear_rows(tetris_copy)
            transiciones_filas = random.randint(0, 6)
            transiciones_columnas = random.randint(0, 8)
            huecos = random.randint(0, 8)
            pozos = random.randint(0, 8)

            A = (-4.500158825082766, 3.4181268101392694, -3.2178882868487753, -9.348695305445199, -7.899265427351652, -3.3855972247263626)
            score = altura*A[0] + filas*A[1] #+ transiciones_filas*A[2] + transiciones_columnas*A[3] + huecos*A[4] + pozos*A[5]

            moves[score] = actions
            # time.sleep(1)
            # print(score, moves[score], "ALTURA = ", altura)
    t.score = prevscore
    print("MAX = ", max(moves.keys()))
    return moves[max(moves.keys())]
