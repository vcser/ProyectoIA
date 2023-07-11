from Tetris.tetris import Tetris
from copy import deepcopy
from Tetris.global_variables import LEFT_KEY, RIGHT_KEY, DOWN_KEY, PULL_DOWN_KEY, ROTATE_KEY
from Tetris.piece import Piece
import random
#from utils_model import *
import time


#calcular betta
def clear_rows(t: Tetris):
    matrix = t.get_grid_state()
    contador=0
    num_cleared_rows=0
    for i in range(len(matrix)):
        contador=0
        for j in range(len(matrix[i])):
            if(matrix[i][j]==1):contador+=1
            else: break
            if(contador==10):num_cleared_rows+=1
        contador+=1
    return num_cleared_rows

def calculate_Tfilas():
    pass

def calculate_Tcolumnas():
    pass


#calcular gamma
def calculate_Trans_rows(t:Tetris):
    matrix = t.get_grid_state()
    contador=0
    for i in range(len(matrix)):
        for j in range(0,len(matrix[i])-1):
                if((matrix[i][j]==0 and matrix[i][j+1]!=0) or (matrix[i][j]!=0 and matrix[i][j+1]==0)):
                    contador+=1
    return contador



#calcular delta
def calculate_Trans_columns(t:Tetris):
    matrix = t.get_grid_state()
    contador=0

    for i in range(len(matrix[0])):# matrix de 0 debido a que todas filas tienen la misma cantidad de bloques
        for j in range(0,len(matrix)-1):
                if((matrix[j][i]==0 and matrix[j+1][i]!=0) or (matrix[j][i]!=0 and matrix[j+1][i]==0)):
                    contador+=1
    return contador

#calcular zeta
def calculate_num_pozos(t:Tetris):
    matrix = t.get_grid_state()
    num_pozos=0
    es_Pozo=0
    
    for i in range(1,len(matrix[0])-1):
        for j in range(len(matrix)):
            if(matrix[j][i]==0):
                if(matrix[j][i-1]!=0 and matrix[j][i+1]!=0):
                        es_Pozo+=1
            else:
                if(es_Pozo>=2):
                    num_pozos+=1
                es_Pozo=0
    return num_pozos

#calcular epsylon            
def calculate_num_hol(t:Tetris):
    matrix=t.get_grid_state()
    contador=0
    for i in range(1,len(matrix)):
        for j in range(len(matrix[i])):
            if(matrix[i][j]==0 and matrix[i-1][j]!=0):
                contador+=1                
    return contador

#calcular alfa x
def calculate_highest_peak(t:Tetris,altura):
    matrix = t.get_grid_state()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if(matrix[i][j]!=0):
                return(len(matrix)-i)
    return 0

next_piece: Piece = None

def calculate_possible_moves(t: Tetris):
    global next_piece
    piece = deepcopy(t.current_piece)
    shape = piece.shape
    moves = {}
    prevscore = t.score
    next_piece = t.next_piece

    states = {}
    
    # por cada rotacion de la pieza
    for sh in range(0, len(shape)):
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
            tetris_copy.score = prevscore
            
            piece.y = 3
            t.current_piece.y=3
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
            
            states[tetris_copy] = []
    
    print("listo primera simulacion")

    piece = next_piece
    shape =  next_piece.shape
    for state in states.keys():
        t = state
        # por cada rotacion de la pieza
        for sh in range(0, len(shape)):
            # obtener ancho de la pieza
            X = [i for (i, _) in piece.get_formatted_shape()]
            # obtener offset de la rotacion actual
            piece.rotation = sh
            # iterar desde izquierda a derecha
            for i in range(0, 10):
                piece.rotation = 0
                actions = [ROTATE_KEY] * sh
                # t.current_piece = next_piece
                # crear una copia del juego para hacer la simulacion
                tetris_copy = Tetris()
                tetris_copy.locked_pos = deepcopy(t.locked_pos)
                tetris_copy.grid = deepcopy(t.grid)
                tetris_copy.current_piece = piece
                tetris_copy.next_piece = piece
                tetris_copy.score = prevscore
                
                piece.y = 3
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
                states[t].append(tetris_copy)
    

    scores = []
    for state in states.keys():
        score = calculate_score(state)
        best_score = max([calculate_score(x) for x in states[state]])
        scores.append(score + best_score)
    

    moves[score] = actions
    # time.sleep(1)
    #print("ALTURA = ", altura)
    # print ("score es",score)

    t.score = prevscore
    print("MAX = ", max(moves.keys()))
    return moves[max(moves.keys())]

def calculate_score(tetris_copy: Tetris):
    # calcular puntaje generado por las acciones anteriores
    # score = tetris_copy.score
    # grid = tetris_copy.set_grid()
    # calcular altura
    calculate_Trans_columns(tetris_copy)
    altura = 20 - tetris_copy.current_piece.y
    altura = 0 if altura < 0 else altura
    # print("altura = ", altura)
    alfa = calculate_highest_peak(tetris_copy,altura)##falta
    beta = clear_rows(tetris_copy)
    gamma = calculate_Trans_rows(tetris_copy)
    delta = calculate_Trans_columns(tetris_copy)
    epsylon = calculate_num_hol(tetris_copy)
    zetta = calculate_num_pozos(tetris_copy)
    #print("num pozo es: ",zetta)
    filas = clear_rows(tetris_copy)
    transiciones_filas = random.randint(0, 6)
    transiciones_columnas = random.randint(0, 8)
    huecos = random.randint(0, 8)
    pozos = random.randint(0, 8)

    A = (-4.500158825082766, 3.4181268101392694, -3.2178882868487753, -9.348695305445199, -7.899265427351652, -3.3855972247263626)
    
    score = alfa*A[0] + beta*A[1] + gamma*A[2] + delta*A[3] + epsylon*A[4] + zetta*A[5]
    return score