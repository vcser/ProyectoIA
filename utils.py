from Tetris.tetris import Tetris
from copy import deepcopy

def calculate_possible_moves(t: Tetris):
    grid = deepcopy(t.get_grid_state())
    piece = deepcopy(t.current_piece)
    shape = piece.shape
    moves = []
    
    # por cada rotacion de la pieza
    for sh in range(len(shape)):
        # obtener ancho de la pieza
        X = [i for (i, _) in piece.get_formatted_shape()]
        width = max(X) - min(X)
        # obtener offset de la rotacion actual
        piece.rotation = sh
        left = 2 - piece.get_left_bound()
        # iterar desde izquierda a derecha
        for i in range(left, 10 - width + 1):
            piece.y = 0
            piece.x = i
            while piece.in_valid_space(grid):
                piece.y += 1
            # since in valid space is violated now, so fix y coordinate by shifting one block up
            piece.y -= 1
            moves.append(piece.get_formatted_shape())
    
    return moves
