"""
  global_variables.py   :   This file contains the global variables that are used in the game.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   30th July
  Last commit note		:	Added Scoring Vector according to the scoring mentioned at -
  							https://tetris.wiki/Tetris_(BPS)#Scoring
"""

# width of game screen
SCREEN_WIDTH = 850
# height of game screen
SCREEN_HEIGHT = 700
# pixel size of each block
BLOCK_SIZE = 30

# number of rows
NUM_ROWS = 20
# number of columns
NUM_COLUMNS = 10

# width of play screen
PLAY_WIDTH = BLOCK_SIZE * NUM_COLUMNS
# height of play screen
PLAY_HEIGHT = BLOCK_SIZE * NUM_ROWS

# global constants
X = 0
Y = 1
TOP_LEFT = ((SCREEN_WIDTH - PLAY_WIDTH) // 2, (SCREEN_HEIGHT - PLAY_HEIGHT))

LEFT_KEY = 0
RIGHT_KEY = 1
ROTATE_KEY = 2
DOWN_KEY = 3
PULL_DOWN_KEY = 4

# scores corresponding to the number of lines cleared in a move
SCORING_VECTOR = [0, 40, 100, 300, 1200]