# import pygame module
import pygame
# import global variable for the game scope
from Tetris.global_variables import *
from dfs import DFS
from Tetris.tetris import Tetris

# initialize pygame module
pygame.init()
# set caption of game window
pygame.display.set_caption('Tetris')
# load icon for game
icon = pygame.image.load('./.images/game_logo.png')
# set icon for the game
pygame.display.set_icon(icon)


# main_function to be called
def main_game():
    # instantiate game_variable
    t = Tetris()
    dfs = DFS(t)
    # game loop
    while t.game_running and not t.game_over:

        # default action in case no key is pressed
        action = None
        moves = dfs.possible_moves(t.current_piece)
        actions = dfs.best_move(moves)

        # Keyboard motion and quit
        for event in pygame.event.get():
            # kill game, if game window is closed
            if event.type == pygame.QUIT:
                t.game_running = False
                pygame.display.quit()
                quit()

        for action in actions:
          t.play_game(action)

        # Check if user lost
        if t.check_game_over():
            t.game_running = False
            t.game_over = True

    # if game is over and not closed midway
    while t.game_over:
        # fill background
        t.window.fill(color=(100, 100, 100))
        # add and display game over text
        font = pygame.font.SysFont('comicsans', 40)
        game_over_text = font.render("GAME OVER", True, (200, 0, 0))
        game_over_len, game_over_height = game_over_text.get_rect().size
        t.window.blit(game_over_text,
                      (SCREEN_WIDTH / 2 - game_over_len / 2,
                       SCREEN_HEIGHT / 3 + game_over_height / 2))

        # add and display final score text
        final_score_text = font.render("Final Score: " + str(t.score), True, (255, 255, 255))
        final_score_len, final_score_height = final_score_text.get_rect().size
        t.window.blit(final_score_text,
                      (SCREEN_WIDTH / 2 - final_score_len / 2,
                       2 * SCREEN_HEIGHT / 3 + final_score_height / 2))

        pygame.display.update()
        # wait till user presses a key
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                t.game_over = False

    # end game and close window
    pygame.quit()


if __name__ == '__main__':
    main_game()

