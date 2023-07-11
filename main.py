# import pygame module
import pygame
#from dfs import DFS
from Tetris.tetris import Tetris
from Tetris.global_variables import *
from utils import calculate_possible_moves

# initialize pygame module
pygame.init()
# set caption of game window
pygame.display.set_caption('Tetris')
# load icon for game
icon = pygame.image.load('Tetris/.images/game_logo.png')
# set icon for the game
pygame.display.set_icon(icon)

current = None

# main_function to be called
def main_game():
    # instantiate game_variable
    t = Tetris()

    # game loop
    while t.game_running and not t.game_over:

        # default action in case no key is pressed
        action = None
        # receive list of keys that are pressed
        keys = pygame.key.get_pressed()
        # if DOWN ARROW KEY is pressed set action as DOWN
        if keys[pygame.K_DOWN]:
            action = DOWN_KEY

        """
        Note: Here, the keys.get_pressed and event.type == pygame.KEY_DOWN are different
              because the former accounts for press and hold key as multiple input strokes
              but the later registers press and hold of a key as one input stroke. 
        """

        # Keyboard motion and quit
        for event in pygame.event.get():
            # kill game, if game window is closed
            if event.type == pygame.QUIT:
                t.game_running = False
                pygame.display.quit()
                quit()

            # if key stroke registered
            if event.type == pygame.KEYDOWN:
                # kill game, if escape key is pressed
                if event.key == pygame.K_ESCAPE:
                    t.game_running = False
                    pygame.display.quit()
                    quit()
                # move current piece right for right arrow key
                elif event.key == pygame.K_RIGHT:
                    action = RIGHT_KEY
                # move current piece left for left arrow key
                elif event.key == pygame.K_LEFT:
                    action = LEFT_KEY
                # rotate piece for up arrow key
                elif event.key == pygame.K_UP:
                    action = ROTATE_KEY
                # drop down the piece if space bar is pressed
                elif event.key == pygame.K_SPACE:
                    action = PULL_DOWN_KEY
                # pause game if p is pressed
                elif event.key == pygame.K_p:
                    game_paused = True
                    # lock game in an infinite loop
                    while game_paused:
                        # until p is pressed again
                        for sub_event in pygame.event.get():
                            if sub_event.type == pygame.KEYDOWN and sub_event.key == pygame.K_p:
                                game_paused = False

        t.play_game(action)
        best = []
        # take the decided action if any
        global current
        if t.current_piece != current:
            best = calculate_possible_moves(t)
            print(best)
            current = t.current_piece
        
        for action in best:
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
