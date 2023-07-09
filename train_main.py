import argparse
# import os for file-directory modifications and manipulations
import os
# Local ONLY (directorio para utilizar los nucleos CUDA de mi GPU)
os.add_dll_directory("C:/Archivos de programa/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Archivos de programa/NVIDIA/zlib123dllx64/dll_x64")

import shutil
from random import random, randint, sample

# import pickle to save and reload models
import pickle

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

# import the pygame module for the Game UI
import pygame

# import the Tetris class to be used as the main game variable
from Tetris.tetris import Tetris
# import global variable for the game scope
from Tetris.global_variables import ROTATE_KEY, RIGHT_KEY, LEFT_KEY, DOWN_KEY
# import utility function to choose the best solution
from utils_model import TetrisParams

from collections import deque

# initialize pygame module
pygame.init()
# set caption of game window
pygame.display.set_caption('Tetris')
# load icon for game
icon = pygame.image.load('Tetris/.images/game_logo.png')
# set icon for the game
pygame.display.set_icon(icon)

# global variable to keep track on generations
gen_index = 0
# global variable to keep track of best solution over all generations
max_fitness = 0


# driver method
def main_game(genomes, config):
    
    actions = [ROTATE_KEY, RIGHT_KEY, LEFT_KEY, DOWN_KEY]
    
    t = Tetris()
    g = 1
    # run until all tetris instances are not over
    while True: #TODO: Definir condicion de salida
        # iterate through each instance of tetris, model and genome
            
        # get list possible moves along with the respective current and future fitness
        possible_moves_result = try_possible_moves(t, model)
        # if list is not empty
        if possible_moves_result:
            # best moves correspond to 0th position because of descending sort
            best_rotation, x_position, _, _ = possible_moves_result[0]

            # while current_rotation does not match the best rotation
            while t.current_piece.rotation != best_rotation:
                # keep rotating
                t.play_game(ROTATE_KEY)

            # while min x coord does not match the best x coord keep shifting accordingly
            while x_position != min([x for x, _ in t.current_piece.get_formatted_shape()]):
                # if it's toward right
                if x_position > min([x for x, _ in t.current_piece.get_formatted_shape()]):
                    # move right
                    t.play_game(RIGHT_KEY)
                # if it's toward left
                else:
                    # move left
                    t.play_game(LEFT_KEY)

            # pull down the piece to the bottom-most possible position
            t.play_game(DOWN_KEY)
            # play one frame of game
            t.play_game(None)
        # if the possible moves list is empty, means that no possible moves left
        else:
            # exit game
            t.game_over = True

        # assign the fitness as score, implying that higher score means more fitness
        g.fitness = t.score

        # if current fitness is better than global max_fitness
        if g.fitness > max_fitness:
            # re-assign global max_fitness
            max_fitness = g.fitness

            # empty the directory with the last global high scorer model
            max_fit_model_dir = os.listdir("./max_fit_model/")
            # iterate through each file
            for file_name in max_fit_model_dir:
                # delete each file
                os.remove("./max_fit_model/" + file_name)

            # create a file for global high scorer model
            with open("max_fit_model/max_fit_model_" + str(t.score) + ".pickle", 'wb') as model_file:
                # save the model
                pickle.dump(g, model_file)

        # if game is over 
        if t.check_game_over() or t.game_over:
            # get global index from the populations
            removed_index = [genome_id for genome_id, genome in genomes if genome == g][0]
            # print stats for reference
            print("Model Killed: {}, Models Left: {}, Generation: {}, Fitness: {}".format
                    (removed_index, len(models) - 1, gen_index - 1, t.score))
            # remove the tetris instance
            tetrises.pop(tetrises.index(t))
            # remove model instance
            models.pop(models.index(m))
            # remove genome instance
            gen.pop(gen.index(g))

# Aqui se definen todos los parametros a utilizar
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    #parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    #parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    #parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=2000) # 3000
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris()  # Partida de Tetris
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = TetrisParams(env.get_grid_state()).get_state()  #! Es de la instancia en ese momento
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        next_actions, next_steps = TetrisParams.get_next_states(env)  #! Acciones y estados basados en las posibles acciones disponibles
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        #next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=True)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            TetrisParams(env).get_num_lines_cleared()
            #state = env.reset()
            env = Tetris()                                              #!
            state = TetrisParams(env.get_grid_state()).get_state()      #!
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/tetris".format(opt.saved_path))


# execute the following only if this is the calling module
if __name__ == '__main__':
    #main_game()
    #opt = get_args()
    #train(opt)
    t = Tetris()
    u = TetrisParams(t.get_grid_state())
    print(u.get_num_column_with_holes())