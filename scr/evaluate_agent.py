from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ple_custom'))

import neat
from random import randint
import numpy as np
from ple import PLE
from ple.games import FlappyBird
from PyQt5.QtWidgets import QApplication, QMainWindow
from SBGames_Paper_0.score_display_window import Ui_MainWindow  # Assuming this contains your custom UI class


# === CONFIGURATION ===

NUM_SCENARIOS = 1
NUM_PIPES = 50000

# Random gaps for pipe generation in each scenario
random_gaps = [randint(0, 170) for _ in range(NUM_PIPES)]
scenario_gaps = [[randint(0, 160) for _ in range(NUM_PIPES)] for _ in range(NUM_SCENARIOS)]

# Initialize game environment
game = FlappyBird(pipe_gap_config=True, init_gap=random_gaps, pipe_count=NUM_PIPES)
env = PLE(game)

# === NEAT SETUP ===

# Load NEAT configuration file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'flappy_neat_feedforward_config.txt')
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Restore checkpoint from trained population
checkpoint = neat.Checkpointer.restore_checkpoint('neat-checkpoint-111')
genomes = [g for g in checkpoint.population.values() if g.fitness is not None]

# Select the best genome by highest fitness
best_genome = max(genomes, key=lambda g: g.fitness)
neural_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

# === UI SETUP ===

app = QApplication([])
window = QMainWindow()
score_ui = Ui_MainWindow()
score_ui.setupUi(window)
window.show()

# === GAME LOOP ===

"""
Main evaluation loop for the best NEAT agent.
Continuously runs the Flappy Bird environment and displays score updates.
"""

score = 0

while True:
    # Extract current game state
    game_state = env.game.getGameState()

    # Build input vector for the neural network
    player_y = game_state["player_y"]
    pipe_bottom_y = game_state["next_pipe_bottom_y"]
    pipe_center_y = (pipe_bottom_y - game_state["next_pipe_top_y"]) / 2

    # Network activation
    output = neural_net.activate((player_y, pipe_bottom_y, pipe_center_y))

    # Decide action based on threshold
    action = 119 if output[0] >= 0.4 else None
    reward = env.act(action)

    # Display game screen
    env.display_screen = True
    env.force_fps = False

    # Update score and UI if action received reward
    if reward > 0:
        score += 1
        score_ui.Adicionar_Score(score)
