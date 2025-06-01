from __future__ import print_function
import os
import sys

# Add custom PLE path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ple_custom'))

import neat
from random import randint
import numpy as np
from ple import PLE
from ple.games import FlappyBird
from PyQt5.QtWidgets import QApplication, QMainWindow
from score_display_window import Ui_MainWindow  # Custom PyQt5 UI class for score display

# === CONFIGURATION ===

# Number of different pipe-gap scenarios to be tested
NUM_SCENARIOS = 1

# Total number of pipes to be generated in the environment
NUM_PIPES = 50000

# Randomly generated pipe gaps to simulate a variety of scenarios
random_gaps = [randint(0, 170) for _ in range(NUM_PIPES)]
scenario_gaps = [[randint(0, 160) for _ in range(NUM_PIPES)] for _ in range(NUM_SCENARIOS)]

# === GAME ENVIRONMENT SETUP ===

# Initialize Flappy Bird game environment with custom pipe gap configuration
game = FlappyBird(pipe_gap_config=True, init_gap=random_gaps, pipe_count=NUM_PIPES)
env = PLE(game)

# === NEAT SETUP ===

# Path to NEAT configuration file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'flappy_neat_feedforward_config.txt')

# Load NEAT configuration for evolutionary algorithm
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Load a previously saved checkpoint from a trained NEAT population
checkpoint = neat.Checkpointer.restore_checkpoint('neat-checkpoint-111')

# Filter genomes that already have an evaluated fitness score
genomes = [g for g in checkpoint.population.values() if g.fitness is not None]

# Select the best genome based on maximum fitness
best_genome = max(genomes, key=lambda g: g.fitness)

# Create a feedforward neural network using the best genome
neural_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

# === UI SETUP ===

# Initialize PyQt5 application and setup main score display window
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
    # Extract current game state features
    game_state = env.game.getGameState()

    # Prepare input vector for the neural network
    player_y = game_state["player_y"]
    pipe_bottom_y = game_state["next_pipe_bottom_y"]
    pipe_center_y = (pipe_bottom_y - game_state["next_pipe_top_y"]) / 2

    # Get network output (single neuron output)
    output = neural_net.activate((player_y, pipe_bottom_y, pipe_center_y))

    # Choose whether to flap based on thresholded output
    action = 119 if output[0] >= 0.4 else None

    # Apply the selected action and receive the reward
    reward = env.act(action)

    # Enable screen rendering and disable FPS limit
    env.display_screen = True
    env.force_fps = False

    # If the action resulted in positive reward, increment score and update UI
    if reward > 0:
        score += 1
        score_ui.Adicionar_Score(score)
