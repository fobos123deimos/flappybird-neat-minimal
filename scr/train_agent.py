from __future__ import print_function
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ple_custom'))

import neat
import matplotlib.pyplot as plt
from numpy import *
from ple import PLE
from ple.games import FlappyBird
import SBGames_Paper_0.neat_visualizations as neat_visualizations


# === GLOBAL METRICS FOR PLOTTING ===
avg_fitness_per_gen = []
best_fitness_per_gen = []
avg_score_per_gen = []
best_score_per_gen = []

# === SCENARIO SETTINGS ===
NUM_SCENARIOS = 3
NUM_PIPES = 3

# === WEIGHTS ===
score = 0.0
distance = 0.0
y_factor = 0.0

WEIGHT_DISTANCE = 1.0
WEIGHT_SCORE = 1.0
WEIGHT_Y_FACTOR = 0.08

WEIGHT_SCENARIO_1 = 1.0
WEIGHT_SCENARIO_2 = 2.0
WEIGHT_SCENARIO_3 = 6.0

# === ENVIRONMENT SETUP ===
initial_gaps = [25 for _ in range(NUM_PIPES)]
game = FlappyBird(pipe_gap_config=True, init_gap=initial_gaps, pipe_count=NUM_PIPES)
env = PLE(game)

scenario_gaps = [
    [25 if i % 2 == 0 else 25 + 80 * j for i in range(NUM_PIPES)]
    for j in range(NUM_SCENARIOS)
]

print(game.height)


def eval_genomes(genomes, config):
    """
    Evaluates each genome by running it through all defined scenarios
    and assigning a fitness score based on pipe traversal and proximity.
    """
    networks = []
    scenario_fitness_scores = []
    scenario_raw_scores = []

    for genome_id, genome in genomes:
        networks.append(neat.nn.FeedForwardNetwork.create(genome, config))

    for net, (genome_id, genome) in zip(networks, genomes):
        scenario_scores = []
        raw_scores = []
        env.init()

        for i in range(NUM_SCENARIOS):
            score = 0.0
            distance = 0.0
            y_factor = 0.0

            while True:
                state = env.game.getGameState()
                inp_y = state["player_y"]
                inp_pipe_bottom = state["next_pipe_bottom_y"]
                inp_pipe_center = (state["next_pipe_bottom_y"] - state["next_pipe_top_y"]) / 2

                output = net.activate((inp_y, inp_pipe_bottom, inp_pipe_center))
                action = 119 if output[0] >= 0.4 else None
                result = env.act(action)

                env.display_screen = True
                env.force_fps = True

                if result > 0:
                    score += 1
                    if score == NUM_PIPES:
                        if i < NUM_SCENARIOS:
                            env.reset_game(scenario_gaps[i], NUM_PIPES)
                            break
                        break

                distance += 1.0

                if env.game_over():
                    post_state = env.game.getGameState()
                    if i < NUM_SCENARIOS:
                        y_factor = abs(
                            post_state["player_y"]
                            - (post_state["next_pipe_top_y"]
                               + (post_state["next_pipe_bottom_y"]
                                  - post_state["next_pipe_top_y"]) / 2)
                        )
                        env.reset_game(scenario_gaps[i], NUM_PIPES)
                        break
                    break

            normalized_score = (
                WEIGHT_DISTANCE * (distance / 195)
                + WEIGHT_SCORE * (score / 3)
                - WEIGHT_Y_FACTOR * (y_factor / game.height)
            )
            scenario_scores.append(normalized_score)
            raw_scores.append(score)

        weighted_fitness = round(
            (
                scenario_scores[0] * WEIGHT_SCENARIO_1
                + scenario_scores[1] * WEIGHT_SCENARIO_2
                + scenario_scores[2] * WEIGHT_SCENARIO_3
            ) / (WEIGHT_SCENARIO_1 + WEIGHT_SCENARIO_2 + WEIGHT_SCENARIO_3),
            4
        )

        genome.fitness = weighted_fitness
        scenario_fitness_scores.append(weighted_fitness)
        scenario_raw_scores.append(round(sum(raw_scores) / len(raw_scores), 4))

    avg_fitness_per_gen.append(round(sum(scenario_fitness_scores) / len(scenario_fitness_scores), 4))
    best_fitness_per_gen.append(max(scenario_fitness_scores))
    avg_score_per_gen.append(round(sum(scenario_raw_scores) / len(scenario_raw_scores), 4))
    best_score_per_gen.append(max(scenario_raw_scores))

    print("Best fitness:", max(scenario_fitness_scores))
    print("Avg fitness:", avg_fitness_per_gen[-1])
    print("Best score:", max(scenario_raw_scores))
    print("Avg score:", avg_score_per_gen[-1])


def run(config_file):
    """
    Main NEAT run routine, initializes configuration and population,
    runs the evolutionary loop, and generates visualizations.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))

    winner = population.run(eval_genomes, 100)

    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-3: 'Player_Y', -2: 'Pipe_Bottom_Y', -1: 'Gap_Center_Y', 0: 'Jump_Prob'}
    neat_visualizations.draw_net(config, winner, view=True, node_names=node_names)
    neat_visualizations.plot_species(stats, view=True)

    # Fitness Plot
    plt.plot(range(1, len(avg_fitness_per_gen) + 1), avg_fitness_per_gen, linewidth=2, color='blue', label='Average Fitness')
    plt.plot(range(1, len(best_fitness_per_gen) + 1), best_fitness_per_gen, linewidth=2, color='red', label='Best Fitness')
    plt.title("Fitness over Generations", fontsize=18)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("Fitness Score", fontsize=12)
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('Fitness.png')
    plt.show()

    # Score Plot
    plt.plot(range(1, len(avg_score_per_gen) + 1), avg_score_per_gen, linewidth=2, color='blue', label='Average Score')
    plt.plot(range(1, len(best_score_per_gen) + 1), best_score_per_gen, linewidth=2, color='red', label='Best Score')
    plt.title("Score over Generations", fontsize=18)
    plt.xlabel("Generations", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig('Score.png')
    plt.show()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'flappy_neat_feedforward_config.txt')
    run(config_path)
