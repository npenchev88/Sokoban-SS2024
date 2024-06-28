import gym
import neat
import numpy as np
import gym_sokoban
import pickle
import modal

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install("gym", "neat-python", "numpy", "gym_sokoban")

# Define the Modal app with mounts for the script and configuration file
app = modal.App(
    "NEAT model for Sokoban",
    image=image,
    mounts=[
        modal.Mount.from_local_file("modal_neat.py"),
        modal.Mount.from_local_file("config-feedforward-4", remote_path="/root/config-feedforward-4"),
    ],
)

file_name = 'winner_test_modal.pkl'

def process_observation(environment):
    arr_walls, arr_goals, arr_boxes, arr_player = environment.render(mode='raw')
    combined = np.ones_like(arr_walls)
    combined[arr_walls == 0] = 0
    combined[arr_goals == 1] = 3
    combined[arr_boxes == 1] = 2
    combined[(arr_boxes == 1) & (arr_goals == 1)] = 4
    combined[arr_player == 1] = 5
    return combined.flatten()

def fitness(genomes, config):
    env = gym.make('Sokoban-small-v1')
    for genome_id, genome in genomes:
        observation = env.reset()
        observation = process_observation(env)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        done = False

        while not done:
            nn_actions = net.activate(observation)
            action = np.argmax(nn_actions)
            observation, reward, done, info = env.step(action)
            observation = process_observation(env)
            fitness += reward

        genome.fitness = fitness
    env.close()

# Load configuration
config_path = './config-feedforward-4'
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

@app.function(gpu="any")
def run():
    winner = p.run(fitness, 5)
    with open(file_name, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Result saved to {file_name}")

@app.local_entrypoint()
def main():
    run.remote()
