import os

import neat
import visualize
import pickle
from tqdm import tqdm

from hog import sus_update, average_win_rate, final_strategy, run_experiments

NUM_ROLLS = 6
GOAL = 100

def updated_final_strategy(score, opponent_score, num_rolls=NUM_ROLLS, goal=GOAL):
    return final_strategy(score, opponent_score, num_rolls, goal)

def eval_genomes(genomes, config):
    for genome_id, genome in tqdm(genomes):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        def genome_strategy(score, opponent_score, num_rolls=NUM_ROLLS, goal=GOAL):
            boar_and_sus_score = sus_update(0, score, opponent_score) - score
            inputs = (score, 
                      opponent_score, 
                      goal - score, 
                      goal - opponent_score, 
                      boar_and_sus_score)
            output = net.activate(inputs)
            return max([i for i in range(len(output))], key=lambda i: output[i])
        genome.fitness = average_win_rate(genome_strategy, baseline=updated_final_strategy)
        
def run(config_file):
    # load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    print(f"Average win rate of the best genome is: {winner.fitness}")
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    
    p = neat.Checkpointer.restore_checkpoint('hog-checkpoints/neat-checkpoint-4')
    p.run(eval_genomes, 10)

def main(*args):
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Hog Strategy")
    parser.add_argument('--run_experiments', '-r', action='store_true',
                        help='Runs strategy experiments')
    parser.add_argument('--create_strategy', '-c', action='store_true',
                        help='Run 300 generations to create a new strategy')
    parser.add_argument('--load_strategy', '-l', action='store_true',
                        help='Load previously created strategy')
    parser.add_argument('--save_strategy', '-s', action='store_true',
                        help='Save created strategy')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize neural network')
    
    args = parser.parse_args()
    
    
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'hog_config')
    config = neat.Config(neat.DefaultGenome, 
                        neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, 
                        neat.DefaultStagnation, 
                        config_path)
    node_names = {-1: "Player Score",
                -2: "Opponent Score",
                -3: "Player Distance to Goal",
                -4: "Opponent Distance to Goal",
                -5: "Boar + Sus Score",
                0: "0 Rolls", 1: "1 Rolls", 2: "2 Rolls", 3: "3 Rolls",
                4: "4 Rolls", 5: "5 Rolls", 6: "6 Rolls"}
    
    if args.create_strategy:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'hog_config')
        run(config_path)
        
    elif args.save_strategy:
        p = neat.Checkpointer.restore_checkpoint('hog-checkpoints/neat-checkpoint-299')
        winner = p.run(eval_genomes, 1)
        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        print(f"Average win rate of the best genome is: {winner.fitness}")
        
        with open("winning_genome.pkl", "wb") as file:
            pickle.dump(winner, file)
        
    elif args.load_strategy:
        with open("winning_genome.pkl", "rb") as file:
            winner = pickle.load(file) 
        
        def winning_genome_strategy(score, opponent_score, num_rolls=NUM_ROLLS, goal=GOAL):
            net = neat.nn.FeedForwardNetwork.create(winner, config)
            boar_and_sus_score = sus_update(0, score, opponent_score) - score
            inputs = (score, 
                      opponent_score, 
                      goal - score, 
                      goal - opponent_score, 
                      boar_and_sus_score)
            output = net.activate(inputs)
            return max([i for i in range(len(output))], key=lambda i: output[i])
        # Run experiments
        if args.run_experiments:  
            run_experiments(baseline=winning_genome_strategy)    
               
    if args.visualize:
        visualize.draw_net(config, winner, True, node_names=node_names)    

        
            


if __name__ == "__main__":
    main()