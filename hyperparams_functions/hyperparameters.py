import wandb
import random
import datetime
import numpy as np


###### ALL THE FUNCTIONS USED FOR HPYERPARAMETER TUNING ######

def run_agent_wandb(DuelingDQN, state_shape, action_shape):
    # Trial run to check if the algorithm runs and saves the data

    for i in range(20):
        # 🐝 initialise a wandb run
        wandb.init(
            project="DuelingDQN_Max_CartPole_195",
            config={
                "BATCH_SIZE": random.randint(32, 160),
                "LR": random.uniform(5*10**-4, 10**-3),
                "UPDATE_EVERY": random.randint(4,40),
                })
        
        # Copy your config 
        config = wandb.config
        begin_time = datetime.datetime.now()

        # Assuming env, TutorialAgent, and other required variables are defined
        agent = Agent(state_size=state_shape, action_size=action_shape, seed=0, DuelingDQN = DuelingDQN, LR = config.LR, BATCH_SIZE = config.BATCH_SIZE, UPDATE_EVERY = config.UPDATE_EVERY)
        scores_array = dqn(agent, wandb_flag = True)

        time_taken = datetime.datetime.now() - begin_time
        print(time_taken)

        wandb.finish()

def wandb_sweep():
    wandb.login()
    sweep_config = {
        'method': 'bayes'
        }
    metric = {
        'name': 'Loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric
    parameters_dict = ({
        'LR': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 2.5e-5,
            'max': 5e-4
        },
        'BATCH_SIZE': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 120,
            'max': 136,
        },
        'UPDATE_EVERY': {
            # 
            'distribution': 'q_log_uniform_values',
            'min': 4,
            'max': 16
        },
        })
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="MaxCartPoleDuelingDQN195")

    def train(config=sweep_config):
        # Initialize a new wandb run
        with wandb.init(config=config):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            agent = Agent(state_size=state_shape, action_size=action_shape, seed=0, DuelingDQN = MaxDuelingDQN, LR = config.LR, BATCH_SIZE = config.BATCH_SIZE, UPDATE_EVERY = config.UPDATE_EVERY)
            scores_array = dqn(agent, wandb_flag = True)
            # wandb.log({"loss": env., "epoch": epoch})           
    wandb.agent(sweep_id, train, count=30)
    # 5rk4j0g8 MaxCartpoleDuelingDQN

def genetic_dqn(DuelingDQN, state_shape, action_shape, hyperparameters, name,p=5, c=0.8, m=0.1, e=500, iterations=10):
    # GENE POPULATION
    p = p
    # RATE OF CROSSOVER
    c = c
    # RATE OF MUTATION
    m = m
    # EVALUATION EPISODES
    e = e

    # HYPERPARAMETERS
    hyperparameters_array = {}
    for i in range(p):
        for j in range(p):
            hyperparameters_array[i] = {}
            hyperparameters_array[i]["BATCH_SIZE"] = 130
            hyperparameters_array[i]["LR"] = 0.001
            hyperparameters_array[i]["UPDATE_EVERY"] = 4
            hyperparameters_array[i]["hidden_size"] = 128
    hyperparameters
    for _ in range(iterations):
        print(hyperparameters_array)
        for episode in range(p):
            wandb.init(
            project=name,
            config={
                "BATCH_SIZE": hyperparameters_array[episode]['BATCH_SIZE'],
                "LR": hyperparameters_array[episode]['LR'],
                "UPDATE_EVERY": hyperparameters_array[episode]['UPDATE_EVERY'],
                "hidden_size": hyperparameters_array[episode]["hidden_size"]
                })
            config = wandb.config
            begin_time = datetime.datetime.now()

            # Assuming env, TutorialAgent, and other required variables are defined
            average_scores = []
            for i in range(5):
                agent = Agent(state_size=state_shape, action_size=action_shape, seed=np.random.randint(1,100), DuelingDQN = DuelingDQN, LR = config.LR, BATCH_SIZE = config.BATCH_SIZE, UPDATE_EVERY = config.UPDATE_EVERY, hidden_size = config.hidden_size)
                scores_array = dqn(agent, wandb_flag = True)
                average_scores.append(scores_array)
            average_scores = np.mean(average_scores, axis=0)
            for i in average_scores:
                wandb.log({"Average Score": i})
            regret = env.spec.reward_threshold*len(average_scores) - np.sum(average_scores)
            wandb.log({"Regret": regret})
            time_taken = datetime.datetime.now() - begin_time
            hyperparameters_array[episode]['fitness'] = scores_array[-1]
            wandb.finish()
        # ROULETTE WHEEL SELECTION
        fitness_values = [hyperparameters_array[i]['fitness'] for i in range(p)]
        softmax_values = np.exp(fitness_values) / np.sum(np.exp(fitness_values))
        selection_probabilities = softmax_values / np.sum(softmax_values)

        # SELECT PARENTS
        parents = np.random.choice(range(p), size=2, replace=False, p=selection_probabilities)

        # PERFORM CROSSOVER
        offspring = {}
        for hyperparameter in hyperparameters:
            if random.random() < c:
                offspring[hyperparameter] = hyperparameters_array[parents[0]][hyperparameter]
            else:
                offspring[hyperparameter] = hyperparameters_array[parents[1]][hyperparameter]

        # PERFORM MUTATION
        if random.random() < m:
            offspring["BATCH_SIZE"] = 65
        if random.random() < m:
            offspring["LR"] = 0.0001328080938018766
        if random.random() < m:
            offspring["UPDATE_EVERY"] = 24
        if random.random() < m:
            offspring["hidden_size"] = 128

        # Update hyperparameters_array for the next iteration
        hyperparameters_array[np.argmin(selection_probabilities)] = offspring
    
    return hyperparameters_array