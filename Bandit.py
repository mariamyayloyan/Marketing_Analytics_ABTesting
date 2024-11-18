"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm



class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    """
    Abstract base class for bandit algorithms.

    This class defines the blueprint for multi-armed bandit algorithms.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initializes the bandit with a given probability.

        Args:
            p (float): True probability of success for the bandit arm.
        """

        pass

    @abstractmethod
    def __repr__(self):
        """Returns a string representation of the bandit."""
        pass

    @abstractmethod
    def pull(self):
        """Simulates pulling the bandit's arm to generate a reward."""
        pass

    @abstractmethod
    def update(self):
        """
        Updates the estimated success rate based on the observed reward.

        Args:
            x (float): Observed reward from the bandit's arm.
        """
        pass

    @abstractmethod
    def experiment(self):
        """Runs the experiment for the bandit algorithm."""
        pass

    @abstractmethod
    def report(self):
        """
        Generates a report summarizing the experiment results.
        """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization:
    """
    Visualization class for plotting results of bandit experiments.
    """

    def plot1(self, N, results, algorithm='EpsilonGreedy'):

        """
        Plots the performance of the bandit algorithm.

        Args:
            N (int): Number of trials.
            results (tuple): Experiment results containing rewards and estimates.
            algorithm (str): Name of the bandit algorithm (default is 'EpsilonGreedy').
        """
        
        # Visualize the performance of each bandit: linear and log

        #Retrieving the bandits and Cumulative Average Reward
        
        cumulative_reward_average = results[0]
        bandits = results[3]
        
        ## LINEAR SCALE
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Linear Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.show()

        ## LOG SCALE
        plt.plot(cumulative_reward_average, label='Cumulative Average Reward')
        plt.plot(np.ones(N) * max([b.p for b in bandits]), label='Optimal Reward')
        plt.legend()
        plt.title(f"Win Rate Convergence for {algorithm} - Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Estimated Reward")
        plt.xscale("log")
        plt.show()

    def plot2(self, results_eg, results_ts):
        """
        Compares Epsilon-Greedy and Thompson Sampling algorithms.

        Args:
            results_eg (tuple): Results from the Epsilon-Greedy algorithm.
            results_ts (tuple): Results from the Thompson Sampling algorithm.
        """
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        
        # Retrieving Cumulative reward and regret
        cumulative_rewards_eps = results_eg[1]
        cumulative_rewards_th = results_ts[1]
        cumulative_regret_eps = results_eg[2]
        cumulative_regret_th = results_ts[2]

        ## Cumulative Reward
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        ## Cumulative Regret
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.title("Cumulative Regret Comparison Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()

        #Cumulative Reward Log
        plt.plot(cumulative_rewards_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_rewards_th, label='Thompson Sampling')
        plt.legend()
        plt.yscale("log")
        plt.title("Cumulative Reward Comparison Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.show()

        ## Cumulative Regret Log
        plt.plot(cumulative_regret_eps, label='Epsilon-Greedy')
        plt.plot(cumulative_regret_th, label='Thompson Sampling')
        plt.legend()
        plt.yscale("log")
        plt.title("Cumulative Regret Comparison Log Scale")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.show()



#--------------------------------------#

class BanditArm(Bandit):
    """
    This class is for initializing bandit arms.

    Parameters:
    p (float): The true win rate of the arm.

    Attributes:
    p (float): The true win rate of the arm.
    p_estimate (float): The estimated win rate.
    N (int): The number of pulls.

    Methods:
    pull(): Pull the arm and return the sampled reward.
    update(): Update the estimated win rate with a new reward value.
    experiment(): Run the experiment..
    report(): Generate a report with statistics about the experiment.
    """

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0 #average reward estimate
        self.N = 0 
        self.r_estimate = 0 #average regret estimate
        logger.info(f"Initialized Bandit with true win rate: {self.p}")
    
    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'
   
    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N

    def experiment(self):
        pass

    def report(self, N, results, algorithm = "Epsilon Greedy"):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results 
        else:
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward = results 
        
        # Save experiment data in a CSV file
        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_df.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save Final Results in a CSV file
        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        data_df1.to_csv(f'{algorithm}_Final.csv', index=False)

        for b in range(len(bandits)):
           logger.debug(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
           logger.debug("--------------------------------------------------")
        
        
        logger.debug(f"Cumulative Reward : {sum(reward)}")
        
        logger.debug(" ")
        
        logger.debug(f"Cumulative Regret : {cumulative_regret[-1]}")
              
        logger.debug(" ")
        
        if algorithm == 'EpsilonGreedy':                            
            logger.debug(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")

         
   
    

#--------------------------------------#
    
class EpsilonGreedy(Bandit):
    """Epsilon-Greedy multi-armed bandit algorithm.
    
    Methods:
    pull(): Pull the arm and return the sampled reward.
    update(x): Update the estimated win rate with a new reward value.
    experiment(BANDIT_REWARDS, N, t=1): Run the experiment.
    report(N, results): Generate a report with statistics about the experiment.

    """

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0 #average reward estimate
        self.N = 0 
        self.r_estimate = 0 #average regret estimate
        logger.info(f"Initialized Bandit with true win rate: {self.p}")
    
    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    def pull(self):
        reward = np.random.randn() + self.p
        #logger.debug(f"Pulled Bandit with true win rate {self.p}. Reward: {reward:.2f}")
        return reward

    def update(self, x):
        self.N += 1.
        old_estimate = self.p_estimate
        self.p_estimate = (1 - 1.0/self.N) * self.p_estimate + 1.0/ self.N * x
        self.r_estimate = self.p - self.p_estimate
        #logger.debug(f"Updated Bandit: True win rate {self.p}, Old estimate {old_estimate:.2f}, "
                 #f"New estimate {self.p_estimate:.2f}, Regret {self.r_estimate:.2f}")
    
    def experiment(self, BANDIT_REWARDS, N, t = 1):
        logger.info(f"Starting experiment with {len(BANDIT_REWARDS)} bandits and {N} trials.")
        #Initializing Bandits
        bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
        means = np.array(BANDIT_REWARDS)
        true_best = np.argmax(means)  
        count_suboptimal = 0
        EPS = 1/t
    
        reward = np.empty(N)
        chosen_bandit = np.empty(N)


        for i in range(N):
            p = np.random.random()
            
            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            x = bandits[j].pull()
            
            bandits[j].update(x)
    

            if j != true_best:
                count_suboptimal += 1
            
            reward[i] = x
            chosen_bandit[i] = j
            
            t+=1
            EPS = 1/t
            # Log progress every 1000 trials
            if (i + 1) % 1000 == 0:
                logger.debug(f"Trial {i + 1}/{N}: Chosen bandit {j}, Estimated win rate {bandits[j].p_estimate:.2f}, "
                            f"Reward {x:.2f}, Cumulative reward {np.sum(reward):.2f}")

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        
        cumulative_regret = np.empty(N)
        for i in range(len(reward)):
            cumulative_regret[i] = N*max(means) - cumulative_reward[i]

        return cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal
    

    def report(self, N, results, algorithm = "Epsilon Greedy"):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results 
        else:
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward = results 
        
        # Save experiment data in a CSV file
        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_df.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save Final Results in a CSV file
        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        data_df1.to_csv(f'{algorithm}_Final.csv', index=False)

        for b in range(len(bandits)):
            logger.info(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            logger.info("--------------------------------------------------")
        
        
        logger.info(f"Cumulative Reward : {sum(reward)}")
        

        logger.info(" ")
        
        logger.info(f"Cumulative Regret : {cumulative_regret[-1]}")
              
        logger.info(" ")
        
        if algorithm == 'EpsilonGreedy':                            
            logger.info(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")


#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    ThompsonSampling is a class for implementing the Thompson Sampling algorithm for multi-armed bandit problems.

    Methods:
    - pull(): Pull the bandit arm and return the observed reward.
    - sample(): Sample from the posterior distribution of the bandit arm's win rate.
    - update(x): Update the bandit arm's parameters and estimated win rate based on the observed reward.
    - plot(bandits, trial): Plot the probability distribution of the bandit arm's win rate after a given number of trials.
    - experiment(BANDIT_REWARDS, N): Run an experiment to estimate cumulative reward and regret for Thompson Sampling.
    """
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0 #average reward estimate
        self.N = 0 
        self.r_estimate = 0 #average regret estimate
        self.lambda_ = 1  # Initial precision of the prior
        self.tau = 1  # Precision of the reward noise
        logger.info(f"Initialized Thompson Sampling Bandit with true win rate: {self.p}")

    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    def pull(self):
        reward = np.random.randn() / np.sqrt(self.tau) + self.p
        #logger.debug(f"Pulled Bandit with true win rate {self.p}. Reward: {reward:.2f}")
        return reward
    
    def sample(self):
        sample = np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate
        #logger.debug(f"Sampled from Bandit with estimated win rate {self.p_estimate:.2f}: {sample:.2f}")
        return sample

    def update(self, x):
        old_estimate = self.p_estimate
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate
        #logger.debug(f"Updated Bandit: True win rate {self.p}, Old estimate {old_estimate:.2f}, "
                     #f"New estimate {self.p_estimate:.2f}, Regret {self.r_estimate:.2f}")
    
    def plot(self, bandits, trial):
        x = np.linspace(-3, 6, 200)
        logger.info(f"Plotting bandit distributions after {trial} trials.")
        for b in bandits:
            y = norm.pdf(x, b.p_estimate, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label=f"Real mean: {b.p:.4f}, Plays: {b.N}")
        plt.title(f"Bandit distributions after {trial} trials")
        plt.legend()
        plt.show()

    def experiment(self, BANDIT_REWARDS, N):
        logger.info(f"Starting Thompson Sampling experiment with {len(BANDIT_REWARDS)} bandits and {N} trials.")
        bandits = [ThompsonSampling(m) for m in BANDIT_REWARDS]

        sample_points = [5, 20, 50, 100, 200, 500, 1000, 1999, 5000, 10000, 19999]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)
        
        for i in range(N):
            # Choose the bandit with the highest sample value
            j = np.argmax([b.sample() for b in bandits])

            # Plot distributions at sample points
            if i in sample_points:
                self.plot(bandits, i)

            # Pull the chosen bandit's arm
            x = bandits[j].pull()
            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

            # Log progress every 1000 trials
            if (i + 1) % 1000 == 0:
                logger.debug(f"Trial {i + 1}/{N}: Chosen bandit {j}, Estimated win rate {bandits[j].p_estimate:.2f}, "
                            f"Reward {x:.2f}, Cumulative reward {np.sum(reward):.2f}")

        cumulative_reward_average = np.cumsum(reward) / (np.arange(N) + 1)
        cumulative_reward = np.cumsum(reward)
        cumulative_regret = np.empty(N)
        
        for i in range(len(reward)):
            cumulative_regret[i] = N * max([b.p for b in bandits]) - cumulative_reward[i]

        logger.info("Thompson Sampling experiment completed.")
        return cumulative_reward_average, cumulative_reward, cumulative_regret, bandits, chosen_bandit, reward


    def report(self, N, results, algorithm = "Epsilon Greedy"):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        
        if algorithm == 'EpsilonGreedy':
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward, count_suboptimal = results 
        else:
            cumulative_reward_average, cumulative_reward,  cumulative_regret, bandits, chosen_bandit, reward = results 
        
        # Save experiment data in a CSV file
        data_df = pd.DataFrame({
            'Bandit': [b for b in chosen_bandit],
            'Reward': [r for r in reward],
            'Algorithm': algorithm
        })

        data_df.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save Final Results in a CSV file
        data_df1 = pd.DataFrame({
            'Bandit': [b for b in bandits],
            'Reward': [p.p_estimate for p in bandits],
            'Algorithm': algorithm
        })


        data_df1.to_csv(f'{algorithm}_Final.csv', index=False)

        for b in range(len(bandits)):
            logger.info(f'Bandit with True Win Rate {bandits[b].p} - Pulled {bandits[b].N} times - Estimated average reward - {round(bandits[b].p_estimate, 4)} - Estimated average regret - {round(bandits[b].r_estimate, 4)}')
            logger.info("--------------------------------------------------")
        
        
        logger.info(f"Cumulative Reward : {sum(reward)}")
        
        logger.info(" ")
        
        logger.info(f"Cumulative Regret : {cumulative_regret[-1]}")
              
        logger.info(" ")
        
        if algorithm == 'EpsilonGreedy':                            
            logger.info(f"Percent suboptimal : {round((float(count_suboptimal) / N), 4)}")





def comparison(N, results_eg, results_ts):

    """
    Compare performance of Epsilon Greedy and Thompson Sampling algorithms in terms of cumulative average reward.

    Prints:
    Linear and log scale plots of cumulative average reward and optimal reward of both algorithms.
    """  
    # think of a way to compare the performances of the two algorithms VISUALLY and 

    #Retrieving the bandits and Cumulative Average Reward

    cumulative_reward_average_eg = results_eg[0]
    cumulative_reward_average_ts = results_ts[0]
    bandits_eg = results_eg[3]
    reward_eg = results_eg[5]
    reward_ts = results_ts[5]
    regret_eg = results_eg[2][-1]
    regret_ts = results_ts[2][-1]

    
    logger.info(f"Total Reward Epsilon Greedy : {sum(reward_eg)}")
    logger.info(f"Total Reward Thomspon Sampling : {sum(reward_ts)}")
        
    logger.info(" ")
        
    logger.info(f"Total Regret Epsilon Greedy : {regret_eg}")
    logger.info(f"Total Regret Thomspon Sampling : {regret_ts}")
        

    plt.figure(figsize=(12, 5))

    ## LINEAR SCALE
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Linear Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")


    ## LOG SCALE
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_reward_average_eg, label='Cumulative Average Reward Epsilon Greedy')
    plt.plot(cumulative_reward_average_ts, label='Cumulative Average Reward Thompson Sampling')
    plt.plot(np.ones(N) * max([b.p for b in bandits_eg]), label='Optimal Reward')
    plt.legend()
    plt.title(f"Comparison of Win Rate Convergence  - Log Scale")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Reward")
    plt.xscale("log")
    
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
