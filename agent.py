import numpy as np
import matplotlib.pyplot as plt
from environment import StockTradingEnv
from tqdm import tqdm
import random

random.seed(42)

class QLearningAgent():
    def __init__(
                    self, 
                    n_actions: int, 
                    n_states: int,
                    df,
                    exploration_prob=0.3,
                    decay=0.001, 
                    min_exploration_prob=0.01,
                    gamma=0.9,
                    lr=0.1
    ):
        self.n_actions = n_actions
        self.n_states = n_states
        self.q_table = np.zeros((n_states, n_actions))
        self.total_rewards = list()
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.env = StockTradingEnv(df, n_actions, n_states, starting_value=5000)
        self.lr = lr
        self.gamma = gamma
        self.decay = decay

    def episode(self, current_state, episode_num):
        if episode_num == 0:
            action = self.env.sample_action(episode_num)
        else:
            if np.random.uniform(0, 1) < self.exploration_prob:
                action = self.env.sample_action(episode_num)
            else:
                action = np.argmax(self.q_table[current_state,:])
                action = self.env.verify_action(action, episode_num)
        
        next_state, reward = self.env.step(action, episode_num)

        self.q_table[current_state, action] = self.q_table[current_state, action] - (1- self.lr) * self.q_table[current_state, action] + self.lr * (reward + self.gamma * max(self.q_table[next_state, :]))

        return next_state, reward
    
    def simulate(self):
        current_state = 6
        self.env.reset()
        for e in range(self.env.num_episodes):
            # print(f"==== Episode {e} ====")
            next_state, reward = self.episode(current_state, e)
            self.exploration_prob = max(self.min_exploration_prob, np.exp(-self.decay * e))
            current_state = next_state

    def train_agent(self):
        num_epochs = 1000
        self.total_rewards = []
        for ep in tqdm(range(num_epochs)):
            self.simulate()
            total = sum(self.env.reward_historical)
            self.total_rewards.append(total)
        plt.plot(self.total_rewards)
        plt.show()

    
    def render(self):
        print(self.q_table)
        # plt.plot(self.env.reward_pct_historical)
        # plt.show()

        # plt.plot(self.env.reward_historical)
        # plt.show()

        fig, ax1 = plt.subplots()
        # Plot the first line chart on the primary y-axis
        ax1.plot(self.env.portfolio_value_historical, 'b-', label='Portfolio Value')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis
        ax2 = ax1.twinx()
        ax2.plot(self.env.historical_close, 'r-', label='Daily Closing Price')
        ax2.set_ylabel('Daily Close', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add a title and show the plot
        plt.title('Portfolio Value Performance vs Daily Closing')
        # plt.show()


if __name__ == "__main__":
    MAX_TRANSACTIONS_QTY = 3

    n_actions = (2 * MAX_TRANSACTIONS_QTY) + 1


    import pandas as pd

    df = pd.read_csv("/Users/benho/Documents/mcomp/CS5344_Project/processed_data/sti_processed.csv")
    df = df[df['Date'] > '2024-01-01']
    # df['Pct_Change'] = df['Pct_Change'].apply(lambda x: float(x[:-1]))
    # df.to_csv("/Users/benho/Documents/mcomp/CS5344_Project/processed_data/sti_processed.csv")
    agent = QLearningAgent(df=df, n_actions=n_actions, n_states=9)
    # agent.simulate()
    agent.train_agent()
    agent.render()




