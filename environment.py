import numpy as np
import random
import math

random.seed(42)

class StockTradingEnv():
    def __init__(self, df, n_actions, n_states, starting_value):
        self.df = df
        self.num_episodes = len(df)
        self.stock_value = 0
        self.num_stocks = 0
        self.n_actions = n_actions
        self.starting_value = starting_value
        self.cash_value = starting_value
        self.portfolio_value = self.cash_value + self.stock_value
        self.reward_historical = []
        self.reward_pct_historical = []
        self.pct_change_historical = []
        self.historical_close = []
        self.portfolio_value_historical = [self.portfolio_value]
        self.state_map = np.array([[0, 1, 2],[3, 4, 5], [6, 7, 8]])
        self.action_map = {
            0: "Buy 1",
            1: "Buy 2",
            2: "Buy 3",
            3: "Sell 1",
            4: "Sell 2",
            5: "Sell 3",
            6: "Hold"
        }

    def reset(self):
        self.reward_historical = []
        self.reward_pct_historical = []
        self.pct_change_historical = []
        self.historical_close = []
        self.stock_value = 0
        self.num_stocks = 0
        self.n_actions = self.n_actions
        self.cash_value = self.starting_value
        self.portfolio_value = self.cash_value + self.stock_value
        self.portfolio_value_historical = [self.portfolio_value]

    
    def verify_action(self, action, e):
        current_price = self.df.iloc[e]['Close/Last']
        # print(f"Sampled action: {action} - ", self.action_map[action])
        # print("Num stocks: ", self.num_stocks)
        # print("Stock Value: ", self.stock_value)
        # print("Cash Value: ", self.cash_value)
        # print("Portfolio Value: ", self.portfolio_value)

        if action in [0, 1, 2]:
            max_stocks_possible = math.floor(self.cash_value / current_price)
            num_stocks_to_buy = action + 1
            # print(f"Max stocks buyable: {max_stocks_possible}")
            # print(f"Number of stocks to buy: {num_stocks_to_buy}")

            if max_stocks_possible < num_stocks_to_buy:
                while max_stocks_possible < num_stocks_to_buy and action != 0:
                    num_stocks_to_buy -= 1
                    action -= 1

            # print(f"Final action decision: {action} - {self.action_map[action]}")          
        elif action in [3, 4, 5]:
            max_stocks_possible = self.num_stocks
            num_stocks_to_sell = {3: 1, 4: 2, 5: 3}[action]
            # print(f"Number of stocks sellable: {max_stocks_possible}")
            # print(f"Number of stocks to sell: {num_stocks_to_sell}")
            if max_stocks_possible < num_stocks_to_sell:
                while max_stocks_possible < num_stocks_to_sell and action != 3:
                    num_stocks_to_sell -= 1
                    action -= 1
            
            # print(f"Final action decision: {action} - {self.action_map[action]}") 

        return action

    def sample_action(self, e):
        action = np.random.choice(a=self.n_actions)
        return self.verify_action(action, e)
        
    def step(self, action, e):
        current_day = self.df.iloc[e]
        current_price = current_day['Close/Last']
        pct_change = current_day['Pct_Change']

        # new_stock_value = self.stock_value
        # new_cash_value = self.cash_value

        if action == 0: # Buy 1
            new_cash_value = self.cash_value - (current_price * 1)
            self.num_stocks += 1
            new_stock_value = self.num_stocks * current_price
        if action == 1: # Buy 2
            new_cash_value = self.cash_value - (current_price * 2)
            self.num_stocks += 2
            new_stock_value = self.num_stocks * current_price
        if action == 2: # Buy 5
            new_cash_value = self.cash_value - (current_price * 3)
            self.num_stocks += 3
            new_stock_value = self.num_stocks * current_price
        if action == 3: # Sell 1
            new_cash_value = self.cash_value + (current_price * 1)
            self.num_stocks -= 1
            new_stock_value = self.num_stocks * current_price
        if action == 4: # Sell 2
            new_cash_value = self.cash_value + (current_price * 2)
            self.num_stocks -= 2
            new_stock_value = self.num_stocks * current_price
        if action == 5: # Sell 5
            new_cash_value = self.cash_value + (current_price * 3)
            self.num_stocks -= 3
            new_stock_value = self.num_stocks * current_price
        if action == 6: # Hold
            new_cash_value = self.cash_value
            new_stock_value = self.num_stocks * current_price
        
        reward = new_stock_value + new_cash_value - self.portfolio_value
        # print("Current portfolio value: ", self.portfolio_value)
        # print("New portfolio value: ", new_cash_value + new_stock_value)
        # print("Reward: ", reward)
        reward_pct = reward / self.portfolio_value
        # print("Reward Percentage: ", reward_pct)

        self.cash_value = new_cash_value
        self.stock_value = new_stock_value

        self.portfolio_value = self.cash_value + self.stock_value
        self.portfolio_value_historical.append(self.portfolio_value)

        if reward_pct >= 0.001:
            x = 0
        elif reward_pct < 0.001 and reward_pct > -0.001:
            x = 1
        elif reward_pct <= -0.001:
            x = 2

        self.reward_historical.append(reward)
        self.reward_pct_historical.append(reward_pct)

        pct_change = pct_change / 100
        if pct_change >= 0.002:
            y = 0
        elif pct_change < 0.002 and pct_change > -0.002:
            y = 1
        elif pct_change <= -0.002:
            y = 2

        self.pct_change_historical.append(pct_change)
        self.historical_close.append(current_price)
        
        next_state = self.state_map[x, y]

        return next_state, reward
        










        