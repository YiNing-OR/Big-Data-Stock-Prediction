import numpy as np

class ActionSpace():
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def sample(self):
        return np.random.choice(a=self.n_actions)

class StockTradingEnv():
    def __init__(self, df, n_actions, n_states, starting_value):
        self.df = df
        self.num_episodes = len(df)
        self.stock_value = 0
        self.num_stocks = 0
        self.n_actions = n_actions
        self.cash_value = starting_value
        self.portfolio_value = self.cash_value + self.stock_value
        self.portfolio_value_historical = [self.portfolio_value]
        self.state_map = np.array([[0, 1, 2],[3, 4, 5], [6, 7, 8]])

    def sample_action(self, e):
        current_price = self.df.iloc[e]['Close/Last']
        if current_price > self.cash_value:
            return np.random.choice(a=[3, 6], p=[0.3, 0.7])
        elif current_price <= self.cash_value:
            return np.random.choice(a=self.n_actions)
        
    def step(self, action, e):
        current_day = self.df.iloc[e]
        current_price = current_day['Close/Last']
        pct_change = current_day['Pct_Change']

        new_stock_value = self.stock_value
        new_cash_value = self.cash_value

        if action == 0: # Buy 1
            if current_price <= self.cash_value:
                new_cash_value = self.cash_value - (current_price * 1)
                self.num_stocks += 1
                new_stock_value = self.num_stocks * current_price
        if action == 1: # Buy 2
            if current_price <= self.cash_value:
                new_cash_value = self.cash_value - (current_price * 2)
                self.num_stocks += 2
                new_stock_value = self.num_stocks * current_price
        if action == 2: # Buy 5
            if current_price <= self.cash_value:
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
        reward_pct = (reward - self.portfolio_value) / self.portfolio_value
        self.portfolio_value = new_stock_value + new_cash_value
        self.portfolio_value_historical.append(self.portfolio_value)

        if reward_pct >= 0.05:
            x = 0
        elif reward_pct < 0.05 and reward_pct > -0.05:
            x = 1
        elif reward_pct <= -0.05:
            x = 2

        pct_change = pct_change / 100
        if pct_change >= 0.05:
            y = 0
        elif pct_change < 0.05 and pct_change > -0.05:
            y = 1
        elif pct_change <= -0.05:
            y = 2
        
        next_state = self.state_map[x, y]

        return next_state, reward
        










        