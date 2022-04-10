# encoding=utf-8
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import List
matplotlib.use("Agg")
# mycode
TEST=False
# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class PairsTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: List[int],
        buy_cost_pct: List[float],
        sell_cost_pct: List[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: List[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool =False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        # mycode
        # short_share_limit=1e4,#1万
    ):
        self.shares_per_hand=100
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax #最多持有的股票数
        self.num_stock_shares=num_stock_shares
        self.initial_amount = initial_amount # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling # 1e-4
        self.state_space = state_space
        # self.action_space = action_space
        # 每次只对一只股票给出action，另一支通过hedge ratio计算
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.tech_indicator_list = tech_indicator_list
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.data = self.df.loc[self.day, :]
        self.terminal = False #是否结束
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        '''
                # for multiple stock
                # 将所有东西转换成一个list表示的state
                state = (
                    [self.initial_amount] # 初始空余资金
                    + self.data.close.values.tolist() #data[self.day]当天股票价格
                    + self.initial_list[1:] #初始各股票持有share
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ], #当天各股票因子list，shape==tech num*data len？
                        [],
                    )
                ) # append initial stocks_share to initial state, instead of all zero 
        '''
        self.state = self._initiate_state() #

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount+np.sum(np.array(self.num_stock_shares)*np.array(self.state[1:1+self.stock_dim]))] # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory=[] # we need sometimes to preserve the state in the middle of trading process 
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

        # mycode
        self.short_share_limit=self.hmax
        self.score_rewards_memory=[]
        self.penalties_memory=[]

        # kalman
        self.delta = 0.0001
        self.Vw = self.delta / (1 - self.delta) * np.eye(2)
        self.Ve = 0.001

        self.beta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.R = np.zeros((2, 2))

        self.position_type = None  # long or short

        self.data0=self.df.pivot_table(index=['date'],columns=['tic'],values=['close']).iloc[:,0]
        self.data1=self.df.pivot_table(index=['date'],columns=['tic'],values=['close']).iloc[:,1]

    def _sell_stock(self, index, action:int):
        """
        :params action, int, sell num of shares

        :return sell_num_shares, a positive value
        """
        assert action<=0
        def _do_sell_normal():
            if self.state[index + 2*self.stock_dim + 1]!=True : # check if the stock is able to sell, for simlicity we just add it in techical index
            # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                # 有可以卖空的空间
                if self.state[index + self.stock_dim + 1] > -self.short_share_limit:
                    if TEST and self.state[index + self.stock_dim + 1]+action<0:
                        if self.day<10:
                            print("going short")
                    
                    sell_num_shares = min(abs(action),self.state[index + self.stock_dim + 1] +self.short_share_limit)
                    # 卖的钱
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    print(f"selling hits the limit {self.state[index + self.stock_dim + 1]} {action}")
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares


        def _do_sell_go_short():
            pass


        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        assert action >=0
        def _do_buy():
            if self.state[index + 2*self.stock_dim+ 1] !=True: # check if the stock is able to buy
            # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] / (self.state[index + 1]*(1 + self.buy_cost_pct[index])) # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def _calc_exposure(self,actions=None):
        """
        计算风险暴露，并在reward加入penalty
        actions in step [-0.1783969  0.7176521]
        后面*100再取整就是买入卖出的股票数量

        """
        exposure=sum(abs(np.array(self.state_memory[0][1+self.stock_dim:1+2*self.stock_dim])-np.array(self.state_memory[-1][1+self.stock_dim:1+2*self.stock_dim])))

        return exposure

    def _take_action(self,action:int):
        """
        action是针对第一个股票的操作
        """
        hedge_ratio=None
        if action<0:
            self._sell_stock(0, action)
            self._buy_stock(1, -hedge_ratio*action)
        elif action>0:
            self._buy_stock(0, action)
            self._sell_stock(1, -hedge_ratio*action)

        pass

    def step(self, actions):
        # print(f"actions in step {actions}")
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            ) # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print(f"train reward {self.score_rewards_memory}")
                print(f"penalties {self.penalties_memory}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)
            
            # mycode
            # final_exposure=self._calc_exposure()
            # print(f"final_exposure {final_exposure}")
            # self.reward-=final_exposure
            return self.state, self.reward, self.terminal, {}

        else:
            # 利用对冲比率计算另一支股票的交易量
            x = np.asarray([self.data0[self.day], 1.0]).reshape((1, 2))
            y = self.data1[self.day]

            self.R = self.P + self.Vw  # state covariance prediction
            yhat = x.dot(self.beta)  # measurement prediction

            Q = x.dot(self.R).dot(x.T) + self.Ve  # measurement variance

            e = y - yhat  # measurement prediction error

            K = self.R.dot(x.T) / Q  # Kalman gain

            self.beta += K.flatten() * e  # State update
            self.P = self.R - K * x.dot(self.R)

            sqrt_Q = np.sqrt(Q)
            
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            temp=np.array(actions).tolist() #actions变成交易的股票数
            temp.append(-self.beta[0]*actions)
            actions=temp
            actions=(np.array(actions)//self.shares_per_hand) *self.shares_per_hand
            actions[np.where(actions<0)]=actions[np.where(actions<0)]+self.shares_per_hand
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.day<5:
                print(type(actions),actions)


            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    # todo 有卖空
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) # 当天self.day收盘价
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]) #每个个股的持股数
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            #np.argsort([-70,  31,  -7, -91,   0]) => array([3, 0, 2, 4, 1])
            argsort_actions = np.argsort(actions)
            # 为什么不直接使用np.where的结果，是为了保持顺序？
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            # 必须先买后卖，因为假设每次最多购买现有balance的股票，防止先卖后买无上限。但这么做也无法阻止第一天卖无限，第二天买无限。
            # 那就限定卖空也最多卖空此时手上有的balance
            # update: 没有现金只有股票资产
            sell_num_stocks=0
            # 这里的for只会执行一次
            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                sell_num_stocks = self._sell_stock(index, actions[index]) 
                actions[index] = -sell_num_stocks
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                if sell_num_stocks !=0:
                    actions[index] = self._buy_stock(index, actions[index])
                else:
                    actions[index] = 0



            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset# 获得的利润值
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling # 或许是为了使得结果不要太大，每次的收益也就几块钱几十块钱，放缩后就是0.01级别，方便梯度下降？
            
            # mycode exposure penalty，鼓励多交易
            # self.score_rewards_memory.append(self.reward)
            # penalty=(abs(sum(actions))-sum(abs(actions))*0.1)* self.reward_scaling
            # self.penalties_memory.append(penalty)
            # self.reward -= penalty

            # final_exposure=self._calc_exposure()* self.reward_scaling
            # print(f"final_exposure {final_exposure}")
            # self.reward-=final_exposure

            # self.reward -= abs(sum(actions))-sum(abs(actions))*0.2

            self.state_memory.append(self.state) # add current state in state_recorder for each step
        # print(f"self.reward in step {self.reward}")
        # self.reward in step 0.006148119274992496
        # self.reward in step -0.0008418249999987893
        # self.reward in step 0.04269511025000829
        # self.reward in step 0.022832062799984124
        # self.reward in step -0.014518103500001598
        # self.reward in step -0.006606224324984942
        # self.reward in step -0.05840524760000408
        # self.reward in step 0.008869894000003115
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount+np.sum(np.array(self.num_stock_shares)*np.array(self.state[1:1+self.stock_dim]))]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        # mycode
        self.score_rewards_memory=[]
        self.penalties_memory=[]

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if TEST: 
            print(f"self.initial in _initiate_state() {self.initial}")
            print(f"self.previous_state {self.previous_state}")

        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                # 将所有东西转换成一个list表示的state
                state = (
                    [self.initial_amount] # 初始空余资金
                    + self.data.close.values.tolist() #data[self.day]当天股票价格
                    + self.num_stock_shares #初始各股票持有share
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ], #当天各股票因子list，macd在最初几天均为0
                        [],
                    )
                ) # append initial stocks_share to initial state, instead of all zero 
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process 
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(state_list,columns=['cash','Bitcoin_price','Gold_price','Bitcoin_num','Gold_num','Bitcoin_Disable','Gold_Disable'])
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


# Streaming output truncated to the last 5000 lines.
# actions in step [-0.1783969  0.7176521]
# actions in step [0.12283203 0.23639394]
# actions in step [-1. -1.]
# actions in step [ 0.5086391 -0.5585535]
# actions in step [ 0.65995204 -0.80262953]
# actions in step [ 0.655211 -1.      ]
# actions in step [ 0.67246556 -0.9702243 ]
# actions in step [1.         0.01292537]
# actions in step [-1.          0.09557958]
# actions in step [0.15242529 0.20743884]
# actions in step [0.58685267 1.        ]
# actions in step [-0.02940753 -1.        ]
# actions in step [1.         0.07644849]
# actions in step [ 1.        -0.5155239]
# actions in step [ 0.5126176 -0.1305682]
# actions in step [-0.24939942  0.22324024]
# actions in step [-1.          0.02099888]
# actions in step [1.         0.34709275]
# actions in step [ 1. -1.]
# actions in step [-1.  1.]
# actions in step [1.         0.21426173]
# actions in step [-0.6563884 -1.       ]