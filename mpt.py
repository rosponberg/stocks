from group import SecurityGroup
from security import SecurityData

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import sqrt

class ModernPortfolioTheory(SecurityGroup):
    """
    ModernPortfolioTheory.__init__() constructs the ModernPortfolioTheory object
    securities          The SecurityData objects to group (*SecurityData)
    returns             The constructed ModernPortfolioTheory object (ModernPortfolioTheory)
    """
    def __init__(self, *securities):
        super().__init__(*securities)
        self._securities = self.overlap().normalize().securities
        
        self._security_len = self._securities[0].len
        self._years = self._securities[0].years
        self._days = self._securities[0].days

        # Compute covariance matrix
        data = np.array([sec.df.Close.values for sec in self._securities])
        data = np.transpose(data)
        index = self._securities[0].df.index
        columns = [sec.ticker for sec in self._securities]
        df = pd.DataFrame(data, columns=columns, index=index) 
        df = df.pct_change()

        self._cov = df.cov()
        self._returns = [sec.ret(annualize=True) for sec in self._securities]

    @property
    def securities(self):
        return self._securities

    @property
    def security_len(self):
        return self._security_len

    @property
    def years(self):
        return self._years

    @property
    def days(self):
        return self._days

    @property
    def cov(self):
        return self._cov

    """
    ModernPortfolioTheory.from_tickers() constructs the ModernPortfolioTheory object using tickers
    tickers             The tickers to scrape data from (*str)
    returns             The constructed SecurityGroup object
    """
    def from_tickers(*tickers):
        securities = []
        for t in tickers:
            try:
                s = SecurityData.from_file(t)
            except:
                s = SecurityData.from_scrape(t)
                s.serialize()
            securities.append(s)
        return ModernPortfolioTheory(*securities)

    """ 
    ModernPortfolioTheory.vol() returns the expected volaility based on a certain weight vector
    weights              The weight vector (list)
    returns             The expected volaility for the weight vector (float)
    """
    def vol(self, weights):
        var = self.cov.mul(weights, axis=0)
        var = var.mul(weights, axis=1)
        var = var.sum().sum()
        return sqrt(var * round(252 * self.security_len / self.days))

    """ 
    ModernPortfolioTheory.ret() returns the expected return based on a certain weight vector
    weights             The weight vector (list)
    returns             The expected return for the weight vector (float)
    """
    def ret(self, weights):
        return np.dot(weights, self._returns)

    """ 
    ModernPortfolioTheory.sharpe() returns the expected sharpe ratio based on a certain weight vector
    weights             The weight vector (list)
    rfr                 The risk-free rate (float)
    returns             The expected sharpe ratio for the weight vector (float)
    """
    def sharpe(self, weights, rfr=0):
        return (self.ret(weights) - rfr) / self.vol(weights)

    """
    ModernPortfolioTheory.optimize() optimizes a weight vector to a certain target
    Targets:
    If ret=0 and vol=None, maximizes the sharpe ratio
    If ret != 0 and vol=None, minimizes the volatility at ret
    If ret=0 and vol != None, maximizes the return at vol
    
    ret                 The return target (float)
    vol                 The volatility target (float)
    bounds              The bounds of the individiuals weights (tuple)
    returns the result of scipy.minimize
    """
    def optimize(self, ret=0, vol=None, bounds=(0,1)):
        sumToOneConstraint = {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }
        returnConstraint = {
            "type": "ineq",
            "fun": lambda w: self.ret(w) - ret
        }
        constraints = [sumToOneConstraint, returnConstraint]

        fun = lambda w: -self.ret(w) / self.vol(w)
        if vol != None:
            fun = lambda w: -self.ret(w)
            volatilityConstraint = {
                "type": "eq",
                "fun": lambda w: vol - self.vol(w) 
            }
            constraints.append(volatilityConstraint)

        guess = np.ones(self.len) / self.len
        boundsTuple = (bounds for _ in range(self.len))
        return minimize(fun, guess, bounds=boundsTuple, constraints=constraints)

    """
    ModernPortfolioTheory.iterative_optimize() iteratively optimizes to certain constraints
    sampleSize              The amount of stocks to try each iteration (int)
    iters                   The number of iterations to go through
    ret                     The return target
    vol                     The volatility target
    *ret and vol behave the same as in ModernPortfolioTheory.optimize()
    bounds                  The bounds of the individual weights
    rebalanceWeeks          The rebalance period to backtest
    verbose                 Whether to print the progress of the optimization
    epsilon                 The minimium weight that will not be chopped at each iteration
    """
    def iterative_optimize(self, sampleSize, iters, ret=0, vol=None, bounds=(0,1), rebalanceWeeks=12, verbose=True, epsilon=0.01):
        portfolio = []
        portfolio_weights = []
        max_sharpe = 0

        for _ in range(iters):
            sample = np.random.choice(self.securities, size=sampleSize).tolist()
            sample.extend(portfolio)
            sample = list(set(sample)) 

            mpt = ModernPortfolioTheory(*sample)
            res = mpt.optimize(ret=ret, vol=vol, bounds=bounds)
            weights = res.x
            mpt_sharpe = mpt.sharpe(weights)

            sample = [sample[i] for i in range(len(weights)) if weights[i] >= epsilon]
            weights = [weights[i] for i in range(len(weights)) if weights[i] >= epsilon]

            backtest = SecurityGroup(*sample).portfolio(weights, rebalanceWeeks)
            real_sharpe = backtest.sharpe()

            if real_sharpe > max_sharpe:
                max_sharpe = real_sharpe
                portfolio = sample
                portfolio_weights = weights

                # Print
                if verbose:
                    tickers = [sec.ticker for sec in portfolio]
                    print("Trial No.", _ + 1)
                    print("\t".join(tickers))
                    print("\t".join([str(round(w*100, 2)) for w in weights]))
                    print("Backtest Sharpe:", round(real_sharpe, 3))
                    print("Theoretical Sharpe:", round(mpt_sharpe, 3))
                    print()

        return portfolio, portfolio_weights

    """
    ModernPortfolioTheory.__graph() creates a scatter plot of sharpe values
    x               The volatilities (list)
    y               The returns (list)
    maxSharpe       The maximum sharpe ratio of the dataset (float)
    plt             The matplotlib.pyplot object (matplotlib.pyplot)
    """
    def __graph(x, y, maxSharpe, plt):
        plt.figure(facecolor='black')
        ax = plt.axes()
        ax.set_facecolor("black")
        ax.spines['bottom'].set_color('red')
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='x', colors='red')
        ax.tick_params(axis='y', colors='red')
        ax.yaxis.label.set_color('#FF0000')
        ax.xaxis.label.set_color('#FF0000')
        ax.title.set_color('#FF0000')

        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Return")
        plt.axline((0, 0), slope=maxSharpe, color="#00FF00")
        #plt.axline((0, 0), slope=0.5553554683802386, color="#333333") # SPY
        plt.scatter(x, y, s=5, color="red")
        plt.show()

    """
    ModernPortfolioThoery.__get_efficient_frontier() returns the efficient frontier
    points              The number of points to include (int)
    """
    def __get_efficient_frontier(self, points = 25):
        returns = [sec.ret(annualize=True) for sec in self._securities]
        maxReturn = max(returns)
        minReturn = min(returns)

        y = np.linspace(minReturn, maxReturn, points)
        x = [self.vol(self.optimize(ret=ret).x) for ret in y]
        return x, y

    """
    ModernPortfolioTheory.show_efficent_frontier() graphs the efficient frontier
    plt                 The matplotlib.pyplot object (matplotlib.pyplot)
    points              The number of points to include (int)
    """
    def show_efficent_frontier(self, plt, points=25):
        x, y = self.__get_efficient_frontier(points=points)
        maxSharpe = max(y[i] / x[i] for i in range(len(x)))
        ModernPortfolioTheory.__graph(x, y, maxSharpe, plt)

    """
    ModernPortfolioTheory.show() graphs random portfolios
    plt             The matplotlib.pyplot object (matplotlib.pyplot)
    samples         The number of points to graph (int)
    """
    def show(self, plt, samples=500):
        weights = np.random.rand(samples, self.len)
        vigorish = weights.sum(axis=1)
        weights = weights / vigorish[:,None]

        x = np.apply_along_axis(self.vol, 1, weights)
        y = np.apply_along_axis(self.ret, 1, weights)
        maxSharpe = max(y / x)

        ModernPortfolioTheory.__graph(x, y, maxSharpe, plt)