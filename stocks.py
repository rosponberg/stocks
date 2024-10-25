import numpy as np
import pandas as pd
import yfinance as yf
import os
from datetime import date
from math import floor, log10, sqrt
from scipy.optimize import minimize

PROJECT_FOLDER = os.path.join("~/", "Goatberg")
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")

class Security:
    # ============= Constructor Methods =============
    """
    This class is designed to represent the history of a certain security

    Security.__init__() constructs the Security object
    ticker          the ticker of the security (str)
    df              the dataframe of the daily historical data of the security (pd.DataFrame)

    For this object to work correctly the dataframe must have the following columns (in order)
    Date (index of dataframe), Open, High, Low, Close, and Volume
    returns         The constructed Security object (Security)
    """
    def __init__(self, ticker, df):
        self.ticker = ticker
        self._df = df
        self._priceCols = ["Open", "High", "Low", "Close"]

        start = self._df.index[0]
        end = self._df.index[-1]
        difference = end-start

        self._len = len(df.index)
        self._years = (difference.days + difference.seconds / 86400) / 365.2425
        self._days = floor(self._years * 252)
        self._periodsPerYear = round(252 * self.len / self.days)

    """
    Security.from_scrape() returns the historical data of a security scraped from yahoo finance
    ticker          the ticker of the security to scrape (str)
    dropna          whether to keep or drop rows which have empty values (boolean)
    returns         A Security object corresponding to what was scraped from yahoo finance (Security)
    """
    def from_scrape(ticker, dropna = True):
        company = yf.Ticker(ticker)
        df = company.history(period="max")

        dropColumns = df.columns.values.tolist()
        for v in ["Open", "High", "Low", "Close", "Volume"]:
            dropColumns.remove(v)

        # Dont give a DAMN
        df = df.drop(columns=dropColumns)

        df.index = df.index.date    # Remove times
        df.index.name = "Date"

        if dropna:
            df.replace(0, None)
            df.dropna(axis=1, how="any")

        return Security(ticker, df)

    """
    Security.from_file() returns the historical security data parsed from file
    ticker          the ticker of the security to scrape (str)
    data_folder     the folder which contains the data file (str)
    dropna          whether to keep or drop rows which have empty values (boolean)

    The method reads the data from the file named "TICKER.csv" where "TICKER" is the ticker parameter
    returns         A Security object corresponding to what was read from the file (Security)
    """
    def from_file(ticker, dataFolder=DATA_FOLDER, dropna = True):
        path = os.path.join(dataFolder, ticker + ".csv")
        df = pd.read_csv(path, parse_dates=True, date_format="%Y%m/%d")
        df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)

        df.index = df.index.date
        df.index.name = "Date"

        if dropna:
            df.replace(0, None)
            df.dropna(axis=1, how="any")

        return Security(ticker, df)

    # ============= Getters =============
    @property
    def df(self):
        return self._df

    @property
    def len(self):
        return self._len

    @property
    def years(self):
        return self._years

    @property
    def days(self):
        return self._days

    @property
    def periodsPerYear(self):
        return self._periodsPerYear

    # ============= Display Methods =============
    """
    Security.__repr__() returns the string for the data and the ticker
    full            Whether to add every single row in the dataframe (boolean)
    returns         None
    """
    def __repr__(self):
        string = "Historical Data for" +  self.ticker + "..."
        string += repr(self.df)
        return string
        
    """
    Security.serialize() writes the dataframe to a file named "(self.ticker).csv"
    data_folder     The folder where the file is to be located at
    returns         None
    """
    def serialize(self, data_folder=DATA_FOLDER):
        path = os.path.join(data_folder, self.ticker + ".csv")
        self.df.to_csv(path)

    """
    Security.graph() adds the security data to a matplotlib graph
    plt             The pyplot object (matplotlib.pyplot)
    col             The column to plot (str)
    normalize       Whether to normalize the data before plotting it (boolean)
    label           Whether to label the data (boolean)
    returns         None
    """
    def graph(self, plt, col="Close", normalize=False, label=True, log=True):
        series = self.df[col]

        if log: plt.yscale("log")
        if label: plt.plot(series, label = self.ticker)
        else: plt.plot(series)

    #  ============= Data Manipulation Methods =============
    """
    Security.normalize_price_data() normalizes all the price columns to start at 1
    returns a new Security object corresponding to the normalized data (Security)
    """
    def normalize(self):
        start = self.df.Open.iloc[0]
        return Security(self.ticker, self.df / start)

    """
    Security.slice() slices the Security object
    start           The starting index of the slice (int)
    end             The ending index of the slice (int)

    If end is None, the data is sliced from start until the end of the data
    returns the new Security object corresponding to the sliced data (Security)
    """
    def slice(self, start, end=None):
        if end == None:
            return Security(self.ticker, self.df.iloc[start:])
        return Security(self.ticker, self.df.iloc[start:end])

    """
    Security.slice_by_date() slices the Security object
    start           The starting date of the slice (datetime.date)
    end             The ending date of the slice (datetime.date)

    If end is None, the data is sliced from start until the end of the data
    returns the new Security object corresponding to the sliced data (Security)
    """
    def slice_loc(self, start, end=None):
        if end == None:
            return Security(self.ticker, self.df.loc[start:])
        return Security(self.ticker, self.df.loc[start:end])

    # ============= Computations =============
    """
    Security.vol() calculates the volatility of the security
    col             The column to calculate for (str)
    annualize       Whether to annualize the data (boolean)
    returns the volatility of the security data (float)
    """
    def vol(self, col="Close", annualize=False):
        price = self.df[col]
        pctChange = price.pct_change().iloc[1:]
        vol = np.std(pctChange.to_numpy(), ddof=1)
        
        n = self.periodsPerYear if annualize else self.len
        return vol*sqrt(n)

    """
    Security.ret() calculates the historical return of the security
    col             The column to calculate for (str)
    annualize       Whether to annualize the data (boolean)
    returns the return of the security data (float)
    """
    def ret(self, col="Close", annualize=False):
        price = self.df[col]
        pctChange = price.pct_change().iloc[1:]
        dailyReturn = np.mean(pctChange)

        n = self.periodsPerYear if annualize else self.len
        return (1 + dailyReturn) ** n - 1

    """
    Security.sharpe() calculates the annualized sharpe ratio of the security
    rfr             The current risk-free rate (float)
    col             The column to calculate for (str)
    returns the sharpe ratio of the security data (float)
    """
    def sharpe(self, rfr=0, col="Close"):
        return (self.ret(col, True) - rfr) / self.vol(col, True)

    """
    Security.invert() 
    returns the short of the stock (Security)
    """
    def invert(self):
        df = self.df
        principal = df.Close.iloc[0]
        df = 2*principal - df
        return Security("Short " + self.ticker, df)



class SecurityGroup:
    """
    This class is designed to represent a collection of Security objects
    
    SecurityGroup.__init__() constructs the SecurityGroup object
    securities          The Security objects to group (*Security)
    returns the constructed SecurityGroup object (SecurityGroup)
    """
    def __init__(self, *securities):
        self._securities = list(securities)
        self._len = len(securities)

    """
    SecurityGroup.from_tickers() constructs the SecurityGroup object using tickers
    tickers             The tickers to scrape data from (*str)
    returns the constructed SecurityGroup object (SecurityGroup)
    """
    def from_tickers(*tickers):
        securities = []
        for t in tickers:
            try:
                s = Security.from_file(t)
            except:
                s = Security.from_scrape(t)
            securities.append(s)
        return SecurityGroup(*securities)

    """
    SecurityGroup.normalize() normalizes all the security data to start at 1
    returns the corresponding Security object (Security)
    """
    def normalize(self):
        securities = [sec.normalize() for sec in self._securities]
        return SecurityGroup(*securities)

    # ============= Getters =============
    @property
    def securities(self):
        return self._securities

    @property
    def len(self):
        return self._len

    # ============= Interfacing Methods =============
    """
    SecurityGroup.show() displays all the security data using a matplotlib graph
    plt                 The plot (matplotlib.pyplot)
    col                 The column to display (str)
    returns             None
    """
    def show(self, plt, col="Close", log=True):
        for sec in self._securities:
            sec.graph(plt, col=col, log=log)
        plt.legend(loc="upper left")
        plt.show()

    """
    SecurityGroup.__repr__() summarizes the data for all securities in the group
    It shows the annualized return, volatility, and sharpe ratio
    returns             None
    """
    def __repr__(self):
        returns = []
        volatilty = []
        sharpe = []

        string = "\t"
        for sec in self._securities:
            string += sec.ticker + "\t"

            returns.append(str(round(sec.ret(annualize=True), 4)))
            volatilty.append(str(round(sec.vol(annualize=True), 4)))
            sharpe.append(str(round(sec.sharpe(0), 4)))

        string += "\ncagr\t" + "\t".join(returns) + "\n"
        string += "vol\t" + "\t".join(volatilty) + "\n"
        string += "sharpe\t" + "\t".join(sharpe)
        return string

    # ============= Data Manipulation Methods =============
    """
    SecurityGroup.trim() trims all the data so it starts on the same day and has the same len
    returns the corresponding Security object (Security)
    """
    def trim_data(self):
        lens = [sec.days for sec in self._securities]
        minLenSecurity = self._securities[lens.index(min(lens))]
        maxDate = minLenSecurity.df.index[0]

        securities = [sec.slice_by_date(maxDate) for sec in self._securities]
        return SecurityGroup(*securities)

    """
    SecurityGroup.overlap() trims all the data so it only includes rows with overlapping dates
    returns The corresponding Security object (Security)
    """
    def overlap(self):
        overlapIndex = self._securities[0].df.index
        for i in range(1, self._len):
            sec = self._securities[i]
            overlapIndex = overlapIndex.intersection(sec.df.index)
            
        newSecurities = []
        for sec in self._securities:
            df = sec.df.filter(overlapIndex, axis=0)
            newSecurities.append(Security(sec.ticker, df))
        return SecurityGroup(*newSecurities)

    """
    SecurityGroup.copy() copies the SecurityGroup object
    returns the copy (SecurityGroup)
    """
    def copy(self):
        return SecurityGroup(*self.securities)

    # ============= Computations =============
    """
    SecurityGroup.iterative_optimize() iteratively optimizes a portfolio to certain constraints
    sampleSize              The amount of stocks to try each iteration (int)
    iters                   The number of iterations to go through
    ret                     The return target
    vol                     The volatility target
    *ret and vol behave the same as in ModernPortfolioTheory.optimize()
    bounds                  The bounds of the individual weights
    rebalanceWeeks          The rebalance period to backtest
    verbose                 Whether to print the progress of the optimization
    epsilon                 The maximum weight that will be chopped at each iteration
    returns two lists corresponding to the securities and the weights of the optimized values (Security[], float[])
    """
    def iterative_optimize(self, sampleSize, iters, ret=0, vol=None, bounds=(0,1), rebalanceWeeks=12, verbose=True, epsilon=0.01, secs=[]):
        weights = []
        maxSharpe = 0

        for trial in range(iters):
            sampleSecs = np.random.choice(self.securities, size=sampleSize).tolist()
            sampleSecs.extend(secs)
            sampleSecs = list(set(sampleSecs)) 

            sampleMPT = ModernPortfolioTheory(*sampleSecs)
            res = sampleMPT.optimize(ret=ret, vol=vol, bounds=bounds)
            sampleWeights = res.x

            mptSharpe = sampleMPT.sharpe(sampleWeights)

            sampleSecs = [sampleSecs[i] for i in range(len(sampleWeights)) if sampleWeights[i] >= epsilon]
            sampleWeights = [sampleWeights[i] for i in range(len(sampleWeights)) if sampleWeights[i] >= epsilon]

            backtest = SecurityGroup(*sampleSecs).portfolio(sampleWeights, rebalanceWeeks)
            realSharpe = backtest.sharpe()

            # If the best portfolio was found
            if realSharpe > maxSharpe:
                maxSharpe = realSharpe
                secs = sampleSecs
                weights = sampleWeights

                # Print
                if verbose:
                    tickers = [sec.ticker for sec in secs]
                    print("Trial No.", trial + 1)
                    print("\t".join(tickers))
                    print("\t".join([str(round(w*100, 2)) for w in sampleWeights]))
                    print("Backtest Sharpe:", round(realSharpe, 3))
                    print("Theoretical Sharpe:", round(mptSharpe, 3))
                    print()

        return secs, weights

    """
    SecurityGroup.portfolio() gets the backtest of a portfolio based on a certain weight vector
    weights             The weight vector (list)
    rebalanceWeeks      How many weeks to wait until rebalance (int)
    returns the backtest (StockData object)
    """
    def portfolio(self, weights, rebalanceWeeks=0):
        c = self.copy().overlap().normalize()
        dfs = [sec.df for sec in c.securities]
        dfs = [dfs[i] for i in range(len(dfs)) if weights[i] > 0]
        weights = [w for w in weights if w > 0]

        cols = dfs[0].columns.to_list()
        index = dfs[0].index

        rebalanceDays = [index[0]]
        lastRebalance = index[0]

        if rebalanceWeeks > 0:
            for i in range(1, len(index)):
                if (index[i] - lastRebalance).days >= rebalanceWeeks * 7:
                    rebalanceDays.append(index[i])
                    lastRebalance = index[i]
        
        if rebalanceDays[-1] != index[-1]:
            rebalanceDays.append(index[-1])

        value = len(dfs)
        dfs = [dfs[i] * weights[i] / dfs[i].Open.iloc[0] for i in range(len(dfs))]
        sections = [[] for _ in dfs]
        for day in rebalanceDays:
            value = 0
            for i in range(len(dfs)):
                section = dfs[i].loc[:day].iloc[:-1]
                sections[i].append(section)
                dfs[i] = dfs[i].loc[day:]
                value += dfs[i].Open.iloc[0]
            
            for i in range(len(dfs)):
                dfs[i] /= dfs[i].Open.iloc[0]
                dfs[i] *= value
                dfs[i] *= weights[i]

        sections = [pd.concat(s) for s in sections]
        data = sum([df.to_numpy() for df in sections])
        df = pd.DataFrame(data, columns=cols, index=sections[0].index)
        portfolio = Security("pfolio", df)
        return portfolio

class ModernPortfolioTheory(SecurityGroup):
    """
    This class is made to perform portfolio optimization according to Markowitz Theory on a group of securities

    ModernPortfolioTheory.__init__() constructs the ModernPortfolioTheory object
    securities          The Security objects to group (*Security)
    returns the constructed ModernPortfolioTheory object (ModernPortfolioTheory)
    """
    def __init__(self, *securities):
        super().__init__(*securities)
        self._securities = self.overlap().normalize().securities

        
        self._securityLen = self._securities[0].len
        self._years = self._securities[0].years
        self._days = self._securities[0].days

        # Compute covariance matrix & returns vector
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
    def securityLen(self):
        return self._securityLen

    @property
    def years(self):
        return self._years

    @property
    def days(self):
        return self._days

    @property
    def cov(self):
        return self._cov

    @property
    def returns(self):
        return self._returns

    """
    ModernPortfolioTheory.from_tickers() constructs the ModernPortfolioTheory object using tickers
    tickers             The tickers to scrape data from (*str)
    returns the constructed SecurityGroup object
    """
    def from_tickers(*tickers):
        securities = []
        for t in tickers:
            try:
                s = Security.from_file(t)
            except:
                s = Security.from_scrape(t)
                s.serialize()
            securities.append(s)
        return ModernPortfolioTheory(*securities)

    """ 
    ModernPortfolioTheory.vol() returns the expected volaility based on a certain weight vector
    weights              The weight vector (list)
    returns the expected volaility for the weight vector (float)
    """
    def vol(self, weights):
        var = self.cov.mul(weights, axis=0)
        var = var.mul(weights, axis=1)
        var = var.sum().sum()
        return sqrt(var * round(252 * self.securityLen / self.days))

    """ 
    ModernPortfolioTheory.ret() returns the expected return based on a certain weight vector
    weights             The weight vector (list)
    returns the expected return for the weight vector (float)
    """
    def ret(self, weights):
        return np.dot(weights, self.returns)

    """ 
    ModernPortfolioTheory.sharpe() returns the expected sharpe ratio based on a certain weight vector
    weights             The weight vector (list)
    rfr                 The risk-free rate (float)
    returns the expected sharpe ratio for the weight vector (float)
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
    ModernPortfolioTheory.__graph() creates a scatter plot of sharpe values
    x               The volatilities (list)
    y               The returns (list)
    maxSharpe       The maximum sharpe ratio of the dataset (float)
    plt             The matplotlib.pyplot object (matplotlib.pyplot)
    returns         None
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
    returns the lists corresponding to the x and y values of the points (float[], float[])
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
    returns             None
    """
    def show_efficent_frontier(self, plt, points=25):
        x, y = self.__get_efficient_frontier(points=points)
        maxSharpe = max(y[i] / x[i] for i in range(len(x)))
        ModernPortfolioTheory.__graph(x, y, maxSharpe, plt)

    """
    ModernPortfolioTheory.show() graphs random portfolios
    plt             The matplotlib.pyplot object (matplotlib.pyplot)
    samples         The number of points to graph (int)
    returns         None
    """
    def show(self, plt, samples=500):
        weights = np.random.rand(samples, self.len)
        vigorish = weights.sum(axis=1)
        weights = weights / vigorish[:,None]

        x = np.apply_along_axis(self.vol, 1, weights)
        y = np.apply_along_axis(self.ret, 1, weights)
        maxSharpe = max(y / x)

        ModernPortfolioTheory.__graph(x, y, maxSharpe, plt)