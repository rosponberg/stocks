from security import SecurityData
import pandas as pd

class SecurityGroup:
    """
    SecurityGroup.__init__() constructs the SecurityGroup object
    securities          The SecurityData objects to group (*SecurityData)
    returns             The constructed SecurityGroup object (SecurityGroup)
    """
    def __init__(self, *securities):
        self._securities = list(securities)
        self._len = len(securities)

    """
    SecurityGroup.from_tickers() constructs the SecurityGroup object using tickers
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
            securities.append(s)
        return SecurityGroup(*securities)

    """
    SecurityGroup.normalize() normalizes all the security data to start at 1
    returns             The corresponding SecurityData object (SecurityData)
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
    returns             The corresponding SecurityData object (SecurityData)
    """
    def trim_data(self):
        lens = [sec.days for sec in self._securities]
        minLenSecurity = self._securities[lens.index(min(lens))]
        maxDate = minLenSecurity.df.index[0]

        securities = [sec.slice_by_date(maxDate) for sec in self._securities]
        return SecurityGroup(*securities)

    """
    SecurityGroup.overlap() trims all the data so it only includes rows with overlapping dates
    returns             The corresponding SecurityData object (SecurityData)
    """
    def overlap(self):
        overlapIndex = self._securities[0].df.index

        for i in range(1, self._len):
            sec = self._securities[i]
            overlapIndex = overlapIndex.intersection(sec.df.index)

        newSecurities = []
        for sec in self._securities:
            df = sec.df.filter(overlapIndex, axis=0)
            newSecurities.append(SecurityData(sec.ticker, df))
        return SecurityGroup(*newSecurities)

    """
    SecurityGroup.copy() copies the SecurityGroup object
    """
    def copy(self):
        return SecurityGroup(*self.securities)

    """
    SecurityGroup.portfolio() gets the backtest of a portfolio based on a certain weight vector
    weights             The weight vector (list)
    rebalanceWeeks      How many weeks to wait until rebalance (int)
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
        portfolio = SecurityData("pfolio", df)
        return portfolio