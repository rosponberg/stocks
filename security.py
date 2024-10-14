import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import date
from math import floor, log10, sqrt

PROJECT_FOLDER = os.path.join("~/", "Goatberg")
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")

class SecurityData:
    # ============= Constructor Methods =============
    """
    SecurityData.__init__() constructs the SecurityData object
    ticker          the ticker of the security (str)
    df              the dataframe of the daily historical data of the security (pd.DataFrame)

    For this object to work correctly the dataframe must have the following columns (in order)
    Date (index of dataframe), Open, High, Low, Close, and Volume
    returns         The constructed SecurityData object (SecurityData)
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
    SecurityData.from_scrape() returns the historical data of a security scraped from yahoo finance
    ticker          the ticker of the security to scrape (str)
    dropna          whether to keep or drop rows which have empty values (boolean)
    returns         A SecurityData object corresponding to what was scraped from yahoo finance (SecurityData)
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

        return SecurityData(ticker, df)

    """
    SecurityData.from_file() returns the historical security data parsed from file
    ticker          the ticker of the security to scrape (str)
    data_folder     the folder which contains the data file (str)
    dropna          whether to keep or drop rows which have empty values (boolean)

    The method reads the data from the file named "TICKER.csv" where "TICKER" is the ticker parameter
    returns         A SecurityData object corresponding to what was read from the file (SecurityData)
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

        return SecurityData(ticker, df)

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
    SecurityData.__repr__() returns the string for the data and the ticker
    full            Whether to add every single row in the dataframe (boolean)
    returns         None
    """
    def __repr__(self):
        string = "Historical Data for" +  self.ticker + "..."
        string += repr(self.df)
        return string
        
    """
    SecurityData.serialize() writes the dataframe to a file named "(self.ticker).csv"
    data_folder     The folder where the file is to be located at
    returns         None
    """
    def serialize(self, data_folder=DATA_FOLDER):
        path = os.path.join(data_folder, self.ticker + ".csv")
        self.df.to_csv(path)

    """
    SecurityData.graph() adds the security data to a matplotlib graph
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
    SecurityData.normalize_price_data() normalizes all the price columns to start at 1
    returns         A new SecurityData object corresponding to the normalized data (SecurityData)
    """
    def normalize(self):
        start = self.df.Open.iloc[0]
        return SecurityData(self.ticker, self.df / start)

    """
    SecurityData.slice() slices the SecurityData object
    start           The starting index of the slice (int)
    end             The ending index of the slice (int)

    If end is None, the data is sliced from start until the end of the data
    returns         The new SecurityData object corresponding to the sliced data (SecurityData)
    """
    def slice(self, start, end=None):
        if end == None:
            return SecurityData(self.ticker, self.df.iloc[start:])
        return SecurityData(self.ticker, self.df.iloc[start:end])

    """
    SecurityData.slice_by_date() slices the SecurityData object
    start           The starting date of the slice (datetime.date)
    end             The ending date of the slice (datetime.date)

    If end is None, the data is sliced from start until the end of the data
    returns         The new SecurityData object corresponding to the sliced data (SecurityData)
    """
    def slice_loc(self, start, end=None):
        if end == None:
            return SecurityData(self.ticker, self.df.loc[start:])
        return SecurityData(self.ticker, self.df.loc[start:end])

    # ============= Metrics =============
    """
    SecurityData.vol() calculates the volatility of the security
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
    SecurityData.ret() calculates the historical return of the security
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
    SecurityData.sharpe() calculates the annualized sharpe ratio of the security
    rfr             The current risk-free rate (float)
    col             The column to calculate for (str)
    returns the sharpe ratio of the security data (float)
    """
    def sharpe(self, rfr=0, col="Close"):
        return (self.ret(col, True) - rfr) / self.vol(col, True)

    def invert(self):
        df = self.df
        principal = df.Close.iloc[0]
        df = 2*principal - df
        return SecurityData("Short " + self.ticker, df)


