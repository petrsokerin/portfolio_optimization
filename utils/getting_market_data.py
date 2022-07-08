from abc import ABCMeta, abstractmethod
import datetime
import pandas as pd
import numpy as np


class MetaAbstractGetPreprocMarketData(ABCMeta):
    """
    A meta class used to validate types and values
    of inherited classes' attributes
    """

    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super(
            MetaAbstractGetPreprocMarketData,
            self).__call__(
            *args,
            **kwargs)

        for attr_name in obj.required_attributes:
            # validating if an object has the given named attribute
            if not hasattr(obj, attr_name):
                raise ValueError('required attribute (%s) not set' % attr_name)

            attr_val = getattr(obj, attr_name)
            # validating if a required attribute has correct type
            if not isinstance(attr_val, obj.required_attributes[attr_name]):
                raise TypeError(
                    'required attribute (%s) has incorrect type' %
                    attr_name)

        # validating if required 'start_date' and 'finish_date' have correct
        # datetime types
        for date in ['start_date', 'finish_date']:
            date_value = getattr(obj, date)
            try:
                datetime.datetime.strptime(date_value, '%Y-%m-%d')
            except ValueError:
                raise ValueError(
                    'required attribute (%s) has incorrect datetime type' %
                    date_value)

        return obj


class AbstractGetPreprocMarketData(metaclass=MetaAbstractGetPreprocMarketData):
    """
    An abstract class used to represent getting and preprocessing market data.
    Market data contains time-series data of assets' prices.

    Attributes
    ----------
    (static) required_attributes : list
      a dict where keys are attributes' names,
      values are required types
      of required attributes
    (static) dict_urls_keys : dict
      a container with keys required for dict_urls
    dict_urls : dict
      a dict where keys are identificators of URLs, values are URLs
      for grabbing market data
    tickers : list
      a list of short identificators of market assets
    start_date : str
      a string in format 'YYYY-MM-DD' for URL to get market data
      from this date
    finish_date : str
      a string in format 'YYYY-MM-DD' for URL to get market data
      until this date

    Methods
    ----------
    get_data(self) -> dict
      Returns a dict where keys are assets' tickers,
      values are time-series dataframes
      with 2 following columns:
        tradedate : datetime64[ns],
        close : float64.
    preprocess_datа(dict_market_prices: dict) -> dict
      Returns a dict where keys are assets' tickers,
      values are preprocessed time-series dataframes
    main(self) -> dict
      A class method used for getting and preprocessing market data.
      Returns a market prices' dict.
    """

    dict_urls_keys = []

    def __init__(
            self,
            dict_urls: dict,
            dict_slices: dict,
            tickers: list,
            start_date: str,
            finish_date: str):

        self.dict_urls = dict_urls
        self.dict_slices = dict_slices
        self.tickers = tickers
        self.start_date = start_date
        self.finish_date = finish_date

    @abstractmethod
    def get_data(self) -> dict:
        pass

    @staticmethod
    def preprocess_data(dict_market_prices: dict) -> dict:
        """Returns a dict where keys are assets' tickers,
           values are preprocessed time-series dataframes

        Parameters
        ----------
        dict_market_prices : dict
            a dict where keys are assets' tickers,
            values are time-series dataframes
            with 2 following columns:
              tradedate : datetime64[ns],
              close : float64.
        """
        for ticker in dict_market_prices:
            dict_market_prices[ticker] = dict_market_prices[ticker][[
                'TRADEDATE', 'CLOSE']]
            dict_market_prices[ticker]['tradedate'] = pd.to_datetime(
                dict_market_prices[ticker]['TRADEDATE'])
            dict_market_prices[ticker] = dict_market_prices[ticker].replace(
                '', '0')
            dict_market_prices[ticker]['close'] = dict_market_prices[ticker][
                'CLOSE'].astype('float64')
            dict_market_prices[ticker] = dict_market_prices[ticker].drop(
                ['CLOSE', 'TRADEDATE'], axis=1)
            dict_market_prices[ticker] = dict_market_prices[ticker].replace(
                0.0, np.nan)
            dict_market_prices[ticker] = dict_market_prices[ticker].fillna(
                method='ffill')

        return dict_market_prices

    def main(self) -> dict:
        """a class method used for getting and preprocessing market data
            Returns a market prices' dict.
        """
        dict_market_prices = self.get_data()
        dict_market_prices = self.preprocess_data(dict_market_prices)

        return dict_market_prices
