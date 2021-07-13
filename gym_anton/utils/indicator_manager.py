import talib.abstract as ta


def bollingerBands(close_prices, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0):
    upperband, middleband, lowerband = ta.BBANDS(close_prices, timeperiod, nbdevup, nbdevdn, matype)
    return (upperband, middleband, lowerband)


def macd(close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macdsignal, macdhist = ta.MACD(close_prices, fastperiod, slowperiod, signalperiod)
    return (macd, macdsignal, macdhist)


def rsi(close_prices, timeperiod=14):
    real = ta.RSI(close_prices, timeperiod)
    return real


def adx(high_prices, low_prices, close_prices, timeperiod=14):
    real = ta.ADX(high_prices, low_prices, close_prices, timeperiod)
    return real


def plus_di(high_prices, low_prices, close_prices, timeperiod=14):
    real = ta.PLUS_DI(high_prices, low_prices, close_prices, timeperiod)
    return real


def minus_di(high_prices, low_prices, close_prices, timeperiod=14):
    real = ta.MINUS_DI(high_prices, low_prices, close_prices, timeperiod)
    return real


def ma(close_prices, timeperiod=30, matype=0):
    real = ta.MA(close_prices, timeperiod, matype)
    return real
