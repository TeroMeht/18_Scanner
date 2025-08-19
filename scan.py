from ib_insync import *
util.startLoop()
import pandas as pd
from AdjustTimezone import adjust_timezone_IB_data

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

sub = ScannerSubscription(numberOfRows=50, instrument='STK',locationCode='STK.US.MAJOR',
                          scanCode="HOT_BY_VOLUME",marketCapAbove=1000,abovePrice=5,aboveVolume=300000,stockTypeFilter='CORP')


scanData = ib.reqScannerData(sub)



def display_with_stock_symbol(scanData):
    df=util.df(scanData)
    df['contract'] = df.apply(lambda l:l['contractDetails'].contract, axis=1)
    df['symbol'] = df.apply(lambda l:l['contract'].symbol, axis=1)
    return df[['rank','contractDetails','contract','symbol']]

#print(display_with_stock_symbol(scanData))


ticker_dict = {}
for contract in display_with_stock_symbol(scanData)['contract'].tolist():
    ticker_dict[contract] = ib.reqMktData(contract=contract, genericTickList="",snapshot=True,regulatorySnapshot=False)
ib.sleep(3)

#print(util.df(ticker_dict.values())[['close', 'last', 'bid', 'ask']])



def display_stock_with_marketdata(scanData, ticker_dict):
    # Base stock info
    df = display_with_stock_symbol(scanData)  # should have 'rank' and 'symbol'

    # Build market data DataFrame
    market_data_list = []
    for ticker in ticker_dict.values():
        contract = ticker.contract
        market_data_list.append({
            'symbol': contract.symbol,
            'close': getattr(ticker, 'close', None),
            'last': getattr(ticker, 'last', None),
            'bid': getattr(ticker, 'bid', None),
            'ask': getattr(ticker, 'ask', None)
        })
    market_data_df = pd.DataFrame(market_data_list)

    # Merge
    df_merged = df.merge(market_data_df, on='symbol', how='left')

    # Compute % change safely
    df_merged['% Change'] = ((df_merged['last'] - df_merged['close']) / df_merged['close'] * 100).round(2)

    return df_merged[['rank','symbol','close','last','bid','ask','% Change']]


#print(display_stock_with_marketdata(scanData, ticker_dict))



def handle_incoming_dataframe_intraday(
    bars_df, 
    symbol, 
    adjust_timezone_IB_data # pass this function as well
):
    try:
        if bars_df is not None and not bars_df.empty:
            for col in ['average', 'barCount']:
                if col in bars_df.columns:
                    bars_df = bars_df.drop(columns=[col])

            if 'date' in bars_df.columns:
                bars_df['date'] = bars_df['date'].apply(adjust_timezone_IB_data)

            bars_df.columns = [col.capitalize() for col in bars_df.columns]
            bars_df['Symbol'] = symbol

            bars_df = bars_df[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


            # Split Date into Date and Time
            if bars_df['Date'].dtype == object:
                bars_df[['Date', 'Time']] = bars_df['Date'].str.split(' ', expand=True)
            else:
                bars_df['Date'] = bars_df['Date'].astype(str)
                bars_df[['Date', 'Time']] = bars_df['Date'].str.split(' ', expand=True)

            data = bars_df[['Symbol', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

            return data
        else:
            print("empty data")

    except Exception as e:
        print(f"An error occurred while processing the data: {e}")



def fetch_total_volume(df_stocks, ib):
    """
    Fetch 2-min bars for each symbol and compute total daily volume 
    using processed DataFrame (with timezone & cleaning applied).
    
    Returns:
        pd.DataFrame: DataFrame with 'symbol' and 'total_volume'
    """
    total_volumes = []

    for symbol in df_stocks['symbol']:
        # Create an IB contract
        contract = Stock(symbol, 'SMART', 'USD')
        contract.primaryExchange = 'ARCA'

        # Request historical 2-min bars (full day)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr='1 D',
            barSizeSetting='2 mins',
            whatToShow='TRADES',
            useRTH=False,   # include premarket
            formatDate=1
        )

        if not bars:
            print(f"No historical data for {symbol}")
            total_volumes.append({'symbol': symbol, 'total_volume': None})
            continue

        # Convert to DataFrame
        bars_df = pd.DataFrame([bar.__dict__ for bar in bars])

        # ✅ Apply preprocessing (timezone + cleaning)
        processed_df = handle_incoming_dataframe_intraday(
            bars_df=bars_df,
            symbol=contract.symbol,
            adjust_timezone_IB_data=adjust_timezone_IB_data
        )

        if processed_df is None or processed_df.empty:
            print(f"No processed data for {symbol}")
            total_volumes.append({'Symbol': symbol, 'total_volume': None})
            continue

        # ✅ Sum volume from processed DataFrame
        total_volume = processed_df['Volume'].sum()
        print(f"\n{symbol} - Total processed volume: {total_volume}")

        # ✅ Print processed bars (cleaned)
        print(processed_df[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']])

        total_volumes.append({'Symbol': symbol, 'total_volume': total_volume})

    return pd.DataFrame(total_volumes)



def fetch_historydata(df_stocks, ib, days=5):
    """
    Fetch historical intraday data for each symbol and calculate
    the average volume per time bucket over the past N days.

    Returns a single DataFrame with columns: ['symbol', 'Time', 'avg_volume_Xd']
    """
    all_results = []

    for symbol in df_stocks['symbol']:
        # Create an IB contract
        contract = Stock(symbol, 'SMART', 'USD')
        contract.primaryExchange = 'ARCA'

        # Request historical 2-min bars (past N days)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f'{days} D',
            barSizeSetting='2 mins',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )

        if not bars:
            print(f"No historical data for {symbol}")
            continue

        # Convert to DataFrame
        bars_df = pd.DataFrame([bar.__dict__ for bar in bars])

        # Apply preprocessing (your timezone & cleaning logic)
        processed_df = handle_incoming_dataframe_intraday(
            bars_df=bars_df,
            symbol=contract.symbol,
            adjust_timezone_IB_data=adjust_timezone_IB_data
        )

        if processed_df is None or processed_df.empty:
            print(f"No processed data for {symbol}")
            continue

        # Ensure proper datetime
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])

        # Calculate average volume per time bucket
        avg_vol_df = (
            processed_df.groupby('Time')['Volume']
            .mean()
            .reset_index()
            .rename(columns={'Volume': f'avg_volume_{days}d'})
        )

        avg_vol_df['Symbol'] = symbol
        all_results.append(avg_vol_df)

    if not all_results:
        return pd.DataFrame(columns=['Symbol', 'Time', f'avg_volume_{days}d'])

    # Concatenate all symbols into a single DataFrame
    return pd.concat(all_results, ignore_index=True)




def calculate_total_avg_volume(all_processed_dfs, avg_volumes_dict):
    """
    Loop through each symbol's intraday data, merge with its
    5-day average volume DataFrame, and compute totals.
    
    Parameters:
        all_processed_dfs (dict): {symbol: intraday_df}
        avg_volumes_dict (dict): {symbol: avg_volume_df}
    
    Returns:
        pd.DataFrame: summary with actual and 5-day average totals
    """
    results = []

    for symbol, processed_df in all_processed_dfs.items():
        if symbol not in avg_volumes_dict:
            print(f"⚠️ No 5-day avg volume data for {symbol}, skipping")
            continue

        avg_volume_df = avg_volumes_dict[symbol]

        # Merge on Time
        merged = processed_df.merge(avg_volume_df, on="Time", how="inner")

        # Totals
        actual_total = merged["Volume"].sum()
        avg_total = merged["avg_volume_5d"].sum()

        print(f"\n--- {symbol} ---")
        print(merged[['Date', 'Time', 'Volume', 'avg_volume_5d']].head(10))
        print(f"Actual total volume: {actual_total}")
        print(f"Total 5-day avg volume: {avg_total}")

        results.append({
            "symbol": symbol,
            "actual_total_volume": actual_total,
            "avg_total_volume": avg_total
        })

    return pd.DataFrame(results)





df_stocks = display_stock_with_marketdata(scanData, ticker_dict)
df_historydatas = fetch_historydata(df_stocks, ib)

#print(df_historydatas)

df_volumes = fetch_total_volume(df_stocks, ib)

# # 3️⃣ Merge total volumes into df_stocks
df_stocks = df_stocks.merge(df_volumes, on='symbol', how='left')

# # Sort by total volume descending
df_stocks_sorted = df_stocks.sort_values(by='total_volume', ascending=False)

#print(df_stocks_sorted)
print(df_historydatas)
print(df_stocks_sorted)