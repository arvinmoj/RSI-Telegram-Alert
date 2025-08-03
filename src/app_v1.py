import ccxt
import pandas as pd
import ta.momentum
import asyncio
import telegram
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
INTERVALS = ['3m', '5m', '15m']
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # Replace with your bot token
CHAT_ID = 'YOUR_CHAT_ID'  # Replace with your Telegram chat ID
CHECK_INTERVAL = 60  # Seconds between checks

# Initialize exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Initialize Telegram bot
bot = telegram.Bot(token=TELEGRAM_TOKEN)

async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
        logger.info(f"Telegram message sent: {message}")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

def fetch_ohlcv(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol} at {timeframe}: {e}")
        return None

def calculate_rsi(df, period=14):
    try:
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=period).rsi()
        df['rsi'] = rsi
        return df
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None

def check_rsi_signals(df, symbol, timeframe):
    if df is None or len(df) < 2:
        return None
    
    current_rsi = df['rsi'].iloc[-1]
    previous_rsi = df['rsi'].iloc[-2]
    timestamp = df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
    
    signals = []
    
    # Check for RSI crossing above 30 (buy signal)
    if previous_rsi < RSI_OVERSOLD and current_rsi >= RSI_OVERSOLD:
        message = (f"🚀 {symbol} {timeframe} - RSI Buy Signal!\n"
                   f"RSI crossed above {RSI_OVERSOLD}: {current_rsi:.2f}\n"
                   f"Time: {timestamp}")
        signals.append(message)
    
    # Check for RSI crossing below 70 (sell signal)
    if previous_rsi > RSI_OVERBOUGHT and current_rsi <= RSI_OVERBOUGHT:
        message = (f"📉 {symbol} {timeframe} - RSI Sell Signal!\n"
                   f"RSI crossed below {RSI_OVERBOUGHT}: {current_rsi:.2f}\n"
                   f"Time: {timestamp}")
        signals.append(message)
    
    return signals

async def monitor_rsi():
    while True:
        for symbol in SYMBOLS:
            for timeframe in INTERVALS:
                # Fetch data
                df = fetch_ohlcv(symbol, timeframe, limit=RSI_PERIOD + 1)
                if df is None:
                    continue
                
                # Calculate RSI
                df = calculate_rsi(df, RSI_PERIOD)
                if df is None:
                    continue
                
                # Check for signals
                signals = check_rsi_signals(df, symbol, timeframe)
                if signals:
                    for signal in signals:
                        await send_telegram_message(signal)
        
        # Wait before next check
        await asyncio.sleep(CHECK_INTERVAL)

async def main():
    logger.info("Starting RSI monitoring bot...")
    try:
        await monitor_rsi()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())