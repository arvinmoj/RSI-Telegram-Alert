import os
import sys
import math
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import telegram
import pandas as pd
from dotenv import load_dotenv
import ccxt.async_support as ccxt
from ta.momentum import RSIIndicator

# ─── Configuration ────────────────────────────────────────────────────────── #

load_dotenv()  # read .env if present

@dataclass
class Config:
    SYMBOLS: List[str] = field(
        default_factory=lambda: os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
    )
    INTERVALS: List[str] = field(
        default_factory=lambda: os.getenv("INTERVALS", "3m,5m,15m").split(",")
    )
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", 14))
    RSI_OVERBOUGHT: int = int(os.getenv("RSI_OVERBOUGHT", 70))
    RSI_OVERSOLD: int = int(os.getenv("RSI_OVERSOLD", 30))
    TELEGRAM_TOKEN: Optional[str] = os.getenv("TELEGRAM_TOKEN")
    CHAT_ID: Optional[str] = os.getenv("CHAT_ID")
    EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance").lower()
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", 2))          # seconds
    BUFFER_SECONDS: int = int(os.getenv("BUFFER_SECONDS", 5))    # after close
    ALERT_COOLDOWN: int = int(os.getenv("ALERT_COOLDOWN", 300))  # seconds
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        missing = [n for n in ("TELEGRAM_TOKEN", "CHAT_ID") if getattr(self, n) is None]
        if missing:
            raise ValueError(f"Missing required env variables: {', '.join(missing)}")

CONFIG = Config()
CONFIG.validate()

# ─── Logging ──────────────────────────────────────────────────────────────── #

logging.basicConfig(
    level=getattr(logging, CONFIG.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("RSI_Bot")

# ─── Telegram & Exchange ──────────────────────────────────────────────────── #

bot = telegram.Bot(token=CONFIG.TELEGRAM_TOKEN)

exchange_class = getattr(ccxt, CONFIG.EXCHANGE_ID, None)
if exchange_class is None:
    log.critical("Unknown exchange %s", CONFIG.EXCHANGE_ID)
    sys.exit(1)

exchange = exchange_class({"enableRateLimit": True})


# ─── Helpers ─────────────────────────────────────────────────────────────── #

def seconds_from_timeframe(tf: str) -> int:
    """Use CCXT’s parse_timeframe for robustness."""
    return exchange.parse_timeframe(tf)


def next_candle_close_seconds(tf_seconds: int) -> float:
    """Return seconds until the current candle *closes* (UTC)."""
    now = datetime.now(timezone.utc).timestamp()
    return tf_seconds - (now % tf_seconds)


class AlertManager:
    """Prevent duplicate alerts within a cooldown window."""

    def __init__(self, cooldown: int) -> None:
        self.cooldown = cooldown
        self._last: Dict[Tuple[str, str, str], float] = {}

    def allow(self, symbol: str, interval: str, kind: str) -> bool:
        k = (symbol, interval, kind)
        now = datetime.now().timestamp()
        last = self._last.get(k, 0.0)
        if now - last >= self.cooldown:
            self._last[k] = now
            return True
        return False


alerts = AlertManager(CONFIG.ALERT_COOLDOWN)


async def tg_send(msg: str) -> None:
    """Send Telegram message with one retry on timeout."""
    try:
        await bot.send_message(chat_id=CONFIG.CHAT_ID, text=msg)
        log.info("Sent TG alert: %s", msg.splitlines()[0])
    except telegram.error.TimedOut:
        log.warning("Telegram timeout – retrying once…")
        await asyncio.sleep(2)
        await bot.send_message(chat_id=CONFIG.CHAT_ID, text=msg)
    except Exception as e:  # pylint: disable=broad-except
        log.error("Telegram send failed: %s", e, exc_info=True)


async def robust_fetch_ohlcv(
    symbol: str, interval: str, limit: int
) -> List[List]:
    """Retry fetch_ohlcv with exponential back-off."""
    for attempt in range(1, CONFIG.MAX_RETRIES + 1):
        try:
            return await exchange.fetch_ohlcv(symbol, interval, limit=limit)
        except (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable) as e:
            if attempt == CONFIG.MAX_RETRIES:
                log.error("fetch_ohlcv failed after %d attempts: %s", attempt, e)
                return []
            delay = CONFIG.RETRY_DELAY * attempt
            log.warning(
                "%s/%s network issue (%s). Retry %d/%d in %ds",
                symbol,
                interval,
                type(e).__name__,
                attempt,
                CONFIG.MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as e:  # pylint: disable=broad-except
            log.error("Unexpected error %s: %s", type(e).__name__, e, exc_info=True)
            return []
    return []


# ─── Core check function ─────────────────────────────────────────────────── #

async def check_rsi(symbol: str, interval: str) -> None:
    tf_seconds = seconds_from_timeframe(interval)

    # Always align to candle close
    await asyncio.sleep(next_candle_close_seconds(tf_seconds) + CONFIG.BUFFER_SECONDS)

    while True:
        try:
            limit = CONFIG.RSI_PERIOD + 3  # extra candles for safety
            ohlcv = await robust_fetch_ohlcv(symbol, interval, limit)
            if not ohlcv:
                # Wait one candle and try again
                await asyncio.sleep(tf_seconds)
                continue

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["rsi"] = RSIIndicator(close=df["close"], window=CONFIG.RSI_PERIOD).rsi()

            # Use the last *closed* candle’s RSI (index -2)
            prev_rsi, curr_rsi = df["rsi"].iloc[-3:-1]

            oversold_exit = (
                prev_rsi < CONFIG.RSI_OVERSOLD and curr_rsi >= CONFIG.RSI_OVERSOLD
            )
            overbought_exit = (
                prev_rsi > CONFIG.RSI_OVERBOUGHT and curr_rsi <= CONFIG.RSI_OVERBOUGHT
            )

            if oversold_exit and alerts.allow(symbol, interval, "oversold_exit"):
                await tg_send(
                    f"📈 *RSI Oversold Exit*\n"
                    f"{symbol} | {interval}\n"
                    f"{prev_rsi:.2f} ➜ {curr_rsi:.2f} crosses ↑ {CONFIG.RSI_OVERSOLD}"
                )

            elif overbought_exit and alerts.allow(
                symbol, interval, "overbought_exit"
            ):
                await tg_send(
                    f"📉 *RSI Overbought Exit*\n"
                    f"{symbol} | {interval}\n"
                    f"{prev_rsi:.2f} ➜ {curr_rsi:.2f} crosses ↓ {CONFIG.RSI_OVERBOUGHT}"
                )
            else:
                log.debug(
                    "%s/%s RSI: prev=%5.2f cur=%5.2f",
                    symbol,
                    interval,
                    prev_rsi,
                    curr_rsi,
                )

        except Exception as e:  # pylint: disable=broad-except
            log.error("Error in %s/%s loop: %s", symbol, interval, e, exc_info=True)

        # Sleep until *next* candle close
        await asyncio.sleep(next_candle_close_seconds(tf_seconds) + CONFIG.BUFFER_SECONDS)


# ─── Boot-time validation ────────────────────────────────────────────────── #

async def validate_config() -> None:
    await exchange.load_markets()
    # Symbols
    available = set(exchange.symbols)
    for s in CONFIG.SYMBOLS:
        if s not in available:
            raise ValueError(f"Symbol '{s}' not available on {CONFIG.EXCHANGE_ID}")
    # Timeframes
    tf_available = exchange.timeframes.keys()
    for tf in CONFIG.INTERVALS:
        if tf not in tf_available:
            raise ValueError(
                f"Timeframe '{tf}' not supported on {CONFIG.EXCHANGE_ID}"
            )


# ─── Main asyncio entry-point ────────────────────────────────────────────── #

async def main() -> None:
    try:
        await validate_config()
    except ValueError as e:
        log.critical("Configuration error: %s", e)
        return

    log.info(
        "Started RSI bot on %s | symbols=%s | intervals=%s",
        CONFIG.EXCHANGE_ID,
        ",".join(CONFIG.SYMBOLS),
        ",".join(CONFIG.INTERVALS),
    )

    tasks = [
        asyncio.create_task(check_rsi(sym, tf))
        for sym in CONFIG.SYMBOLS
        for tf in CONFIG.INTERVALS
    ]

    # Wait until any task raises (they shouldn’t) or Ctrl-C happens
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        log.info("Shutting down…")
        await exchange.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user, closing…")
