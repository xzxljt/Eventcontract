# --- START OF FILE market_client.py ---

import os

from binance_client import BinanceClient
from gate_client import GateClient


def get_market_client():
    exchange = os.getenv("EXCHANGE", "binance").lower()
    if exchange in {"gate", "gateio", "gate.io"}:
        return GateClient()
    return BinanceClient()

# --- END OF FILE market_client.py ---
