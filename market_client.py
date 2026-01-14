# --- START OF FILE market_client.py ---

import os


def get_market_client():
    exchange = os.getenv("EXCHANGE", "binance").lower()
    if exchange in {"gate", "gateio", "gate.io"}:
        from gate_client import GateClient
        return GateClient()
    from binance_client import BinanceClient
    return BinanceClient()

# --- END OF FILE market_client.py ---
