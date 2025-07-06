# -*- coding: utf-8 -*-
import os
import itertools
import multiprocessing
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 确保可以从项目根目录正确导入模块
try:
    from binance_client import BinanceClient
    from strategies import RsiBollingerBandsStrategy as FlexibleSignalStrategy
except ImportError:
    print("请确保 optimizer.py 在项目根目录下运行，以便正确导入 binance_client 和 strategies。")
    exit()

# --- 1. 参数空间定义 ---
PARAM_SPACE = {
    'use_rsi': [True, False],
    'use_bbands': [True, False],
    'use_td_sequential': [False],  # 根据要求，暂时不启用 TD Sequential
    'rsi_period': range(10, 31, 5),  # 步长为5以减少组合数
    'bb_period': range(15, 36, 5),   # 步长为5
    'bb_std_dev': [1.5, 2.0, 2.5],
    'contract_duration': ['10m', '30m', '1h'],
    # 扩展RSI阈值的搜索范围
    'rsi_oversold': range(20, 41, 5),  # 20, 25, 30, 35, 40
    'rsi_overbought': range(60, 81, 5), # 60, 65, 70, 75, 80
}

# 事件合约收益/亏损率
PAYOUT_RATES = {
    '10m': {'win': 0.80, 'loss': -1.0},
    '30m': {'win': 0.85, 'loss': -1.0},
    '1h': {'win': 0.85, 'loss': -1.0},
}
DAYS_OF_DATA = 30 # 获取过去30天的数据

def resample_data(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """将1分钟K线数据重采样到指定的时间周期"""
    if not df_1m.index.is_monotonic_increasing:
        df_1m = df_1m.sort_index()

    resampling_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # 使用 closed='left', label='left' 来确保K线时间戳是周期的开始
    resampled_df = df_1m.resample(timeframe, closed='left', label='left').agg(resampling_rules)
    resampled_df.dropna(inplace=True)
    return resampled_df

def run_backtest_for_params(args: Tuple[Dict[str, Any], pd.DataFrame]) -> Tuple[Dict[str, Any], float, float, int]:
    """
    为单个参数组合运行回测。
    这是将在多进程中执行的工作函数。
    """
    params, df_1m = args
    
    # 过滤掉无效的参数组合
    if not params.get('use_rsi') and not params.get('use_bbands') and not params.get('use_td_sequential'):
        return (params, 0, 0, 0)
    if not params.get('use_rsi'):
        params['rsi_period'] = 14 # 如果不使用RSI，提供一个默认值以避免策略类出错
    if not params.get('use_bbands'):
        params['bb_period'] = 20 # 如果不使用BBands，提供一个默认值
        params['bb_std_dev'] = 2.0

    try:
        duration_str = params['contract_duration']
        
        # 2. 重采样数据
        df_resampled = resample_data(df_1m, duration_str)
        if len(df_resampled) < max(params.get('rsi_period', 0), params.get('bb_period', 0)) + 5:
            return (params, 0, 0, 0) # 数据不足以计算指标

        # 3. 实例化策略并生成信号
        # RsiBollingerBandsStrategy 在 strategies.py 中被重命名为 FlexibleSignalStrategy
        strategy_params = {
            'use_bb': params['use_bbands'],
            'use_rsi': params['use_rsi'],
            'use_td_seq': params['use_td_sequential'],
            'bb_period': params['bb_period'],
            'bb_std_dev': params['bb_std_dev'],
            'rsi_period': params['rsi_period'],
            'rsi_oversold': params['rsi_oversold'],
            'rsi_overbought': params['rsi_overbought'],
        }
        strategy = FlexibleSignalStrategy(params=strategy_params)
        df_signals = strategy.generate_signals(df_resampled)

        signals = df_signals[df_signals['signal'] != 0]
        if signals.empty:
            return (params, 0, 0, 0)

        # 4. 模拟事件合约
        wins = 0
        losses = 0
        total_signals = len(signals)
        payout = PAYOUT_RATES[duration_str]
        
        # 将1分钟数据的时间戳转换为与信号数据相同的时区
        if df_1m.index.tz is None:
            df_1m.index = df_1m.index.tz_localize('UTC')
        df_1m.index = df_1m.index.tz_convert(signals.index.tz)

        for signal_time, signal_row in signals.iterrows():
            signal_direction = signal_row['signal']
            
            # 找到信号时间之后的价格
            future_time = signal_time + pd.to_timedelta(duration_str)
            
            # 在1分钟数据中寻找最接近未来时间的收盘价
            future_candles = df_1m.loc[df_1m.index >= future_time]
            if not future_candles.empty:
                future_price = future_candles.iloc[0]['close']
                entry_price = signal_row['close']
                
                if signal_direction == 1: # 预测涨
                    if future_price > entry_price:
                        wins += 1
                    else:
                        losses += 1
                elif signal_direction == -1: # 预测跌
                    if future_price < entry_price:
                        wins += 1
                    else:
                        losses += 1
        
        if total_signals == 0:
            return (params, 0, 0, 0)

        win_rate = wins / total_signals if total_signals > 0 else 0
        total_return = (wins * payout['win']) + (losses * payout['loss'])
        
        return (params, total_return, win_rate, total_signals)

    except Exception as e:
        # print(f"参数 {params} 的回测出错: {e}")
        return (params, 0, 0, 0)


def main():
    """主优化器函数"""
    print("--- 开始策略参数优化 ---")
    
    # 1. 获取数据
    print(f"正在获取过去 {DAYS_OF_DATA} 天的 ETHUSDT 1分钟 K线数据...")
    client = BinanceClient()
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_OF_DATA)
    
    try:
        df_1m = client.get_historical_klines(
            symbol="ETHUSDT",
            interval="1m",
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000)
        )
        if df_1m.empty:
            print("获取K线数据失败，请检查网络或API设置。")
            return
        print(f"成功获取 {len(df_1m)} 条1分钟K线数据。")
    except Exception as e:
        print(f"获取数据时发生错误: {e}")
        return

    # 2. 生成参数组合
    keys, values = zip(*PARAM_SPACE.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 过滤掉没有启用任何指标的组合
    param_combinations = [p for p in param_combinations if p['use_rsi'] or p['use_bbands']]
    
    print(f"将要测试 {len(param_combinations)} 个参数组合。")

    # 3. 并行处理
    # 将1分钟数据框与每个参数组合打包
    tasks = [(params, df_1m) for params in param_combinations]
    
    results = []
    # 使用tqdm显示进度条
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(tasks), desc="优化进度") as pbar:
            for result in pool.imap_unordered(run_backtest_for_params, tasks):
                if result:
                    results.append(result)
                pbar.update()

    if not results:
        print("\n优化完成，但没有找到任何有效的信号。")
        return

    # 4. 结果分析
    print("\n--- 优化结果分析 ---")
    
    # 过滤: 平均每天至少0.5个信号 (降低门槛)
    min_signals = int(DAYS_OF_DATA * 0.5)
    filtered_results = [res for res in results if res[3] >= min_signals]
    
    if not filtered_results:
        print(f"所有参数组合产生的信号总数都少于 {min_signals}。无法提供最佳参数建议。")
        print("可以尝试放宽参数范围或减少过滤条件。")
        return
        
    print(f"找到 {len(filtered_results)} 个满足“日均1信号”条件的参数组合。")

    # 排序
    # 按胜率最高排序
    best_win_rate_combo = sorted(filtered_results, key=lambda x: x[2], reverse=True)[0]
    # 按总回报率最高排序
    best_return_combo = sorted(filtered_results, key=lambda x: x[1], reverse=True)[0]

    # 5. 输出报告
    print("\n--- 最佳参数报告 ---")
    
    print("\n【胜率最高】的参数组合:")
    params, total_return, win_rate, total_signals = best_win_rate_combo
    print(f"  - 参数: {params}")
    print(f"  - 胜率: {win_rate:.2%}")
    print(f"  - 总回报率: {total_return:.2f} (基于每次1单位投注)")
    print(f"  - 信号总数: {total_signals} (过去{DAYS_OF_DATA}天)")

    print("\n【总回报率最高】的参数组合:")
    params, total_return, win_rate, total_signals = best_return_combo
    print(f"  - 参数: {params}")
    print(f"  - 总回报率: {total_return:.2f} (基于每次1单位投注)")
    print(f"  - 胜率: {win_rate:.2%}")
    print(f"  - 信号总数: {total_signals} (过去{DAYS_OF_DATA}天)")
    
    print("\n--- 优化完成 ---")


if __name__ == '__main__':
    # 在Windows上，必须将多进程代码放在 if __name__ == '__main__': 块中
    multiprocessing.freeze_support() 
    main()