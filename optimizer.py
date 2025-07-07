import pandas as pd
from multiprocessing import Pool, cpu_count
from itertools import product
import time
from datetime import datetime, timedelta
import logging

# 假设这些模块在PYTHONPATH中
import json
from backtester import run_single_backtest
from binance_client import BinanceClient
from strategies import get_available_strategies

# --- 配置模块 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPTIMIZATION_CONFIG = {
    "symbol": "ETHUSDT",
    "backtest_start_date": "2025-06-01",
    "backtest_end_date": "2025-06-30",
    "scenarios": [
        {"contract_period": "10m", "kline_intervals": ["1m", "3m"]},
        # {"contract_period": "30m", "kline_intervals": ["5m", "15m"]},
        # {"contract_period": "1h", "kline_intervals": ["15m", "30m"]},
    ],
    "strategy_id": "flexible_signal",
    "fixed_backtest_params": {
        'initial_balance': 1000.0,
        'profit_rate_pct': 80.0,
        'loss_rate_pct': 100.0,
        'investment_strategy_id': 'fixed',
        'investment_strategy_params': {'amount': 10},
        'min_investment_amount': 5.0,
    },
    "analysis": {
        "min_avg_daily_trades": 0.0,
        "max_avg_daily_trades": 100.0,
    }
}

# 使用较小的步长进行迭代
STRATEGY_PARAM_RANGES = {
    'use_bb': [True],
    'use_rsi': [True],
    'use_td_seq': [False],
    'bb_period': range(18, 23, 2),  # 18, 20, 22
    'bb_std_dev': [i * 0.1 for i in range(19, 22)],  # 1.9, 2.0, 2.1
    'rsi_period': range(6, 20, 1),  # 12, 14, 16
    'rsi_oversold': range(15, 35, 1), # 28, 30, 32
    'rsi_overbought': range(60, 85, 1), # 68, 70, 72
    # --- TD Sequential 参数 ---
    'td_seq_buy_setup': [9, 13],
    'td_seq_sell_setup': [9, 13],
}

# --- 数据加载模块 ---
def load_all_data(symbol: str, intervals: list[str], start_date: str, end_date: str) -> dict:
    """
    一次性加载所有需要的回测和指数价格数据。
    """
    client = BinanceClient()
    all_data = {}
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    # 需要为结算获取额外数据，这里多加一天
    end_dt_obj = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_ms = int(end_dt_obj.timestamp() * 1000)

    for interval in set(intervals):
        logger.info(f"Loading {interval} kline data for {symbol}...")
        df_kline = client.get_historical_klines(symbol, interval, start_time=start_ms, end_time=end_ms)
        
        logger.info(f"Loading {interval} index price data for {symbol}...")
        df_index = client.get_index_price_klines(symbol, interval, start_time=start_ms, end_time=end_ms)
        
        if df_kline.empty or df_index.empty:
            raise ValueError(f"Failed to load data for interval {interval}. Kline data or index price data is empty.")
            
        all_data[interval] = {
            "kline": df_kline,
            "index_price": df_index
        }
    logger.info("All data loaded successfully.")
    return all_data

# --- 任务执行模块 ---
def generate_param_combinations(param_ranges: dict) -> list[dict]:
    """从范围生成所有参数组合的列表。"""
    keys = param_ranges.keys()
    values = param_ranges.values()
    combinations = list(product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

def worker_function(args: tuple) -> dict:
    """
    每个子进程执行的回测函数，增加了错误处理。
    """
    param_combo, df_kline, df_index_price, strategy_id, backtest_config = args
    
    try:
        # 将场景特定的配置与固定的回测参数合并
        full_config = {**backtest_config, **OPTIMIZATION_CONFIG['fixed_backtest_params']}

        return run_single_backtest(
            df_kline=df_kline,
            df_index_price=df_index_price,
            strategy_id=strategy_id,
            strategy_params=param_combo,
            backtest_config=full_config
        )
    except Exception as e:
        # 如果回测失败，返回一个包含错误信息的字典
        return {
            'params': param_combo,
            'win_rate': -1,
            'roi_percentage': -1,
            'total_predictions': 0,
            'error': str(e)
        }

# --- 主执行逻辑 ---
if __name__ == "__main__":
    main_start_time = time.time()

    # 1. 加载所有需要的数据
    all_intervals = [item for scenario in OPTIMIZATION_CONFIG['scenarios'] for item in scenario['kline_intervals']]
    try:
        all_data = load_all_data(
            OPTIMIZATION_CONFIG['symbol'],
            all_intervals,
            OPTIMIZATION_CONFIG['backtest_start_date'],
            OPTIMIZATION_CONFIG['backtest_end_date']
        )
    except ValueError as e:
        logger.error(f"Data loading failed: {e}")
        exit()

    # 2. 生成所有参数组合
    param_combinations = generate_param_combinations(STRATEGY_PARAM_RANGES)
    logger.info(f"Generated {len(param_combinations)} parameter combinations to test.")

    # 3. 遍历所有场景进行优化
    for scenario in OPTIMIZATION_CONFIG['scenarios']:
        for interval in scenario['kline_intervals']:
            scenario_start_time = time.time()
            logger.info(f"\n{'='*80}\nOptimizing for contract_period='{scenario['contract_period']}' on kline_interval='{interval}'\n{'='*80}")

            # 4. 准备任务列表
            scenario_config = {
                "symbol": OPTIMIZATION_CONFIG['symbol'],
                "interval": interval,
                "event_period": scenario['contract_period']
            }
            
            tasks = [(
                combo, 
                all_data[interval]['kline'], 
                all_data[interval]['index_price'],
                OPTIMIZATION_CONFIG['strategy_id'],
                scenario_config
            ) for combo in param_combinations]

            # 5. 使用多进程执行
            num_processes = max(1, cpu_count() - 1)
            logger.info(f"Starting parallel execution with {num_processes} processes...")
            
            with Pool(processes=num_processes) as pool:
                results = pool.map(worker_function, tasks)
            logger.info("--- Individual Backtest Results ---")
            for res in results:
                if not res:
                    logger.warning("  Received an empty result from a worker.")
                    continue
                
                # 检查是否有错误信息从worker返回
                if 'error' in res:
                    logger.error(
                        f"  Backtest CRASHED for params: {res.get('params', {})}. "
                        f"Error: {res.get('error')}"
                    )
                # 检查是否是正常失败（例如没有交易）
                elif res.get('win_rate', -1) == -1:
                    logger.warning(f"  Backtest completed with no trades or invalid data for params: {res.get('params', {})}")
                # 成功的分数
                else:
                    logger.info(
                        f"  Params: {res.get('params', {})}, "
                        f"Win Rate: {res.get('win_rate', 0):.2f}%, "
                        f"ROI: {res.get('roi_percentage', 0):.2f}%, "
                        f"Trades: {res.get('total_predictions', 0)}"
                    )
            logger.info("--- End of Individual Results ---")

            
            logger.info(f"Finished parallel execution. Received {len(results)} results.")

            # 6. 分析和报告结果
            if not results:
                logger.warning("No results returned from backtesting.")
                continue

            results_df = pd.DataFrame(results)
            
            # 过滤失败的运行
            valid_results_df = results_df[results_df['win_rate'] != -1].copy()
            if valid_results_df.empty:
                logger.warning("All backtest runs failed for this scenario.")
                continue

            # 计算每日平均交易次数
            start_dt = datetime.strptime(OPTIMIZATION_CONFIG['backtest_start_date'], "%Y-%m-%d")
            end_dt = datetime.strptime(OPTIMIZATION_CONFIG['backtest_end_date'], "%Y-%m-%d")
            num_days = (end_dt - start_dt).days
            if num_days == 0: num_days = 1
            
            valid_results_df['avg_daily_trades'] = valid_results_df['total_predictions'] / num_days

            # 应用交易频率约束
            min_trades = OPTIMIZATION_CONFIG['analysis']['min_avg_daily_trades']
            max_trades = OPTIMIZATION_CONFIG['analysis']['max_avg_daily_trades']
            
            constrained_df = valid_results_df[
                (valid_results_df['avg_daily_trades'] >= min_trades) &
                (valid_results_df['avg_daily_trades'] <= max_trades)
            ].copy()

            logger.info(f"Total valid runs: {len(valid_results_df)}. Runs meeting trade frequency constraints ({min_trades}-{max_trades} trades/day): {len(constrained_df)}")

            if constrained_df.empty:
                logger.warning("No results met the trade frequency constraints. Showing best from all valid runs.")
                # 如果没有满足约束的，就从所有有效结果中选
                target_df = valid_results_df
            else:
                target_df = constrained_df

            # 找到最优结果
            # --- 7. 筛选并保存多个优秀结果 ---
            # 定义“优秀”的标准：胜率和ROI都在前10%
            top_10_percent_count = max(1, int(len(target_df) * 0.1))

            top_by_win_rate = target_df.sort_values(by='win_rate', ascending=False).head(top_10_percent_count)
            top_by_roi = target_df.sort_values(by='roi_percentage', ascending=False).head(top_10_percent_count)

            # 合并两个列表并去重
            combined_top_results = pd.concat([top_by_win_rate, top_by_roi]).drop_duplicates().sort_values(by='win_rate', ascending=False)
            
            # 将参数字典转换为可读的字符串以便于查看
            if 'params' in combined_top_results.columns:
                 combined_top_results['params_str'] = combined_top_results['params'].apply(lambda p: json.dumps(p, sort_keys=True))

            # 保存到JSON文件
            results_filename = f"optimization_results_{scenario['contract_period']}_{interval}.json"
            try:
                combined_top_results.to_json(results_filename, orient='records', indent=4, lines=False)
                logger.info(f"已将 {len(combined_top_results)} 个优秀结果保存到文件: {results_filename}")
            except Exception as e:
                logger.error(f"保存结果到 {results_filename} 时失败: {e}")


            # --- 8. 打印报告 (仍然打印最优的两个) ---
            best_win_rate = top_by_win_rate.iloc[0]
            best_roi = top_by_roi.iloc[0]

            logger.info(f"\n--- Optimization Report for {scenario['contract_period']} @ {interval} ---")
            logger.info(f"Highest Win Rate (constrained):")
            logger.info(f"  - Win Rate: {best_win_rate['win_rate']:.2f}%")
            logger.info(f"  - ROI: {best_win_rate['roi_percentage']:.2f}%")
            logger.info(f"  - Total Trades: {best_win_rate['total_predictions']}")
            logger.info(f"  - Avg Daily Trades: {best_win_rate['avg_daily_trades']:.2f}")
            logger.info(f"  - Parameters: {best_win_rate['params']}")

            logger.info(f"\nHighest ROI (constrained):")
            logger.info(f"  - Win Rate: {best_roi['win_rate']:.2f}%")
            logger.info(f"  - ROI: {best_roi['roi_percentage']:.2f}%")
            logger.info(f"  - Total Trades: {best_roi['total_predictions']}")
            logger.info(f"  - Avg Daily Trades: {best_roi['avg_daily_trades']:.2f}")
            logger.info(f"  - Parameters: {best_roi['params']}")
            
            scenario_end_time = time.time()
            logger.info(f"Scenario finished in {scenario_end_time - scenario_start_time:.2f} seconds.")

    main_end_time = time.time()
    logger.info(f"\nTotal optimization process finished in {main_end_time - main_start_time:.2f} seconds.")