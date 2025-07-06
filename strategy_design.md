# 交易策略优化方案

## 1. 概述

本方案旨在对现有的 `RsiBollingerBandsStrategy` 进行功能增强和重构，以集成 **TD Sequential 指标**，并建立一个灵活的**可插拔指标配置系统**。这将允许用户通过参数动态启用或禁用策略中的任何一个指标（RSI、布林带、TD Sequential），从而实现对策略行为的精细控制。

## 2. 核心设计思想

*   **配置驱动 (Configuration-Driven):** 策略的行为将完全由传入的参数 `params` 控制。我们添加了 `use_rsi`、`use_bb`、`use_td_seq` 等布尔型参数来决定是否启用相应指标的计算和信号生成。
*   **模块化计算 (Modular Calculation):** 在 `calculate_all_indicators` 方法中，每个指标的计算将是独立的，并由其对应的 `use_*` 参数触发。这确保了不计算未使用的指标，从而提高效率。
*   **清晰的信号逻辑 (Clear Signal Logic):** 在 `generate_signals_from_indicators_on_window` 方法中，我们实现了一个明确的信号组合逻辑。默认采用 **“全票通过” (AND Logic)** 原则，即只有当所有**已启用**的指标同时发出同向信号时，最终的交易信号才会被触发。

## 3. 建议的代码结构

我们修改了 `RsiBollingerBandsStrategy` 类。为了反映其增强的功能，其在 `get_available_strategies` 中的ID已更新为 `flexible_signal`。

### 3.1. `__init__` (构造函数)

我们在这里添加了新的配置参数，以控制各个指标的开关。

```python
# 在 RsiBollingerBandsStrategy 内部
def __init__(self, params: Dict[str, Any] = None):
    default_params = {
        # --- Indicator Switches ---
        'use_bb': True,
        'use_rsi': True,
        'use_td_seq': False, # Default to off

        # --- Bollinger Bands Params ---
        'bb_period': 20,
        'bb_std_dev': 2.0,

        # --- RSI Params ---
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,

        # --- TD Sequential Params ---
        'td_seq_buy_setup': 9,  # Can be 9 or 13
        'td_seq_sell_setup': 9, # Can be 9 or 13
    }
    if params: default_params.update(params)
    super().__init__(default_params)
    
    active_indicators = []
    if self.params.get('use_bb'): active_indicators.append('BB')
    if self.params.get('use_rsi'): active_indicators.append('RSI')
    if self.params.get('use_td_seq'): active_indicators.append('TD')
    self.name = f"Flexible ({'+'.join(active_indicators)})"
    
    # ... (min_history_periods calculation) ...
```

### 3.2. `calculate_all_indicators` (指标计算)

此方法将根据配置参数，按需计算指标。

```python
# 在 RsiBollingerBandsStrategy 内部
def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    df = super().calculate_all_indicators(df)
    try:
        if self.params.get('use_bb'):
            df.ta.bbands(length=self.params['bb_period'], std=self.params['bb_std_dev'], append=True)

        if self.params.get('use_rsi'):
            df.ta.rsi(length=self.params['rsi_period'], append=True)

        if self.params.get('use_td_seq'):
            df.ta.td_seq(append=True)

    except Exception as e:
        logger.error(f"({self.name}) Error calculating indicators: {e}", exc_info=True)
    return df
```

## 4. 更新后的策略逻辑

### 4.1. 信号生成流程图 (Mermaid)

```mermaid
graph TD
    A[开始] --> B{检查配置};
    B --> C{启用布林带?};
    C -- 是 --> D[计算BB信号 (bb_signal)];
    C -- 否 --> E;
    D --> E;
    E{启用RSI?};
    E -- 是 --> F[计算RSI信号 (rsi_signal)];
    E -- 否 --> G;
    F --> G;
    G{启用TD Sequential?};
    G -- 是 --> H[计算TD信号 (td_signal)];
    G -- 否 --> I;
    H --> I;
    I[组合信号: AND 逻辑] --> J{生成最终信号};
    J --> K[结束];

    subgraph "信号计算"
        D: "价格 <= BBL ? 1 : (价格 >= BBU ? -1 : 0)"
        F: "RSI <= 超卖 ? 1 : (RSI >= 超买 ? -1 : 0)"
        H: "出现TD9/13买入? 1 : (出现TD9/13卖出? -1 : 0)"
    end

    subgraph "信号组合"
        I: "1. 收集所有已启用的信号 [s1, s2...]\n2. 如果所有信号都为1, final_signal = 1\n3. 如果所有信号都为-1, final_signal = -1\n4. 否则, final_signal = 0"
    end
```

### 4.2. 信号生成伪代码

这是 `generate_signals_from_indicators_on_window` 方法的核心逻辑。

```pseudocode
function generate_signals_from_indicators_on_window(df_window):
    // 初始化
    final_signal = 0
    active_signals = []
    last_row = df_window.iloc[-1]

    // 1. 计算布林带信号 (如果启用)
    if params['use_bb'] is True:
        bb_signal = 0
        if last_row['close'] <= last_row['BBL']:
            bb_signal = 1
        elif last_row['close'] >= last_row['BBU']:
            bb_signal = -1
        active_signals.append(bb_signal)

    // 2. 计算RSI信号 (如果启用)
    if params['use_rsi'] is True:
        rsi_signal = 0
        if last_row['RSI'] <= params['rsi_oversold']:
            rsi_signal = 1
        elif last_row['RSI'] >= params['rsi_overbought']:
            rsi_signal = -1
        active_signals.append(rsi_signal)

    // 3. 计算TD Sequential信号 (如果启用)
    if params['use_td_seq'] is True:
        td_signal = 0
        // 检查TD买入信号 (TD9或TD13)
        if last_row['TD_SEQ_UPa'] == params['td_seq_buy_setup']:
             td_signal = 1
        // 检查TD卖出信号 (TD9或TD13)
        elif last_row['TD_SEQ_DNa'] == params['td_seq_sell_setup']:
             td_signal = -1
        active_signals.append(td_signal)

    // 4. 组合信号 (AND 逻辑)
    if not active_signals: // 如果没有启用任何指标
        return 0 // 无信号

    is_all_buy = all(s == 1 for s in active_signals)
    is_all_sell = all(s == -1 for s in active_signals)

    if is_all_buy:
        final_signal = 1
    elif is_all_sell:
        final_signal = -1
    else:
        final_signal = 0

    // 设置最终信号和置信度
    set_signal_on_last_row(final_signal)
    return df_window
```

## 5. TD Sequential 指标实现方案

### 5.1. 主方案: `pandas-ta`

我们已采用 `pandas-ta` 库作为主方案，因为它简洁高效。

*   **调用:** `df.ta.td_seq(append=True)`
*   **关键列:**
    *   `TD_SEQ_UPa`: TD 买入序列计数 (当达到 `td_seq_buy_setup` 中定义的9或13时触发信号)。
    *   `TD_SEQ_DNa`: TD 卖出序列计数 (当达到 `td_seq_sell_setup` 中定义的9或13时触发信号)。

### 5.2. 备用方案: 手动实现

如果 `pandas-ta` 存在问题或不可用，以下是不依赖任何库的TD Sequential核心逻辑伪代码。这可以被实现为一个私有方法 `_calculate_td_sequential_backup(df)`。

```pseudocode
function _calculate_td_sequential_backup(dataframe):
    // 初始化列
    dataframe['td_setup_up'] = 0
    dataframe['td_setup_down'] = 0

    // --- 1. 计算 TD Setup (TDST) ---
    for i from 4 to len(dataframe):
        // 买入结构 (Buy Setup)
        if dataframe['close'][i] > dataframe['close'][i-4]:
            dataframe['td_setup_up'][i] = dataframe['td_setup_up'][i-1] + 1
        else:
            dataframe['td_setup_up'][i] = 0

        // 卖出结构 (Sell Setup)
        if dataframe['close'][i] < dataframe['close'][i-4]:
            dataframe['td_setup_down'][i] = dataframe['td_setup_down'][i-1] + 1
        else:
            dataframe['td_setup_down'][i] = 0

    // --- 2. 识别 TD Setup 完成信号 (TD9) ---
    // (为简化，这里只展示TD9，TD13逻辑类似，但基于TD Countdown)
    dataframe['td9_buy_signal'] = 0
    dataframe['td9_sell_signal'] = 0

    for i from 1 to len(dataframe):
        // 完美的买入信号 TD9
        if dataframe['td_setup_up'][i-1] == 8 and dataframe['td_setup_up'][i] == 9:
            // 完美条件: 第8或第9根K线的低点必须低于第6和第7根K线的低点
            low_8 = dataframe['low'][i-1]
            low_9 = dataframe['low'][i]
            low_6 = dataframe['low'][i-3]
            low_7 = dataframe['low'][i-2]
            if low_9 < min(low_6, low_7) or low_8 < min(low_6, low_7):
                dataframe['td9_buy_signal'][i] = 1

        // 完美的卖出信号 TD9
        if dataframe['td_setup_down'][i-1] == 8 and dataframe['td_setup_down'][i] == 9:
            // 完美条件: 第8或第9根K线的高点必须高于第6和第7根K线的高点
            high_8 = dataframe['high'][i-1]
            high_9 = dataframe['high'][i]
            high_6 = dataframe['high'][i-3]
            high_7 = dataframe['high'][i-2]
            if high_9 > max(high_6, high_7) or high_8 > max(high_6, high_7):
                dataframe['td9_sell_signal'][i] = -1

    return dataframe
```
*注意：完整的TD Sequential逻辑还包括Countdown阶段，上述伪代码为核心的Setup阶段，已能满足大部分TD9信号需求。*