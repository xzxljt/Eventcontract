# 策略参数优化引擎

## 概述

策略参数优化引擎是一个强大的后端模块，用于自动化寻找最优的策略参数组合。它通过网格搜索和多线程并行计算，为用户提供基于历史数据的最佳策略参数。

## 主要功能

### 1. 网格搜索优化
- 支持多参数同时优化
- 自动生成所有可能的参数组合
- 可配置参数范围和步长

### 2. 多线程并行计算
- 利用多核CPU加速计算
- 可配置最大线程数
- 支持优雅停止机制

### 3. 综合评估体系
- **收益率指标**: 总收益率、年化收益率、夏普比率
- **风险指标**: 最大回撤、波动率、VaR
- **交易指标**: 胜率、盈利因子、交易次数、平均持仓时间
- **综合评分**: 可自定义权重的综合评分系统

### 4. 结果可视化
- 胜率vs收益率散点图
- 参数排名表格
- 详细的优化统计信息

### 5. 实用功能
- 参数预设（保守型、平衡型、激进型）
- 进度实时监控
- 结果导出（CSV格式）
- 资源限制保护

## 使用方法

### 1. Web界面使用

1. 启动服务器：
   ```bash
   python main.py
   ```

2. 访问优化页面：
   ```
   http://localhost:8000/optimization
   ```

3. 配置优化参数：
   - 选择交易对和K线周期
   - 设置回测时间范围
   - 选择策略和参数范围
   - 可选择预设参数或自定义

4. 开始优化并监控进度

5. 查看结果和散点图

### 2. API使用

#### 启动优化
```bash
POST /api/optimization/start
```

请求体示例：
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "start_date": "2024-01-01",
  "end_date": "2024-01-02",
  "strategy_id": "simple_rsi",
  "strategy_params_ranges": {
    "rsi_period": {"min": 10, "max": 20, "step": 2},
    "rsi_overbought": {"min": 65, "max": 80, "step": 5},
    "rsi_oversold": {"min": 20, "max": 35, "step": 5}
  },
  "max_combinations": 1000,
  "min_trades": 10
}
```

#### 获取进度
```bash
GET /api/optimization/progress/{optimization_id}
```

#### 获取结果
```bash
GET /api/optimization/results/{optimization_id}
```

#### 停止优化
```bash
POST /api/optimization/stop/{optimization_id}
```

### 3. 编程接口使用

```python
from optimization_engine import get_optimization_engine

# 获取引擎实例
engine = get_optimization_engine()

# 配置优化参数
config = {
    'symbol': 'BTCUSDT',
    'interval': '1m',
    'start_date': '2024-01-01',
    'end_date': '2024-01-02',
    'strategy_id': 'simple_rsi',
    'strategy_params_ranges': {
        'rsi_period': {'min': 10, 'max': 20, 'step': 2},
        'rsi_overbought': {'min': 65, 'max': 80, 'step': 5},
        'rsi_oversold': {'min': 20, 'max': 35, 'step': 5}
    }
}

# 启动优化
optimization_id = engine.optimize_strategy(config)

# 监控进度
progress = engine.get_optimization_progress(optimization_id)

# 获取结果
results = engine.get_optimization_results(optimization_id)
```

## 支持的策略

### 1. 简单RSI策略 (simple_rsi)
- **参数**: rsi_period, rsi_overbought, rsi_oversold
- **适用**: 基础RSI超买超卖策略

### 2. 加强版RSI策略 (enhanced_rsi)
- **参数**: rsi_period, rsi_overbought, rsi_oversold, ema_fast, ema_slow, adx_period等
- **适用**: 1分钟K线和10分钟事件合约，智能趋势识别

### 3. 灵活信号组合策略 (flexible_signal)
- **参数**: bb_period, bb_std_dev, rsi_period, rsi_overbought, rsi_oversold等
- **适用**: 结合RSI、布林带和TD Sequential指标

## 参数预设

每个策略都提供三种预设：

### 保守型 (Conservative)
- 适合稳定收益
- 参数范围较窄
- 风险较低

### 平衡型 (Balanced)
- 收益与风险平衡
- 中等参数范围
- 推荐使用

### 激进型 (Aggressive)
- 追求高收益
- 参数范围较宽
- 风险较高

## 评估指标说明

### 综合评分计算
默认权重：
- 总收益率: 25%
- 胜率: 25%
- 夏普比率: 20%
- 最大回撤: 15%
- 盈利因子: 15%

### 关键指标
- **总收益率**: 整个回测期间的总收益百分比
- **胜率**: 盈利交易占总交易的比例
- **夏普比率**: 风险调整后的收益率
- **最大回撤**: 最大的资金回撤百分比
- **盈利因子**: 总盈利/总亏损的比值

## 性能优化

### 资源限制
- 默认最大组合数: 10,000
- 默认最大线程数: CPU核心数+4
- 最小交易次数要求: 10

### 优化建议
1. 合理设置参数范围，避免组合爆炸
2. 使用参数预设作为起点
3. 根据系统性能调整最大线程数
4. 定期清理优化数据

## 故障排除

### 常见问题

1. **优化启动失败**
   - 检查参数范围是否合理
   - 确认日期格式正确
   - 验证策略ID是否存在

2. **优化速度慢**
   - 减少参数组合数量
   - 缩短回测时间范围
   - 调整线程数设置

3. **内存不足**
   - 减少最大组合数
   - 缩短回测时间范围
   - 及时清理优化数据

### 日志查看
优化过程的详细日志记录在 `logs/service.log` 中。

## 测试

### 快速测试
```bash
python quick_test_optimization.py
```

### API测试
```bash
python test_optimization_api.py
```

### 完整功能测试
```bash
python test_optimization_engine.py
```

## 技术架构

### 核心组件
- **OptimizationEngine**: 主引擎类
- **GridSearchOptimizer**: 网格搜索优化器
- **EvaluationMetrics**: 评估指标计算器
- **ProgressTracker**: 进度跟踪器
- **ResultsManager**: 结果管理器
- **ParameterValidator**: 参数验证器

### 数据流
1. 参数验证 → 2. 数据准备 → 3. 组合生成 → 4. 并行回测 → 5. 结果评估 → 6. 排序输出

## 扩展开发

### 添加新策略
1. 在 `strategies.py` 中实现策略类
2. 在 `get_available_strategies()` 中注册
3. 在 `optimization_engine.py` 中添加参数预设

### 自定义评估指标
1. 继承 `EvaluationMetrics` 类
2. 重写 `calculate_all_metrics()` 方法
3. 调整 `calculate_composite_score()` 权重

### 优化算法扩展
1. 实现新的优化器类
2. 继承 `GridSearchOptimizer` 接口
3. 在 `OptimizationEngine` 中集成

## 许可证

本项目仅供学习和研究使用，不构成投资建议。
