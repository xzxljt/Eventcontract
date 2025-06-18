from typing import Dict, Any, List, Optional, Union

class BaseInvestmentStrategy:
    """投资策略基类"""
    def __init__(self, params: Dict[str, Any], min_amount: float = 5.0, max_amount: float = 250.0):
        self.params = params or {}
        self.name = "基础投资策略"
        self.description = "基础投资策略描述"
        # minAmount 和 maxAmount 应该从 params 中获取，如果存在的话，否则使用默认值
        # 这允许在策略实例化时通过 params 覆盖全局的 min/max
        self.min_amount = float(self.params.get('minAmount', min_amount))
        self.max_amount = float(self.params.get('maxAmount', max_amount))


    def calculate_investment(
        self,
        current_balance: float,
        previous_trade_result: Optional[bool] = None, # True for win, False for loss, None for first trade
        base_investment_from_settings: float = 20.0 # 用户在UI设置的基础金额
    ) -> float:
        """
        计算本次应投资的金额。

        参数:
            current_balance (float): 当前账户余额。
            previous_trade_result (Optional[bool]): 上一次交易的结果。
            base_investment_from_settings (float): 用户在UI设置的基础投资额，用作某些策略的起点。

        返回:
            float: 计算出的投资金额，已应用min/max约束。
        """
        raise NotImplementedError

    def _apply_bounds(self, amount: float) -> float:
        """确保投资金额在最小和最大限制之间"""
        # 四舍五入到最接近的整数，然后应用最小和最大限制
        rounded_amount = round(amount)
        bounded_amount = max(self.min_amount, min(self.max_amount, rounded_amount))
        return bounded_amount

    def reset_state(self):
        """重置策略的内部状态（例如，马丁格尔序列的步骤）"""
        pass # 默认无状态

class FixedAmountStrategy(BaseInvestmentStrategy):
    def __init__(self, params: Dict[str, Any], **kwargs): # kwargs 会捕获 min_amount, max_amount
        super().__init__(params, **kwargs) # 将 params 和 kwargs 都传递给基类
        self.name = "固定金额策略"
        self.description = "每次投资固定金额。"
        self.amount = float(self.params.get('amount', 20.0)) # 从参数或默认值获取固定金额


    def calculate_investment(
        self,
        current_balance: float,
        previous_trade_result: Optional[bool] = None,
        base_investment_from_settings: float = 20.0
    ) -> float:
        # 对于固定金额策略，我们使用其自身参数中的'amount'，而不是base_investment_from_settings
        return self._apply_bounds(self.amount)

class MartingaleStrategy(BaseInvestmentStrategy):
    """
    自定义马丁格尔序列策略。
    参数 'sequence' 是一系列投资金额。
    如果上次输了，则使用序列中的下一个金额；如果赢了或首次交易，则使用序列的第一个金额。
    如果序列结束仍未赢，则重置到序列开始或按参数'reset_on_sequence_end'行动。
    """
    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self.name = "马丁格尔序列策略"
        self.description = "根据预设序列和输赢情况调整投资额。"
        self.sequence: List[Union[int, float]] = self.params.get('sequence', [5, 10, 30, 90, 250])
        if not self.sequence or not all(isinstance(x, (int, float)) and x > 0 for x in self.sequence): # 确保金额大于0
            raise ValueError("Martingale 'sequence' 参数必须是非空正数数字列表。")
        self.reset_on_sequence_end = self.params.get('reset_on_sequence_end', True)
        self.current_step = 0


    def calculate_investment(
        self,
        current_balance: float,
        previous_trade_result: Optional[bool] = None,
        base_investment_from_settings: float = 20.0
    ) -> float:
        if previous_trade_result is None: # 首次交易
            self.current_step = 0
        elif previous_trade_result is True: # 上次赢了
            self.current_step = 0
        elif previous_trade_result is False: # 上次输了
            self.current_step += 1
            if self.current_step >= len(self.sequence):
                if self.reset_on_sequence_end:
                    self.current_step = 0
                else:
                    self.current_step = len(self.sequence) - 1 # 保持在序列最后一个
        
        amount_from_sequence = self.sequence[self.current_step]
        return self._apply_bounds(amount_from_sequence)

    def reset_state(self):
        self.current_step = 0

class AntiMartingaleStrategy(BaseInvestmentStrategy):
    """
    反马丁格尔策略。
    参数 'base_amount' 是基础投资额。
    参数 'multiplier' 是胜利后的乘数。
    参数 'max_streak_increase' 是连续胜利增加投资的最大次数。
    如果上次赢了，则增加投资额；如果输了或首次交易，则使用基础金额。
    """
    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self.name = "反马丁格尔策略"
        self.description = "胜利后增加投资额，失败后恢复基础投资额。"
        self.base_amount = float(self.params.get('base_amount', 20.0))
        self.multiplier = float(self.params.get('multiplier', 1.5))
        self.max_streak_increase = int(self.params.get('max_streak_increase', 3))
        self.win_streak = 0
        self.current_investment = self.base_amount # 初始化 current_investment


    def calculate_investment(
        self,
        current_balance: float,
        previous_trade_result: Optional[bool] = None,
        base_investment_from_settings: float = 20.0 # 对于此策略，通常使用其内部的 base_amount
    ) -> float:
        if previous_trade_result is None:
            self.win_streak = 0
            self.current_investment = self.base_amount
        elif previous_trade_result is True:
            self.win_streak += 1
            if self.win_streak <= self.max_streak_increase: # 只在未达到最大连胜次数时才增加投资
                self.current_investment *= self.multiplier
            # 如果达到或超过最大连胜增加次数，则 current_investment 保持不变（即上一次胜利放大后的金额）
            # 或者也可以设计为：达到最大连胜后，投资额重置为 base_amount 或保持上一次的额度。当前是保持。
        elif previous_trade_result is False:
            self.win_streak = 0
            self.current_investment = self.base_amount
        
        return self._apply_bounds(self.current_investment)

    def reset_state(self):
        self.win_streak = 0
        self.current_investment = self.base_amount

class PercentageOfBalanceStrategy(BaseInvestmentStrategy):
    """
    账户余额百分比策略。
    参数 'percentage' 是投资当前余额的百分比。
    """
    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self.name = "账户百分比策略"
        self.description = "每次投资当前账户余额的固定百分比。"
        # 修正：优先从 'percentage' 键获取值，以匹配文档字符串和可能的调用约定，
        # 同时保留 'percentageOfBalance' 作为备用，以兼容前端UI定义。
        percentage_val = self.params.get('percentage', self.params.get('percentageOfBalance'))
        if percentage_val is None:
            # 如果两个键都不存在，则使用默认值
            percentage_val = 10.0
        
        self.percentage = float(percentage_val)
        if not (0 < self.percentage <= 100):
            raise ValueError("百分比参数 ('percentage' 或 'percentageOfBalance') 必须在 (0, 100] 之间。")


    def calculate_investment(
        self,
        current_balance: float,
        previous_trade_result: Optional[bool] = None,
        base_investment_from_settings: float = 20.0
    ) -> float:
        if current_balance <= 0:
            amount = 0 
        else:
            amount = current_balance * (self.percentage / 100.0)
        
        return self._apply_bounds(amount)

class PercentageStreakMultiplierStrategy(BaseInvestmentStrategy):
    """
    百分比连赢/连亏乘数策略。
    基于账户余额百分比投资，并在连续胜利或失败达到阈值时应用乘数。
    """
    def __init__(self, params: Dict[str, Any], **kwargs):
        super().__init__(params, **kwargs)
        self.name = "百分比连赢/连亏乘数策略"
        self.description = "基于账户余额百分比投资，并在连续胜利或失败达到指定次数后调整投资额。"
        
        self.percentage = float(params.get('percentageOfBalance', 10.0))
        if not (0 < self.percentage <= 100):
            raise ValueError("'percentageOfBalance' 参数必须在 (0, 100] 之间。")

        self.streak_threshold_win = int(params.get('streak_threshold_win', 3))
        self.streak_multiplier_win = float(params.get('streak_multiplier_win', 1.5))
        self.streak_threshold_loss = int(params.get('streak_threshold_loss', 3))
        self.streak_multiplier_loss = float(params.get('streak_multiplier_loss', 0.8))

        if self.streak_threshold_win < 1:
            raise ValueError("'streak_threshold_win' 必须至少为 1。")
        if self.streak_multiplier_win <= 0:
            raise ValueError("'streak_multiplier_win' 必须为正数。")
        if self.streak_threshold_loss < 1:
            raise ValueError("'streak_threshold_loss' 必须至少为 1。")
        if self.streak_multiplier_loss <= 0:
            raise ValueError("'streak_multiplier_loss' 必须为正数。")

        self.win_streak = 0
        self.loss_streak = 0

    def calculate_investment(
        self,
        current_balance: float,
        previous_trade_result: Optional[bool] = None,
        base_investment_from_settings: float = 20.0 
    ) -> float:
        if previous_trade_result == True:
            self.win_streak += 1
            self.loss_streak = 0
        elif previous_trade_result == False:
            self.loss_streak += 1
            self.win_streak = 0
        elif previous_trade_result is None: # 首次交易或状态重置后
            pass # 保持当前的连胜/连败计数

        if current_balance <= 0:
            base_investment = 0.0
        else:
            base_investment = current_balance * (self.percentage / 100.0)
        
        current_investment = base_investment

        if self.win_streak >= self.streak_threshold_win:
            current_investment *= self.streak_multiplier_win
        elif self.loss_streak >= self.streak_threshold_loss: # 使用 elif 确保只有一个乘数生效
            current_investment *= self.streak_multiplier_loss
            
        return self._apply_bounds(current_investment)

    def reset_state(self):
        self.win_streak = 0
        self.loss_streak = 0

def get_available_investment_strategies() -> List[Dict[str, Any]]:
    """获取所有可用投资策略的列表及其元数据"""
    strategies = [
        {
            'id': 'fixed',
            'name': '固定金额',
            'class': FixedAmountStrategy,
            'description': '每次交易投资固定金额。',
            'parameters': [
                {'name': 'amount', 'type': 'float', 'default': 20.0, 'min': 1.0, 'max': 10000.0, 'step': 0.01, 'description': '固定投资金额 (USDT)'},
            ]
        },
        {
            'id': 'martingale_custom_sequence_original', 
            'name': '马丁格尔序列 (经典)',
            'class': MartingaleStrategy,
            'description': '使用 [5, 10, 30, 90, 250] 序列。输则进阶，赢则重置。',
            'parameters': [
                {'name': 'sequence', 'type': 'list_float', 'default': [5, 10, 30, 90, 250], 'description': '投资金额序列', 'readonly': True, 'advanced': True}, 
                {'name': 'reset_on_sequence_end', 'type': 'boolean', 'default': True, 'description': '序列结束后是否重置到初始金额', 'advanced': True}
            ]
        },
        {
            'id': 'martingale_custom_sequence_1', 
            'name': '马丁格尔序列 (自定义1)',
            'class': MartingaleStrategy, 
            'description': '使用 [5, 8, 18, 40, 90, 203] 序列。输则进阶，赢则重置。',
            'parameters': [
                {'name': 'sequence', 'type': 'list_float', 'default': [5, 8, 18, 40, 90, 203], 'description': '投资金额序列', 'readonly': True, 'advanced': True},
                {'name': 'reset_on_sequence_end', 'type': 'boolean', 'default': True, 'description': '序列结束后是否重置到初始金额', 'advanced': True}
            ]
        },
         { 
            'id': 'martingale_user_defined', # 这个在前端可能需要特殊处理参数输入
            'name': '马丁格尔序列 (用户定义)',
            'class': MartingaleStrategy,
            'description': '用户可定义投资序列。输则进阶，赢则重置。',
            'parameters': [
                {'name': 'sequence', 'type': 'list_float', 'default': [10,20,40], 'description': '自定义投资金额序列 (如: 10,20,40)', 'advanced': False, 'editor': 'text_list'},
                {'name': 'reset_on_sequence_end', 'type': 'boolean', 'default': True, 'description': '序列结束后是否重置到初始金额'}
            ]
        },
        {
            'id': 'anti_martingale',
            'name': '反马丁格尔',
            'class': AntiMartingaleStrategy,
            'description': '胜利后增加投资，失败后恢复基础投资。',
            'parameters': [
                {'name': 'base_amount', 'type': 'float', 'default': 20.0, 'min': 1.0, 'max': 1000.0, 'step': 0.01, 'description': '基础投资金额 (USDT)'},
                {'name': 'multiplier', 'type': 'float', 'default': 1.5, 'min': 1.01, 'max': 3.0, 'step': 0.01, 'description': '胜利后的乘数'}, 
                {'name': 'max_streak_increase', 'type': 'int', 'default': 3, 'min':0, 'max':10, 'description':'最大连续增加投资次数 (0表示无限)'}
            ]
        },
        {
            'id': 'percentage_of_balance',
            'name': '账户百分比',
            'class': PercentageOfBalanceStrategy,
            'description': '投资当前账户余额的一定百分比。',
            'parameters': [
                {'name': 'percentageOfBalance', 'type': 'float', 'default': 10.0, 'min': 0.1, 'max': 50.0, 'step': 0.1, 'description': '投资账户余额的百分比 (%)'},
            ]
        },
        {
            'id': 'percentage_streak_multiplier',
            'name': '百分比连赢/连亏乘数',
            'class': PercentageStreakMultiplierStrategy,
            'description': '基于账户余额百分比投资，并在连胜/连败达到指定次数后调整投资额。',
            'parameters': [
                {'name': 'percentageOfBalance', 'type': 'float', 'default': 10.0, 'min': 0.1, 'max': 50.0, 'step': 0.1, 'description': '基础投资占账户余额的百分比 (%)'},
                {'name': 'streak_threshold_win', 'type': 'int', 'default': 3, 'min': 1, 'max': 10, 'description': '触发连胜乘数的连胜次数'},
                {'name': 'streak_multiplier_win', 'type': 'float', 'default': 1.5, 'min': 0.1, 'max': 5.0, 'step': 0.1, 'description': '连胜时的投资乘数'},
                {'name': 'streak_threshold_loss', 'type': 'int', 'default': 3, 'min': 1, 'max': 10, 'description': '触发连败乘数的连败次数'},
                {'name': 'streak_multiplier_loss', 'type': 'float', 'default': 0.8, 'min': 0.1, 'max': 5.0, 'step': 0.1, 'description': '连败时的投资乘数 (例如0.8减少风险, 1.2增加风险)'},
            ]
        },
    ]
    return strategies

if __name__ == '__main__':
    # 测试代码保持不变，用于快速验证策略行为
    print("测试投资策略:")

    print("\n--- 固定金额策略 ---")
    fixed_params_from_global = {'minAmount': 5, 'maxAmount': 100} 
    fixed_strategy_specific_params = {'amount': 25.0} 
    merged_fixed_params = {**fixed_params_from_global, **fixed_strategy_specific_params}
    fixed_strategy = FixedAmountStrategy(params=merged_fixed_params)
    print(f"固定策略: 初始参数 amount={fixed_strategy.amount}, min={fixed_strategy.min_amount}, max={fixed_strategy.max_amount}")
    print(f"  投资(余额1000): {fixed_strategy.calculate_investment(1000)}")
    # Backtester 层面会用 min(balance, calculated_investment)
    # 所以这里测试的是策略本身的计算和_apply_bounds
    print(f"  投资(余额10): {fixed_strategy.calculate_investment(10)}") # 策略计算25, _apply_bounds(25) -> 25 (因为 min_amount 5, max_amount 100)。
                                                                  # Backtester 会再根据余额调整为10.

    print("\n--- 马丁格尔序列策略 (经典) ---")
    martingale_global_params = {'minAmount': 1, 'maxAmount': 50}
    martingale_strategy_orig = MartingaleStrategy(params=martingale_global_params) 
    print(f"马丁格尔(经典): min={martingale_strategy_orig.min_amount}, max={martingale_strategy_orig.max_amount}, seq={martingale_strategy_orig.sequence}")
    print(f"  首次投资(余额1000): {martingale_strategy_orig.calculate_investment(1000, None)}") 
    print(f"  上次输(余额995), 本次: {martingale_strategy_orig.calculate_investment(995, False)}") 
    print(f"  上次输(余额985), 本次: {martingale_strategy_orig.calculate_investment(985, False)}") 
    print(f"  上次赢(余额1015), 本次: {martingale_strategy_orig.calculate_investment(1015, True)}") 
    martingale_strategy_orig.reset_state()
    print(f"  重置后, 上次赢(余额1000), 本次: {martingale_strategy_orig.calculate_investment(1000, True)}") 

    print("\n--- 测试马丁格尔在余额不足时的行为（由_apply_bounds处理） ---")
    martingale_low_balance_params = {'minAmount': 5, 'maxAmount': 15} 
    martingale_lb_strat = MartingaleStrategy(params=martingale_low_balance_params)
    print(f"马丁格尔(低余额约束): min={martingale_lb_strat.min_amount}, max={martingale_lb_strat.max_amount}, seq={martingale_lb_strat.sequence}")
    print(f"  首次(余额100): {martingale_lb_strat.calculate_investment(100, None)}") 
    print(f"  上次输(余额95), 本次: {martingale_lb_strat.calculate_investment(95, False)}") 
    print(f"  上次输(余额85), 本次: {martingale_lb_strat.calculate_investment(85, False)}") 
    print(f"  上次输(余额70), 本次: {martingale_lb_strat.calculate_investment(70, False)}") 
    print(f"  上次赢(余额85), 本次: {martingale_lb_strat.calculate_investment(85, True)}") 


    print("\n--- 反马丁格尔策略 ---")
    anti_m_global_params = {'minAmount': 5, 'maxAmount': 100}
    anti_m_specific_params = {'base_amount': 10, 'multiplier': 2, 'max_streak_increase': 2}
    merged_anti_m_params = {**anti_m_global_params, **anti_m_specific_params}
    anti_martingale_strategy = AntiMartingaleStrategy(params=merged_anti_m_params)
    print(f"反马丁格尔: base={anti_martingale_strategy.base_amount}, mult={anti_martingale_strategy.multiplier}, max_streak={anti_martingale_strategy.max_streak_increase}, min={anti_martingale_strategy.min_amount}, max={anti_martingale_strategy.max_amount}")
    print(f"  首次(余额1000): {anti_martingale_strategy.calculate_investment(1000, None)}") 
    print(f"  上次赢(余额1010), 本次: {anti_martingale_strategy.calculate_investment(1010, True)}") 
    print(f"  上次赢(余额1030), 本次: {anti_martingale_strategy.calculate_investment(1030, True)}") 
    print(f"  上次赢(余额1070, 已达最大连胜), 本次: {anti_martingale_strategy.calculate_investment(1070, True)}") 
    print(f"  上次输(余额1030), 本次: {anti_martingale_strategy.calculate_investment(1030, False)}") 

    print("\n--- 账户百分比策略 ---")
    percent_global_params = {'minAmount': 10, 'maxAmount': 200}
    percent_specific_params = {'percentageOfBalance': 5} 
    merged_percent_params = {**percent_global_params, **percent_specific_params}
    percentage_strategy = PercentageOfBalanceStrategy(params=merged_percent_params)
    print(f"账户百分比: perc={percentage_strategy.percentage}%, min={percentage_strategy.min_amount}, max={percentage_strategy.max_amount}")
    print(f"  余额1000, 投资: {percentage_strategy.calculate_investment(1000)}") 
    print(f"  余额2000, 投资: {percentage_strategy.calculate_investment(2000)}") 
    print(f"  余额100 (计算结果5 < minAmount 10), 投资: {percentage_strategy.calculate_investment(100)}") 
    print(f"  余额5000 (计算结果250 > maxAmount 200), 投资: {percentage_strategy.calculate_investment(5000)}") 
    print(f"  余额0, 投资: {percentage_strategy.calculate_investment(0)}")


    print("\n--- 获取可用策略 ---")
    available_strats = get_available_investment_strategies()
    for s in available_strats:
        print(f"ID: {s['id']}, Name: {s['name']}, Class: {s['class'].__name__}")

    print("\n--- 测试百分比连赢/连亏乘数策略 ---")
    streak_params_global = {'minAmount': 5, 'maxAmount': 500}
    streak_params_specific = {
        'percentageOfBalance': 10.0, # 10%
        'streak_threshold_win': 2,    # 2连胜触发
        'streak_multiplier_win': 1.5, # 赢则投资 *1.5
        'streak_threshold_loss': 3,   # 3连败触发
        'streak_multiplier_loss': 0.7 # 输则投资 *0.7
    }
    merged_streak_params = {**streak_params_global, **streak_params_specific}
    streak_strategy = PercentageStreakMultiplierStrategy(params=merged_streak_params)
    
    print(f"策略参数: perc={streak_strategy.percentage}%, win_thresh={streak_strategy.streak_threshold_win}, win_mult={streak_strategy.streak_multiplier_win}, loss_thresh={streak_strategy.streak_threshold_loss}, loss_mult={streak_strategy.streak_multiplier_loss}, min={streak_strategy.min_amount}, max={streak_strategy.max_amount}")

    balance = 1000.0
    print(f"初始余额: {balance}")

    # 首次交易
    inv = streak_strategy.calculate_investment(balance, None)
    print(f"  首次交易 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: 1000 * 10% = 100)") # 100
    balance -= inv # 假设亏损以便测试连败

    # 第一次失败
    inv = streak_strategy.calculate_investment(balance, False) # loss_streak = 1
    print(f"  上次输 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: 900 * 10% = 90)") # 90
    balance -= inv

    # 第二次失败
    inv = streak_strategy.calculate_investment(balance, False) # loss_streak = 2
    print(f"  上次输 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: 810 * 10% = 81)") # 81
    balance -= inv

    # 第三次失败 (触发连败乘数)
    inv = streak_strategy.calculate_investment(balance, False) # loss_streak = 3
    print(f"  上次输 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: (729 * 10%) * 0.7 = 72.9 * 0.7 = 51.03 -> 51)") # 51
    balance += inv # 假设这次赢了

    # 第一次胜利 (重置连败，win_streak = 1)
    inv = streak_strategy.calculate_investment(balance, True) # win_streak = 1, loss_streak = 0
    print(f"  上次赢 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: (729+51) * 10% = 780 * 10% = 78)") # 78
    balance += inv # 假设又赢了

    # 第二次胜利 (触发连胜乘数)
    inv = streak_strategy.calculate_investment(balance, True) # win_streak = 2
    print(f"  上次赢 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: ((780+78) * 10%) * 1.5 = (858 * 10%) * 1.5 = 85.8 * 1.5 = 128.7 -> 129)") # 129
    balance -= inv # 假设这次输了

    # 再次失败 (重置连胜, loss_streak = 1)
    inv = streak_strategy.calculate_investment(balance, False) # loss_streak = 1, win_streak = 0
    print(f"  上次输 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: (858-129) * 10% = 729 * 10% = 72.9 -> 73)") # 73

    streak_strategy.reset_state()
    print(f"  状态重置后, win_streak={streak_strategy.win_streak}, loss_streak={streak_strategy.loss_streak}")
    inv = streak_strategy.calculate_investment(balance, None)
    print(f"  重置后首次交易 (余额 {balance:.2f}), 投资: {inv:.2f} (预期: 729 * 10% = 72.9 -> 73)")