# 修复前端“固定金额”策略参数加载问题的计划

## 问题描述

在前端实时监控页面，当加载一个已有的、使用“固定金额”投资策略的活动配置时，投资金额（`amount`）没有被正确地填充到UI的输入框中，导致显示的仍然是默认值。

## 根本原因分析

1.  **UI数据绑定**: “固定金额”策略的金额输入框，其值绑定在 `investmentStrategyParams.value.amount`。
2.  **配置加载逻辑**: 在 `frontend/static/js/live-test-scripts.js` 的 `populateUiFromConfigDetails` 函数中，从服务器配置 (`invSettingsFromServer`) 中读取的 `amount` 值被赋给了 `monitorSettings.value.investment.amount`。
3.  **逻辑断层**: 缺少一个关键步骤，即将 `invSettingsFromServer.amount` 的值赋给 `investmentStrategyParams.value.amount`。数据在传递过程中中断了。

## 修复方案

我们将通过修改 `frontend/static/js/live-test-scripts.js` 文件来解决这个问题。

### 修改步骤

1.  **文件**: [`frontend/static/js/live-test-scripts.js`](frontend/static/js/live-test-scripts.js)
2.  **函数**: `populateUiFromConfigDetails`
3.  **定位**: 在以下代码行之后：
    ```javascript
    updateInvestmentStrategyParams(invStrategy); // Use the new reusable function
    ```
4.  **操作**: 插入一段新的逻辑，专门处理“固定金额”策略的 `amount` 参数。

    ```javascript
    // 之前的代码...
    updateInvestmentStrategyParams(invStrategy); // Use the new reusable function
    
    // --- 新增修复逻辑 START ---
    // 特别处理：如果策略是“固定金额”，需要将顶层的 amount
    // 同步到策略参数中，以正确填充UI输入框。
    if (invStrategy.id === 'fixed' && invSettingsFromServer.amount !== undefined) {
        investmentStrategyParams.value.amount = invSettingsFromServer.amount;
    }
    // --- 新增修复逻辑 END ---

    // Now, override with any specific params from the server config
    const paramsFromServer = invSettingsFromServer.investment_strategy_specific_params || {};
    investmentStrategyParams.value = { ...investmentStrategyParams.value, ...paramsFromServer };
    // ... 之后的代码
    ```

### 预期结果

修复后，当加载任何包含“固定金额”策略的配置时，正确的投资金额应该能够被无误地填充到UI的相应输入框中。