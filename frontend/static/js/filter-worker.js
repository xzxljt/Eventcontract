// frontend/static/js/filter-worker.js

self.onmessage = function(event) {
    const { results: originalResults, filters } = event.data;

    if (!originalResults) {
        self.postMessage(null);
        return;
    }

    const { filterStartTime, filterEndTime, minConfidence, excludedWeekdays } = filters;
    const predictions = originalResults.predictions || [];

    // 1. Filter predictions
    const filteredPredictions = predictions.filter(p => {
        // Confidence filter
        if (p.confidence < minConfidence) {
            return false;
        }

        // Time filter
        if (filterStartTime && filterEndTime) {
            const signalTime = p.signal_time.substring(11, 16);
            const isInTimeRange = (filterStartTime <= filterEndTime)
                ? (signalTime >= filterStartTime && signalTime <= filterEndTime) // Normal case
                : (signalTime >= filterStartTime || signalTime <= filterEndTime); // Cross-day case
            if (!isInTimeRange) {
                return false;
            }
        }

        // Weekday filter
        if (excludedWeekdays && excludedWeekdays.length > 0) {
            const signalDay = new Date(p.signal_time).getDay().toString();
            if (excludedWeekdays.includes(signalDay)) {
                return false;
            }
        }

        return true;
    });

    // 2. Recalculate everything based on filtered predictions
    const initialBalance = originalResults.initial_balance;
    let currentBalance = initialBalance;
    const dailyPnl = {};
    let totalPnlAmount = 0;
    let totalWins = 0;
    let longPredictions = 0;
    let longWins = 0;
    let shortPredictions = 0;
    let shortWins = 0;
    let totalGrossProfit = 0;
    let totalGrossLoss = 0;
    let maxBalance = initialBalance;
    let maxDrawdown = 0;
    let consecutiveWins = 0;
    let consecutiveLosses = 0;
    let maxConsecutiveWins = 0;
    let maxConsecutiveLosses = 0;

    // Sort predictions by time to ensure correct balance calculation
    filteredPredictions.sort((a, b) => new Date(a.signal_time) - new Date(b.signal_time));

    const recalculatedPredictions = [];

    for (const pred of filteredPredictions) {
        const dateStr = pred.signal_time.split('T')[0];
        if (!dailyPnl[dateStr]) {
            dailyPnl[dateStr] = { pnl: 0, trades: 0, balance: 0, daily_return_pct: 0, start_balance_of_day: currentBalance };
        }
        
        const pnl = pred.pnl_amount || 0;
        const isWin = pnl > 0; // Recalculate result as a boolean
        // Calculate price change percentage correctly based on trade side
        let pnlPercentage = 0;
        if (pred.signal_price !== 0 && pred.end_price !== undefined && pred.end_price !== null) {
            if (pred.side === 'BUY' || pred.side === 'LONG') {
                pnlPercentage = ((pred.end_price - pred.signal_price) / pred.signal_price) * 100;
            } else if (pred.side === 'SELL' || pred.side === 'SHORT') {
                pnlPercentage = ((pred.signal_price - pred.end_price) / pred.signal_price) * 100;
            }
        }
        // Normalize confidence from (e.g. 85) to (e.g. 0.85)
        const normalizedConfidence = pred.confidence > 1 ? pred.confidence / 100 : pred.confidence;

        totalPnlAmount += pnl;
        currentBalance += pnl;

        dailyPnl[dateStr].pnl += pnl;
        dailyPnl[dateStr].trades += 1;
        dailyPnl[dateStr].balance = currentBalance;

        if (isWin) {
            totalWins++;
            consecutiveWins++;
            consecutiveLosses = 0;
            maxConsecutiveWins = Math.max(maxConsecutiveWins, consecutiveWins);
            totalGrossProfit += pnl;
        } else { // loss
            consecutiveLosses++;
            consecutiveWins = 0;
            maxConsecutiveLosses = Math.max(maxConsecutiveLosses, consecutiveLosses);
            totalGrossLoss += Math.abs(pnl);
        }

        if (pred.signal === 1) {
            longPredictions++;
            if (isWin) longWins++;
        } else if (pred.signal === -1) {
            shortPredictions++;
            if (isWin) shortWins++;
        }

        maxBalance = Math.max(maxBalance, currentBalance);
        const drawdown = (maxBalance - currentBalance) / maxBalance;
        maxDrawdown = Math.max(maxDrawdown, drawdown);

        recalculatedPredictions.push({
            ...pred,
            pnl_amount: pnl,
            pnl_percentage: pnlPercentage,
            confidence: normalizedConfidence, // Use normalized confidence
            result: isWin, // Use recalculated boolean result
            balance_after_trade: currentBalance,
            final_balance: currentBalance
        });
    }

    for (const dateStr in dailyPnl) {
        const dayData = dailyPnl[dateStr];
        if (dayData.start_balance_of_day > 0) {
            dayData.daily_return_pct = (dayData.pnl / dayData.start_balance_of_day) * 100;
        }
    }

    const finalBalance = currentBalance;
    const roiPercentage = initialBalance > 0 ? (totalPnlAmount / initialBalance) * 100 : 0;
    const profitFactor = totalGrossLoss > 0 ? (totalGrossProfit / totalGrossLoss).toFixed(2) : '∞';

    const finalResult = {
        ...originalResults,
        predictions: recalculatedPredictions,
        total_predictions: recalculatedPredictions.length,
        total_wins: totalWins,
        win_rate: recalculatedPredictions.length > 0 ? (totalWins / recalculatedPredictions.length) * 100 : 0,
        long_predictions: longPredictions,
        long_win_rate: longPredictions > 0 ? (longWins / longPredictions) * 100 : 0,
        short_predictions: shortPredictions,
        short_win_rate: shortPredictions > 0 ? (shortWins / shortPredictions) * 100 : 0,
        initial_balance: initialBalance,
        final_balance: finalBalance,
        total_pnl_amount: totalPnlAmount,
        roi_percentage: roiPercentage,
        max_drawdown_percentage: maxDrawdown * 100,
        profit_factor: profitFactor,
        max_consecutive_wins: maxConsecutiveWins,
        max_consecutive_losses: maxConsecutiveLosses,
        daily_pnl: dailyPnl,
    };

    self.postMessage(finalResult);
};