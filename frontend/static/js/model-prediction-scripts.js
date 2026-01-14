let ws = null;
let trainingTaskId = null;
let predictionChart = null;
let performanceChart = null;

function generateTaskId() {
    return 'task_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
}

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        return;
    }
    
    trainingTaskId = generateTaskId();
    const wsUrl = `ws://${window.location.host}/ws/training?task_id=${trainingTaskId}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('WebSocket连接已打开');
        document.getElementById('training-status-text').textContent = '就绪';
        document.getElementById('model-status').innerHTML = '<span class="model-status-indicator status-ready"></span>就绪';
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateTrainingStatus(data);
    };
    
    ws.onclose = function() {
        console.log('WebSocket连接已关闭');
        document.getElementById('training-status-text').textContent = '断开连接';
        document.getElementById('model-status').innerHTML = '<span class="model-status-indicator status-error"></span>断开连接';
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket错误:', error);
        document.getElementById('training-status-text').textContent = '连接错误';
        document.getElementById('model-status').innerHTML = '<span class="model-status-indicator status-error"></span>连接错误';
    };
}

function updateTrainingStatus(data) {
    if (data.progress !== undefined) {
        document.getElementById('progress-fill').style.width = `${data.progress}%`;
        document.getElementById('progress-percent').textContent = `${data.progress}%`;
    }
    
    if (data.stage) {
        document.getElementById('current-stage').textContent = data.stage;
    }
    
    if (data.status) {
        document.getElementById('training-status-text').textContent = data.status;
        
        // 更新模型状态指示器
        const modelStatus = document.getElementById('model-status');
        if (data.status === '训练中') {
            modelStatus.innerHTML = '<span class="model-status-indicator status-training"></span>训练中';
        } else if (data.status === '训练完成') {
            modelStatus.innerHTML = '<span class="model-status-indicator status-ready"></span>就绪';
        } else if (data.status === '训练失败') {
            modelStatus.innerHTML = '<span class="model-status-indicator status-error"></span>错误';
        }
    }
    
    if (data.time_remaining) {
        document.getElementById('estimated-time').textContent = data.time_remaining;
    }
    
    if (data.error) {
        alert('训练错误: ' + data.error);
        document.getElementById('model-status').innerHTML = '<span class="model-status-indicator status-error"></span>错误';
    }
    
    if (data.log) {
        const trainingLog = document.getElementById('training-log');
        const logEntry = document.createElement('p');
        logEntry.textContent = data.log;
        trainingLog.appendChild(logEntry);
        trainingLog.scrollTop = trainingLog.scrollHeight;
    }
}

function getFormParams() {
    const modelSystem = document.getElementById('model-system').value;
    const params = {
        model_system: modelSystem,
        scaler_type: document.getElementById('scaler-type').value,
        use_feature_selection: document.getElementById('feature-selection').checked,
        n_features: parseInt(document.getElementById('n-features').value),
        strategy_mode: document.getElementById('strategy-mode').value,
        adaptive_strategy: document.getElementById('adaptive-strategy').checked
    };
    
    if (modelSystem === 'single') {
        params.model_type = document.getElementById('single-model-type').value;
    } else {
        params.model_count = parseInt(document.getElementById('model-count').value);
        params.model_types = [];
        if (document.getElementById('include-rf').checked) params.model_types.push('random_forest');
        if (document.getElementById('include-xgb').checked) params.model_types.push('xgboost');
    }
    
    return params;
}

function startTraining() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        setTimeout(startTraining, 1000);
        return;
    }
    
    const params = getFormParams();
    
    const trainingData = {
        action: 'start_training',
        params: {
            model_count: params.model_system === 'ensemble' ? params.model_count : 1,
            model_types: params.model_system === 'ensemble' ? params.model_types.join(',') : '',
            model_params: {
                model_type: params.model_system === 'single' ? params.model_type : 'random_forest'
            },
            preprocessor_params: {
                scaler_type: params.scaler_type,
                use_feature_selection: params.use_feature_selection,
                n_features: params.n_features
            },
            strategy_mode: params.strategy_mode,
            adaptive_strategy: params.adaptive_strategy
        }
    };
    
    ws.send(JSON.stringify(trainingData));
    document.getElementById('training-status-text').textContent = '训练中';
    document.getElementById('model-status').innerHTML = '<span class="model-status-indicator status-training"></span>训练中';
    
    // 清空训练日志
    const trainingLog = document.getElementById('training-log');
    trainingLog.innerHTML = '<p>开始训练...</p>';
}

function stopTraining() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    
    const stopData = {
        action: 'stop_training'
    };
    
    ws.send(JSON.stringify(stopData));
    document.getElementById('training-status-text').textContent = '停止中';
}

function runPrediction() {
    fetch('/api/prediction/run', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayPredictionResults(data.data);
            updatePredictionChart(data.data);
            updatePerformanceChart(data.data);
            updateFeatureImportance(data.data);
            updateModelStatus(data.data);
        } else {
            alert('预测失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('预测错误:', error);
        alert('预测请求失败');
    });
}

function clearCache() {
    fetch('/api/prediction/clear-cache', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('缓存已清除');
        } else {
            alert('清除缓存失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('清除缓存错误:', error);
        alert('清除缓存请求失败');
    });
}

function loadModel() {
    fetch('/api/prediction/load-model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('模型加载成功');
            updateModelStatus(data.data);
        } else {
            alert('模型加载失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('加载模型错误:', error);
        alert('加载模型请求失败');
    });
}

function displayPredictionResults(results) {
    const resultsContainer = document.getElementById('prediction-results');
    
    if (!results) {
        resultsContainer.innerHTML = '<div class="results-placeholder"><p>预测结果为空</p></div>';
        return;
    }
    
    let html = '';
    
    // 智能组合预测结果
    if (results.smart_combined_prediction) {
        const smartPred = results.smart_combined_prediction;
        html += '<div class="mb-4">';
        html += '<h3>智能组合预测</h3>';
        html += '<div class="prediction-stats">';
        html += '<div class="stat-card">';
        html += '<div class="stat-value">' + (smartPred.final_signal === 1 ? '买入' : smartPred.final_signal === -1 ? '卖出' : '持有') + '</div>';
        html += '<div class="stat-label">最终信号</div>';
        html += '</div>';
        html += '<div class="stat-card">';
        html += '<div class="stat-value">' + (smartPred.final_confidence * 100).toFixed(1) + '%</div>';
        html += '<div class="stat-label">置信度</div>';
        html += '</div>';
        html += '<div class="stat-card">';
        html += '<div class="stat-value">' + (smartPred.consistency * 100).toFixed(1) + '%</div>';
        html += '<div class="stat-label">模型一致性</div>';
        html += '</div>';
        html += '</div>';
        
        if (smartPred.market_conditions) {
            html += '<h4>市场条件</h4>';
            html += '<div class="market-conditions">';
            for (const [key, value] of Object.entries(smartPred.market_conditions)) {
                html += '<p><strong>' + key + ':</strong> ' + value.toFixed(4) + '</p>';
            }
            html += '</div>';
        }
        html += '</div>';
    }
    
    // 机器学习预测
    if (results.ml_predictions) {
        html += '<div class="mb-4">';
        html += '<h3>机器学习模型预测</h3>';
        html += '<div class="table-responsive">';
        html += '<table class="table table-sm">';
        html += '<thead><tr><th>模型</th><th>信号</th><th>置信度</th></tr></thead>';
        html += '<tbody>';
        results.ml_predictions.forEach((pred, index) => {
            html += '<tr>';
            html += '<td>' + pred[0] + '</td>';
            html += '<td>' + (pred[1] === 1 ? '买入' : pred[1] === -1 ? '卖出' : '持有') + '</td>';
            html += '<td>' + (pred[2] * 100).toFixed(1) + '%</td>';
            html += '</tr>';
        });
        html += '</tbody>';
        html += '</table>';
        html += '</div>';
        html += '</div>';
    }
    
    // 技术指标预测
    if (results.technical_predictions) {
        html += '<div class="mb-4">';
        html += '<h3>技术指标预测</h3>';
        html += '<div class="table-responsive">';
        html += '<table class="table table-sm">';
        html += '<thead><tr><th>指标</th><th>信号</th><th>置信度</th></tr></thead>';
        html += '<tbody>';
        results.technical_predictions.forEach((pred, index) => {
            html += '<tr>';
            html += '<td>' + pred[0] + '</td>';
            html += '<td>' + (pred[1] === 1 ? '买入' : pred[1] === -1 ? '卖出' : '持有') + '</td>';
            html += '<td>' + (pred[2] * 100).toFixed(1) + '%</td>';
            html += '</tr>';
        });
        html += '</tbody>';
        html += '</table>';
        html += '</div>';
        html += '</div>';
    }
    
    // 模型信息
    if (results.model_info) {
        html += '<div class="mb-4">';
        html += '<h3>模型信息</h3>';
        html += '<div class="model-info">';
        if (results.use_ensemble) {
            html += '<p><strong>模型类型:</strong> 集成模型</p>';
            if (results.model_info.ensemble_info) {
                html += '<p><strong>模型数量:</strong> ' + results.model_info.ensemble_info.model_count + '</p>';
                html += '<p><strong>模型类型:</strong> ' + results.model_info.ensemble_info.model_types.join(', ') + '</p>';
                if (results.model_info.ensemble_info.metrics) {
                    html += '<p><strong>MSE:</strong> ' + results.model_info.ensemble_info.metrics.mse.toFixed(4) + '</p>';
                    html += '<p><strong>RMSE:</strong> ' + results.model_info.ensemble_info.metrics.rmse.toFixed(4) + '</p>';
                    html += '<p><strong>R²:</strong> ' + results.model_info.ensemble_info.metrics.r2.toFixed(4) + '</p>';
                }
            }
        } else {
            html += '<p><strong>模型类型:</strong> ' + results.model_info.model_type + '</p>';
            if (results.model_info.metrics) {
                html += '<p><strong>MSE:</strong> ' + (results.model_info.metrics.mse ? results.model_info.metrics.mse.toFixed(4) : 'N/A') + '</p>';
                html += '<p><strong>RMSE:</strong> ' + (results.model_info.metrics.rmse ? results.model_info.metrics.rmse.toFixed(4) : 'N/A') + '</p>';
                html += '<p><strong>R²:</strong> ' + (results.model_info.metrics.r2 ? results.model_info.metrics.r2.toFixed(4) : 'N/A') + '</p>';
            }
        }
        html += '</div>';
        html += '</div>';
    }
    
    resultsContainer.innerHTML = html;
}

function updatePredictionChart(results) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    const labels = Array.from({length: 10}, (_, i) => `Point ${i+1}`);
    
    let datasets = [];
    
    if (results.comparison && results.comparison.actual_values) {
        datasets.push({
            label: '实际值',
            data: results.comparison.actual_values.slice(0, 10),
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderWidth: 2
        });
    }
    
    if (results.predictions) {
        datasets.push({
            label: '预测值',
            data: results.predictions.slice(0, 10),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderWidth: 2,
            borderDash: [5, 5]
        });
    }
    
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: '价格'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '数据点'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '预测值与实际值对比'
                }
            }
        }
    });
}

function updatePerformanceChart(results) {
    const ctx = document.getElementById('performance-chart').getContext('2d');
    
    // 销毁现有图表
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    let datasets = [];
    let labels = [];
    
    // 准备模型性能数据
    if (results.model_info) {
        if (results.use_ensemble && results.model_info.model_infos) {
            labels = results.model_info.model_infos.map(info => info.info.model_type);
            const mseData = results.model_info.model_infos.map(info => info.info.metrics.mse);
            const r2Data = results.model_info.model_infos.map(info => info.info.metrics.r2);
            
            datasets = [
                {
                    label: 'MSE',
                    data: mseData,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1
                },
                {
                    label: 'R²',
                    data: r2Data,
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1
                }
            ];
        } else if (!results.use_ensemble) {
            labels = ['当前模型'];
            datasets = [
                {
                    label: 'MSE',
                    data: [results.model_info.metrics ? results.model_info.metrics.mse : 0],
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgb(255, 99, 132)',
                    borderWidth: 1
                },
                {
                    label: 'R²',
                    data: [results.model_info.metrics ? results.model_info.metrics.r2 : 0],
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1
                }
            ];
        }
    }
    
    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '性能指标'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模型'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '模型性能对比'
                }
            }
        }
    });
}

function updateFeatureImportance(results) {
    const featureImportanceContainer = document.getElementById('feature-importance');
    
    if (!results.feature_importance) {
        featureImportanceContainer.innerHTML = '<div class="text-center text-muted py-3"><p>特征重要性将显示在这里...</p></div>';
        return;
    }
    
    // 按重要性排序
    const sortedFeatures = Object.entries(results.feature_importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // 只显示前10个特征
    
    let html = '<div class="feature-importance-list">';
    sortedFeatures.forEach(([feature, importance]) => {
        const importancePercent = (importance * 100).toFixed(1);
        html += '<div class="mb-2">';
        html += '<div class="d-flex justify-content-between mb-1">';
        html += '<span>' + feature + '</span>';
        html += '<span>' + importancePercent + '%</span>';
        html += '</div>';
        html += '<div class="feature-importance-bar" style="width: ' + importancePercent + '%"></div>';
        html += '</div>';
    });
    html += '</div>';
    
    featureImportanceContainer.innerHTML = html;
}

function updateModelStatus(data) {
    // 更新模型状态
    if (data) {
        if (data.model_info) {
            if (data.use_ensemble) {
                document.getElementById('current-model-type').textContent = '集成模型';
            } else {
                document.getElementById('current-model-type').textContent = data.model_info.model_type;
            }
            
            if (data.model_info.metrics) {
                const metrics = data.use_ensemble ? data.model_info.ensemble_info.metrics : data.model_info.metrics;
                if (metrics) {
                    document.getElementById('model-performance').textContent = 'MSE: ' + metrics.mse.toFixed(4) + ', R²: ' + metrics.r2.toFixed(4);
                }
            }
        }
        
        if (data.timestamp) {
            const date = new Date(data.timestamp * 1000);
            document.getElementById('last-training-time').textContent = date.toLocaleString();
        }
        
        if (data.market_conditions) {
            document.getElementById('market-trend').textContent = data.market_conditions.trend_strength > 0.5 ? '上升趋势' : data.market_conditions.trend_strength < -0.5 ? '下降趋势' : '横盘';
            document.getElementById('market-volatility').textContent = data.market_conditions.volatility > 0.5 ? '高' : data.market_conditions.volatility < 0.3 ? '低' : '中等';
            document.getElementById('market-volume').textContent = data.market_conditions.volume_strength > 0.5 ? '高' : data.market_conditions.volume_strength < 0.3 ? '低' : '正常';
        }
        
        if (data.smart_combined_prediction) {
            document.getElementById('model-consistency').textContent = '高 (' + (data.smart_combined_prediction.consistency * 100).toFixed(1) + '%)';
            document.getElementById('prediction-confidence').textContent = '中等 (' + (data.smart_combined_prediction.final_confidence * 100).toFixed(1) + '%)';
            document.getElementById('market-signal').textContent = data.smart_combined_prediction.final_signal === 1 ? '买入信号' : data.smart_combined_prediction.final_signal === -1 ? '卖出信号' : '持有信号';
        }
        
        document.getElementById('current-strategy').textContent = '平衡型';
        document.getElementById('strategy-adjustment').textContent = '自适应开启';
    }
}

function setupEventListeners() {
    document.getElementById('start-training').addEventListener('click', startTraining);
    document.getElementById('stop-training').addEventListener('click', stopTraining);
    document.getElementById('run-prediction').addEventListener('click', runPrediction);
    document.getElementById('clear-cache').addEventListener('click', clearCache);
    document.getElementById('load-model').addEventListener('click', loadModel);
    
    // 模型系统选择事件
    document.getElementById('model-system').addEventListener('change', function() {
        const system = this.value;
        if (system === 'single') {
            document.getElementById('single-model-config').style.display = 'block';
            document.getElementById('ensemble-model-config').style.display = 'none';
        } else {
            document.getElementById('single-model-config').style.display = 'none';
            document.getElementById('ensemble-model-config').style.display = 'block';
        }
    });
}

// 页面加载完成后初始化
window.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    connectWebSocket();
    
    // 初始化图表
    const predictionCtx = document.getElementById('prediction-chart').getContext('2d');
    predictionChart = new Chart(predictionCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: '价格'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '数据点'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '预测值与实际值对比'
                }
            }
        }
    });
    
    const performanceCtx = document.getElementById('performance-chart').getContext('2d');
    performanceChart = new Chart(performanceCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '性能指标'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模型'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '模型性能对比'
                }
            }
        }
    });
    
    // 初始化模型状态
    updateModelStatus();
});

// 页面关闭时清理WebSocket连接
window.addEventListener('beforeunload', function() {
    if (ws) {
        ws.close();
    }
});