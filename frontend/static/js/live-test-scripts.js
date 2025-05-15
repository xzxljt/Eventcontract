// frontend/static/js/live-test-scripts.js

const { createApp, ref, onMounted, onUnmounted, computed, watch, getCurrentInstance } = Vue;

const app = createApp({
    setup() {
        // --- 从 utils.js 引入或准备使用的函数 ---
        const utilFormatDateTime = typeof formatDateTime === 'function' ? formatDateTime : (val) => val || '-';
        const utilGetWinRateClass = typeof getWinRateClass === 'function' ? getWinRateClass : () => '';
        const utilGetPnlClass = typeof getPnlClass === 'function' ? getPnlClass : () => '';
        const utilGetSignalStatusClass = typeof getSignalStatusClass === 'function' ? getSignalStatusClass : () => '';
        const utilFormatDateForInput = typeof formatDateForInput === 'function' ? formatDateForInput : (d) => d.toISOString().slice(0,16);

        // --- State Variables ---
        const symbols = ref([]);
        const favoriteSymbols = ref([]); // 待提取到 useFavoriteSymbols

        const predictionStrategies = ref([]);
        const selectedPredictionStrategy = ref(null); // 整个策略对象
        const predictionStrategyParams = ref({}); // 策略的参数键值对

        const investmentStrategies = ref([]);
        const selectedInvestmentStrategy = ref(null); // 整个策略对象
        const investmentStrategyParams = ref({}); // 策略的特定参数键值对

        const allSavedParams = ref({ // 从后端加载的所有已保存策略参数
            prediction_strategies: {},
            investment_strategies: {}
        });

        const liveSignals = ref([]); // 所有接收到的信号
        const loadingInitialData = ref(false); // 控制初始加载动画
        const error = ref(null); // 通用错误消息
        const serverMessage = ref(''); // 通用服务器消息提示
        let serverMessageTimer = null;

        // WebSocket related
        const socket = ref(null);
        const socketStatus = ref('disconnected'); // 'disconnected', 'connecting', 'connected', 'error'
        const currentConfigId = ref(null); // 当前活动的后端配置ID

        // Button loading states
        const applyingConfig = ref(false); 
        const stoppingTest = ref(false);

        // Live Stats
        const stats = ref({
            total_signals: 0, total_verified: 0, total_correct: 0,
            win_rate: 0, total_pnl_pct: 0, average_pnl_pct: 0,
            total_profit_amount: 0
        });

        // Frontend Monitor Settings (UI Bound)
        const monitorSettings = ref({
            symbol: 'all', // 'all' 或特定交易对
            interval: 'all', // 'all' 或特定K线周期
            // prediction_strategy_id 由 selectedPredictionStrategy 动态设置
            confidence_threshold: 50,
            event_period: '10m',
            enableSound: false,
            investment: { // 这些是全局的投资设定，会被策略参数覆盖或补充
                amount: 20.0, // 默认基础投资额，用于'fixed'等策略
                // strategy_id 由 selectedInvestmentStrategy 动态设置
                minAmount: 5.0,
                maxAmount: 250.0,
                percentageOfBalance: 10.0, // 用于'percentage_of_balance'策略
                profitRate: 80.0,
                lossRate: 100.0,
                simulatedBalance: 1000.0, // 用于'percentage_of_balance'策略
            }
        });

        // --- Computed Properties ---
        const latestSignals = computed(() => {
            if (!Array.isArray(liveSignals.value)) return [];
            // 保持原样，按时间倒序取前5条
            return [...liveSignals.value]
                .sort((a, b) => {
                    const timeA = a.signal_time ? new Date(a.signal_time).getTime() : 0;
                    const timeB = b.signal_time ? new Date(b.signal_time).getTime() : 0;
                    if (isNaN(timeA) || isNaN(timeB)) return 0; // 处理无效日期
                    return timeB - timeA;
                })
                .slice(0, 5); // 显示最近的5条信号
        });

        const sortedSymbols = computed(() => { // 用于监控交易对下拉列表
            return [...symbols.value].sort((a, b) => {
                const aIsFav = isFavorite(a);
                const bIsFav = isFavorite(b);
                if (aIsFav && !bIsFav) return -1;
                if (!aIsFav && bIsFav) return 1;
                return a.localeCompare(b); // 非收藏的按字母排序
            });
        });

        // --- Server Message Handling ---
        const showServerMessage = (message, isError = false, duration = 5000) => {
            serverMessage.value = message;
            if (isError) error.value = message; // 同时更新 error ref 如果是错误
            else error.value = null;
            
            if (serverMessageTimer) clearTimeout(serverMessageTimer);
            serverMessageTimer = setTimeout(() => {
                serverMessage.value = '';
                if (isError) error.value = null;
            }, duration);
        };

        // --- Favorite Symbols Logic (待提取到 useFavoriteSymbols) ---
        const loadFavorites = () => {
            const stored = localStorage.getItem('favoriteSymbols');
            if (stored) favoriteSymbols.value = JSON.parse(stored);
        };
        const saveFavorites = () => localStorage.setItem('favoriteSymbols', JSON.stringify(favoriteSymbols.value));
        const isFavorite = (symbol) => favoriteSymbols.value.includes(symbol);
        const toggleFavorite = (symbol) => {
            if (!symbol || symbol === 'all') return;
            const index = favoriteSymbols.value.indexOf(symbol);
            if (index === -1) favoriteSymbols.value.push(symbol);
            else favoriteSymbols.value.splice(index, 1);
            saveFavorites();
            // sortSymbols() 会在 sortedSymbols computed 属性中自动处理
        };

        // --- Initialization and Data Fetching (部分待提取到 apiService) ---
        const fetchAllSavedParameters = async () => {
            try {
                const response = await axios.get('/api/load_all_strategy_parameters');
                allSavedParams.value = response.data || { prediction_strategies: {}, investment_strategies: {} };
            } catch (err) {
                console.error('LiveTest: Failed to fetch all saved parameters:', err);
                showServerMessage('加载已保存的策略参数失败。', true);
                allSavedParams.value = { prediction_strategies: {}, investment_strategies: {} };
            }
        };
        const fetchSymbols = async () => {
            try {
                const response = await axios.get('/api/symbols');
                symbols.value = response.data;
                // 默认选择 BTCUSDT 或第一个，如果 'all' 不是当前选项
                if (monitorSettings.value.symbol !== 'all' && symbols.value.length > 0) {
                     if (!symbols.value.includes(monitorSettings.value.symbol)) {
                        monitorSettings.value.symbol = symbols.value.includes('BTCUSDT') ? 'BTCUSDT' : symbols.value[0];
                     }
                }
            } catch (err) { console.error('获取交易对失败:', err); showServerMessage('获取交易对失败', true); }
        };
        const fetchPredictionStrategies = async () => {
            try {
                const response = await axios.get('/api/prediction-strategies');
                predictionStrategies.value = response.data;
            } catch (err) { console.error('获取预测策略失败:', err); showServerMessage('获取预测策略失败', true); }
        };
        const fetchInvestmentStrategies = async () => {
            try {
                const response = await axios.get('/api/investment-strategies');
                investmentStrategies.value = response.data;
            } catch (err) { console.error('获取投资策略失败:', err); showServerMessage('获取投资策略失败', true); }
        };
        
        // --- UI Population from Server Config ---
        const populateUiFromConfigDetails = (configDetails) => {
            if (!configDetails) return;

            monitorSettings.value.symbol = configDetails.symbol || 'all';
            monitorSettings.value.interval = configDetails.interval || 'all';
            monitorSettings.value.confidence_threshold = configDetails.confidence_threshold ?? 50; // 使用 ?? 保留 0
            monitorSettings.value.event_period = configDetails.event_period || '10m';

            // 设置预测策略
            const predStrategy = predictionStrategies.value.find(s => s.id === configDetails.prediction_strategy_id);
            if (predStrategy) {
                selectedPredictionStrategy.value = predStrategy; // 这会触发 watch 来填充参数 (默认+全局保存)
                // 然后用 configDetails 中的参数覆盖
                if (configDetails.prediction_strategy_params) {
                    const paramsFromServer = { ...configDetails.prediction_strategy_params };
                    // 合并：(默认+全局保存) -> 服务器活动配置参数 (服务器活动配置覆盖前者)
                    // 确保只覆盖策略定义中存在的非高级参数
                    const currentDefaultsAndSaved = { ...predictionStrategyParams.value };
                    const finalParams = { ...currentDefaultsAndSaved };
                    if (predStrategy.parameters) {
                        predStrategy.parameters.forEach(pDef => {
                            if (!pDef.advanced && paramsFromServer.hasOwnProperty(pDef.name)) {
                                let val = paramsFromServer[pDef.name];
                                if (pDef.type === 'int') val = parseInt(val, 10);
                                else if (pDef.type === 'float') val = parseFloat(val);
                                else if (pDef.type === 'boolean') val = (val === true || String(val).toLowerCase() === 'true');
                                finalParams[pDef.name] = val;
                            }
                        });
                    }
                    predictionStrategyParams.value = finalParams;
                }
            } else {
                selectedPredictionStrategy.value = null; // 清空选择和参数
            }
            
            // 设置投资策略及参数
            if (configDetails.investment_settings) {
                const invSettingsFromServer = configDetails.investment_settings;
                // 更新全局投资设置
                monitorSettings.value.investment.amount = invSettingsFromServer.amount ?? monitorSettings.value.investment.amount;
                monitorSettings.value.investment.minAmount = invSettingsFromServer.minAmount ?? monitorSettings.value.investment.minAmount;
                monitorSettings.value.investment.maxAmount = invSettingsFromServer.maxAmount ?? monitorSettings.value.investment.maxAmount;
                monitorSettings.value.investment.profitRate = invSettingsFromServer.profitRate ?? monitorSettings.value.investment.profitRate;
                monitorSettings.value.investment.lossRate = invSettingsFromServer.lossRate ?? monitorSettings.value.investment.lossRate;
                monitorSettings.value.investment.percentageOfBalance = invSettingsFromServer.percentageOfBalance ?? monitorSettings.value.investment.percentageOfBalance;
                monitorSettings.value.investment.simulatedBalance = invSettingsFromServer.simulatedBalance ?? monitorSettings.value.investment.simulatedBalance;

                const invStrategy = investmentStrategies.value.find(s => s.id === invSettingsFromServer.strategy_id);
                if (invStrategy) {
                    selectedInvestmentStrategy.value = invStrategy; // 触发 watch (默认+全局保存)
                    if (invSettingsFromServer.investment_strategy_specific_params) {
                        let specificParamsFromServer = { ...invSettingsFromServer.investment_strategy_specific_params };
                        if (invStrategy.id === 'martingale_user_defined' && Array.isArray(specificParamsFromServer.sequence)) {
                            specificParamsFromServer.sequence = specificParamsFromServer.sequence.join(','); // 转为字符串供UI
                        }
                        // 合并：(默认+全局保存) -> 服务器活动配置特定参数
                        const currentDefaultsAndSavedSpecific = { ...investmentStrategyParams.value };
                        const finalSpecificParams = { ...currentDefaultsAndSavedSpecific };

                        if (invStrategy.parameters) {
                             invStrategy.parameters.forEach(pDef => {
                                if (!pDef.advanced && !pDef.readonly &&
                                    pDef.name !== 'minAmount' && pDef.name !== 'maxAmount' && pDef.name !== 'amount' && // 这些是全局的
                                    specificParamsFromServer.hasOwnProperty(pDef.name)) {
                                    
                                    let val = specificParamsFromServer[pDef.name];
                                    // 类型转换可以根据pDef.type添加，但martingale的sequence特殊处理已完成
                                    finalSpecificParams[pDef.name] = val;
                                }
                            });
                        }
                        investmentStrategyParams.value = finalSpecificParams;
                    }
                } else {
                    selectedInvestmentStrategy.value = null;
                }
            }
            showServerMessage("已使用服务器上的活动配置更新当前设置。", false, 3000);
        };


        // --- Watchers for Strategy Selection and Parameter Population (待提取到 useStrategyManagement) ---
        watch(selectedPredictionStrategy, (newStrategy, oldStrategy) => {
            // 仅当策略实际改变时才重置参数，避免 populateUiFromConfigDetails 设置后被 watch 覆盖
            if (newStrategy?.id !== oldStrategy?.id) {
                if (newStrategy && newStrategy.id) {
                    // monitorSettings.value.prediction_strategy_id = newStrategy.id; // 不需要，config中会包含
                    const savedGlobalParams = allSavedParams.value.prediction_strategies?.[newStrategy.id] || {};
                    const defaultParams = {};
                    if (newStrategy.parameters) {
                        newStrategy.parameters.forEach(param => {
                            if (!param.advanced) {
                                defaultParams[param.name] = param.type === 'boolean' ? (param.default === true) : param.default;
                            }
                        });
                    }
                    predictionStrategyParams.value = { ...defaultParams, ...savedGlobalParams };
                } else {
                    // monitorSettings.value.prediction_strategy_id = '';
                    predictionStrategyParams.value = {};
                }
            }
        });

        watch(selectedInvestmentStrategy, (newStrategy, oldStrategy) => {
            if (newStrategy?.id !== oldStrategy?.id) {
                if (newStrategy && newStrategy.id) {
                    // monitorSettings.value.investment.strategy_id = newStrategy.id; // 不需要
                    const savedGlobalParams = allSavedParams.value.investment_strategies?.[newStrategy.id] || {};
                    const defaultParams = {};
                    if (newStrategy.parameters) {
                        newStrategy.parameters.forEach(param => {
                            if (!param.advanced && !param.readonly &&
                                param.name !== 'minAmount' && param.name !== 'maxAmount' && param.name !== 'amount') { // 这些是全局投资设置
                                if (param.editor === 'text_list' && Array.isArray(param.default)) {
                                    defaultParams[param.name] = param.default.join(',');
                                } else if (param.type === 'boolean') {
                                    defaultParams[param.name] = (param.default === true);
                                } else {
                                    defaultParams[param.name] = param.default;
                                }
                            }
                        });
                    }
                    let mergedParams = { ...defaultParams, ...savedGlobalParams };
                    if (newStrategy.id === 'martingale_user_defined' && mergedParams.sequence && Array.isArray(mergedParams.sequence)) {
                        mergedParams.sequence = mergedParams.sequence.join(',');
                    }
                    investmentStrategyParams.value = mergedParams;

                    // 如果选的是固定金额策略，尝试从其参数或全局保存值或默认值更新UI上的全局amount
                    if (newStrategy.id === 'fixed') {
                         monitorSettings.value.investment.amount = mergedParams?.amount || savedGlobalParams?.amount || defaultParams?.amount || monitorSettings.value.investment.amount;
                    }
                } else {
                    // monitorSettings.value.investment.strategy_id = '';
                    investmentStrategyParams.value = {};
                }
            }
        });

        // --- WebSocket Logic ---
        const connectWebSocket = () => {
            // ... (与之前版本类似，但 message handler 中调用 populateUiFromConfigDetails)
            if (socket.value && (socket.value.readyState === WebSocket.OPEN || socket.value.readyState === WebSocket.CONNECTING)) {
                showServerMessage("WebSocket 已经连接或正在连接中。", false, 2000);
                return;
            }
            error.value = null; serverMessage.value = '';
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            const wsUrl = `${protocol}//${host}/ws/live-test`;

            socket.value = new WebSocket(wsUrl);
            socketStatus.value = 'connecting';

            socket.value.onopen = () => {
                socketStatus.value = 'connected';
                showServerMessage("WebSocket 连接成功。", false, 3000);
                const localConfigId = localStorage.getItem('liveTestConfigId');
                if (localConfigId) { // 优先使用 localStorage 中的 ID
                    currentConfigId.value = localConfigId; // 更新 ref
                    socket.value.send(JSON.stringify({ type: 'restore_session', data: { config_id: currentConfigId.value } }));
                } else if (currentConfigId.value) { // 如果 localStorage 没有，但 ref 中有（例如页面刚加载，从旧的localStorage恢复的）
                     socket.value.send(JSON.stringify({ type: 'restore_session', data: { config_id: currentConfigId.value } }));
                }
                // 如果都没有，则不尝试恢复会话
            };

            socket.value.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log("LiveTest: Received WebSocket message:", message);
                    applyingConfig.value = false; stoppingTest.value = false; // 通用清除按钮加载状态

                    switch (message.type) {
                        case "initial_signals": handleInitialSignals(message.data); break;
                        case "initial_stats": 
                        case "stats_update": 
                            stats.value = { ...stats.value, ...message.data }; 
                            break;
                        case "new_signal":
                            handleNewSignal(message.data);
                            if (monitorSettings.value.enableSound) playSound();
                            break;
                        case "verified_signal": handleVerifiedSignal(message.data); break;
                        case "config_set_confirmation":
                            if (message.data.success) {
                                currentConfigId.value = message.data.config_id;
                                localStorage.setItem('liveTestConfigId', currentConfigId.value); // 持久化
                                showServerMessage(message.data.message || '配置已成功应用！', false, 5000);
                                if(message.data.applied_config) { // 服务器返回了实际应用的配置
                                    populateUiFromConfigDetails(message.data.applied_config);
                                }
                            } else {
                                showServerMessage(message.data.message || '配置应用失败。', true, 5000);
                            }
                            break;
                        case "session_restored":
                            currentConfigId.value = message.data.config_id; 
                            localStorage.setItem('liveTestConfigId', currentConfigId.value); // 确保ID同步和持久化
                            populateUiFromConfigDetails(message.data.config_details); // 用服务器的配置更新UI
                            showServerMessage(`会话已恢复 (ID: ${currentConfigId.value.substring(0,8)}...)`, false, 4000);
                            break;
                        case "session_not_found":
                            showServerMessage(`未能恢复会话 (ID: ${message.data.config_id ? message.data.config_id.substring(0,8) : 'N/A'}...)，请重新应用配置。`, true, 6000);
                            currentConfigId.value = null; // 清除ID
                            localStorage.removeItem('liveTestConfigId'); // 从localStorage清除
                            break;
                        case "test_stopped_confirmation":
                            if (message.data.success) {
                                showServerMessage(`测试 (ID: ${message.data.stopped_config_id ? message.data.stopped_config_id.substring(0,8) : 'N/A'}...) 已成功停止。`, false, 5000);
                                currentConfigId.value = null;
                                localStorage.removeItem('liveTestConfigId');
                            } else {
                                showServerMessage(message.data.message || '停止测试失败。', true, 5000);
                            }
                            break;
                        case "error": 
                            showServerMessage(message.data.message || "收到来自服务器的 WebSocket 错误", true, 6000);
                            break;
                        default: console.warn("LiveTest: Received unknown message type:", message.type);
                    }
                } catch (e) {
                    console.error("LiveTest: Failed to parse message or handle it:", e, event.data);
                    showServerMessage("处理服务器消息失败: " + e.message, true);
                } finally {
                     applyingConfig.value = false; stoppingTest.value = false; // 确保按钮状态恢复
                }
            };

            socket.value.onclose = (event) => {
                socketStatus.value = 'disconnected';
                socket.value = null;
                applyingConfig.value = false; stoppingTest.value = false;
                if (!event.wasClean) {
                    showServerMessage("WebSocket 连接意外断开。如果后台有测试在运行，它仍将继续。", true, 7000);
                } else {
                     showServerMessage("WebSocket 连接已关闭。", false, 3000);
                }
                 // 可选：尝试自动重连，但要注意避免无限循环
                // setTimeout(() => { if (!socket.value) connectWebSocket(); }, 5000);
            };
            socket.value.onerror = (errEvent) => {
                console.error("LiveTest: WebSocket error: ", errEvent);
                socketStatus.value = 'error';
                applyingConfig.value = false; stoppingTest.value = false;
                showServerMessage("WebSocket 连接错误。请检查服务是否运行或网络连接。", true, 7000);
                if (socket.value) { socket.value.close(); socket.value = null; }
            };
        };

        const sendRuntimeConfig = () => {
            if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
                showServerMessage("WebSocket 未连接。请先连接服务。", true); return;
            }
            if (!selectedPredictionStrategy.value?.id) {
                showServerMessage("请先选择一个预测策略！", true); return;
            }
            if (!selectedInvestmentStrategy.value?.id) {
                showServerMessage("请先选择一个投资策略！", true); return;
            }

            applyingConfig.value = true; error.value = null; serverMessage.value = '';

            // 准备 prediction_strategy_params，确保类型正确
            let finalPredictionParams = {};
            if (selectedPredictionStrategy.value.parameters) {
                selectedPredictionStrategy.value.parameters.forEach(paramDef => {
                    if (predictionStrategyParams.value.hasOwnProperty(paramDef.name)) {
                        let val = predictionStrategyParams.value[paramDef.name];
                        if (paramDef.type === 'int') val = parseInt(val, 10);
                        else if (paramDef.type === 'float') val = parseFloat(val);
                        else if (paramDef.type === 'boolean') val = (val === true || String(val).toLowerCase() === 'true');
                        finalPredictionParams[paramDef.name] = val;
                    }
                });
            }
            
            // 准备 investment_strategy_specific_params
            let finalInvestmentSpecificParams = { ...investmentStrategyParams.value };
            if (selectedInvestmentStrategy.value.id === 'martingale_user_defined' && finalInvestmentSpecificParams.sequence) {
                if (typeof finalInvestmentSpecificParams.sequence === 'string') {
                    try {
                        const parsedSequence = finalInvestmentSpecificParams.sequence.split(',')
                            .map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0);
                        if (parsedSequence.length > 0) {
                            finalInvestmentSpecificParams.sequence = parsedSequence;
                        } else {
                           const defaultSeqDef = selectedInvestmentStrategy.value.parameters.find(p => p.name === 'sequence');
                           finalInvestmentSpecificParams.sequence = defaultSeqDef?.default && Array.isArray(defaultSeqDef.default) ? defaultSeqDef.default : [10,20,40];
                           showServerMessage("马丁格尔序列无效，已使用默认值发送。", true, 3000);
                        }
                    } catch (e) {
                        const defaultSeqDef = selectedInvestmentStrategy.value.parameters.find(p => p.name === 'sequence');
                        finalInvestmentSpecificParams.sequence = defaultSeqDef?.default && Array.isArray(defaultSeqDef.default) ? defaultSeqDef.default : [10,20,40];
                        showServerMessage("解析马丁格尔序列错误，已使用默认值发送。", true, 3000);
                    }
                } // 如果已经是数组，则直接使用
            }
            
            const investmentSettingsPayload = {
                strategy_id: selectedInvestmentStrategy.value.id,
                investment_strategy_specific_params: finalInvestmentSpecificParams,
                minAmount: parseFloat(monitorSettings.value.investment.minAmount),
                maxAmount: parseFloat(monitorSettings.value.investment.maxAmount),
                profitRate: parseFloat(monitorSettings.value.investment.profitRate),
                lossRate: parseFloat(monitorSettings.value.investment.lossRate),
            };
            // 根据策略类型条件性地添加amount或percentageOfBalance/simulatedBalance
            if (selectedInvestmentStrategy.value.id === 'percentage_of_balance') {
                investmentSettingsPayload.percentageOfBalance = parseFloat(monitorSettings.value.investment.percentageOfBalance);
                investmentSettingsPayload.simulatedBalance = parseFloat(monitorSettings.value.investment.simulatedBalance);
            } else {
                 investmentSettingsPayload.amount = parseFloat(monitorSettings.value.investment.amount);
            }
            
            const configPayload = {
                type: 'set_runtime_config',
                data: {
                    prediction_strategy_id: selectedPredictionStrategy.value.id,
                    prediction_strategy_params: finalPredictionParams,
                    confidence_threshold: parseFloat(monitorSettings.value.confidence_threshold),
                    event_period: monitorSettings.value.event_period,
                    symbol: monitorSettings.value.symbol,
                    interval: monitorSettings.value.interval,
                    investment_settings: investmentSettingsPayload
                }
            };
            socket.value.send(JSON.stringify(configPayload));
        };

        const stopCurrentTest = () => {
            if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
                showServerMessage("WebSocket 未连接。", true); return;
            }
            if (!currentConfigId.value) { // 只有当有已知的 currentConfigId 时才尝试停止
                showServerMessage("没有活动的测试配置可停止。请先应用一个配置。", true); return;
            }
            stoppingTest.value = true; error.value = null; serverMessage.value = '';
            // 后端会使用 websocket_to_config_id_map (或类似机制) 找到对应的 config_id 来停止
            // 如果 currentConfigId.value 是由服务器确认的，则可以发送它以确保停止正确的任务
            socket.value.send(JSON.stringify({ type: 'stop_current_test', data: { config_id_to_stop: currentConfigId.value } }));
        };

        const startLiveTestService = () => connectWebSocket();
        const stopLiveTestService = () => {
            if (socket.value) socket.value.close(); 
            showServerMessage("WebSocket 连接已断开。如果后台有测试在运行，它仍将继续。", false, 4000);
        };

        const saveStrategyParameters = async () => {
            // ... (与 index-scripts.js 中的 saveStrategyParameters 逻辑非常相似，确保类型转换)
            let savedCount = 0;
            let errorMessages = [];
            let hasAttemptedSave = false;

            if (selectedPredictionStrategy.value && selectedPredictionStrategy.value.id) {
                hasAttemptedSave = true;
                let finalPredictionParamsToSave = {};
                 if (selectedPredictionStrategy.value.parameters) {
                    selectedPredictionStrategy.value.parameters.forEach(paramDef => {
                        if (predictionStrategyParams.value.hasOwnProperty(paramDef.name)) {
                             let val = predictionStrategyParams.value[paramDef.name];
                            if (paramDef.type === 'int') val = parseInt(val, 10);
                            else if (paramDef.type === 'float') val = parseFloat(val);
                            else if (paramDef.type === 'boolean') val = (val === true || String(val).toLowerCase() === 'true');
                            finalPredictionParamsToSave[paramDef.name] = val;
                        }
                    });
                }
                const payload = {
                    strategy_type: 'prediction',
                    strategy_id: selectedPredictionStrategy.value.id,
                    params: finalPredictionParamsToSave
                };
                try {
                    await axios.post('/api/save_strategy_parameter_set', payload);
                    if (!allSavedParams.value.prediction_strategies) allSavedParams.value.prediction_strategies = {};
                    allSavedParams.value.prediction_strategies[payload.strategy_id] = { ...payload.params };
                    savedCount++;
                } catch (err) {
                    errorMessages.push(`预测策略 "${selectedPredictionStrategy.value.name}": ${err.response?.data?.detail || err.message}`);
                }
            }

            if (selectedInvestmentStrategy.value && selectedInvestmentStrategy.value.id) {
                hasAttemptedSave = true;
                let paramsToSave = { ...investmentStrategyParams.value };
                if (selectedInvestmentStrategy.value.id === 'martingale_user_defined' && paramsToSave.sequence) {
                    if (typeof paramsToSave.sequence === 'string') {
                         try { // 转为数组再保存
                            const parsedSequence = paramsToSave.sequence.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0);
                            if (parsedSequence.length > 0) paramsToSave.sequence = parsedSequence;
                            else delete paramsToSave.sequence; 
                        } catch (e) { delete paramsToSave.sequence; }
                    } // 如果已是数组，则直接使用
                }
                const payload = {
                    strategy_type: 'investment',
                    strategy_id: selectedInvestmentStrategy.value.id,
                    params: paramsToSave
                };
                try {
                    await axios.post('/api/save_strategy_parameter_set', payload);
                    if (!allSavedParams.value.investment_strategies) allSavedParams.value.investment_strategies = {};
                    allSavedParams.value.investment_strategies[payload.strategy_id] = { ...payload.params }; // 保存转换后的参数
                    savedCount++;
                } catch (err) {
                    errorMessages.push(`投资策略 "${selectedInvestmentStrategy.value.name}": ${err.response?.data?.detail || err.message}`);
                }
            }

            if (!hasAttemptedSave) {
                showServerMessage("没有选定任何策略或没有参数可保存。", true); return;
            }
            if (errorMessages.length > 0) {
                showServerMessage(`部分策略参数保存失败：\n${errorMessages.join("\n")}` + (savedCount > 0 ? "\n其余成功！" : ""), true, 8000);
            } else if (savedCount > 0) {
                showServerMessage("选定策略的参数已成功保存为全局默认值！", false, 5000);
            }
        };

        // --- Signal Handling ---
        const sanitizeSignal = (signal) => {
            const sanitized = { ...signal };
            const numericFields = [
                'confidence', 'signal_price', 'actual_end_price', 
                'price_change_pct', 'pnl_pct', 'investment_amount', 
                'actual_profit_loss_amount', 'profit_rate_pct', 'loss_rate_pct',
                'potential_profit', 'potential_loss' // 添加这两个计算字段
            ];
            numericFields.forEach(field => {
                if (field in sanitized && sanitized[field] !== null && sanitized[field] !== undefined) {
                    const num = parseFloat(sanitized[field]);
                    sanitized[field] = isNaN(num) ? null : num; // 如果不能转为数字，则设为null
                } else if (field in sanitized && (sanitized[field] === null || sanitized[field] === undefined)) {
                    sanitized[field] = null; // 确保是 null 而不是 undefined
                }
            });
            sanitized.verified = typeof sanitized.verified === 'boolean' ? sanitized.verified : false;
            if (sanitized.result !== null && sanitized.result !== undefined) {
                sanitized.result = (sanitized.result === true || String(sanitized.result).toLowerCase() === 'true' || sanitized.result === 1);
            } else {
                sanitized.result = null; // 明确设为 null
            }
            return sanitized;
        };

        const handleInitialSignals = (signalsArray) => {
            if (!Array.isArray(signalsArray)) {
                liveSignals.value = []; return;
            }
            liveSignals.value = signalsArray.map(s => sanitizeSignal(s)).sort((a, b) => new Date(b.signal_time).getTime() - new Date(a.signal_time).getTime());
            updateAllTimeRemaining();
        };
        
        const handleNewSignal = (signalData) => {
            const newSignal = sanitizeSignal(signalData);
            
            // 根据投资额和盈亏率计算潜在盈亏 (如果数据中没有提供)
            if (newSignal.investment_amount && newSignal.profit_rate_pct && newSignal.potential_profit === null) {
                newSignal.potential_profit = parseFloat((newSignal.investment_amount * (newSignal.profit_rate_pct / 100)).toFixed(2));
            }
            if (newSignal.investment_amount && newSignal.loss_rate_pct && newSignal.potential_loss === null) {
                newSignal.potential_loss = parseFloat((newSignal.investment_amount * (newSignal.loss_rate_pct / 100)).toFixed(2));
            }

            const existingIdx = liveSignals.value.findIndex(s => s.id === newSignal.id);
            if (existingIdx === -1) {
                liveSignals.value.unshift(newSignal); // 加到最前面
                 if (liveSignals.value.length > 100) { // 限制列表长度
                    liveSignals.value.pop();
                }
            } else { // 更新已存在的信号
                liveSignals.value[existingIdx] = { ...liveSignals.value[existingIdx], ...newSignal };
            }
            updateAllTimeRemaining();
        };

        const handleVerifiedSignal = (signalData) => {
            const verifiedSignal = sanitizeSignal(signalData);
            verifiedSignal.verified = true; 
            const index = liveSignals.value.findIndex(s => s.id === verifiedSignal.id);
            if (index !== -1) {
                liveSignals.value[index] = { ...liveSignals.value[index], ...verifiedSignal };
            } else { // 如果因为某种原因初始信号没收到，验证信号来了也添加进去
                liveSignals.value.unshift(verifiedSignal);
                 if (liveSignals.value.length > 100) { liveSignals.value.pop(); }
            }
            updateAllTimeRemaining();
        };

        // --- UI Helpers ---
        const audioContext = ref(null);
        const playSound = () => {
            if (!monitorSettings.value.enableSound || (!window.AudioContext && !window.webkitAudioContext)) return;
            if (!audioContext.value) audioContext.value = new (window.AudioContext || window.webkitAudioContext)();
            try {
                const oscillator = audioContext.value.createOscillator();
                const gainNode = audioContext.value.createGain();
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.value.destination);
                oscillator.type = 'sine'; 
                oscillator.frequency.setValueAtTime(440, audioContext.value.currentTime); 
                gainNode.gain.setValueAtTime(0.1, audioContext.value.currentTime); 
                oscillator.start();
                oscillator.stop(audioContext.value.currentTime + 0.2);
            } catch(e) { console.warn("Error playing sound:", e); }
        };

        const getStrategyName = (strategyId, type = 'prediction') => {
            const list = type === 'prediction' ? predictionStrategies.value : investmentStrategies.value;
            const strategy = list.find(s => s.id === strategyId);
            return strategy ? strategy.name : (strategyId || "未知");
        };

        // --- Countdown Timer for Signals ---
        const timeRemainingRefs = ref({}); // { signalId: "mm分ss秒" }
        let countdownInterval = null;

        const updateAllTimeRemaining = () => {
            liveSignals.value.forEach(signal => {
                if (!signal.verified && signal.expected_end_time) {
                    try {
                        const endTime = new Date(signal.expected_end_time);
                        const now = new Date();
                        const diffMs = endTime.getTime() - now.getTime();

                        if (diffMs <= 0) {
                            timeRemainingRefs.value[signal.id] = '待验证';
                        } else {
                            const totalSeconds = Math.floor(diffMs / 1000);
                            const minutes = Math.floor(totalSeconds / 60);
                            const seconds = totalSeconds % 60;
                            timeRemainingRefs.value[signal.id] = `${String(minutes).padStart(2, '0')}分${String(seconds).padStart(2, '0')}秒`;
                        }
                    } catch (e) {
                        timeRemainingRefs.value[signal.id] = '时间错误';
                    }
                } else if (signal.verified) {
                    timeRemainingRefs.value[signal.id] = '已验证';
                } else {
                    timeRemainingRefs.value[signal.id] = 'N/A';
                }
            });
        };
        const getTimeRemaining = (signal) => timeRemainingRefs.value[signal.id] || (signal.verified ? '已验证' : '计算中...');
        const isTimeRemainingRelevant = (signal) => !signal.verified && signal.expected_end_time;


        // --- Historical Analysis (remains mostly client-side based on loaded liveSignals) ---
        const analysisFilter = ref({ 
            startDate: '', endDate: '', minConfidence: 0, maxConfidence: 100,
            direction: 'all', symbol: 'all', interval: 'all'
        });
        const analysisResults = ref([]);
        const hasAnalyzed = ref(false);
        const filteredStats = ref({
            total_verified: 0, win_rate: 0, long_signals:0, long_win_rate:0, short_signals:0, short_win_rate:0,
            profit_count: 0, loss_count: 0, total_profit: 0, total_loss: 0, avg_profit:0, avg_loss:0, profit_loss_ratio:0
        });
        
        const setDefaultDateRange = () => {
            const now = new Date(); 
            const sevenDaysAgo = new Date();
            sevenDaysAgo.setDate(now.getDate() - 7);
            analysisFilter.value.endDate = utilFormatDateForInput(now); // 使用 utils.js 中的函数
            analysisFilter.value.startDate = utilFormatDateForInput(sevenDaysAgo);
        };
        const analyzeHistoricalData = () => {
            hasAnalyzed.value = true;
            try {
                const startDate = analysisFilter.value.startDate ? new Date(analysisFilter.value.startDate) : null;
                const endDate = analysisFilter.value.endDate ? new Date(analysisFilter.value.endDate) : null;
                
                const filtered = liveSignals.value.filter(signal => {
                    if (!signal.verified) return false; 
                    if (startDate && endDate) {
                        const signalDate = new Date(signal.signal_time);
                        if (isNaN(signalDate.getTime()) || signalDate < startDate || signalDate > endDate) return false;
                    }
                    const confidence = parseFloat(signal.confidence);
                    if (isNaN(confidence) || confidence < analysisFilter.value.minConfidence || confidence > analysisFilter.value.maxConfidence) return false;

                    if (analysisFilter.value.direction !== 'all') {
                        if (analysisFilter.value.direction === 'long' && signal.signal !== 1) return false;
                        if (analysisFilter.value.direction === 'short' && signal.signal !== -1) return false;
                    }
                    if (analysisFilter.value.symbol !== 'all' && signal.symbol !== analysisFilter.value.symbol) return false;
                    if (analysisFilter.value.interval !== 'all' && signal.interval !== analysisFilter.value.interval) return false;
                    return true;
                });
                analysisResults.value = filtered.sort((a,b) => new Date(b.signal_time) - new Date(a.signal_time));
                calculateFilteredStats(filtered);
            } catch (err) { console.error('分析历史数据失败:', err); showServerMessage("分析数据时出错: " + err.message, true); } 
        };
        const calculateFilteredStats = (signals) => {
            const verified = signals.filter(s => s.verified); // 确保只统计已验证的
            const correctSignals = verified.filter(s => s.result === true);
            const longSignals = verified.filter(s => s.signal === 1);
            const longCorrect = longSignals.filter(s => s.result === true);
            const shortSignals = verified.filter(s => s.signal === -1);
            const shortCorrect = shortSignals.filter(s => s.result === true);
            
            let totalProfitVal = 0, totalLossVal = 0, profitCountVal = 0, lossCountVal = 0;
            verified.forEach(signal => {
                const pnl = parseFloat(signal.actual_profit_loss_amount);
                if (isNaN(pnl)) return;
                if (signal.result === true) { // 盈利
                    profitCountVal++; 
                    totalProfitVal += Math.abs(pnl); // 确保盈利额是正数
                } else if (signal.result === false) { // 亏损
                    lossCountVal++; 
                    totalLossVal += Math.abs(pnl); // 确保亏损额是正数（代表亏损的绝对值）
                }
            });
            const avgProfit = profitCountVal > 0 ? totalProfitVal / profitCountVal : 0;
            // avgLoss 计算的是平均亏损的绝对值
            const avgLoss = lossCountVal > 0 ? totalLossVal / lossCountVal : 0; 
            
            filteredStats.value = {
                total_verified: verified.length,
                win_rate: verified.length > 0 ? parseFloat((correctSignals.length / verified.length * 100).toFixed(2)) : 0,
                long_signals: longSignals.length,
                long_win_rate: longSignals.length > 0 ? parseFloat((longCorrect.length / longSignals.length * 100).toFixed(2)) : 0,
                short_signals: shortSignals.length,
                short_win_rate: shortSignals.length > 0 ? parseFloat((shortCorrect.length / shortSignals.length * 100).toFixed(2)) : 0,
                profit_count: profitCountVal, 
                loss_count: lossCountVal,
                total_profit: parseFloat(totalProfitVal.toFixed(2)), 
                total_loss: parseFloat(totalLossVal.toFixed(2)), // 这是亏损的总额（绝对值）
                avg_profit: parseFloat(avgProfit.toFixed(2)), // 平均盈利额
                avg_loss: parseFloat(avgLoss.toFixed(2)),   // 平均亏损额（绝对值）
                // 盈亏比 = 平均盈利额 / 平均亏损额（绝对值）
                profit_loss_ratio: avgLoss > 0 ? parseFloat((avgProfit / avgLoss).toFixed(2)) : (avgProfit > 0 ? Infinity : 0)
            };
        };

        // --- Lifecycle Hooks ---
        onMounted(async () => {
            loadingInitialData.value = true;
            loadFavorites();
            setDefaultDateRange();
            
            const storedConfigId = localStorage.getItem('liveTestConfigId');
            if (storedConfigId) currentConfigId.value = storedConfigId;

            await fetchAllSavedParameters(); // 先加载保存的参数
            await Promise.all([ // 并行获取其他数据
                fetchSymbols(),
                fetchPredictionStrategies(),
                fetchInvestmentStrategies()
            ]);

            // 设置默认选中的策略（在所有策略数据加载完毕后）
            if (predictionStrategies.value.length > 0 && !selectedPredictionStrategy.value) {
                const defaultPred = predictionStrategies.value.find(s => s.id === 'simple_rsi') || predictionStrategies.value[0];
                selectedPredictionStrategy.value = defaultPred;
            }
            if (investmentStrategies.value.length > 0 && !selectedInvestmentStrategy.value) {
                const defaultInv = investmentStrategies.value.find(s => s.id === 'fixed') || investmentStrategies.value[0];
                selectedInvestmentStrategy.value = defaultInv;
            }
            
            loadingInitialData.value = false;
            // connectWebSocket(); // 用户可以手动点击连接按钮，或在这里自动连接
            countdownInterval = setInterval(updateAllTimeRemaining, 1000);
        });

        onUnmounted(() => {
            if (socket.value) socket.value.close();
            if (countdownInterval) clearInterval(countdownInterval);
            if (serverMessageTimer) clearTimeout(serverMessageTimer);
        });

        return {
            // State
            symbols, favoriteSymbols, sortedSymbols,
            predictionStrategies, selectedPredictionStrategy, predictionStrategyParams,
            investmentStrategies, selectedInvestmentStrategy, investmentStrategyParams,
            // allSavedParams, // 主要内部使用
            liveSignals, latestSignals, loadingInitialData, error, serverMessage,
            socketStatus, stats, monitorSettings, currentConfigId,
            applyingConfig, stoppingTest,

            // Methods
            toggleFavorite, isFavorite,
            startLiveTestService, stopLiveTestService,
            sendRuntimeConfig, stopCurrentTest, saveStrategyParameters,
            getStrategyName,
            getTimeRemaining, isTimeRemainingRelevant,
            
            // Analysis
            analysisFilter, analysisResults, hasAnalyzed, filteredStats, analyzeHistoricalData,

            // Utils for template (will be registered to globalProperties)
            // formatDateTime, getWinRateClass, getPnlClass, getSignalStatusClass (已通过 globalProperties 注册)
        };
    }
});

// 注册全局属性
const utilsToRegister = { formatDateTime, getWinRateClass, getPnlClass, getSignalStatusClass };
for (const key in utilsToRegister) {
    if (typeof utilsToRegister[key] === 'function') {
        app.config.globalProperties[key] = utilsToRegister[key];
    } else {
        app.config.globalProperties[key] = (val) => val || (key.includes('Class') ? '' : '-'); // 提供回退
        console.error(`utils.js: ${key} function is not defined globally.`);
    }
}

app.mount('#app');