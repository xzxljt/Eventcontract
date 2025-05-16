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
        const favoriteSymbols = ref([]);

        const predictionStrategies = ref([]);
        const selectedPredictionStrategy = ref(null);
        const predictionStrategyParams = ref({});

        const investmentStrategies = ref([]);
        const selectedInvestmentStrategy = ref(null);
        const investmentStrategyParams = ref({});

        const allSavedParams = ref({
            prediction_strategies: {},
            investment_strategies: {}
        });

        const liveSignals = ref([]); 
        const loadingInitialData = ref(false);
        const error = ref(null);
        const serverMessage = ref('');
        let serverMessageTimer = null;

        const socket = ref(null);
        const socketStatus = ref('disconnected');
        const currentConfigId = ref(null);

        const applyingConfig = ref(false);
        const stoppingTest = ref(false);

        const stats = ref({
            total_signals: 0, total_verified: 0, total_correct: 0,
            win_rate: 0, total_pnl_pct: 0, average_pnl_pct: 0,
            total_profit_amount: 0
        });

        const monitorSettings = ref({
            symbol: 'all',
            interval: 'all',
            confidence_threshold: 50,
            event_period: '10m',
            enableSound: false,
            investment: {
                amount: 20.0,
                minAmount: 5.0,
                maxAmount: 250.0,
                percentageOfBalance: 10.0,
                profitRate: 80.0,
                lossRate: 100.0,
                simulatedBalance: 1000.0, // 这个是用户在UI上设置的初始模拟本金
            }
        });
        
        // +++ START: ADDED/MODIFIED STATE +++
        const activeTestConfigDetails = ref(null); // 用于存储当前活动配置的完整细节，包括余额
        // +++ END: ADDED/MODIFIED STATE +++

        const selectedSignalIds = ref([]); 
        const deletingSignals = ref(false); 

        const signalManagementFilter = ref({ 
            symbol: 'all',
            interval: 'all',
            direction: 'all', 
            verifiedStatus: 'all', 
            minConfidence: 0,
            maxConfidence: 100,
        });

        // --- Computed Properties ---
        // 原 latestSignals 已被 displayedManagedSignals 取代主要功能
        // 如果仍有其他地方需要仅显示最新几条，可以保留或调整
        // const latestSignals = computed(() => { ... }); 

        const displayedManagedSignals = computed(() => {
            if (!Array.isArray(liveSignals.value)) return [];
            
            let filtered = [...liveSignals.value];

            // 应用筛选条件
            if (signalManagementFilter.value.symbol !== 'all') {
                filtered = filtered.filter(s => s.symbol === signalManagementFilter.value.symbol);
            }
            if (signalManagementFilter.value.interval !== 'all') {
                filtered = filtered.filter(s => s.interval === signalManagementFilter.value.interval);
            }
            if (signalManagementFilter.value.direction !== 'all') {
                const dir = signalManagementFilter.value.direction === 'long' ? 1 : -1;
                filtered = filtered.filter(s => s.signal === dir);
            }
            if (signalManagementFilter.value.verifiedStatus !== 'all') {
                const isVerified = signalManagementFilter.value.verifiedStatus === 'verified';
                filtered = filtered.filter(s => (s.verified || false) === isVerified); // 确保 s.verified 存在
            }
            if (signalManagementFilter.value.minConfidence > 0 || signalManagementFilter.value.maxConfidence < 100) {
                filtered = filtered.filter(s => {
                    const conf = parseFloat(s.confidence);
                    return !isNaN(conf) && conf >= signalManagementFilter.value.minConfidence && conf <= signalManagementFilter.value.maxConfidence;
                });
            }
            // TODO: 如果添加日期筛选器，在这里实现
            // if (signalManagementFilter.value.startDate && signalManagementFilter.value.endDate) { ... }

            // 按信号时间倒序排列
            return filtered.sort((a, b) => {
                const timeA = a.signal_time ? new Date(a.signal_time).getTime() : 0;
                const timeB = b.signal_time ? new Date(b.signal_time).getTime() : 0;
                if (isNaN(timeA) || isNaN(timeB)) return 0; // 处理无效日期
                return timeB - timeA;
            });
            // .slice(0, 50); // 可选：如果信号过多，可以限制初始显示数量或实现分页
        });

        const areAllDisplayedManagedSignalsSelected = computed({
            get: () => {
                const displayedIds = displayedManagedSignals.value.map(s => s.id);
                if (displayedIds.length === 0) return false;
                return displayedIds.every(id => selectedSignalIds.value.includes(id));
            },
            set: (value) => {
                // 这个 set 会被 v-model 调用，value 是 checkbox 的新状态 (true/false)
                toggleSelectAllDisplayedManagedSignals(value);
            }
        });

        const sortedSymbols = computed(() => {
            return [...symbols.value].sort((a, b) => {
                const aIsFav = isFavorite(a);
                const bIsFav = isFavorite(b);
                if (aIsFav && !bIsFav) return -1;
                if (!aIsFav && bIsFav) return 1;
                return a.localeCompare(b);
            });
        });

        // --- Server Message Handling ---
        const showServerMessage = (message, isError = false, duration = 5000) => {
            serverMessage.value = message;
            if (isError) error.value = message;
            else error.value = null;
            
            if (serverMessageTimer) clearTimeout(serverMessageTimer);
            serverMessageTimer = setTimeout(() => {
                serverMessage.value = '';
                if (isError) error.value = null;
            }, duration);
        };

        // --- Favorite Symbols Logic ---
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
        };

        // --- Initialization and Data Fetching ---
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
            // ... (这部分逻辑保持不变, 用于填充左侧配置表单) ...
            if (!configDetails) return;

            monitorSettings.value.symbol = configDetails.symbol || 'all';
            monitorSettings.value.interval = configDetails.interval || 'all';
            monitorSettings.value.confidence_threshold = configDetails.confidence_threshold ?? 50;
            monitorSettings.value.event_period = configDetails.event_period || '10m';

            const predStrategy = predictionStrategies.value.find(s => s.id === configDetails.prediction_strategy_id);
            if (predStrategy) {
                selectedPredictionStrategy.value = predStrategy;
                if (configDetails.prediction_strategy_params) {
                    const paramsFromServer = { ...configDetails.prediction_strategy_params };
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
                selectedPredictionStrategy.value = null;
            }
            
            if (configDetails.investment_settings) {
                const invSettingsFromServer = configDetails.investment_settings;
                monitorSettings.value.investment.amount = invSettingsFromServer.amount ?? monitorSettings.value.investment.amount;
                monitorSettings.value.investment.minAmount = invSettingsFromServer.minAmount ?? monitorSettings.value.investment.minAmount;
                monitorSettings.value.investment.maxAmount = invSettingsFromServer.maxAmount ?? monitorSettings.value.investment.maxAmount;
                monitorSettings.value.investment.profitRate = invSettingsFromServer.profitRate ?? monitorSettings.value.investment.profitRate;
                monitorSettings.value.investment.lossRate = invSettingsFromServer.lossRate ?? monitorSettings.value.investment.lossRate;
                monitorSettings.value.investment.percentageOfBalance = invSettingsFromServer.percentageOfBalance ?? monitorSettings.value.investment.percentageOfBalance;
                monitorSettings.value.investment.simulatedBalance = invSettingsFromServer.simulatedBalance ?? monitorSettings.value.investment.simulatedBalance;

                const invStrategy = investmentStrategies.value.find(s => s.id === invSettingsFromServer.strategy_id);
                if (invStrategy) {
                    selectedInvestmentStrategy.value = invStrategy;
                    if (invSettingsFromServer.investment_strategy_specific_params) {
                        let specificParamsFromServer = { ...invSettingsFromServer.investment_strategy_specific_params };
                        if (invStrategy.id === 'martingale_user_defined' && Array.isArray(specificParamsFromServer.sequence)) {
                            specificParamsFromServer.sequence = specificParamsFromServer.sequence.join(',');
                        }
                        const currentDefaultsAndSavedSpecific = { ...investmentStrategyParams.value };
                        const finalSpecificParams = { ...currentDefaultsAndSavedSpecific };

                        if (invStrategy.parameters) {
                             invStrategy.parameters.forEach(pDef => {
                                if (!pDef.advanced && !pDef.readonly &&
                                    pDef.name !== 'minAmount' && pDef.name !== 'maxAmount' && pDef.name !== 'amount' &&
                                    specificParamsFromServer.hasOwnProperty(pDef.name)) {
                                    finalSpecificParams[pDef.name] = specificParamsFromServer[pDef.name];
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

        // --- Watchers for Strategy Selection and Parameter Population ---
        watch(selectedPredictionStrategy, (newStrategy, oldStrategy) => {
            if (newStrategy?.id !== oldStrategy?.id) {
                if (newStrategy && newStrategy.id) {
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
                    predictionStrategyParams.value = {};
                }
            }
        });

        watch(selectedInvestmentStrategy, (newStrategy, oldStrategy) => {
            if (newStrategy?.id !== oldStrategy?.id) {
                if (newStrategy && newStrategy.id) {
                    const savedGlobalParams = allSavedParams.value.investment_strategies?.[newStrategy.id] || {};
                    const defaultParams = {};
                    if (newStrategy.parameters) {
                        newStrategy.parameters.forEach(param => {
                            if (!param.advanced && !param.readonly &&
                                param.name !== 'minAmount' && param.name !== 'maxAmount' && param.name !== 'amount') {
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
                    if (newStrategy.id === 'fixed') {
                         monitorSettings.value.investment.amount = mergedParams?.amount || savedGlobalParams?.amount || defaultParams?.amount || monitorSettings.value.investment.amount;
                    }
                } else {
                    investmentStrategyParams.value = {};
                }
            }
        });

        // --- WebSocket Logic ---
        const connectWebSocket = () => {
            if (socket.value && (socket.value.readyState === WebSocket.OPEN || socket.value.readyState === WebSocket.CONNECTING)) {
                showServerMessage("WebSocket 已经连接或正在连接中。", false, 2000); return;
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
                if (localConfigId) {
                    currentConfigId.value = localConfigId;
                    socket.value.send(JSON.stringify({ type: 'restore_session', data: { config_id: currentConfigId.value } }));
                } else if (currentConfigId.value) {
                     socket.value.send(JSON.stringify({ type: 'restore_session', data: { config_id: currentConfigId.value } }));
                }
            };

        socket.value.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                console.log("LiveTest: Received WebSocket message:", message);
                applyingConfig.value = false; stoppingTest.value = false;

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
                    
                    // +++ START: MODIFIED CASES +++
                    case "config_set_confirmation":
                        if (message.data.success) {
                            currentConfigId.value = message.data.config_id;
                            localStorage.setItem('liveTestConfigId', currentConfigId.value);
                            showServerMessage(message.data.message || '配置已成功应用！', false, 5000);
                            if(message.data.applied_config) {
                                populateUiFromConfigDetails(message.data.applied_config);
                                activeTestConfigDetails.value = message.data.applied_config; // 更新活动配置细节
                            }
                        } else {
                            showServerMessage(message.data.message || '配置应用失败。', true, 5000);
                            activeTestConfigDetails.value = null; // 清理
                        }
                        break;
                    case "session_restored":
                        currentConfigId.value = message.data.config_id; 
                        localStorage.setItem('liveTestConfigId', currentConfigId.value);
                        populateUiFromConfigDetails(message.data.config_details);
                        activeTestConfigDetails.value = message.data.config_details; // 更新活动配置细节
                        showServerMessage(`会话已恢复 (ID: ${currentConfigId.value.substring(0,8)}...)`, false, 4000);
                        break;
                    case "session_not_found":
                        showServerMessage(`未能恢复会话 (ID: ${message.data.config_id ? message.data.config_id.substring(0,8) : 'N/A'}...)，请重新应用配置。`, true, 6000);
                        currentConfigId.value = null;
                        localStorage.removeItem('liveTestConfigId');
                        activeTestConfigDetails.value = null; // 清理
                        break;
                    case "test_stopped_confirmation":
                        if (message.data.success) {
                            showServerMessage(`测试 (ID: ${message.data.stopped_config_id ? message.data.stopped_config_id.substring(0,8) : 'N/A'}...) 已成功停止。`, false, 5000);
                            currentConfigId.value = null;
                            localStorage.removeItem('liveTestConfigId');
                            activeTestConfigDetails.value = null; // 清理活动配置细节
                        } else {
                            showServerMessage(message.data.message || '停止测试失败。', true, 5000);
                        }
                        break;
                    // +++ END: MODIFIED CASES +++
                    
                    // +++ START: NEW CASE HANDLER +++
                    case "config_specific_balance_update":
                        if (message.data && message.data.config_id === currentConfigId.value) {
                            if (activeTestConfigDetails.value) {
                                activeTestConfigDetails.value.current_balance = message.data.new_balance;
                            }
                            // 可选: 短暂显示余额更新消息
                            // showServerMessage(`当前测试余额更新为: ${message.data.new_balance.toFixed(2)} USDT (本次盈亏: ${message.data.last_pnl_amount.toFixed(2)})`, false, 3000);
                        }
                        break;
                    // +++ END: NEW CASE HANDLER +++

                    case "signals_deleted_notification": 
                        if (message.data && Array.isArray(message.data.deleted_ids)) {
                            liveSignals.value = liveSignals.value.filter(s => !message.data.deleted_ids.includes(s.id));
                            selectedSignalIds.value = selectedSignalIds.value.filter(id => !message.data.deleted_ids.includes(id));
                            showServerMessage(message.data.message || `${message.data.deleted_ids.length} 个信号已被其他操作删除。`, false, 4000);
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
                 applyingConfig.value = false; stoppingTest.value = false;
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
            // ... (这部分逻辑保持不变)
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
                }
            }
            
            const investmentSettingsPayload = {
                strategy_id: selectedInvestmentStrategy.value.id,
                investment_strategy_specific_params: finalInvestmentSpecificParams,
                minAmount: parseFloat(monitorSettings.value.investment.minAmount),
                maxAmount: parseFloat(monitorSettings.value.investment.maxAmount),
                profitRate: parseFloat(monitorSettings.value.investment.profitRate),
                lossRate: parseFloat(monitorSettings.value.investment.lossRate),
            };
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
            // ... (这部分逻辑保持不变)
            if (!socket.value || socket.value.readyState !== WebSocket.OPEN) {
                showServerMessage("WebSocket 未连接。", true); return;
            }
            if (!currentConfigId.value) {
                showServerMessage("没有活动的测试配置可停止。请先应用一个配置。", true); return;
            }
            stoppingTest.value = true; error.value = null; serverMessage.value = '';
            socket.value.send(JSON.stringify({ type: 'stop_current_test', data: { config_id_to_stop: currentConfigId.value } }));
        };

        const startLiveTestService = () => connectWebSocket();
        const stopLiveTestService = () => {
            if (socket.value) socket.value.close(); 
            showServerMessage("WebSocket 连接已断开。如果后台有测试在运行，它仍将继续。", false, 4000);
        };

        const saveStrategyParameters = async () => {
            // ... (这部分逻辑保持不变)
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
                         try {
                            const parsedSequence = paramsToSave.sequence.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n) && n > 0);
                            if (parsedSequence.length > 0) paramsToSave.sequence = parsedSequence;
                            else delete paramsToSave.sequence; 
                        } catch (e) { delete paramsToSave.sequence; }
                    }
                }
                const payload = {
                    strategy_type: 'investment',
                    strategy_id: selectedInvestmentStrategy.value.id,
                    params: paramsToSave
                };
                try {
                    await axios.post('/api/save_strategy_parameter_set', payload);
                    if (!allSavedParams.value.investment_strategies) allSavedParams.value.investment_strategies = {};
                    allSavedParams.value.investment_strategies[payload.strategy_id] = { ...payload.params };
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
            // ... (这部分逻辑保持不变)
            const sanitized = { ...signal };
            const numericFields = [
                'confidence', 'signal_price', 'actual_end_price',
                'price_change_pct', 'pnl_pct', 'investment_amount',
                'actual_profit_loss_amount', 'profit_rate_pct', 'loss_rate_pct',
                'potential_profit', 'potential_loss', 'balance_after_trade'
            ];
            numericFields.forEach(field => {
                if (field in sanitized && sanitized[field] !== null && sanitized[field] !== undefined) {
                    const num = parseFloat(sanitized[field]);
                    sanitized[field] = isNaN(num) ? null : num;
                } else if (field in sanitized && (sanitized[field] === null || sanitized[field] === undefined)) {
                    sanitized[field] = null;
                }
            });
            sanitized.verified = typeof sanitized.verified === 'boolean' ? sanitized.verified : false;
            if (sanitized.result !== null && sanitized.result !== undefined) {
                sanitized.result = (sanitized.result === true || String(sanitized.result).toLowerCase() === 'true' || sanitized.result === 1);
            } else {
                sanitized.result = null;
            }
            return sanitized;
        };
        
        const handleInitialSignals = (signalsArray) => { // 修改: 不再预排序和切片
            if (!Array.isArray(signalsArray)) {
                liveSignals.value = []; return;
            }
            // liveSignals 存储原始/完整列表，displayedManagedSignals 会负责过滤和排序
            liveSignals.value = signalsArray.map(s => sanitizeSignal(s));
            updateAllTimeRemaining();
        };
        
        const handleNewSignal = (signalData) => {
            // ... (这部分逻辑保持不变)
            const newSignal = sanitizeSignal(signalData);
            
            if (newSignal.investment_amount && newSignal.profit_rate_pct && newSignal.potential_profit === null) {
                newSignal.potential_profit = parseFloat((newSignal.investment_amount * (newSignal.profit_rate_pct / 100)).toFixed(2));
            }
            if (newSignal.investment_amount && newSignal.loss_rate_pct && newSignal.potential_loss === null) {
                newSignal.potential_loss = parseFloat((newSignal.investment_amount * (newSignal.loss_rate_pct / 100)).toFixed(2));
            }

            const existingIdx = liveSignals.value.findIndex(s => s.id === newSignal.id);
            if (existingIdx === -1) {
                liveSignals.value.unshift(newSignal);
                 if (liveSignals.value.length > 2000) { // 限制列表总长度 (按需调整)
                    liveSignals.value.splice(1000); // 例如，保留最新的1000条，移除旧的1000条
                }
            } else {
                liveSignals.value[existingIdx] = { ...liveSignals.value[existingIdx], ...newSignal };
            }
            updateAllTimeRemaining();
        };

        const handleVerifiedSignal = (signalData) => {
            // ... (这部分逻辑保持不变)
            const verifiedSignal = sanitizeSignal(signalData);
            verifiedSignal.verified = true; 
            const index = liveSignals.value.findIndex(s => s.id === verifiedSignal.id);
            if (index !== -1) {
                liveSignals.value[index] = { ...liveSignals.value[index], ...verifiedSignal };
            } else {
                liveSignals.value.unshift(verifiedSignal);
                 if (liveSignals.value.length > 2000) { liveSignals.value.splice(1000); }
            }
            updateAllTimeRemaining();
        };

        // --- 新增: 信号管理方法 ---
        const toggleSelectAllDisplayedManagedSignals = (forceValue = undefined) => {
            const displayedIds = displayedManagedSignals.value.map(s => s.id);
            // 如果 forceValue 是布尔值 (来自 v-model set)，则使用它
            // 否则 (来自 @change 事件或直接调用无参数)，则切换当前状态
            const shouldSelectAll = (typeof forceValue === 'boolean') ? forceValue : !areAllDisplayedManagedSignalsSelected.value;

            if (shouldSelectAll) {
                displayedIds.forEach(id => {
                    if (!selectedSignalIds.value.includes(id)) {
                        selectedSignalIds.value.push(id);
                    }
                });
            } else {
                selectedSignalIds.value = selectedSignalIds.value.filter(id => !displayedIds.includes(id));
            }
        };

        const deleteSelectedSignals = async () => {
            if (selectedSignalIds.value.length === 0) {
                showServerMessage("没有选中任何信号。", true);
                return;
            }
            if (!confirm(`确定要删除选中的 ${selectedSignalIds.value.length} 个信号吗？此操作不可恢复。`)) {
                return;
            }

            deletingSignals.value = true;
            try {
                const response = await axios.post('/api/live-signals/delete-batch', {
                    signal_ids: selectedSignalIds.value // 发送ID数组
                });

                if (response.data.status === 'success' || response.data.status === 'warning') {
                    const deletedActuallyCount = response.data.deleted_count || 0;
                    if (deletedActuallyCount > 0) {
                        // 从本地 liveSignals 中移除 (如果后端没有通过 WebSocket 通知所有客户端)
                        // 从本地 liveSignals 中移除
                        const remainingSignals = liveSignals.value.filter(signal => !selectedSignalIds.value.includes(signal.id));
                        liveSignals.value = remainingSignals; // 强制更新引用以触发 Vue 响应性
                        // 注意: 后端现在会通过 'signals_deleted_notification' 来通知，所以这里可以不主动移除
                        // 但为了即时性，可以保留，或者依赖WebSocket消息。
                        // 如果后端广播了 'signals_deleted_notification', 客户端的 WebSocket 处理器会处理。
                    }
                    selectedSignalIds.value = []; // 清空选择
                    showServerMessage(response.data.message || `${deletedActuallyCount} 个信号已处理。`, false, 5000);
                    // stats 会通过WebSocket的 "stats_update" 消息自动更新
                } else {
                    showServerMessage(response.data.message || "删除信号失败。", true);
                }
            } catch (err) {
                console.error("删除信号错误:", err);
                showServerMessage(err.response?.data?.detail || err.message || "删除信号时发生网络错误。", true);
            } finally {
                deletingSignals.value = false;
            }
        };

        // --- UI Helpers ---
        const audioContext = ref(null);
        const playSound = () => {
             // ... (这部分逻辑保持不变)
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
            // ... (这部分逻辑保持不变)
            const list = type === 'prediction' ? predictionStrategies.value : investmentStrategies.value;
            const strategy = list.find(s => s.id === strategyId);
            return strategy ? strategy.name : (strategyId || "未知");
        };

        // --- Countdown Timer for Signals ---
        const timeRemainingRefs = ref({});
        let countdownInterval = null;

        const updateAllTimeRemaining = () => {
            // ... (这部分逻辑保持不变)
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
            // ... (这部分逻辑保持不变)
            const now = new Date(); 
            const sevenDaysAgo = new Date();
            sevenDaysAgo.setDate(now.getDate() - 7);
            analysisFilter.value.endDate = utilFormatDateForInput(now);
            analysisFilter.value.startDate = utilFormatDateForInput(sevenDaysAgo);
        };
        const analyzeHistoricalData = () => {
            // ... (这部分逻辑保持不变)
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
            // ... (这部分逻辑保持不变)
            const verified = signals.filter(s => s.verified);
            const correctSignals = verified.filter(s => s.result === true);
            const longSignals = verified.filter(s => s.signal === 1);
            const longCorrect = longSignals.filter(s => s.result === true);
            const shortSignals = verified.filter(s => s.signal === -1);
            const shortCorrect = shortSignals.filter(s => s.result === true);
            
            let totalProfitVal = 0, totalLossVal = 0, profitCountVal = 0, lossCountVal = 0;
            verified.forEach(signal => {
                const pnl = parseFloat(signal.actual_profit_loss_amount);
                if (isNaN(pnl)) return;
                if (signal.result === true) {
                    profitCountVal++; 
                    totalProfitVal += Math.abs(pnl);
                } else if (signal.result === false) {
                    lossCountVal++; 
                    totalLossVal += Math.abs(pnl);
                }
            });
            const avgProfit = profitCountVal > 0 ? totalProfitVal / profitCountVal : 0;
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
                total_loss: parseFloat(totalLossVal.toFixed(2)),
                avg_profit: parseFloat(avgProfit.toFixed(2)),
                avg_loss: parseFloat(avgLoss.toFixed(2)),
                profit_loss_ratio: avgLoss > 0 ? parseFloat((avgProfit / avgLoss).toFixed(2)) : (avgProfit > 0 ? Infinity : 0)
            };
        };

        // --- Lifecycle Hooks ---
        onMounted(async () => {
            loadingInitialData.value = true;
            loadFavorites();
            setDefaultDateRange(); // For analysis filter
            
            const storedConfigId = localStorage.getItem('liveTestConfigId');
            if (storedConfigId) currentConfigId.value = storedConfigId;

            await fetchAllSavedParameters();
            await Promise.all([
                fetchSymbols(),
                fetchPredictionStrategies(),
                fetchInvestmentStrategies()
            ]);

            if (predictionStrategies.value.length > 0 && !selectedPredictionStrategy.value) {
                const defaultPred = predictionStrategies.value.find(s => s.id === 'simple_rsi') || predictionStrategies.value[0];
                selectedPredictionStrategy.value = defaultPred;
            }
            if (investmentStrategies.value.length > 0 && !selectedInvestmentStrategy.value) {
                const defaultInv = investmentStrategies.value.find(s => s.id === 'fixed') || investmentStrategies.value[0];
                selectedInvestmentStrategy.value = defaultInv;
            }
            
            loadingInitialData.value = false;
            // connectWebSocket(); // 用户可以手动点击连接按钮
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
            liveSignals, loadingInitialData, error, serverMessage,
            socketStatus, stats, monitorSettings, currentConfigId,
            applyingConfig, stoppingTest,

            selectedSignalIds,
            deletingSignals,
            signalManagementFilter,

            // +++ START: ADDED TO RETURN +++
            activeTestConfigDetails, 
            // +++ END: ADDED TO RETURN +++

            // Computed
            displayedManagedSignals, 
            areAllDisplayedManagedSignalsSelected, 

            // Methods
            toggleFavorite, isFavorite,
            startLiveTestService, stopLiveTestService,
            sendRuntimeConfig, stopCurrentTest, saveStrategyParameters,
            getStrategyName,
            getTimeRemaining, isTimeRemainingRelevant,
            
            toggleSelectAllDisplayedManagedSignals,
            deleteSelectedSignals,
            
            analysisFilter, analysisResults, hasAnalyzed, filteredStats, analyzeHistoricalData,
        };
    }
});

// 注册全局属性 (保持不变)
const utilsToRegister = { formatDateTime, getWinRateClass, getPnlClass, getSignalStatusClass };
for (const key in utilsToRegister) {
    if (typeof utilsToRegister[key] === 'function') {
        app.config.globalProperties[key] = utilsToRegister[key];
    } else {
        app.config.globalProperties[key] = (val) => val || (key.includes('Class') ? '' : '-');
        console.error(`utils.js: ${key} function is not defined globally.`);
    }
}

app.mount('#app');