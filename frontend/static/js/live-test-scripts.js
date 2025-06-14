// frontend/static/js/live-test-scripts.js

const { createApp, ref, onMounted, onUnmounted, computed, watch, getCurrentInstance } = Vue;

const app = createApp({
    setup() {
        // --- From utils.js 引入或准备使用的函数 ---
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
        const signalDisplayMode = ref('current'); // 'current' 或 'historical'
        const loadingInitialData = ref(false);
        const error = ref(null);
        const serverMessage = ref('');
        let serverMessageTimer = null;
        const validationErrors = ref([]); // 用于存储后端返回的字段级别错误详情
// --- Validation Error Helper ---
        // 根据路径查找并返回特定的验证错误信息
        const getValidationError = (path) => {
            if (!Array.isArray(validationErrors.value) || validationErrors.value.length === 0 || !Array.isArray(path) || path.length === 0) {
                return null;
            }
            // 查找 loc 数组与给定 path 数组完全匹配的错误
            const error = validationErrors.value.find(err => {
                if (!Array.isArray(err.loc) || err.loc.length !== path.length + 1) { // +1 because err.loc includes 'body' or 'query'
                    return false;
                }
                // 比较路径的每个元素，跳过 err.loc 的第一个元素
                for (let i = 0; i < path.length; i++) {
                    if (err.loc[i + 1] !== path[i]) {
                         return false;
                    }
                }
                return true;
            });
            return error ? error.msg : null;
        };

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
            interval: '1m', // Default to 1 minute interval
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
                min_trade_interval_minutes: 0,
            }
        });
        
        // +++ START: ADDED/MODIFIED STATE +++
        // 用于存储当前活动配置的完整细节，包括余额和盈亏
        const activeTestConfigDetails = ref(null);
        // +++ END: ADDED/MODIFIED STATE +++

        const selectedSignalIds = ref([]); 
        const deletingSignals = ref(false); 

        const signalManagementFilter = ref({
            symbol: 'all',
            interval: '1m', // Default filter interval to 1 minute
            direction: 'all',
            verifiedStatus: 'all',
            minConfidence: 0,
            maxConfidence: 100,
            startDate: '', // 新增：开始日期时间
            endDate: '',   // 新增：结束日期时间
        });


        // +++ START: PAGINATION STATE +++
        const currentPage = ref(1);
        const itemsPerPage = ref(20); // Default items per page
        // +++ END: PAGINATION STATE +++

        // --- Computed Properties ---
        // 原 latestSignals 已被 displayedManagedSignals 取代主要功能
        // 如果仍有其他地方需要仅显示最新几条，可以保留或调整
        // const latestSignals = computed(() => { ... }); 

        const displayedManagedSignals = computed(() => {
            if (!Array.isArray(liveSignals.value)) return [];

            let filtered = [...liveSignals.value];

            // Filter based on display mode
            if (signalDisplayMode.value === 'current') {
                // Only show signals for the current active config
                if (currentConfigId.value) {
                    filtered = filtered.filter(s => s.config_id === currentConfigId.value);
                } else {
                    // If no current config is active, show nothing in 'current' mode
                    return [];
                }
            } else if (signalDisplayMode.value === 'historical') {
                // Show all signals that have a config_id (i.e., not test broadcasts without one)
                // and are verified or pending verification from a past/current config.
                // We exclude 'test_signal_broadcast_all' unless it has a proper config_id assigned.
                filtered = filtered.filter(s => s.config_id && s.config_id !== 'test_signal_broadcast_all');
            } else {
                 // Default or unknown mode, show nothing
                 return [];
            }

            // 应用筛选条件
            if (signalManagementFilter.value.symbol !== 'all') {
                if (signalManagementFilter.value.symbol === 'favorites') {
                    filtered = filtered.filter(s => favoriteSymbols.value.includes(s.symbol));
                } else {
                    filtered = filtered.filter(s => s.symbol === signalManagementFilter.value.symbol);
                }
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
            // 应用日期时间筛选
            let startDate = null;
            if (signalManagementFilter.value.startDate) {
                // 解析 YYYY-MM-DDTHH:mm 格式字符串为本地时间
                const parts = signalManagementFilter.value.startDate.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
                if (parts) {
                    // parts[1]=年, parts[2]=月 (1-12), parts[3]=日, parts[4]=时, parts[5]=分
                    // Date 构造函数中的月份是 0-indexed (0-11)
                    startDate = new Date(parseInt(parts[1]), parseInt(parts[2]) - 1, parseInt(parts[3]), parseInt(parts[4]), parseInt(parts[5]));
                }
            }

            let endDate = null;
            if (signalManagementFilter.value.endDate) {
                 // 解析 YYYY-MM-DDTHH:mm 格式字符串为本地时间
                 const parts = signalManagementFilter.value.endDate.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
                 if (parts) {
                     // Date 构造函数中的月份是 0-indexed (0-11)
                     endDate = new Date(parseInt(parts[1]), parseInt(parts[2]) - 1, parseInt(parts[3]), parseInt(parts[4]), parseInt(parts[5]));
                 }
            }

            if (startDate || endDate) {
                filtered = filtered.filter(s => {
                    // signalTime 是后端提供的 UTC 时间
                    const signalTime = s.signal_time ? new Date(s.signal_time) : null;
                    if (!signalTime || isNaN(signalTime.getTime())) return false; // 忽略无效时间信号

                    let pass = true;
                    // 将本地时间 Date 对象转换为 UTC 毫秒进行比较
                    if (startDate && signalTime.getTime() < startDate.getTime()) {
                        pass = false;
                    }
                    if (endDate && signalTime.getTime() > endDate.getTime()) {
                        pass = false;
                    }
                    return pass;
                });
            }

            // 按信号时间倒序排列
            const sortedFiltered = filtered.sort((a, b) => {
                const timeA = a.signal_time ? new Date(a.signal_time).getTime() : 0;
                const timeB = b.signal_time ? new Date(b.signal_time).getTime() : 0;
                if (isNaN(timeA) || isNaN(timeB)) return 0; // 处理无效日期
                return timeB - timeA;
            });

            // 为每个信号添加一个本地日期字段，供日历分组使用
            const signalsWithLocalDate = sortedFiltered.map(s => {
                const signalTime = s.signal_time ? new Date(s.signal_time) : null;
                let localDate = null;
                if (signalTime && !isNaN(signalTime.getTime())) {
                    // 使用 toLocaleDateString 获取本地日期字符串，然后解析
                    // 这样可以确保获取的是本地时区的日期
                    const localDateString = signalTime.toLocaleDateString('zh-CN', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone // 使用浏览器本地时区
                    });
                    // 解析本地日期字符串为 Date 对象 (时间部分为本地时区的午夜)
                    localDate = new Date(localDateString);
                }
                return {
                    ...s,
                    local_date: localDate // 添加本地日期字段
                };
            });

            console.log("LiveTest: displayedManagedSignals computed property returning", signalsWithLocalDate.length, "signals with local_date."); // Added log
            return signalsWithLocalDate;
            // .slice(0, 50); // 可选：如果信号过多，可以限制初始显示数量或实现分页
        });

        // +++ START: PAGINATION COMPUTED +++
        const totalPages = computed(() => {
            return Math.ceil(displayedManagedSignals.value.length / itemsPerPage.value);
        });

        const paginatedSignals = computed(() => {
            const start = (currentPage.value - 1) * itemsPerPage.value;
            const end = start + itemsPerPage.value;
            return displayedManagedSignals.value.slice(start, end);
        });
        // +++ END: PAGINATION COMPUTED +++

        // --- Computed Properties for Filtered Signals Stats ---
        const filteredWinRateStats = computed(() => {
            const filteredSignals = displayedManagedSignals.value;
            const totalFiltered = filteredSignals.length;
            const verifiedFiltered = filteredSignals.filter(s => s.verified).length;
            const correctFiltered = filteredSignals.filter(s => s.verified && s.result).length;

            const winRateFiltered = verifiedFiltered > 0 ? (correctFiltered / verifiedFiltered) * 100 : 0;

            // Calculate Total and Average Reference PnL (%) for filtered signals
            let totalFilteredPnlPct = 0;
            let pnlCount = 0;
            filteredSignals.forEach(signal => {
                // Only include signals that have a valid numeric pnl_pct
                if (signal.pnl_pct !== null && signal.pnl_pct !== undefined && !isNaN(signal.pnl_pct)) {
                    totalFilteredPnlPct += parseFloat(signal.pnl_pct);
                    pnlCount++;
                }
            });

            const averageFilteredPnlPct = pnlCount > 0 ? totalFilteredPnlPct / pnlCount : 0;

            return {
                total_signals: totalFiltered,
                total_verified: verifiedFiltered,
                total_correct: correctFiltered,
                win_rate: winRateFiltered,
                total_pnl_pct: totalFilteredPnlPct, // Add total PnL (%)
                average_pnl_pct: averageFilteredPnlPct // Add average PnL (%)
            };
        });

        // --- Computed Property for Filtered Total Profit/Loss Amount ---
        const filteredTotalProfitLossAmount = computed(() => {
            const filteredSignals = displayedManagedSignals.value;
            let totalProfitLoss = 0;
            filteredSignals.forEach(signal => {
                // Only include signals that are verified and have a valid actual_profit_loss_amount
                if (signal.verified && signal.actual_profit_loss_amount !== null && signal.actual_profit_loss_amount !== undefined && !isNaN(signal.actual_profit_loss_amount)) {
                    totalProfitLoss += parseFloat(signal.actual_profit_loss_amount);
                }
            });
            return totalProfitLoss;
        });

        const dynamicFilteredBalance = computed(() => {
            const baseBalance = activeTestConfigDetails.value?.investment_settings?.simulatedBalance ?? monitorSettings.value.investment.simulatedBalance;
            const profitLoss = filteredTotalProfitLossAmount.value || 0;
            return parseFloat((baseBalance + profitLoss).toFixed(2));
        });


        watch(displayedManagedSignals, (newValue) => {
            console.log("LiveTest: displayedManagedSignals updated:", newValue);
        }, { immediate: true }); // Watch immediately and on changes

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

                // After fetching symbols, set default monitor and filter symbols based on favorites
                if (symbols.value.length > 0) {
                    const firstFavorite = symbols.value.find(s => isFavorite(s));
                    if (firstFavorite) {
                        // Set monitor symbol to the first favorite
                        monitorSettings.value.symbol = firstFavorite;
                        // Set filter symbol to 'favorites'
                        signalManagementFilter.value.symbol = 'favorites';
                    } else {
                        // If no favorites, default monitor symbol to 'all' (or the first symbol if 'all' is not an option)
                        // 'all' is already the default, but explicitly setting it here for clarity
                        monitorSettings.value.symbol = 'all';
                        // If no favorites, default filter symbol to 'all'
                        signalManagementFilter.value.symbol = 'all';
                    }
                } else {
                     // If no symbols are fetched, default both to 'all'
                     monitorSettings.value.symbol = 'all';
                     signalManagementFilter.value.symbol = 'all';
                }

                console.log("LiveTest: fetchSymbols - monitorSettings.value.symbol after setting:", monitorSettings.value.symbol);
                console.log("LiveTest: fetchSymbols - signalManagementFilter.value.symbol after setting:", signalManagementFilter.value.symbol);

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

            // Only update monitorSettings.value.symbol from configDetails if it's not 'all'
            // This allows the default set by fetchSymbols (first favorite) to persist if the saved config was 'all'
            if (configDetails.symbol && configDetails.symbol !== 'all') {
                 monitorSettings.value.symbol = configDetails.symbol;
            } else if (!configDetails.symbol) {
                 // If configDetails.symbol is null/undefined/empty, explicitly set to 'all' if it wasn't already set by favorites
                 // This handles cases where a config might not have a symbol field
                 if (monitorSettings.value.symbol === 'all') {
                     monitorSettings.value.symbol = 'all';
                 }
            }
            // If configDetails.symbol is 'all', we do nothing here, preserving the value set by fetchSymbols

            // Also update signalManagementFilter.value.symbol based on the loaded config symbol
            const storedFavorites = localStorage.getItem('favoriteSymbols');
            let favoritesExistAndNotEmpty = false;
            if (storedFavorites) {
                try {
                    const parsedFavorites = JSON.parse(storedFavorites);
                    if (Array.isArray(parsedFavorites) && parsedFavorites.length > 0) {
                        favoritesExistAndNotEmpty = true;
                    }
                } catch (e) {
                    console.warn("LiveTest: Could not parse favoriteSymbols from localStorage in populateUiFromConfigDetails", e);
                }
            }

            if (favoritesExistAndNotEmpty) {
                // Favorites exist and are not empty
                // If configDetails.symbol is a specific trading pair (not 'all' and not empty), prioritize it.
                if (configDetails.symbol && configDetails.symbol !== 'all') {
                    signalManagementFilter.value.symbol = configDetails.symbol;
                } else {
                    // Otherwise (configDetails.symbol is 'all', empty, or undefined), set to 'favorites'.
                    signalManagementFilter.value.symbol = 'favorites';
                }
            } else {
                // No favorites, or favorites are empty. Use existing logic.
                if (configDetails.symbol) {
                     signalManagementFilter.value.symbol = configDetails.symbol;
                } else {
                     // If configDetails.symbol is null/undefined/empty, default filter to 'all'
                     signalManagementFilter.value.symbol = 'all';
                }
            }


            monitorSettings.value.interval = configDetails.interval || 'all';
            monitorSettings.value.confidence_threshold = configDetails.confidence_threshold ?? 50;
            monitorSettings.value.event_period = configDetails.event_period || '10m';

            const predStrategy = predictionStrategies.value.find(s => s.id === configDetails.prediction_strategy_id);
            if (predStrategy) {
                selectedPredictionStrategy.value = predStrategy;
                const defaultParams = {};
                if (predStrategy.parameters) {
                    predStrategy.parameters.forEach(param => {
                        if (!param.advanced) {
                            defaultParams[param.name] = param.type === 'boolean' ? (param.default === true) : param.default;
                        }
                    });
                }
                const paramsFromServer = configDetails.prediction_strategy_params || {};
                predictionStrategyParams.value = { ...defaultParams, ...paramsFromServer };
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
                    // Start of new logic: Properly merge default and server params
                    const defaultParams = {};
                    if (invStrategy.parameters) {
                        invStrategy.parameters.forEach(param => {
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

                    const paramsFromServer = invSettingsFromServer.investment_strategy_specific_params || {};
                    let mergedParams = { ...defaultParams, ...paramsFromServer };

                    if (invStrategy.id === 'martingale_user_defined' && mergedParams.sequence && Array.isArray(mergedParams.sequence)) {
                        mergedParams.sequence = mergedParams.sequence.join(',');
                    }
                    
                    investmentStrategyParams.value = mergedParams;
                    // End of new logic
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
                            if (!param.advanced && !param.readonly) {
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

            let connectTimeoutId = null;
            const CONNECTION_TIMEOUT = 15000; // 15 秒连接超时
            let reconnectAttempts = 0;
            const MAX_RECONNECT_ATTEMPTS = 5;
            const RECONNECT_DELAY = 5000; // 5 秒

            const clearConnectTimeout = () => {
                if (connectTimeoutId) {
                    clearTimeout(connectTimeoutId);
                    connectTimeoutId = null;
                }
            };

            const attemptReconnect = () => {
                if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                    reconnectAttempts++;
                    showServerMessage(`WebSocket 连接已关闭，将在 ${RECONNECT_DELAY / 1000} 秒后尝试重新连接 (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`, true, RECONNECT_DELAY);
                    setTimeout(() => {
                        if (socketStatus.value !== 'connected' && socketStatus.value !== 'connecting') {
                             connectWebSocket(); // 重新调用自身以尝试连接
                        }
                    }, RECONNECT_DELAY);
                } else {
                    showServerMessage(`WebSocket 自动重连失败已达上限 (${MAX_RECONNECT_ATTEMPTS} 次)。请手动尝试连接。`, true, 7000);
                    reconnectAttempts = 0; // 重置尝试次数
                }
            };


            socket.value = new WebSocket(wsUrl);
            socketStatus.value = 'connecting';
            showServerMessage("WebSocket 正在连接...", false, CONNECTION_TIMEOUT + 1000); // 显示连接中消息

            connectTimeoutId = setTimeout(() => {
                if (socket.value && socket.value.readyState !== WebSocket.OPEN) {
                    console.error(`LiveTest: WebSocket connection timed out after ${CONNECTION_TIMEOUT / 1000} seconds.`);
                    showServerMessage(`WebSocket 连接超时 (${CONNECTION_TIMEOUT / 1000}秒)。请检查网络或服务器状态。`, true, 7000);
                    socketStatus.value = 'error';
                    if (socket.value) {
                        socket.value.close(1000, "Connection timeout"); // 主动关闭
                    }
                    // onclose 会被触发，可以在那里处理重连逻辑
                }
            }, CONNECTION_TIMEOUT);

            socket.value.onopen = () => {
                clearConnectTimeout();
                reconnectAttempts = 0; // 连接成功，重置重连尝试次数
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
                                    console.log("LiveTest: Received applied_config:", message.data.applied_config);
                                    populateUiFromConfigDetails(message.data.applied_config);
                                    // 更新活动配置细节
                                    activeTestConfigDetails.value = message.data.applied_config;
                                    console.log("LiveTest: activeTestConfigDetails after config_set_confirmation:", activeTestConfigDetails.value);
                                } else {
                                    console.log("LiveTest: config_set_confirmation received, but no applied_config in message.data");
                                }
                            } else {
                                showServerMessage(message.data.message || '配置应用失败。', true, 5000);
                                activeTestConfigDetails.value = null; // 清理
                            }
                            break;
                        case "session_restored":
                            console.log("LiveTest: Received session_recovered message:", message.data);
                            if (message.data && message.data.config_id) {
                                currentConfigId.value = message.data.config_id;
                                localStorage.setItem('liveTestConfigId', currentConfigId.value);
                                if (message.data.config_details) {
                                    populateUiFromConfigDetails(message.data.config_details);
                                    // 使用扩展运算符创建新对象以确保响应性
                                    activeTestConfigDetails.value = { ...message.data.config_details };
                                    console.log("LiveTest: activeTestConfigDetails after session_restored (new object):", activeTestConfigDetails.value);
                                    // 进一步确认关键字段是否已正确设置在 activeTestConfigDetails.value 中
                                    if (activeTestConfigDetails.value) {
                                        console.log("LiveTest: activeTestConfigDetails.value.current_balance:", activeTestConfigDetails.value.current_balance);
                                        console.log("LiveTest: activeTestConfigDetails.value.total_profit_loss_amount:", activeTestConfigDetails.value.total_profit_loss_amount);
                                    }
                                }
                                showServerMessage(`会话已恢复 (ID: ${currentConfigId.value.substring(0,8)}...)`, false, 4000);
                            } else {
                                showServerMessage('会话恢复失败：无效的配置信息', true, 4000);
                            }
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
                                // 清理活动配置细节
                                activeTestConfigDetails.value = null;
                            } else {
                                showServerMessage(message.data.message || '停止测试失败。', true, 5000);
                            }
                            break;
                        // +++ END: MODIFIED CASES +++

                        // 处理活动配置通知 - 新增处理
                        case "active_config_notification":
                            console.log("LiveTest: Received active_config_notification message:", message.data);
                            if (message.data && message.data.config_id && message.data.config) {
                                // 只有当前没有活动配置时才应用收到的配置
                                if (!currentConfigId.value) {
                                    currentConfigId.value = message.data.config_id;
                                    localStorage.setItem('liveTestConfigId', currentConfigId.value);
                                    populateUiFromConfigDetails(message.data.config);
                                    // 更新活动配置细节
                                    activeTestConfigDetails.value = message.data.config;
                                    console.log("LiveTest: activeTestConfigDetails after active_config_notification:", activeTestConfigDetails.value);
                                    showServerMessage(message.data.message || '已连接到活动测试配置', false, 4000);
                                } else {
                                    console.log("LiveTest: 已有活动配置，忽略活动配置通知");
                                }
                            }
                            break;

                        // +++ START: NEW CASE HANDLER +++
                        case "config_specific_balance_update":
                            console.log("LiveTest: Received config_specific_balance_update message:", message.data);
                            console.log("LiveTest: Current config ID:", currentConfigId.value);
                            if (message.data && message.data.config_id === currentConfigId.value) {
                                console.log("LiveTest: Config ID matches. Updating balance and PnL.");
                                if (activeTestConfigDetails.value) {
                                    console.log("LiveTest: activeTestConfigDetails exists.");
                                    if (message.data.new_balance !== undefined) {
                                        activeTestConfigDetails.value.current_balance = message.data.new_balance;
                                        console.log("LiveTest: Updated current_balance to:", activeTestConfigDetails.value.current_balance);
                                    }
                                    if (message.data.total_profit_loss_amount !== undefined) {
                                        activeTestConfigDetails.value.total_profit_loss_amount = message.data.total_profit_loss_amount; // Corrected field name
                                        console.log("LiveTest: Updated total_profit_loss_amount to:", activeTestConfigDetails.value.total_profit_loss_amount);
                                    } else {
                                         console.log("LiveTest: message.data.total_profit_loss_amount is undefined.");
                                    }
                                    console.log("LiveTest: activeTestConfigDetails after balance update:", activeTestConfigDetails.value); // Added log
                                } else {
                                    console.log("LiveTest: activeTestConfigDetails is null.");
                                }
                                // 可选: 短暂显示余额更新消息
                                // showServerMessage(`当前测试余额更新为: ${message.data.new_balance.toFixed(2)} USDT (本次盈亏: ${message.data.last_pnl_amount.toFixed(2)})`, false, 3000);
                            } else {
                                 console.log("LiveTest: Config ID does NOT match or message.data is invalid.");
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
                            // 处理后端返回的结构化错误信息
                            if (message.data && message.data.details && Array.isArray(message.data.details)) {
                                validationErrors.value = message.data.details; // 存储详细错误列表
                                // 清除通用的错误和消息，因为我们将使用字段级别的提示
                                serverMessage.value = '';
                                error.value = null;
                                console.error("LiveTest: Received validation errors:", validationErrors.value);
                                // 可以选择在这里显示一个通用的“请检查表单中的错误”消息
                                // showServerMessage("请检查表单中的错误。", true, 5000);
                            } else {
                                // 原始的非结构化错误处理
                                validationErrors.value = []; // 清空字段级别错误
                                showServerMessage(message.data?.message || "收到来自服务器的 WebSocket 错误", true, 6000);
                                console.error("LiveTest: Received generic error:", message.data);
                            }
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
                clearConnectTimeout(); // 清除连接超时计时器
                socketStatus.value = 'disconnected';
                socket.value = null;
                applyingConfig.value = false; stoppingTest.value = false;

                let closeReason = `代码: ${event.code}, 原因: ${event.reason || '无'}`;
                if (event.code === 1000 && event.reason === "Connection timeout") {
                     // 这是我们自己触发的超时关闭，已经在 connectTimeoutId 中处理了消息
                } else if (!event.wasClean) {
                    console.warn(`LiveTest: WebSocket connection closed unexpectedly. ${closeReason}`);
                    showServerMessage(`WebSocket 连接意外断开 (${closeReason})。如果后台有测试在运行，它仍将继续。`, true, 7000);
                    // 尝试重连，除非是用户主动关闭或达到最大尝试次数
                    if (event.code !== 1000 && event.code !== 1001 && event.code !== 1005) { // 1000=Normal, 1001=Going Away, 1005=No Status Rcvd (often client-side close)
                        attemptReconnect();
                    } else {
                        reconnectAttempts = 0; // 重置，因为这是预期的关闭
                    }
                } else {
                    showServerMessage(`WebSocket 连接已关闭 (${closeReason})。`, false, 3000);
                    reconnectAttempts = 0; // 正常关闭，重置重连尝试
                }
            };

            socket.value.onerror = (errEvent) => {
                clearConnectTimeout(); // 清除连接超时计时器
                console.error("LiveTest: WebSocket error: ", errEvent);
                // 尝试获取更详细的错误信息
                let errorDetails = "未知错误";
                if (errEvent.message) errorDetails = errEvent.message;
                else if (errEvent.type) errorDetails = `类型: ${errEvent.type}`;
                
                socketStatus.value = 'error';
                applyingConfig.value = false; stoppingTest.value = false;
                showServerMessage(`WebSocket 连接错误: ${errorDetails}。请检查服务是否运行或网络连接。`, true, 7000);
                
                // onerror 之后通常会触发 onclose，所以重连逻辑主要放在 onclose 中
                // 但如果 socket 仍然存在，确保它被关闭
                if (socket.value && socket.value.readyState !== WebSocket.CLOSED) {
                    socket.value.close(1011, "WebSocket error occurred"); // 1011 = Internal Error
                }
                socket.value = null; // 确保引用被清除
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
                           showServerMessage("Martin Gauge序列无效，已使用默认值发送。", true, 3000);
                        }
                    } catch (e) {
                        const defaultSeqDef = selectedInvestmentStrategy.value.parameters.find(p => p.name === 'sequence');
                        finalInvestmentSpecificParams.sequence = defaultSeqDef?.default && Array.isArray(defaultSeqDef.default) ? defaultSeqDef.default : [10,20,40];
                        showServerMessage("解析Martin Gauge序列错误，已使用默认值发送。", true, 3000);
                    }
                }
            }
            
            const investmentSettingsPayload = {
                strategy_id: selectedInvestmentStrategy.value.id,
                investment_strategy_specific_params: finalInvestmentSpecificParams,
                // 确保包含 InvestmentStrategySettings Pydantic 模型的所有字段
                amount: selectedInvestmentStrategy.value.id === 'fixed'
                    ? parseFloat(finalInvestmentSpecificParams.amount)
                    : parseFloat(monitorSettings.value.investment.simulatedBalance),
                minAmount: parseFloat(monitorSettings.value.investment.minAmount),
                maxAmount: parseFloat(monitorSettings.value.investment.maxAmount),
                percentageOfBalance: parseFloat(finalInvestmentSpecificParams.percentageOfBalance ?? monitorSettings.value.investment.percentageOfBalance),
                profitRate: parseFloat(monitorSettings.value.investment.profitRate),
                lossRate: parseFloat(monitorSettings.value.investment.lossRate),
                simulatedBalance: parseFloat(monitorSettings.value.investment.simulatedBalance),
                min_trade_interval_minutes: parseFloat(monitorSettings.value.investment.min_trade_interval_minutes) || 0,
            };

            // 在发送新配置之前，清除当前的 configId，以便后端生成新的测试 ID

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

            try {
                socket.value.send(JSON.stringify(configPayload));
                console.log("LiveTest: Sending runtime config:", configPayload);
            } catch (error) {
                console.error("LiveTest: Error sending runtime config:", error);
                showServerMessage("发送配置时发生错误：" + error.message, true);
                applyingConfig.value = false;
            }
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
            // Map origin_config_id to config_id for filtering
            if (sanitized.origin_config_id && !sanitized.config_id) {
                sanitized.config_id = sanitized.origin_config_id;
            }
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
            const sanitizedSignals = signalsArray.map(s => {
                const sanitized = sanitizeSignal(s);
                // Ensure potential profit/loss are calculated/present for initial load too
                 if (sanitized.investment_amount !== null && sanitized.investment_amount !== undefined &&
                    sanitized.profit_rate_pct !== null && sanitized.profit_rate_pct !== undefined) {
                    sanitized.potential_profit = parseFloat((sanitized.investment_amount * (sanitized.profit_rate_pct / 100)).toFixed(2));
                } else {
                     sanitized.potential_profit = null;
                }

                if (sanitized.investment_amount !== null && sanitized.investment_amount !== undefined &&
                    sanitized.loss_rate_pct !== null && sanitized.loss_rate_pct !== undefined) {
                    sanitized.potential_loss = parseFloat((sanitized.investment_amount * (sanitized.loss_rate_pct / 100)).toFixed(2));
                } else {
                    sanitized.potential_loss = null;
                }
                return sanitized;
            });

            liveSignals.value = sanitizedSignals; // Assign the new array
            liveSignals.value = [...liveSignals.value]; // Force reactivity update for the array
            console.log("LiveTest: handleInitialSignals - liveSignals after update:", liveSignals.value);

            updateAllTimeRemaining();
        };
        
        const handleNewSignal = (signalData) => {
            console.log("LiveTest: Raw signal data received:", signalData); // Add this line
            const newSignal = sanitizeSignal(signalData);
            console.log("LiveTest: Sanitized signal data:", newSignal); // Add this line

            console.log("LiveTest: Received new signal:", newSignal); // Add this line
            console.log("LiveTest: Current config ID:", currentConfigId.value); // Add this line

            // Explicitly calculate and set potential profit/loss if components are available
            if (newSignal.investment_amount !== null && newSignal.investment_amount !== undefined &&
                newSignal.profit_rate_pct !== null && newSignal.profit_rate_pct !== undefined) {
                newSignal.potential_profit = parseFloat((newSignal.investment_amount * (newSignal.profit_rate_pct / 100)).toFixed(2));
            } else {
                 newSignal.potential_profit = null; // Ensure it's null if not calculable
            }

            if (newSignal.investment_amount !== null && newSignal.investment_amount !== undefined &&
                newSignal.loss_rate_pct !== null && newSignal.loss_rate_pct !== undefined) {
                newSignal.potential_loss = parseFloat((newSignal.investment_amount * (newSignal.loss_rate_pct / 100)).toFixed(2));
            } else {
                newSignal.potential_loss = null; // Ensure it's null if not calculable
            }

            console.log("LiveTest: Signal object before adding/updating liveSignals:", newSignal); // Add this line

            const existingIdx = liveSignals.value.findIndex(s => s.id === newSignal.id);
            if (existingIdx === -1) {
                liveSignals.value.unshift(newSignal);
                 if (liveSignals.value.length > 2000) { // 限制列表总长度 (按需调整)
                    liveSignals.value.splice(1000); // 例如，保留最新的1000条，移除旧的1000条
                }
            } else {
                // Update existing signal, ensuring reactivity
                // Using Vue.set or creating a new object might be more robust in some Vue versions,
                // but spread syntax should work in Vue 3. Let's stick to spread for now.
                liveSignals.value[existingIdx] = { ...liveSignals.value[existingIdx], ...newSignal };
            }

            // Add this line to force reactivity update for the array
            liveSignals.value = [...liveSignals.value];

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



        // --- Lifecycle Hooks ---
        onMounted(async () => {
            console.log("LiveTest: onMounted - start");
            loadingInitialData.value = true;
            loadFavorites();
            console.log("LiveTest: onMounted - favoriteSymbols after load:", favoriteSymbols.value);

            // The logic for setting default symbols based on favorites is now inside fetchSymbols()
            // If favorites exist, default the signal filter to favorites here as well,
            // in case fetchSymbols was skipped or failed, though fetchSymbols is awaited.
            // This is a fallback/redundancy. The primary logic is in fetchSymbols.
            if (favoriteSymbols.value.length > 0) {
                 signalManagementFilter.value.symbol = 'favorites';
            }
            console.log("LiveTest: onMounted - signalManagementFilter.value.symbol after loadFavorites:", signalManagementFilter.value.symbol);

            const storedConfigId = localStorage.getItem('liveTestConfigId');
            console.log("LiveTest: onMounted - storedConfigId:", storedConfigId);
            if (storedConfigId) {
                currentConfigId.value = storedConfigId;
                console.log("LiveTest: onMounted - currentConfigId set from localStorage:", currentConfigId.value);
            } else {
                 console.log("LiveTest: onMounted - no storedConfigId found.");
            }

            try {
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
            } catch (err) {
                console.error("LiveTest: Error during initial data fetch:", err);
                showServerMessage("加载初始数据失败。请检查后端服务是否运行。", true, 8000);
            } finally {
                loadingInitialData.value = false;
                console.log("LiveTest: onMounted - loadingInitialData set to false.");
            }
            
            console.log("LiveTest: onMounted - connecting WebSocket.");
            connectWebSocket(); // 用户可以手动点击连接按钮
            countdownInterval = setInterval(updateAllTimeRemaining, 1000);
            console.log("LiveTest: onMounted - end.");
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
            validationErrors, // Add validationErrors to the returned object
            deletingSignals,
            signalManagementFilter,

            // +++ START: ADDED TO RETURN +++
            activeTestConfigDetails,
            signalDisplayMode, // Add the new state variable
            // +++ END: ADDED TO RETURN +++

            // Computed
            displayedManagedSignals,
            filteredWinRateStats, // Add the new computed property here
            filteredTotalProfitLossAmount, // Expose the new computed property
            dynamicFilteredBalance, // <-- 新增导出
            // +++ START: PAGINATION RETURN +++
            currentPage,
            itemsPerPage,
            totalPages,
            paginatedSignals,
            // +++ END: PAGINATION RETURN +++
            areAllDisplayedManagedSignalsSelected,

            // Methods
            toggleFavorite, isFavorite,
            startLiveTestService, stopLiveTestService,
            sendRuntimeConfig, stopCurrentTest, saveStrategyParameters,
            getStrategyName,
            getTimeRemaining, isTimeRemainingRelevant,

            toggleSelectAllDisplayedManagedSignals,
            deleteSelectedSignals,

            // Validation Error Method
            getValidationError,
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