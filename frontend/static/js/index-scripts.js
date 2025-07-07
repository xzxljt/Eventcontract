// frontend/static/js/index-scripts.js

const { createApp, ref, onMounted, onUnmounted, watch, computed, getCurrentInstance } = Vue;

const app = createApp({
    setup() {
        const worker = new Worker('./static/js/filter-worker.js');

        // --- Debounce Utility ---
        const debounce = (func, delay) => {
            let timeoutId;
            return (...args) => {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    func.apply(this, args);
                }, delay);
            };
        };

        // --- 从 utils.js 引入或准备使用的函数 ---
        // 这些函数假设 utils.js 已加载并在全局作用域中可用
        // 对于模板中直接使用的，我们将在创建 app 后注册到 globalProperties
        const utilFormatDateTime = typeof formatDateTime === 'function' ? formatDateTime : (val) => val || '-';
        const utilGetWinRateClass = typeof getWinRateClass === 'function' ? getWinRateClass : () => '';
        const utilFormatDateForInput = typeof formatDateForInput === 'function' ? formatDateForInput : (d) => d.toISOString().slice(0,16);


        // --- State Variables ---
        const symbols = ref([]);
        const favoriteSymbols = ref([]); // 将由 useFavoriteSymbols 处理
        
        const predictionStrategies = ref([]);
        const selectedPredictionStrategy = ref(null);
        const predictionStrategyParams = ref({});

        const investmentStrategies = ref([]);
        const selectedInvestmentStrategy = ref(null);
        const investmentStrategyParams = ref({});
        const fieldErrors = ref({}); // <--- 新增：用于存储字段级别的错误 { fieldName: errorMessage }

        const allSavedParams = ref({
            prediction_strategies: {},
            investment_strategies: {}
        });

        const backtestResults = ref(null);
        const loading = ref(false);
        const error = ref(null); // 用于显示一般错误或回测API的文本错误
        const validationError = ref(null); // 用于显示Pydantic校验错误详情


        const backtestParams = ref({
            symbol: '',
            interval: '1m', // Default to 1 minute interval
            start_time: '',
            end_time: '',
            // startTime 和 endTime (camelCase) 仅用于 flatpickr 初始化时的 defaultDate，
            // 实际提交给后端的 payload 使用 snake_case (start_time, end_time)
            prediction_strategy_id: '', 
            eventPeriod: '10m', 
            confidence_threshold: 50,
            investment: {
                initial_balance: 1000.0,
                investment_strategy_id: 'fixed',
                investment_strategy_specific_params: { amount: 20.0 }, // 会被 watch 动态更新
                min_investment_amount: 5.0,
                max_investment_amount: 250.0,
                profit_rate_pct: 80.0,
                loss_rate_pct: 100.0,
                min_trade_interval_minutes: 0
            }
        });

        const calendarView = ref({
            year: new Date().getFullYear(),
            month: new Date().getMonth(),
            weeks: [],
            dailyPnlData: {}
        });

        // --- Favorite Symbols Logic (待提取到 useFavoriteSymbols) ---
        const loadFavorites = () => {
            const stored = localStorage.getItem('favoriteSymbols');
            if (stored) favoriteSymbols.value = JSON.parse(stored);
        };
        const saveFavorites = () => localStorage.setItem('favoriteSymbols', JSON.stringify(favoriteSymbols.value));
        const isFavorite = (symbol) => favoriteSymbols.value.includes(symbol);
        const sortSymbols = () => {
             symbols.value.sort((a, b) => {
                const aIsFav = isFavorite(a); const bIsFav = isFavorite(b);
                if (aIsFav && !bIsFav) return -1; if (!aIsFav && bIsFav) return 1;
                // 如果都不是收藏或都是收藏，则按字母排序 (可选)
                // return a.localeCompare(b);
                return 0; // 保持原有顺序或按 API 返回顺序
            });
        };
        const toggleFavorite = (symbol) => {
            if (!symbol) return;
            const index = favoriteSymbols.value.indexOf(symbol);
            if (index === -1) favoriteSymbols.value.push(symbol);
            else favoriteSymbols.value.splice(index, 1);
            saveFavorites();
            sortSymbols(); // 重新排序以更新UI
        };

        // --- Result Filtering State ---
        const filterStartTime = ref('');
        const filterEndTime = ref('');
        const minConfidence = ref(0);
        const excludedWeekdays = ref([]);

        const weekdays = ref([
           { label: '一', value: '1' }, { label: '二', value: '2' },
           { label: '三', value: '3' }, { label: '四', value: '4' },
           { label: '五', value: '5' }, { label: '六', value: '6' },
           { label: '日', value: '0' }
       ]);

       const toggleWeekday = (dayValue) => {
           const index = excludedWeekdays.value.indexOf(dayValue);
           if (index > -1) {
               excludedWeekdays.value.splice(index, 1);
           } else {
               excludedWeekdays.value.push(dayValue);
           }
       };

        const resetFilters = () => {
            filterStartTime.value = '';
            filterEndTime.value = '';
            minConfidence.value = 0;
            excludedWeekdays.value = [];
        };

        // --- Initialization and Data Fetching (部分待提取到 apiService) ---
        const initDatePickers = () => {
            const endDate = new Date();
            const startDate = new Date();
            startDate.setDate(startDate.getDate() - 7); // 默认7天前

            // 使用 utils.js 中的 formatDateForInput
            const initialStartTimeStr = utilFormatDateForInput(startDate);
            const initialEndTimeStr = utilFormatDateForInput(endDate);
            
            backtestParams.value.start_time = initialStartTimeStr;
            backtestParams.value.end_time = initialEndTimeStr;

            flatpickr("#startTime", {
                enableTime: true, dateFormat: "Y-m-d H:i", locale: "zh",
                defaultDate: initialStartTimeStr,
                onChange: (selectedDates) => {
                    if (selectedDates[0]) {
                        backtestParams.value.start_time = utilFormatDateForInput(selectedDates[0]);
                    }
                }
            });
            flatpickr("#endTime", {
                enableTime: true, dateFormat: "Y-m-d H:i", locale: "zh",
                defaultDate: initialEndTimeStr,
                onChange: (selectedDates) => {
                     if (selectedDates[0]) {
                        backtestParams.value.end_time = utilFormatDateForInput(selectedDates[0]);
                     }
                }
            });
        };

        const fetchAllSavedParameters = async () => {
            try {
                const response = await axios.get('/api/load_all_strategy_parameters');
                allSavedParams.value = response.data || { prediction_strategies: {}, investment_strategies: {} };
            } catch (err) {
                console.error('Failed to fetch all saved parameters:', err);
                error.value = '加载已保存的策略参数失败。';
                allSavedParams.value = { prediction_strategies: {}, investment_strategies: {} };
            }
        };
        
        const fetchSymbols = async () => {
            try {
                const response = await axios.get('/api/symbols');
                symbols.value = response.data;
                sortSymbols(); // 获取后排序，将收藏的排前面
                if (symbols.value.length > 0) {
                    // Default to the first favorite symbol if any exist, otherwise default to the first symbol
                    const firstFavorite = symbols.value.find(s => isFavorite(s));
                    if (firstFavorite) {
                        backtestParams.value.symbol = firstFavorite;
                    } else {
                        backtestParams.value.symbol = symbols.value[0];
                    }
                }
            } catch (err) { console.error('获取交易对失败:', err); error.value = '获取交易对失败'; }
        };

        const fetchPredictionStrategies = async () => {
            try {
                const response = await axios.get('/api/prediction-strategies');
                predictionStrategies.value = response.data;
            } catch (err) { console.error('获取预测策略失败:', err); error.value = '获取预测策略失败'; }
        };

        const fetchInvestmentStrategies = async () => {
            try {
                const response = await axios.get('/api/investment-strategies');
                investmentStrategies.value = response.data;
            } catch (err) { console.error('获取投资策略失败:', err); error.value = '获取投资策略失败'; }
        };

        // --- Strategy Parameter Management ---
        const updatePredictionStrategyParams = (newStrategy) => {
            if (newStrategy && newStrategy.id) {
                backtestParams.value.prediction_strategy_id = newStrategy.id;
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
                backtestParams.value.prediction_strategy_id = '';
                predictionStrategyParams.value = {};
            }
        };

        const updateInvestmentStrategyParams = (newStrategy) => {
            if (newStrategy && newStrategy.id) {
                backtestParams.value.investment.investment_strategy_id = newStrategy.id;
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
            } else {
                backtestParams.value.investment.investment_strategy_id = '';
                investmentStrategyParams.value = {};
            }
        };

        // --- Watchers for Strategy Selection and Parameter Population ---
        watch(selectedPredictionStrategy, updatePredictionStrategyParams, { immediate: true });

        watch(selectedInvestmentStrategy, updateInvestmentStrategyParams, { immediate: true });
        
        // 当UI上的投资策略参数变化时，同步到 backtestParams.investment.investment_strategy_specific_params
        watch(investmentStrategyParams, (newSpecificParams) => {
            let paramsToSync = { ...newSpecificParams };
            // 如果是 martingale_user_defined 且 sequence 是字符串，尝试转换回数组给后端
            if (selectedInvestmentStrategy.value && selectedInvestmentStrategy.value.id === 'martingale_user_defined') {
                const seqParamInfo = selectedInvestmentStrategy.value.parameters.find(p => p.name === 'sequence');
                if (seqParamInfo && typeof newSpecificParams.sequence === 'string') {
                    try {
                        const parsedSequence = newSpecificParams.sequence.split(',')
                            .map(s => parseFloat(s.trim()))
                            .filter(n => !isNaN(n) && n > 0);
                        // 只有当解析后的数组非空时才使用，否则保留原始字符串（可能后端需要原始字符串或有自己的默认）
                        // 或者，如果希望强制使用默认值，可以在这里处理
                        if (parsedSequence.length > 0) {
                            paramsToSync.sequence = parsedSequence;
                        } else {
                            // 如果解析结果为空数组，则使用策略定义中的默认值（如果是数组）
                            paramsToSync.sequence = Array.isArray(seqParamInfo.default) ? seqParamInfo.default : [10,20,40]; //最后的[10,20,40]是硬编码后备
                        }
                    } catch (e) {
                        // 解析出错，也使用默认值
                        paramsToSync.sequence = Array.isArray(seqParamInfo.default) ? seqParamInfo.default : [10,20,40];
                    }
                } else if (seqParamInfo && typeof newSpecificParams.sequence === 'undefined' && Array.isArray(seqParamInfo.default)) {
                    // 如果UI上没有sequence值（例如刚切换策略，且无保存值），则使用默认值
                    paramsToSync.sequence = seqParamInfo.default;
                }
            }
            backtestParams.value.investment.investment_strategy_specific_params = paramsToSync;
        }, { deep: true });


        // --- Core Logic Functions ---
        const ensureGlobalInvestmentSettings = () => {
            const investment = backtestParams.value.investment;
            investment.initial_balance = Number(investment.initial_balance) || 1000.0;
            investment.profit_rate_pct = Number(investment.profit_rate_pct) || 80.0;
            investment.loss_rate_pct = Number(investment.loss_rate_pct) || 100.0;
            investment.min_investment_amount = Number(investment.min_investment_amount) || 5.0;
            investment.max_investment_amount = Number(investment.max_investment_amount) || 250.0;
        };

        const runBacktest = async () => {
            loading.value = true; error.value = null; validationError.value = null; fieldErrors.value = {}; backtestResults.value = null; // 清空 fieldErrors
            calendarView.value.dailyPnlData = {}; calendarView.value.weeks = [];

            try {
                ensureGlobalInvestmentSettings();
                
                // 准备 prediction_strategy_params，确保类型正确
                let finalPredictionParams = {};
                if (selectedPredictionStrategy.value && selectedPredictionStrategy.value.parameters) {
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

                const payload = {
                    symbol: backtestParams.value.symbol,
                    interval: backtestParams.value.interval,
                    start_time: backtestParams.value.start_time,
                    end_time: backtestParams.value.end_time,
                    prediction_strategy_id: backtestParams.value.prediction_strategy_id,
                    prediction_strategy_params: finalPredictionParams,
                    event_period: backtestParams.value.eventPeriod,
                    confidence_threshold: parseFloat(backtestParams.value.confidence_threshold), // 确保是浮点数
                    investment: {
                        ...backtestParams.value.investment, // 包含 initial_balance, strategy_id 等
                        // investment_strategy_specific_params 已经由 watch(investmentStrategyParams) 更新并处理了类型
                        // 确保这里的 specific_params 与后端期望的一致
                    }
                };
                // 移除 investment_strategy_specific_params 中的空值或NaN值，如果需要
                for (const key in payload.investment.investment_strategy_specific_params) {
                    const val = payload.investment.investment_strategy_specific_params[key];
                    if (val === null || val === '' || (typeof val === 'number' && isNaN(val))) {
                        // 根据策略定义，决定是删除此参数还是发送特定值
                        // delete payload.investment.investment_strategy_specific_params[key];
                    }
                }


                console.log('Sending to backtest API:', JSON.stringify(payload, null, 2));
                const response = await axios.post('/api/backtest', payload);
                backtestResults.value = response.data;

                // FIX: Correctly display the entry price from `signal_price`.
                if (backtestResults.value && backtestResults.value.predictions) {
                    backtestResults.value.predictions.forEach(prediction => {
                        // The backend now provides the entry price only in `signal_price`.
                        // To ensure the template displays it correctly without being changed,
                        // we create the field the template expects (`effective_signal_price_for_calc`)
                        // and populate it with the value from `signal_price`.
                        if (prediction.signal_price !== undefined) {
                            prediction.effective_signal_price_for_calc = prediction.signal_price;
                        }
                    });
                }
 
                // 初始计算交给 worker，确保传递的是纯 JS 对象
                worker.postMessage({
                    results: JSON.parse(JSON.stringify(backtestResults.value)),
                    filters: {
                        filterStartTime: '',
                        filterEndTime: '',
                        minConfidence: 0,
                        excludedWeekdays: []
                    }
                });
                resetFilters(); // 重置UI筛选条件
 
                // The calendar is now driven by a watcher on filteredData
                // so we don't need to call generateCalendar here directly.
                } catch (err) {
                    console.error('Backtest failed:', err);
                    // console.log('Full error object for debugging:', JSON.stringify(err, Object.getOwnPropertyNames(err)));
                    // if (err.response) console.log('Error response data for debugging:', JSON.stringify(err.response.data));

                    // Reset errors first
                    validationError.value = null;
                    error.value = null;
                    fieldErrors.value = {};

                    console.log('[DEBUG] Catch Block: Full error object:', err);

                    if (err.response) {
                        const status = err.response.status;
                        const data = err.response.data;
                        console.log(`[DEBUG] Catch Block: Response Status: ${status}`);
                        console.log('[DEBUG] Catch Block: Response Data:', JSON.stringify(data));

                        if ((status === 422 || status === 400) && data && data.detail) {
                            // Likely a validation error from FastAPI/Pydantic
                            if (Array.isArray(data.detail) && data.detail.length > 0) {
                                console.log('[DEBUG] Processing Pydantic array errors. Raw detail:', JSON.stringify(data.detail));
                                const newFieldErrorsLocal = {};
                                const detailedMessagesLocal = [];
                                data.detail.forEach(d => {
                                    const loc = d.loc;
                                    const msg = d.msg;
                                    let fieldKey = '';
                                    let userFriendlyPath = '未知字段';

                                    if (Array.isArray(loc) && loc.length > 1) { // loc[0] is 'body' or 'query'
                                        userFriendlyPath = loc.slice(1).join(' -> ');
                                        if (loc.length === 2 && typeof loc[1] === 'string') { // Top-level field
                                            fieldKey = loc[1]; // General assignment
                                            // Specific mapping for known top-level fields if their model name differs from HTML binding key
                                            if (fieldKey === 'start_time') fieldKey = 'startTime';
                                            else if (fieldKey === 'end_time') fieldKey = 'endTime';
                                            else if (fieldKey === 'prediction_strategy_id') fieldKey = 'predictionStrategy';
                                            // Add other direct mappings if necessary, e.g. 'symbol', 'interval', 'eventPeriod', 'confidence_threshold'
                                        } else if (loc.length > 2 && typeof loc[1] === 'string') { // Nested field
                                            const parentKey = loc[1];
                                            const childKey = loc[2];
                                            if (parentKey === 'prediction_strategy_params' && typeof childKey === 'string') {
                                                fieldKey = `predictionStrategyParams.${childKey}`;
                                            } else if (parentKey === 'investment') {
                                                if (childKey === 'investment_strategy_specific_params' && loc.length > 3 && typeof loc[3] === 'string') {
                                                    fieldKey = `investmentStrategyParams.${loc[3]}`;
                                                } else if (typeof childKey === 'string' && childKey !== 'investment_strategy_specific_params') {
                                                    // For direct children of 'investment' like 'initial_balance'
                                                    fieldKey = childKey; 
                                                }
                                            }
                                        }
                                    }
                                    console.log(`[DEBUG] Pydantic error item: loc=${JSON.stringify(loc)}, msg='${msg}', userFriendlyPath='${userFriendlyPath}', determined fieldKey='${fieldKey}'`);
                                    if (msg) {
                                        detailedMessagesLocal.push(`${userFriendlyPath}: ${msg}`);
                                        if (fieldKey) newFieldErrorsLocal[fieldKey] = msg;
                                    } else {
                                        detailedMessagesLocal.push(`${userFriendlyPath}: (无详细错误信息)`);
                                    }
                                });
                                fieldErrors.value = newFieldErrorsLocal;
                                if (detailedMessagesLocal.length > 0) {
                                    validationError.value = "表单校验失败，请检查以下字段：\n" + detailedMessagesLocal.join("\n");
                                } else {
                                    validationError.value = "表单校验失败，但未获取到详细错误信息。请检查控制台。";
                                }
                            } else if (typeof data.detail === 'string') {
                                // Detail is a single string for 400/422 error
                                validationError.value = data.detail;
                            } else {
                                // Status is 422/400, data.detail exists but is not a non-empty array or string (e.g. empty array [])
                                validationError.value = `校验失败 (代码 ${status})，错误详情格式无法解析或为空。请查看API响应。`;
                            }
                        } else if (data && data.detail && typeof data.detail === 'string') {
                            // Non-422/400 error, but has a 'detail' string (e.g. 500 error with a message)
                            error.value = data.detail;
                        } else if (err.response.statusText) {
                            // Generic HTTP error without a specific 'detail' message
                            error.value = `API 错误 ${status}: ${err.response.statusText}`;
                        } else {
                            // Fallback for err.response existing but no other info
                            error.value = `发生HTTP错误 (代码 ${status})，无更多信息。`;
                        }
                    } else {
                        // Network error or other error without a response object (e.g., err.request was made but no response received)
                        error.value = '回测时发生网络错误或未知客户端错误，请检查网络连接和控制台。';
                    }
                } finally {
                    loading.value = false;
                }
        };
        
        const saveStrategyParameters = async () => {
            let savedCount = 0;
            let errorMessages = [];

            if (selectedPredictionStrategy.value && selectedPredictionStrategy.value.id) {
                // 准备 prediction_strategy_params，确保类型正确 (与 runBacktest 中类似)
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
                    const msg = `保存预测策略 ${selectedPredictionStrategy.value.name} 参数失败: ${err.response?.data?.detail || err.message}`;
                    errorMessages.push(msg);
                }
            }

            if (selectedInvestmentStrategy.value && selectedInvestmentStrategy.value.id) {
                // investment_strategy_specific_params 已经由其 watcher 处理了类型，可以直接使用
                const investmentParamsToSave = { ...backtestParams.value.investment.investment_strategy_specific_params };

                const payload = {
                    strategy_type: 'investment',
                    strategy_id: selectedInvestmentStrategy.value.id,
                    params: investmentParamsToSave 
                };
                 try {
                    await axios.post('/api/save_strategy_parameter_set', payload);
                    if (!allSavedParams.value.investment_strategies) allSavedParams.value.investment_strategies = {};
                    allSavedParams.value.investment_strategies[payload.strategy_id] = { ...payload.params };
                    savedCount++;
                } catch (err) {
                    const msg = `保存投资策略 ${selectedInvestmentStrategy.value.name} 参数失败: ${err.response?.data?.detail || err.message}`;
                    errorMessages.push(msg);
                }
            }

            if (errorMessages.length > 0) alert("部分策略参数保存失败：\n" + errorMessages.join("\n") + (savedCount > 0 ? "\n其余成功！" : ""));
            else if (savedCount > 0) alert("选定策略的参数已成功保存为全局默认值！");
            else alert("没有选定任何策略或没有参数可保存。");
        };

        // --- WebSocket for Live Stats ---
        const connectWebSocket = () => {
            // ... (实现与 live-test-scripts.js 中类似的 WebSocket 连接逻辑，但只关注统计信息)
            // 或者，如果 liveStats 主要用于展示 live-test 页面的状态，这里可以简化或移除，
            // 除非 index 页面也需要独立的实时统计流。
            // 当前 live-test-scripts.js 中的 socket 是为了 /ws/live-test，
            // 如果这里也要连接同一个，则需要协调。
            // 假设这里的 liveStats 是从一个不同的 /ws/global-stats 端点获取，或者只是一个占位符。
            // 为了简化，我们先保持现有的逻辑，它似乎是从 /ws/live-test 获取的，
            // 这意味着如果用户同时打开两个页面，可能会有两个连接到同一个后端 WebSocket 实例。
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            // 注意：这里的 wsUrl 指向 /ws/live-test，与 live-test 页面相同
            // 如果后端设计为每个客户端一个 session，这可能没问题
            // 如果后端是广播，则两个页面都会收到相同数据
            const wsUrl = `${protocol}//${host}/ws/live-test`; 
            
            if (socket.value && (socket.value.readyState === WebSocket.OPEN || socket.value.readyState === WebSocket.CONNECTING)) {
                return; // 已经连接或正在连接
            }

            socket.value = new WebSocket(wsUrl);
            socketStatus.value = 'connecting';

            socket.value.onopen = () => { 
                socketStatus.value = 'connected'; 
                // index 页面通常不发送配置，只接收统计广播 (如果后端这样设计)
            };
            socket.value.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    // 只处理统计相关的消息
                    if (message.type === "initial_stats" || message.type === "stats_update") {
                        liveStats.value = { ...liveStats.value, ...message.data };
                    }
                } catch (e) { console.error("Index WS message error:", e, event.data); }
            };
            socket.value.onclose = () => { 
                socketStatus.value = 'disconnected'; 
                socket.value = null; 
                // 可选：尝试重连
                // setTimeout(connectWebSocket, 5000); 
            };
            socket.value.onerror = (errEvent) => {
                console.error("Index WS error:", errEvent);
                socketStatus.value = 'error'; 
                if (socket.value) socket.value.close(); 
                socket.value = null;
            };
        };

        // --- Calendar Logic ---

        const generateCalendar = () => {
            // Now uses filteredData's daily_pnl via calendarView.dailyPnlData
            const year = calendarView.value.year; const month = calendarView.value.month;
            const dailyPnl = calendarView.value.dailyPnlData || {};
            const firstDay = new Date(year, month, 1);
            const lastDay = new Date(year, month + 1, 0);
            const daysInMonth = lastDay.getDate();
            let startDayOfWeek = firstDay.getDay();
            startDayOfWeek = startDayOfWeek === 0 ? 6 : startDayOfWeek - 1; // 0 (周一) to 6 (周日)

            const weeksArray = []; let currentWeek = [];
            for (let i = 0; i < startDayOfWeek; i++) currentWeek.push({ day: '', pnl: undefined, trades: undefined, balance: undefined, daily_return_pct: undefined, isCurrentMonth: false });
            
            const todayDate = new Date();
            const todayStr = `${todayDate.getFullYear()}-${String(todayDate.getMonth() + 1).padStart(2, '0')}-${String(todayDate.getDate()).padStart(2, '0')}`;

            for (let day = 1; day <= daysInMonth; day++) {
                const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
                const pnlData = dailyPnl[dateStr];
                currentWeek.push({
                    day: day,
                    pnl: pnlData?.pnl,
                    trades: pnlData?.trades,
                    balance: pnlData?.balance,
                    daily_return_pct: pnlData?.daily_return_pct,
                    isCurrentMonth: true,
                    dateString: dateStr,
                    isToday: dateStr === todayStr
                });
                if (currentWeek.length === 7) { weeksArray.push(currentWeek); currentWeek = []; }
            }
            if (currentWeek.length > 0) {
                while (currentWeek.length < 7) currentWeek.push({ day: '', pnl: undefined, trades: undefined, balance: undefined, daily_return_pct: undefined, isCurrentMonth: false });
                weeksArray.push(currentWeek);
            }
            calendarView.value.weeks = weeksArray;
        };
        const changeMonth = (offset) => {
            let newMonth = calendarView.value.month + offset; let newYear = calendarView.value.year;
            if (newMonth < 0) { newMonth = 11; newYear--; }
            else if (newMonth > 11) { newMonth = 0; newYear++; }
            calendarView.value.month = newMonth; calendarView.value.year = newYear;
            generateCalendar();
        };
        const currentMonthYearDisplay = computed(() =>
            new Date(calendarView.value.year, calendarView.value.month).toLocaleDateString('zh-CN', { year: 'numeric', month: 'long' })
        );

        // --- Debounced Filtering Logic ---
        const filteredData = ref(null);
 
        // 监听 Worker 返回的消息
        worker.onmessage = (event) => {
            filteredData.value = event.data;
        };
 
        const triggerWorkerFilter = () => {
            if (!backtestResults.value) {
                return;
            }
            // 确保传递给 worker 的是纯 JS 对象，而不是 Vue 的响应式代理
            worker.postMessage({
                results: JSON.parse(JSON.stringify(backtestResults.value)),
                filters: {
                    filterStartTime: filterStartTime.value,
                    filterEndTime: filterEndTime.value,
                    minConfidence: minConfidence.value,
                    excludedWeekdays: [...excludedWeekdays.value] // 修复：将响应式数组转换为普通数组
                }
            });
        };
 
        const debouncedFilter = debounce(triggerWorkerFilter, 300);
        watch([filterStartTime, filterEndTime, minConfidence, excludedWeekdays], debouncedFilter, { deep: true });

        // Watch for filtered data changes to update the calendar
        watch(filteredData, (newResults) => {
            if (newResults && newResults.daily_pnl) {
                calendarView.value.dailyPnlData = newResults.daily_pnl;
                
                let refDateStr = backtestParams.value.start_time;
                const dailyPnlKeys = Object.keys(newResults.daily_pnl);
                if (dailyPnlKeys.length > 0) {
                    refDateStr = dailyPnlKeys[0] + "T00:00:00";
                } else if (newResults.predictions?.length > 0 && newResults.predictions[0].signal_time) {
                    refDateStr = newResults.predictions[0].signal_time;
                }

                try {
                    const dateObj = new Date(refDateStr);
                    if (!isNaN(dateObj.getTime())) {
                        calendarView.value.year = dateObj.getFullYear();
                        calendarView.value.month = dateObj.getMonth();
                    } else {
                        const today = new Date();
                        calendarView.value.year = today.getFullYear();
                        calendarView.value.month = today.getMonth();
                    }
                } catch (e) {
                    const today = new Date();
                    calendarView.value.year = today.getFullYear();
                    calendarView.value.month = today.getMonth();
                }
                generateCalendar();
            } else {
                // If there are no results (e.g., all filtered out), clear the calendar
                calendarView.value.dailyPnlData = {};
                calendarView.value.weeks = [];
            }
        }, { deep: true });


        // --- Lifecycle Hooks ---
        onMounted(async () => {
            loadFavorites();
            initDatePickers(); 
            ensureGlobalInvestmentSettings(); // 确保投资设置有初始值
            // connectWebSocket(); // 连接 WebSocket 获取实时统计 // Removed live stats websocket connection // Removed live stats websocket connection // Removed live stats websocket connection

            loading.value = true; // 开始加载初始数据
            await fetchAllSavedParameters(); 
            await Promise.all([ // 并行获取
                fetchSymbols(), 
                fetchPredictionStrategies(), 
                fetchInvestmentStrategies()
            ]);
            loading.value = false; // 初始数据加载完成

            // 设置默认选中的策略
            if (predictionStrategies.value.length > 0 && !selectedPredictionStrategy.value) {
                 selectedPredictionStrategy.value = predictionStrategies.value[0];
            }
            if (investmentStrategies.value.length > 0 && !selectedInvestmentStrategy.value) {
                const fixedStrategy = investmentStrategies.value.find(s => s.id === 'fixed');
                selectedInvestmentStrategy.value = fixedStrategy || investmentStrategies.value[0];
            }
        });

        onUnmounted(() => {
            // Removed live stats websocket cleanup
        });

        return {
            // State
            symbols, favoriteSymbols,
            predictionStrategies, selectedPredictionStrategy, predictionStrategyParams,
            investmentStrategies, selectedInvestmentStrategy, investmentStrategyParams,
            allSavedParams,
            backtestParams,
            backtestResults, loading, error, validationError, fieldErrors,

            // Filtering State & Methods
            filterStartTime,
            filterEndTime,
            minConfidence,
            resetFilters,
            excludedWeekdays,

            // Methods
            toggleFavorite, isFavorite,
            runBacktest,
            saveStrategyParameters,
            ensureGlobalInvestmentSettings,
            toggleWeekday, weekdays,

            // Calendar
            calendarView, changeMonth, currentMonthYearDisplay,

            // Computed Properties
            filteredData,

            // Utils for template
            getPnlClass: typeof getPnlClass === 'function' ? getPnlClass : () => '',
        };
    }
});

// 注册全局属性以在模板中使用 utils.js 中的函数
if (typeof formatDateTime === 'function') {
    app.config.globalProperties.formatDateTime = formatDateTime;
} else {
    app.config.globalProperties.formatDateTime = (val) => val || '-';
}
if (typeof getWinRateClass === 'function') {
    app.config.globalProperties.getWinRateClass = getWinRateClass;
} else {
    app.config.globalProperties.getWinRateClass = () => '';
}
if (typeof getPnlClass === 'function') {
    app.config.globalProperties.getPnlClass = getPnlClass;
} else {
    app.config.globalProperties.getPnlClass = () => '';
}

app.mount('#app');