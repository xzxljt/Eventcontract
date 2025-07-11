        // 加载导航栏
        fetch('/templates/navbar.html')
            .then(response => response.text())
            .then(html => {
                document.getElementById('navbar-container').innerHTML = html;
                // 设置当前页面为活跃状态
                const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
                navLinks.forEach(link => {
                    if (link.getAttribute('href') === '/optimization') {
                        link.classList.add('active');
                    }
                });
            })
            .catch(error => console.error('Error loading navbar:', error));



        const { createApp } = Vue;

        createApp({
            data() {
                return {
                    // 基础数据
                    symbols: [],
                    strategies: [],
                    favoriteSymbols: JSON.parse(localStorage.getItem('favoriteSymbols') || '[]'),

                    // 优化参数
                    optimizationParams: {
                        symbol: '',
                        interval: '1m',
                        start_date: (() => {
                            const date = new Date();
                            date.setDate(date.getDate() - 7);
                            return date.toISOString().split('T')[0];
                        })(),
                        end_date: new Date().toISOString().split('T')[0],
                        max_combinations: 1000,
                        min_trades: 10
                    },

                    // 时间和星期过滤
                    timeFilter: {
                        startTime: '',
                        endTime: ''
                    },
                    excludedWeekdays: [], // 默认不排除任何星期
                    weekdays: [
                        { label: '一', value: '1' }, { label: '二', value: '2' },
                        { label: '三', value: '3' }, { label: '四', value: '4' },
                        { label: '五', value: '5' }, { label: '六', value: '6' },
                        { label: '日', value: '0' }
                    ],

                    // 策略相关
                    selectedStrategy: null,
                    selectedPreset: '',
                    parameterRanges: {},
                    strategyPresets: {},

                    // 参数优化配置 - 新增
                    parameterOptimizationConfig: {}, // 存储每个参数的启用状态和固定值

                    // 优化状态
                    isOptimizing: false,
                    currentOptimizationId: null,
                    optimizationProgress: null,
                    optimizationResults: null,
                    progressInterval: null,

                    // 结果显示和排序
                    sortBy: 'composite_score',
                    sortOrder: 'desc',
                    displayedResults: [],
                    selectedResult: null,

                    // 表单验证
                    fieldErrors: {},

                    // 图表
                    scatterChart: null,

                    // 历史记录
                    optimizationHistory: [],
                    showHistoryModal: false
                }
            },

            computed: {
                sortedSymbols() {
                    return [...this.symbols].sort((a, b) => {
                        const aIsFav = this.isFavorite(a);
                        const bIsFav = this.isFavorite(b);
                        if (aIsFav && !bIsFav) return -1;
                        if (!aIsFav && bIsFav) return 1;
                        return 0; // 保持原有顺序
                    });
                },

                canStartOptimization() {
                    return this.selectedStrategy &&
                           this.optimizationParams.symbol &&
                           this.optimizationParams.interval &&
                           this.optimizationParams.start_date &&
                           this.optimizationParams.end_date &&
                           Object.keys(this.parameterRanges).length > 0;
                },

                estimatedCombinations() {
                    if (!Array.isArray(this.selectedStrategy?.parameters) || !Object.keys(this.parameterRanges).length) return 0;

                    let total = 1;
                    for (const param of this.selectedStrategy.parameters) {
                        // 确保param存在且有name属性
                        if (param && param.name) {
                            const config = this.parameterOptimizationConfig[param.name];
                            const range = this.parameterRanges[param.name];

                            // 只计算启用优化的参数
                            if (config?.enabled && range && typeof range.min === 'number' && typeof range.max === 'number' && range.step > 0) {
                                const count = Math.floor((range.max - range.min) / range.step) + 1;
                                total *= Math.max(1, count);
                            }
                        }
                    }
                    return total;
                },

                // 确保参数范围的安全访问
                safeParameterRanges() {
                    const safe = {};
                    if (this.selectedStrategy?.parameters) {
                        this.selectedStrategy.parameters.forEach(param => {
                            if (param?.name) {
                                safe[param.name] = this.parameterRanges[param.name] || {
                                    min: param.min || 0,
                                    max: param.max || 100,
                                    step: param.step || 1
                                };
                            }
                        });
                    }
                    return safe;
                }
            },

            mounted() {
                this.loadStrategies();
                this.loadSymbols();
                this.checkCurrentOptimization();
                this.loadOptimizationHistory();
            },

            methods: {
                getDefaultStartDate() {
                    const date = new Date();
                    date.setDate(date.getDate() - 7);
                    return date.toISOString().split('T')[0];
                },

                getDefaultEndDate() {
                    return new Date().toISOString().split('T')[0];
                },

                async checkCurrentOptimization() {
                    try {
                        const response = await fetch('/api/optimization/current');
                        const result = await response.json();

                        if (result.status === 'success' && result.data) {
                            const currentTask = result.data;

                            // 恢复优化状态
                            this.currentOptimizationId = currentTask.id;
                            this.isOptimizing = currentTask.status === 'running';
                            this.optimizationProgress = currentTask.progress;

                            if (this.isOptimizing) {
                                // 开始监控进度
                                this.startProgressMonitoring();
                                showToast('信息', '检测到正在运行的优化任务，已恢复状态', 'info');
                            } else if (currentTask.status === 'completed') {
                                // 加载完成的结果
                                await this.loadResults();
                                showToast('成功', '检测到已完成的优化任务，已加载结果', 'success');
                            }
                        } else {
                            // 没有当前运行的任务，清理状态
                            this.clearOptimizationState();
                        }
                    } catch (error) {
                        console.error('检查当前优化任务失败:', error);
                        // 出错时也清理状态
                        this.clearOptimizationState();
                    }
                },

                clearOptimizationState() {
                    // 清理优化状态
                    this.currentOptimizationId = null;
                    this.isOptimizing = false;
                    this.optimizationProgress = null;
                    this.optimizationResults = null;

                    // 停止进度监控
                    if (this.progressInterval) {
                        clearInterval(this.progressInterval);
                        this.progressInterval = null;
                    }
                },

                async loadOptimizationHistory() {
                    try {
                        const response = await fetch('/api/optimization/history?limit=10');
                        const result = await response.json();

                        if (result.status === 'success') {
                            this.optimizationHistory = result.data;
                        }
                    } catch (error) {
                        console.error('加载优化历史失败:', error);
                    }
                },

                async deleteOptimizationRecord(recordId) {
                    if (!confirm('确定要删除这条优化记录吗？')) {
                        return;
                    }

                    try {
                        const response = await fetch(`/api/optimization/record/${recordId}`, {
                            method: 'DELETE'
                        });
                        const result = await response.json();

                        if (result.status === 'success') {
                            showToast('成功', '记录删除成功', 'success');
                            await this.loadOptimizationHistory();
                        } else {
                            showToast('错误', '删除失败: ' + result.message, 'danger');
                        }
                    } catch (error) {
                        console.error('删除记录失败:', error);
                        showToast('错误', '删除记录失败: ' + error.message, 'danger');
                    }
                },

                showHistoryModal() {
                    this.showHistoryModal = true;
                    this.$nextTick(() => {
                        const modalElement = document.getElementById('historyModal');
                        if (modalElement) {
                            const modal = new bootstrap.Modal(modalElement);
                            modal.show();
                        }
                    });
                },

                async loadHistoryResults(recordId) {
                    try {
                        // 关闭历史记录模态框
                        const historyModal = bootstrap.Modal.getInstance(document.getElementById('historyModal'));
                        if (historyModal) {
                            historyModal.hide();
                        }

                        // 设置当前优化ID并加载结果
                        this.currentOptimizationId = recordId;
                        await this.loadResults();

                        showToast('成功', '历史结果已加载', 'success');
                    } catch (error) {
                        console.error('加载历史结果失败:', error);
                        showToast('错误', '加载历史结果失败: ' + error.message, 'danger');
                    }
                },

                async loadStrategies() {
                    try {
                        const response = await fetch('/api/prediction-strategies');
                        this.strategies = await response.json();
                        showToast('成功', '策略列表加载成功', 'success');
                    } catch (error) {
                        console.error('加载策略失败:', error);
                        showToast('错误', '加载策略失败: ' + error.message, 'danger');
                    }
                },

                async loadSymbols() {
                    try {
                        const response = await fetch('/api/symbols');
                        this.symbols = await response.json();

                        // 默认选择第一个收藏的交易对，如果没有收藏则选择第一个
                        if (this.symbols.length > 0 && !this.optimizationParams.symbol) {
                            const firstFavorite = this.symbols.find(s => this.isFavorite(s));
                            this.optimizationParams.symbol = firstFavorite || this.symbols[0];
                        }
                    } catch (error) {
                        console.error('加载交易对失败:', error);
                        // 使用默认列表作为备选
                        this.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'XRPUSDT'];
                        if (!this.optimizationParams.symbol) {
                            const firstFavorite = this.symbols.find(s => this.isFavorite(s));
                            this.optimizationParams.symbol = firstFavorite || this.symbols[0];
                        }
                    }
                },

                isFavorite(symbol) {
                    return this.favoriteSymbols.includes(symbol);
                },

                toggleWeekday(dayValue) {
                    // excludedWeekdays存储的是要排除的星期
                    const index = this.excludedWeekdays.indexOf(dayValue);
                    if (index > -1) {
                        this.excludedWeekdays.splice(index, 1); // 取消排除
                    } else {
                        this.excludedWeekdays.push(dayValue); // 添加到排除列表
                    }
                },

                toggleFavorite(symbol) {
                    if (!symbol) return;

                    const index = this.favoriteSymbols.indexOf(symbol);
                    if (index > -1) {
                        this.favoriteSymbols.splice(index, 1);
                        showToast('信息', `已取消收藏 ${symbol}`, 'info');
                    } else {
                        this.favoriteSymbols.push(symbol);
                        showToast('成功', `已收藏 ${symbol}`, 'success');
                    }

                    localStorage.setItem('favoriteSymbols', JSON.stringify(this.favoriteSymbols));
                },

                async onStrategyChange() {
                    if (!this.selectedStrategy) {
                        this.parameterRanges = {};
                        this.parameterOptimizationConfig = {};
                        return;
                    }

                    // 立即初始化参数范围，避免模板访问undefined
                    this.parameterRanges = {};
                    this.parameterOptimizationConfig = {};

                    // 确保parameters存在且是数组
                    if (Array.isArray(this.selectedStrategy.parameters)) {
                        console.log('初始化参数范围，参数列表:', this.selectedStrategy.parameters);
                        this.selectedStrategy.parameters.forEach(param => {
                            // 确保param是对象且有name属性
                            if (param && typeof param === 'object' && param.name) {
                                this.parameterRanges[param.name] = {
                                    min: typeof param.min === 'number' ? param.min : 0,
                                    max: typeof param.max === 'number' ? param.max : 100,
                                    step: typeof param.step === 'number' ? param.step : 1
                                };

                                // 初始化参数优化配置 - 默认启用所有参数
                                this.parameterOptimizationConfig[param.name] = {
                                    enabled: true,
                                    fixedValue: typeof param.default === 'number' ? param.default :
                                              (typeof param.min === 'number' ? param.min : 0)
                                };

                                console.log(`初始化参数 ${param.name}:`, this.parameterRanges[param.name]);
                            }
                        });
                    }

                    console.log('初始化后的parameterRanges:', this.parameterRanges);
                    console.log('初始化后的parameterOptimizationConfig:', this.parameterOptimizationConfig);

                    try {
                        // 获取策略参数范围
                        const response = await fetch(`/api/strategies/${this.selectedStrategy.id}/parameter_ranges`);
                        const data = await response.json();

                        console.log('策略参数API返回数据:', data);

                        if (data.error) {
                            showToast('错误', '获取策略参数失败: ' + data.error, 'danger');
                            return;
                        }

                        // 更新参数范围
                        if (Array.isArray(data.parameters)) {
                            data.parameters.forEach(param => {
                                if (param && typeof param === 'object' && param.name) {
                                    this.parameterRanges[param.name] = {
                                        min: typeof param.min === 'number' ? param.min : 0,
                                        max: typeof param.max === 'number' ? param.max : 100,
                                        step: typeof param.step === 'number' ? param.step : 1
                                    };
                                    console.log(`更新参数 ${param.name}:`, this.parameterRanges[param.name]);
                                }
                            });
                        }

                        console.log('API更新后的parameterRanges:', this.parameterRanges);

                        // 强制Vue重新渲染
                        this.$nextTick(() => {
                            console.log('Vue nextTick: 参数范围已更新');
                        });

                        // 获取参数预设
                        const presetResponse = await fetch(`/api/strategies/${this.selectedStrategy.id}/parameter_presets`);
                        const presetData = await presetResponse.json();

                        if (!presetData.error) {
                            this.strategyPresets = presetData.presets;
                        }

                        showToast('成功', '策略参数加载成功', 'success');
                    } catch (error) {
                        console.error('加载策略参数失败:', error);
                        showToast('错误', '加载策略参数失败: ' + error.message, 'danger');
                    }
                },

                applyPreset() {
                    if (!this.selectedPreset || !this.strategyPresets[this.selectedPreset]) return;

                    const preset = this.strategyPresets[this.selectedPreset];
                    Object.assign(this.parameterRanges, preset.ranges);

                    showToast('成功', `已应用${this.selectedPreset}型参数预设`, 'success');
                },

                // 新增：切换参数优化状态
                toggleParameterOptimization(paramName) {
                    console.log(`切换参数 ${paramName} 的优化状态`);

                    if (!this.parameterOptimizationConfig[paramName]) {
                        console.log(`参数 ${paramName} 的配置不存在`);
                        return;
                    }

                    const currentConfig = this.parameterOptimizationConfig[paramName];
                    const newEnabled = !currentConfig.enabled;

                    console.log(`当前状态: ${currentConfig.enabled}, 新状态: ${newEnabled}`);

                    // 直接修改对象属性（Vue 3中对象是响应式的）
                    this.parameterOptimizationConfig[paramName].enabled = newEnabled;

                    // 如果禁用优化，确保有固定值
                    if (!newEnabled) {
                        const range = this.parameterRanges[paramName];
                        if (range && this.parameterOptimizationConfig[paramName].fixedValue === undefined) {
                            this.parameterOptimizationConfig[paramName].fixedValue = range.min;
                        }
                    }

                    console.log(`参数 ${paramName} 优化状态已切换为:`, this.parameterOptimizationConfig[paramName].enabled);
                },

                // 新增：设置参数固定值
                setParameterFixedValue(paramName, value) {
                    if (!this.parameterOptimizationConfig[paramName]) return;

                    this.parameterOptimizationConfig[paramName].fixedValue = parseFloat(value) || 0;
                    console.log(`参数 ${paramName} 固定值设置为:`, this.parameterOptimizationConfig[paramName].fixedValue);
                },

                // 测试方法：检查参数配置状态
                testParameterConfig() {
                    console.log('当前参数配置状态:');
                    console.log('parameterOptimizationConfig:', this.parameterOptimizationConfig);
                    console.log('parameterRanges:', this.parameterRanges);
                    console.log('selectedStrategy:', this.selectedStrategy);
                },

                async startOptimization() {
                    this.fieldErrors = {};

                    // 验证表单
                    if (!this.validateForm()) {
                        showToast('警告', '请检查表单输入', 'warning');
                        return;
                    }

                    try {
                        // 构建包含优化配置的参数范围数据
                        const strategyParamsRanges = {};
                        for (const paramName in this.parameterRanges) {
                            const range = this.parameterRanges[paramName];
                            const config = this.parameterOptimizationConfig[paramName];

                            if (config?.enabled) {
                                // 启用优化的参数，发送范围配置
                                strategyParamsRanges[paramName] = {
                                    min: range.min,
                                    max: range.max,
                                    step: range.step,
                                    enabled: true
                                };
                            } else {
                                // 禁用优化的参数，发送固定值配置
                                strategyParamsRanges[paramName] = {
                                    enabled: false,
                                    fixed_value: config?.fixedValue || range.min
                                };
                            }
                        }

                        const requestData = {
                            symbol: this.optimizationParams.symbol,
                            interval: this.optimizationParams.interval,
                            start_date: this.optimizationParams.start_date,
                            end_date: this.optimizationParams.end_date,
                            strategy_id: this.selectedStrategy.id,
                            strategy_params_ranges: strategyParamsRanges,
                            max_combinations: this.optimizationParams.max_combinations,
                            min_trades: this.optimizationParams.min_trades
                        };

                        // 添加时间过滤参数 - 现在是包含逻辑，需要转换为排除逻辑
                        if (this.timeFilter.startTime && this.timeFilter.endTime) {
                            // 前端设置的是交易时间段，后端需要的是排除时间段
                            // 我们需要计算出要排除的时间段（即非交易时间段）
                            const startTime = this.timeFilter.startTime;
                            const endTime = this.timeFilter.endTime;

                            // 如果设置了交易时间段，我们需要排除其他时间段
                            // 这里简化处理：将交易时间段作为包含时间段发送给后端
                            // 后端需要相应修改逻辑来处理包含而非排除
                            requestData.include_time_ranges = [{
                                start: startTime,
                                end: endTime
                            }];
                        }

                        // 添加星期过滤参数 - excludedWeekdays存储的是要排除的星期
                        if (this.excludedWeekdays.length > 0) {
                            // 将前端的星期值(0-6)转换为Python的weekday值(0-6，Monday=0)
                            // 前端: 0=周日, 1=周一, ..., 6=周六
                            // Python: 0=周一, 1=周二, ..., 6=周日
                            requestData.exclude_weekdays = this.excludedWeekdays.map(day => {
                                const dayNum = parseInt(day);
                                return dayNum === 0 ? 6 : dayNum - 1; // 0(周日)->6, 1(周一)->0, ..., 6(周六)->5
                            });
                        }

                        const response = await fetch('/api/optimization/start', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(requestData)
                        });

                        const result = await response.json();

                        if (response.ok) {
                            this.currentOptimizationId = result.optimization_id;
                            this.isOptimizing = true;
                            this.optimizationResults = null;
                            this.startProgressMonitoring();
                            // 刷新历史记录
                            await this.loadOptimizationHistory();
                            showToast('成功', '优化任务已启动', 'success');
                        } else {
                            showToast('错误', '启动优化失败: ' + result.detail, 'danger');
                        }
                    } catch (error) {
                        console.error('启动优化失败:', error);
                        showToast('错误', '启动优化失败: ' + error.message, 'danger');
                    }
                },

                validateForm() {
                    let isValid = true;

                    if (!this.optimizationParams.symbol) {
                        this.fieldErrors.symbol = '请选择交易对';
                        isValid = false;
                    }

                    if (!this.selectedStrategy) {
                        this.fieldErrors.strategy_id = '请选择策略';
                        isValid = false;
                    }

                    if (new Date(this.optimizationParams.start_date) >= new Date(this.optimizationParams.end_date)) {
                        this.fieldErrors.end_date = '结束日期必须晚于开始日期';
                        isValid = false;
                    }

                    if (this.estimatedCombinations > this.optimizationParams.max_combinations) {
                        showToast('警告', `参数组合数(${this.estimatedCombinations})超过限制(${this.optimizationParams.max_combinations})`, 'warning');
                        isValid = false;
                    }

                    return isValid;
                },

                startProgressMonitoring() {
                    this.progressInterval = setInterval(async () => {
                        await this.updateProgress();
                    }, 2000);
                },

                async updateProgress() {
                    if (!this.currentOptimizationId) return;

                    try {
                        const response = await fetch(`/api/optimization/progress/${this.currentOptimizationId}`);

                        if (response.status === 404) {
                            // 任务不存在，清理状态
                            console.log('优化任务不存在，清理状态');
                            this.clearOptimizationState();
                            showToast('信息', '优化任务已结束', 'info');
                            return;
                        }

                        this.optimizationProgress = await response.json();

                        if (this.optimizationProgress.status === 'completed') {
                            this.isOptimizing = false;
                            clearInterval(this.progressInterval);
                            await this.loadResults();
                            await this.loadOptimizationHistory();
                            showToast('成功', '优化完成！', 'success');
                        } else if (this.optimizationProgress.status === 'error' || this.optimizationProgress.status === 'stopped') {
                            this.isOptimizing = false;
                            clearInterval(this.progressInterval);
                            await this.loadOptimizationHistory();
                            showToast('警告', `优化${this.optimizationProgress.status === 'error' ? '失败' : '已停止'}: ${this.optimizationProgress.error_message || ''}`, 'warning');
                        }
                    } catch (error) {
                        console.error('更新进度失败:', error);
                        // 如果是网络错误且包含404，也清理状态
                        if (error.message && error.message.includes('404')) {
                            this.clearOptimizationState();
                        }
                    }
                },

                async stopOptimization() {
                    if (!this.currentOptimizationId) return;

                    try {
                        await fetch(`/api/optimization/stop/${this.currentOptimizationId}`, {
                            method: 'POST'
                        });
                        showToast('信息', '已请求停止优化', 'info');
                    } catch (error) {
                        console.error('停止优化失败:', error);
                        showToast('错误', '停止优化失败: ' + error.message, 'danger');
                    }
                },

                async loadResults() {
                    if (!this.currentOptimizationId) return;

                    try {
                        const response = await fetch(`/api/optimization/results/${this.currentOptimizationId}`);
                        this.optimizationResults = await response.json();

                        if (this.optimizationResults.error) {
                            showToast('错误', '获取结果失败: ' + this.optimizationResults.error, 'danger');
                            return;
                        }

                        // 初始化显示结果
                        this.sortResults();

                        // 渲染散点图
                        this.$nextTick(() => {
                            this.renderScatterChart();
                        });
                    } catch (error) {
                        console.error('加载结果失败:', error);
                        showToast('错误', '加载结果失败: ' + error.message, 'danger');
                    }
                },

                renderScatterChart() {
                    if (!this.optimizationResults?.scatter_plot_data?.points?.length) return;

                    const ctx = this.$refs.scatterChart?.getContext('2d');
                    if (!ctx) return;

                    if (this.scatterChart) {
                        this.scatterChart.destroy();
                    }

                    const data = this.optimizationResults.scatter_plot_data.points.map(point => ({
                        x: point.x,
                        y: point.y,
                        rank: point.rank,
                        parameters: point.parameters,
                        composite_score: point.composite_score
                    }));

                    this.scatterChart = new Chart(ctx, {
                        type: 'scatter',
                        data: {
                            datasets: [{
                                label: '参数组合',
                                data: data,
                                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1,
                                pointRadius: 5,
                                pointHoverRadius: 8
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: '胜率 (%)'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: '总收益率 (%)'
                                    }
                                }
                            },
                            plugins: {
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const point = context.raw;
                                            return [
                                                `排名: ${point.rank}`,
                                                `胜率: ${point.x}%`,
                                                `收益率: ${point.y}%`,
                                                `综合评分: ${point.composite_score.toFixed(2)}`,
                                                `参数: ${JSON.stringify(point.parameters)}`
                                            ];
                                        }
                                    }
                                }
                            }
                        }
                    });
                },

                async exportResults() {
                    if (!this.currentOptimizationId) return;

                    try {
                        const response = await fetch(`/api/optimization/export/${this.currentOptimizationId}`, {
                            method: 'POST'
                        });
                        const result = await response.json();

                        if (result.status === 'success') {
                            showToast('成功', '结果导出成功: ' + result.file_path, 'success');
                        } else {
                            showToast('错误', '结果导出失败: ' + result.message, 'danger');
                        }
                    } catch (error) {
                        console.error('导出结果失败:', error);
                        showToast('错误', '导出结果失败: ' + error.message, 'danger');
                    }
                },

                resetOptimization() {
                    this.isOptimizing = false;
                    this.currentOptimizationId = null;
                    this.optimizationProgress = null;
                    this.optimizationResults = null;
                    this.displayedResults = [];
                    this.selectedResult = null;
                    if (this.progressInterval) {
                        clearInterval(this.progressInterval);
                        this.progressInterval = null;
                    }
                    if (this.scatterChart) {
                        this.scatterChart.destroy();
                        this.scatterChart = null;
                    }
                },

                // 结果排序和显示
                sortResults() {
                    if (!this.optimizationResults?.all_results) {
                        this.displayedResults = [];
                        return;
                    }

                    const results = [...this.optimizationResults.all_results];
                    results.sort((a, b) => {
                        // 安全地获取指标值
                        let aVal = a.metrics?.[this.sortBy];
                        let bVal = b.metrics?.[this.sortBy];

                        // 如果值不存在，使用默认值
                        if (aVal === undefined || aVal === null) aVal = 0;
                        if (bVal === undefined || bVal === null) bVal = 0;

                        // 特殊处理最大回撤（越小越好）
                        if (this.sortBy === 'max_drawdown') {
                            return this.sortOrder === 'desc' ? aVal - bVal : bVal - aVal;
                        }

                        return this.sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
                    });

                    // 重新分配排名并添加显示状态
                    this.displayedResults = results.slice(0, 20).map((result, index) => ({
                        ...result,
                        rank: index + 1,
                        showParameters: false
                    }));
                },

                toggleSortOrder() {
                    this.sortOrder = this.sortOrder === 'desc' ? 'asc' : 'desc';
                    this.sortResults();
                },

                toggleParameters(index) {
                    if (this.displayedResults[index]) {
                        this.displayedResults[index].showParameters = !this.displayedResults[index].showParameters;
                    }
                },

                showResultDetails(result) {
                    console.log('显示结果详情:', result);
                    console.log('交易记录数量:', result?.backtest_details?.predictions?.length);
                    this.selectedResult = result;
                    this.$nextTick(() => {
                        console.log('Vue nextTick: selectedResult已设置:', this.selectedResult);
                        const modalElement = document.getElementById('resultDetailsModal');
                        if (modalElement) {
                            const modal = new bootstrap.Modal(modalElement);
                            modal.show();
                            console.log('模态框已显示');
                        } else {
                            console.error('找不到模态框元素');
                        }
                    });
                },

                getStrategyDisplayName(strategyId) {
                    const strategy = this.strategies.find(s => s.id === strategyId);
                    return strategy ? strategy.name : strategyId;
                },

                // 参数显示格式化方法
                getParameterDisplayName(paramKey) {
                    const displayNames = {
                        'rsi_period': 'RSI周期',
                        'rsi_overbought': 'RSI超买线',
                        'rsi_oversold': 'RSI超卖线',
                        'bb_period': '布林带周期',
                        'bb_std_dev': '布林带标准差',
                        'ema_fast': '快速EMA',
                        'ema_slow': '慢速EMA',
                        'adx_period': 'ADX周期',
                        'adx_threshold': 'ADX阈值',
                        'macd_fast': 'MACD快线',
                        'macd_slow': 'MACD慢线',
                        'macd_signal': 'MACD信号线',
                        'volume_period': '成交量周期',
                        'trend_rsi_buy_min': '趋势买入下限',
                        'trend_rsi_buy_max': '趋势买入上限',
                        'trend_rsi_sell_min': '趋势卖出下限',
                        'trend_rsi_sell_max': '趋势卖出上限',
                        'min_confidence_threshold': '最低置信度',
                        'trend_confirmation_periods': '趋势确认周期'
                    };
                    return displayNames[paramKey] || paramKey;
                },

                formatParameterValue(paramKey, value) {
                    // 根据参数类型格式化显示值
                    if (typeof value === 'number') {
                        if (paramKey.includes('std_dev')) {
                            return value.toFixed(1);
                        } else if (paramKey.includes('threshold') || paramKey.includes('confidence')) {
                            return value + '%';
                        } else {
                            return Math.round(value);
                        }
                    }
                    return value;
                },

                getParameterUnit(paramKey) {
                    const units = {
                        'rsi_period': '天',
                        'rsi_overbought': '水平',
                        'rsi_oversold': '水平',
                        'bb_period': '天',
                        'bb_std_dev': '倍数',
                        'ema_fast': '天',
                        'ema_slow': '天',
                        'adx_period': '天',
                        'adx_threshold': '强度',
                        'macd_fast': '天',
                        'macd_slow': '天',
                        'macd_signal': '天',
                        'volume_period': '天',
                        'min_confidence_threshold': '百分比',
                        'trend_confirmation_periods': '周期'
                    };
                    return units[paramKey] || '';
                },

                formatDateTime(dateTimeStr) {
                    if (!dateTimeStr) return '--';
                    try {
                        const date = new Date(dateTimeStr);
                        if (isNaN(date.getTime())) return '--';
                        return date.toLocaleString('zh-CN', {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit'
                        });
                    } catch (e) {
                        return '--';
                    }
                },

                // 工具方法
                formatTime(seconds) {
                    if (!seconds || seconds <= 0) return '0秒';

                    const hours = Math.floor(seconds / 3600);
                    const minutes = Math.floor((seconds % 3600) / 60);
                    const secs = Math.floor(seconds % 60);

                    if (hours > 0) {
                        return `${hours}小时${minutes}分钟`;
                    } else if (minutes > 0) {
                        return `${minutes}分钟${secs}秒`;
                    } else {
                        return `${secs}秒`;
                    }
                },

                getStatusBadgeClass(status) {
                    const classes = {
                        'running': 'bg-primary',
                        'completed': 'bg-success',
                        'error': 'bg-danger',
                        'stopped': 'bg-warning'
                    };
                    return classes[status] || 'bg-secondary';
                },

                getStatusText(status) {
                    const texts = {
                        'running': '运行中',
                        'completed': '已完成',
                        'error': '错误',
                        'stopped': '已停止'
                    };
                    return texts[status] || '未知';
                },

                getPnlClass(value) {
                    if (value > 0) return 'text-success';
                    if (value < 0) return 'text-danger';
                    return 'text-muted';
                },

                getRankBadgeClass(rank) {
                    if (rank === 1) return 'bg-warning text-dark';
                    if (rank <= 3) return 'bg-info';
                    if (rank <= 10) return 'bg-primary';
                    return 'bg-secondary';
                },

                getRankBorderClass(rank) {
                    if (rank === 1) return 'border-warning';
                    if (rank <= 3) return 'border-info';
                    if (rank <= 10) return 'border-primary';
                    return '';
                },

                getRankHeaderClass(rank) {
                    if (rank === 1) return 'bg-warning-subtle';
                    if (rank <= 3) return 'bg-info-subtle';
                    if (rank <= 10) return 'bg-primary-subtle';
                    return '';
                }
            }
        }).mount('#app');