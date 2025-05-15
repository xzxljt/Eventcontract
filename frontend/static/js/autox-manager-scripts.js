// frontend/static/js/autox-manager-scripts.js

const app = Vue.createApp({
    data() {
        return {
            clients: [], // 将存储从 WebSocket 获取的客户端对象列表
            tradeLogs: [],
            loadingClients: true, // 初始加载状态，等待WebSocket连接和数据
            loadingLogs: false,
            tradeLogLimit: 50,
            currentClientForModal: null,
            testCommand: {
                type: 'test_echo',
            },
            manualTrade: {
                symbol: 'ETHUSDT',
                direction: 'up',
                amount: '5',
                signal_id: ''
            },
            toast: {
                title: '',
                message: '',
                type: 'success'
            },
            websocket: null, // WebSocket 实例
            reconnectInterval: null, // 重连定时器
            testCommandModalInstance: null,
            triggerTradeModalInstance: null,
            toastInstance: null,
        };
    },
    methods: {
        connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${protocol}//${window.location.host}/ws/autox_status`;
            this.websocket = new WebSocket(url);

            this.websocket.onopen = () => {
                console.log('WebSocket连接成功:', url);
                this.loadingClients = false; // 连接成功后，等待数据更新
                if (this.reconnectInterval) {
                    clearInterval(this.reconnectInterval);
                    this.reconnectInterval = null;
                }
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === 'autox_clients_update') {
                        console.log('收到客户端列表更新:', message.data);
                        this.clients = message.data;
                    }
                    // 可以根据需要处理其他消息类型
                } catch (error) {
                    console.error('处理WebSocket消息失败:', error);
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket错误:', error);
                this.loadingClients = true; // 错误时显示加载状态或错误信息
                this.showToast('WebSocket连接错误', '与服务器的连接发生错误，尝试重连...', 'danger');
            };

            this.websocket.onclose = (event) => {
                console.log('WebSocket连接关闭:', event.code, event.reason);
                this.loadingClients = true; // 连接关闭时显示加载状态或错误信息
                if (!event.wasClean) {
                    // 连接非正常关闭，尝试重连
                    console.log('WebSocket连接非正常关闭，尝试重连...');
                    this.showToast('WebSocket连接断开', '与服务器的连接已断开，尝试重连...', 'danger');
                    this.scheduleReconnect();
                } else {
                    console.log('WebSocket连接正常关闭。');
                }
            };
        },
        scheduleReconnect() {
            if (!this.reconnectInterval) {
                this.reconnectInterval = setInterval(() => {
                    console.log('尝试重新连接WebSocket...');
                    this.connectWebSocket();
                }, 5000); // 每5秒尝试重连
            }
        },
        async fetchTradeLogs(forceRefresh = false) {
            if (forceRefresh || this.tradeLogs.length === 0 || !this.tradeLogs.length) { // 确保 tradeLogs 有定义
                this.loadingLogs = true;
            }
            try {
                const response = await fetch(`/api/autox/trade_logs?limit=${this.tradeLogLimit}`);
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
                    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                }
                this.tradeLogs = await response.json();
            } catch (error) {
                console.error("获取交易日志失败:", error);
                this.showToast('获取日志失败', error.message, 'danger');
            } finally {
                 if (forceRefresh || !this.tradeLogs || this.tradeLogs.length === 0 ) {
                    this.loadingLogs = false;
                }
            }
        },
        async saveClientNotes(clientId, notes) {
            // 当输入框失焦或内容改变时调用此方法
            // 我们将直接更新该 client 对象的 notes 属性，然后发送 API 请求
            // 如果 API 请求失败，理论上应该回滚前端的 notes 值，但为了简化，暂时不处理回滚
            // 找到对应的客户端对象
            const clientToUpdate = this.clients.find(c => c.client_id === clientId);
            if (clientToUpdate) {
                clientToUpdate.notes = notes; // 立即更新前端显示
            }

            try {
                const response = await fetch(`/api/autox/clients/${clientId}/notes`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ notes: notes })
                });
                const result = await response.json();
                if (!response.ok) {
                    // API失败，显示错误，但前端的 notes 可能已经更新
                    throw new Error(result.detail || `HTTP error! status: ${response.status}`);
                }
                this.showToast('备注已保存', `客户端 ${clientId} 的备注已成功更新。`, 'success');
                // 可选：如果后端返回了更新后的完整 client 对象，可以用来更新 this.clients 中的对应项
                // this.clients = this.clients.map(c => c.client_id === clientId ? result : c);
            } catch (error) {
                console.error(`更新客户端 ${clientId} 备注失败:`, error);
                this.showToast('备注保存失败', `更新客户端 ${clientId} 备注时发生错误: ${error.message}`, 'danger');
                // 考虑是否需要回滚前端的 notes 值
                // WebSocket 会自动推送最新状态，所以通常不需要手动fetchClients
            }
        },
        openTestCommandModal(client) {
            this.currentClientForModal = client;
            this.testCommand.type = 'test_echo';
            if (this.testCommandModalInstance) this.testCommandModalInstance.show();
        },
        async sendTestCommand() {
            if (!this.currentClientForModal || !this.testCommand.type) return;
            const clientId = this.currentClientForModal.client_id;
            try {
                const response = await fetch(`/api/autox/clients/${clientId}/send_test_command?command_type=${encodeURIComponent(this.testCommand.type)}`, {
                    method: 'POST',
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.detail || `HTTP error! status: ${response.status}`);
                }
                this.showToast('测试指令已发送', `成功向 ${clientId} 发送指令 '${this.testCommand.type}'.`, 'success');
                if (this.testCommandModalInstance) this.testCommandModalInstance.hide();
                this.fetchTradeLogs(true);
            } catch (error) {
                console.error("发送测试指令失败:", error);
                this.showToast('测试指令发送失败', error.message, 'danger');
            }
        },
        openTriggerTradeModal(client) {
            this.currentClientForModal = client;
            this.manualTrade.symbol = (client.supported_symbols && client.supported_symbols.length > 0) ? client.supported_symbols[0] : 'ETHUSDT';
            this.manualTrade.direction = 'up';
            this.manualTrade.amount = '5';
            this.manualTrade.signal_id = `manual_${client.client_id.substring(0,4)}_${Date.now().toString().slice(-5)}`;
            if (this.triggerTradeModalInstance) this.triggerTradeModalInstance.show();
        },
        async triggerManualTrade() {
            if (!this.currentClientForModal) return;
            const clientId = this.currentClientForModal.client_id;
            const payload = {
                symbol: this.manualTrade.symbol,
                direction: this.manualTrade.direction,
                amount: String(this.manualTrade.amount),
                signal_id: this.manualTrade.signal_id || null
            };
            try {
                const response = await fetch(`/api/autox/clients/${clientId}/trigger_trade_command`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.detail || `HTTP error! status: ${response.status}`);
                }
                this.showToast('交易指令已发送', `成功向 ${clientId} 发送交易指令. Signal ID: ${result.sent_command?.payload?.signal_id}`, 'success');
                if (this.triggerTradeModalInstance) this.triggerTradeModalInstance.hide();
                // WebSocket 会自动推送最新状态，所以通常不需要手动fetchClients
                await this.fetchTradeLogs(true);
            } catch (error) {
                console.error("手动触发交易失败:", error);
                this.showToast('手动交易失败', error.message, 'danger');
            }
        },
        showToast(title, message, type = 'success') {
            this.toast.title = title;
            this.toast.message = message;
            this.toast.type = type;
            const toastEl = document.getElementById('appToast');
            if (this.toastInstance && toastEl) {
                // 更新头部颜色
                const toastHeader = toastEl.querySelector('.toast-header > strong');
                if(toastHeader){
                    toastHeader.className = 'me-auto'; // Reset classes
                    if(type === 'success') toastHeader.classList.add('text-success');
                    else if(type === 'danger') toastHeader.classList.add('text-danger');
                    else toastHeader.classList.add('text-dark'); // Default
                }
                this.toastInstance.show();
            }
        },
    },
    mounted() {
        const testModalEl = document.getElementById('testCommandModal');
        if (testModalEl) this.testCommandModalInstance = new bootstrap.Modal(testModalEl);

        const triggerModalEl = document.getElementById('triggerTradeModal');
        if (triggerModalEl) this.triggerTradeModalInstance = new bootstrap.Modal(triggerModalEl);

        const toastEl = document.getElementById('appToast');
        if (toastEl) this.toastInstance = new bootstrap.Toast(toastEl);

        // 建立WebSocket连接
        this.connectWebSocket();
        // 首次加载交易日志
        this.fetchTradeLogs(true);
    },
    beforeUnmount() {
        // 关闭WebSocket连接
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.close();
        }
        if (this.reconnectInterval) {
            clearInterval(this.reconnectInterval);
            this.reconnectInterval = null;
        }
        if (this.testCommandModalInstance) this.testCommandModalInstance.dispose();
        if (this.triggerTradeModalInstance) this.triggerTradeModalInstance.dispose();
        if (this.toastInstance) this.toastInstance.dispose();
    }
});

// 全局属性注册 (utils.js 中的函数)
const utilsFunctions = ['formatDateTime', 'getClientStatusClass', 'getTradeStatusClass', 'getLogLevelClass'];
utilsFunctions.forEach(funcName => {
    if (typeof window[funcName] === 'function') {
        app.config.globalProperties[funcName] = window[funcName];
    } else {
        console.warn(`utils.js: ${funcName} function is not defined globally. Using fallback.`);
        // 提供一个非常基础的回退，避免模板渲染时直接报错
        app.config.globalProperties[funcName] = (...args) => {
            if (funcName.includes('Class')) return 'bg-secondary'; // For class functions
            return args[0] || '-'; // For formatting functions
        };
    }
});

app.mount('#app');