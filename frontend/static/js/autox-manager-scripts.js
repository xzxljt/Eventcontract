// frontend/static/js/autox-manager-scripts.js

const app = Vue.createApp({
    data() {
        return {
            clients: [], // 从 WebSocket 获取的客户端对象列表
            tradeLogs: [],
            loadingClients: true,
            loadingLogs: true, // 初始时也设为 true
            tradeLogLimit: 50,
            currentClientForModal: null,
            testCommand: {
                type: 'test_echo',
                payload_str: '', // Payload as JSON string
                payload_error: null, // Error message for payload parsing
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
                type: 'success', // 'success', 'danger', 'warning', 'info'
                time: ''
            },
            websocket: null,
            reconnectInterval: null,
            testCommandModalInstance: null,
            triggerTradeModalInstance: null,
            toastInstance: null,
            // wsClientIdMap: new Map(), // To track client notes edits without overwriting props from WS
        };
    },
    watch: {
        'testCommand.payload_str'(newVal) {
            if (!newVal || newVal.trim() === '') {
                this.testCommand.payload_error = null;
                return;
            }
            try {
                JSON.parse(newVal);
                this.testCommand.payload_error = null;
            } catch (e) {
                this.testCommand.payload_error = '无效的 JSON 格式。';
            }
        },
    },
    methods: {
        connectWebSocket() {
            if (this.websocket && (this.websocket.readyState === WebSocket.OPEN || this.websocket.readyState === WebSocket.CONNECTING)) {
                this.showToast('连接提示', 'WebSocket 已经连接或正在连接中。', 'info');
                return;
            }

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${protocol}//${window.location.host}/ws/autox_status`;
            
            this.loadingClients = true; // Set loading true when attempting to connect
            this.showToast('连接中...', '正在尝试连接到 AutoX 状态服务器...', 'info');

            let connectTimeoutId = null;
            const CONNECTION_TIMEOUT = 15000; // 15 秒连接超时

            // Clear any existing reconnect interval before starting a new connection attempt
            if (this.reconnectInterval) {
                clearInterval(this.reconnectInterval);
                this.reconnectInterval = null;
            }

            this.websocket = new WebSocket(url);

            connectTimeoutId = setTimeout(() => {
                if (this.websocket && this.websocket.readyState !== WebSocket.OPEN) {
                    console.error(`AutoX Status WebSocket: Connection timed out after ${CONNECTION_TIMEOUT / 1000} seconds.`);
                    this.showToast('连接超时', `WebSocket 连接超时 (${CONNECTION_TIMEOUT / 1000}秒)。请检查网络或服务器状态。`, 'danger');
                    if (this.websocket) {
                        this.websocket.close(1000, "Connection timeout"); // Actively close
                    }
                    // onclose will be triggered, which can schedule a reconnect
                }
            }, CONNECTION_TIMEOUT);

            const clearConnectTimeout = () => {
                if (connectTimeoutId) {
                    clearTimeout(connectTimeoutId);
                    connectTimeoutId = null;
                }
            };

            this.websocket.onopen = () => {
                clearConnectTimeout();
                console.log('AutoX Status WebSocket 连接成功:', url);
                this.showToast('连接成功', '已连接到 AutoX 状态服务器。', 'success');
                // loadingClients will be set to false once the first client update is received
                // Reset reconnect attempts if any were made by scheduleReconnect
                if (this.reconnectAttempts) this.reconnectAttempts = 0;
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === 'autox_clients_update') {
                        console.log('收到客户端列表更新:', message.data);
                        
                        const newClientsData = message.data.map(client => {
                            const existingClient = this.clients.find(c => c.client_id === client.client_id);
                            return {
                                ...client,
                                notes_edit: existingClient ? existingClient.notes_edit : (client.notes || ''),
                            };
                        });
                        this.clients = newClientsData;

                        if (this.loadingClients) this.loadingClients = false;
                    }
                } catch (error) {
                    console.error('处理WebSocket消息失败:', error);
                    this.showToast('消息处理错误', `处理来自服务器的消息时出错: ${error.message}`, 'danger');
                }
            };

            this.websocket.onerror = (errorEvent) => {
                clearConnectTimeout();
                console.error('AutoX Status WebSocket 错误:', errorEvent);
                // this.loadingClients remains true or is set true in onclose
                let errorMsg = 'WebSocket 连接发生错误。';
                // Try to get more specific error if possible, though onerror often gives generic events
                if (errorEvent && errorEvent.message) { errorMsg += ` 详情: ${errorEvent.message}`; }
                else if (errorEvent && errorEvent.type) { errorMsg += ` 类型: ${errorEvent.type}`; }
                
                this.showToast('连接错误', `${errorMsg} 将尝试自动重连...`, 'danger');
                // onerror is usually followed by onclose. Reconnect logic is primarily in onclose.
                // Ensure socket is nulled if it exists and isn't closed, to allow scheduleReconnect to work.
                if (this.websocket && this.websocket.readyState !== WebSocket.CLOSED) {
                    this.websocket.close(1011, "WebSocket error occurred"); // 1011 = Internal Error
                }
                this.websocket = null;
            };

            this.websocket.onclose = (event) => {
                clearConnectTimeout();
                console.log('AutoX Status WebSocket 连接关闭:', event.code, event.reason);
                this.loadingClients = true; // Show loading as we are disconnected
                this.websocket = null; // Clear the instance

                let closeReasonMsg = `代码: ${event.code}, 原因: ${event.reason || '无'}`;
                if (event.code === 1000 && event.reason === "Connection timeout") {
                    // This was a client-side timeout, message already shown.
                    // Proceed to scheduleReconnect.
                } else if (!event.wasClean) {
                    this.showToast('连接断开', `WebSocket 连接已断开 (${closeReasonMsg})。尝试自动重连...`, 'warning');
                } else {
                    this.showToast('连接已关闭', `WebSocket 连接已正常关闭 (${closeReasonMsg})。`, 'info');
                    // If it was a clean closure initiated by client (e.g. unmount), don't auto-reconnect.
                    // However, if it's a server-initiated clean close (e.g. server restart), we might still want to.
                    // For simplicity, we'll attempt reconnect unless it's a specific "unmounting" reason.
                    if (event.reason === "Vue component unmounting") {
                        return; // Do not reconnect if unmounting
                    }
                }
                // Schedule reconnect for most close events, except specific client-initiated ones.
                if (event.code !== 1000 || (event.reason && event.reason !== "Vue component unmounting")) {
                     this.scheduleReconnect();
                }
            };
        },
        scheduleReconnect() {
            const MAX_RECONNECT_ATTEMPTS = 5;
            const RECONNECT_DELAY = 5000; // 5 seconds

            if (this.reconnectInterval) return; // Already scheduled

            if (!this.reconnectAttempts) this.reconnectAttempts = 0;

            this.reconnectInterval = setInterval(() => {
                if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                    if (this.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                        this.reconnectAttempts++;
                        console.log(`尝试重新连接WebSocket... (尝试 ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);
                        this.showToast('重连中...', `尝试重新连接 (${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`, 'info');
                        this.connectWebSocket(); // Attempt to reconnect
                    } else {
                        console.log('已达到最大重连次数。停止自动重连。');
                        this.showToast('重连失败', `已达到最大重连次数 (${MAX_RECONNECT_ATTEMPTS})。请手动刷新或检查服务。`, 'danger');
                        clearInterval(this.reconnectInterval);
                        this.reconnectInterval = null;
                        this.reconnectAttempts = 0; // Reset for future manual attempts
                    }
                } else {
                    // If socket exists and is not closed (e.g. OPEN or CONNECTING), clear interval.
                    // This case should ideally be handled by onopen clearing the interval.
                    clearInterval(this.reconnectInterval);
                    this.reconnectInterval = null;
                    this.reconnectAttempts = 0;
                }
            }, RECONNECT_DELAY);
        },
        async fetchTradeLogs(forceRefresh = false, userInitiated = false) {
            if (forceRefresh || this.tradeLogs.length === 0) {
                this.loadingLogs = true;
            }
            try {
                const response = await fetch(`/api/autox/trade_logs?limit=${this.tradeLogLimit}`);
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status} ${response.statusText}` }));
                    throw new Error(errorData.detail || `HTTP ${response.status} ${response.statusText}`);
                }
                const logs = await response.json();
                this.tradeLogs = logs.map(log => ({ ...log, showPayload: false }));
                if (userInitiated) {
                    this.showToast('日志已刷新', `成功获取 ${this.tradeLogs.length} 条最新交易日志。`, 'success');
                }
            } catch (error) {
                console.error("获取交易日志失败:", error);
                this.showToast('获取日志失败', `获取交易日志时发生错误: ${error.message}`, 'danger');
            } finally {
                this.loadingLogs = false;
            }
        },
        refreshClientsManually() {
            this.loadingClients = true;
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                 this.showToast('刷新提示', '客户端列表由服务器实时推送更新。', 'info');
                 // Give a small delay to allow any pending WS messages to arrive.
                 // If no messages arrive, loadingClients might stay true if clients list is empty.
                 // This will be reset to false when 'autox_clients_update' is received.
                 // If after a timeout, clients is still empty and loading is true, it means no clients.
                 setTimeout(() => {
                    if (this.loadingClients && this.clients.length === 0) {
                        // If still loading and no clients, it means WS is connected but no clients reported
                        this.loadingClients = false;
                    }
                 }, 2000);

            } else {
                this.showToast('无法刷新', 'WebSocket 未连接。请检查连接状态。', 'warning');
                // If not connected, loadingClients remains true, and scheduleReconnect should be active
            }
        },
        async saveClientNotes(clientId, notes_to_save) {
            // notes_to_save is the content from notes_edit input
            try {
                const response = await fetch(`/api/autox/clients/${clientId}/notes`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ notes: notes_to_save })
                });
                const result = await response.json(); // This is the updated client object from backend
                if (!response.ok) {
                    throw new Error(result.detail || `HTTP ${response.status}`);
                }
                
                // Update the client in the local list with the full object from the server
                // This ensures 'notes' (source of truth) and 'notes_edit' (UI input) are synced
                const clientIndex = this.clients.findIndex(c => c.client_id === clientId);
                if (clientIndex !== -1) {
                    this.clients[clientIndex] = {
                        ...result, // Server's version of the client data
                        notes_edit: result.notes || '' // Ensure notes_edit reflects the saved notes
                    };
                }
                this.showToast('备注已保存', `客户端 ${this.truncateId(clientId)} 的备注已更新。`, 'success');
            } catch (error) {
                console.error(`更新客户端 ${clientId} 备注失败:`, error);
                this.showToast('备注保存失败', `更新备注时发生错误: ${error.message}`, 'danger');
                // Optionally, find the client and reset its notes_edit to its 'notes' field
                // to discard the failed edit, if desired.
                 const clientToRevert = this.clients.find(c => c.client_id === clientId);
                 if (clientToRevert) {
                    clientToRevert.notes_edit = clientToRevert.notes || ''; // Revert edit field
                 }
            }
        },
        openTestCommandModal(client) {
            this.currentClientForModal = client;
            this.testCommand.type = 'test_echo'; // Default command
            this.testCommand.payload_str = ''; // Reset payload string
            this.testCommand.payload_error = null; // Reset payload error
            if (this.testCommandModalInstance) this.testCommandModalInstance.show();
        },
        async sendTestCommand() {
            if (!this.currentClientForModal || !this.testCommand.type) return;
            if (this.testCommand.payload_error) { // Check for pre-validated JSON error
                this.showToast('Payload错误', '请修正Payload中的JSON格式错误。', 'warning');
                return;
            }

            const clientId = this.currentClientForModal.client_id;
            let payload_to_send = null;
            if (this.testCommand.payload_str && this.testCommand.payload_str.trim() !== '') {
                try {
                    payload_to_send = JSON.parse(this.testCommand.payload_str);
                } catch (e) {
                    // Should have been caught by watcher, but a final check.
                    this.testCommand.payload_error = '无效的 JSON 格式。';
                    this.showToast('Payload错误', '请修正Payload中的JSON格式错误。', 'warning');
                    return;
                }
            }

            try {
                // API requires command_type in query, payload in body
                const response = await fetch(`/api/autox/clients/${clientId}/send_test_command?command_type=${encodeURIComponent(this.testCommand.type)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    // Send payload_to_send. If it's null, JSON.stringify(null) is "null", which is valid JSON.
                    // Backend should handle null payload if it's optional.
                    body: JSON.stringify(payload_to_send)
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.detail || `HTTP ${response.status}`);
                }
                this.showToast('测试指令已发送', `成功向 ${this.truncateId(clientId)} 发送指令 '${this.testCommand.type}'.`, 'success');
                if (this.testCommandModalInstance) this.testCommandModalInstance.hide();
                this.fetchTradeLogs(true, true); // Refresh logs to see command and its processing
            } catch (error) {
                console.error("发送测试指令失败:", error);
                this.showToast('测试指令发送失败', error.message, 'danger');
            }
        },
        openTriggerTradeModal(client) {
            this.currentClientForModal = client;
            // Pre-fill with sensible defaults or first supported symbol
            this.manualTrade.symbol = (client.supported_symbols && client.supported_symbols.length > 0) ? client.supported_symbols[0] : 'ETHUSDT';
            this.manualTrade.direction = 'up';
            this.manualTrade.amount = '5'; // Default amount
            this.manualTrade.signal_id = `manual_${this.truncateId(client.client_id, 4)}_${Date.now().toString().slice(-5)}`;
            if (this.triggerTradeModalInstance) this.triggerTradeModalInstance.show();
        },
        async triggerManualTrade() {
            if (!this.currentClientForModal) return;
            const clientId = this.currentClientForModal.client_id;
            
            // Basic validation
            if (!this.manualTrade.symbol || !this.manualTrade.symbol.trim()) {
                this.showToast('输入无效', '交易对不能为空。', 'warning'); return;
            }
            if (!this.manualTrade.amount || isNaN(parseFloat(this.manualTrade.amount)) || parseFloat(this.manualTrade.amount) <= 0) {
                this.showToast('输入无效', '请输入一个有效的正数金额。', 'warning'); return;
            }

            const payload = {
                symbol: this.manualTrade.symbol.toUpperCase().trim(),
                direction: this.manualTrade.direction,
                amount: String(this.manualTrade.amount), // Backend expects string amount
                signal_id: this.manualTrade.signal_id.trim() || null // Send null if empty
            };

            try {
                const response = await fetch(`/api/autox/clients/${clientId}/trigger_trade_command`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.detail || `HTTP ${response.status}`);
                }
                this.showToast('交易指令已发送', `成功向 ${this.truncateId(clientId)} 发送交易指令. Signal ID: ${result.sent_command?.payload?.signal_id || 'N/A'}`, 'success');
                if (this.triggerTradeModalInstance) this.triggerTradeModalInstance.hide();
                this.fetchTradeLogs(true, true); // Refresh logs to see the command
            } catch (error) {
                console.error("手动触发交易失败:", error);
                this.showToast('手动交易失败', error.message, 'danger');
            }
        },
        showToast(title, message, type = 'success') {
            this.toast.title = title;
            this.toast.message = message; // HTML is allowed via v-html in template
            this.toast.type = type;
            this.toast.time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            
            const toastEl = document.getElementById('appToast');
            if (!this.toastInstance && toastEl) {
                this.toastInstance = new bootstrap.Toast(toastEl, { delay: 5000 });
            }
            if (this.toastInstance) {
                 this.toastInstance.show();
            }
        },
        togglePayloadVisibility(log) {
            log.showPayload = !log.showPayload;
        },
        truncateId(id, length = 8) {
            if (!id) return 'N/A';
            if (id.length <= length) return id;
            return id.substring(0, length) + '...';
        },
        getClientStatusText(status) {
            const map = {
                'idle': '空闲',
                'processing_trade': '交易中',
                'offline': '离线',
            };
            return map[status] || status.charAt(0).toUpperCase() + status.slice(1);
        },
        calculateSymbolHue(symbol) {
            let hash = 0;
            for (let i = 0; i < symbol.length; i++) {
                hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
                hash = hash & hash; // Convert to 32bit integer
            }
            return Math.abs(hash) % 360;
        },
        getSymbolBadgeClass(symbol) {
            // Using HSL for more control over saturation and lightness for badges
            // text-dark is a bootstrap class that sets color to a dark gray
            // For very light backgrounds, text-dark is good.
            // We need to choose a text color that contrasts well with the HSL background.
            // A simple heuristic: if lightness > 50%, use dark text.
            const lightness = 75; // Keep lightness high for badge background
            const textColorClass = lightness > 60 ? 'text-dark' : ''; // Simplified
            return `hsl-badge ${textColorClass}`; // CSS will handle the HSL color via --symbol-hue
        },
        getTradeStatusText(status, error_message, payload) {
            if (error_message) return '失败';
            const statusMap = {
                'command_sent_to_client': '指令已发送',
                'test_command_sent_to_client': '测试指令发送',
                'status_from_client': `客户端: ${payload?.status || '更新'}`,
                'command_received': '客户端已接收',
                'trade_params_set': '参数已设置',
                'trade_confirmed': '交易已确认',
                'trade_execution_succeeded': '执行成功',
                'trade_execution_failed': '执行失败',
                'manual_confirmation_pending': '待人工确认',
                'internal_error': '客户端错误',
                'test_triggered_execute_trade': '手动交易发送'
            };
            return statusMap[status] || status;
        },
        getTradeLogRowClass(status, error_message){
            if (error_message || status === 'trade_execution_failed' || status === 'internal_error') {
                return 'table-danger-soft';
            }
            if (status === 'trade_execution_succeeded') {
                return 'table-success-soft';
            }
            return '';
        }
    },
    async mounted() {
        const testModalEl = document.getElementById('testCommandModal');
        if (testModalEl) this.testCommandModalInstance = new bootstrap.Modal(testModalEl);

        const triggerModalEl = document.getElementById('triggerTradeModal');
        if (triggerModalEl) this.triggerTradeModalInstance = new bootstrap.Modal(triggerModalEl);

        // Toast must be initialized after the element is in the DOM
        const toastEl = document.getElementById('appToast');
        if (toastEl) this.toastInstance = new bootstrap.Toast(toastEl, {delay: 5000});

        this.connectWebSocket();
        await this.fetchTradeLogs(true); // Initial log fetch
    },
    beforeUnmount() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.close(1000, "Vue component unmounting"); // 1000 is normal closure
        }
        if (this.reconnectInterval) {
            clearInterval(this.reconnectInterval);
            this.reconnectInterval = null;
        }
        // Dispose Bootstrap components
        if (this.testCommandModalInstance) this.testCommandModalInstance.dispose();
        if (this.triggerTradeModalInstance) this.triggerTradeModalInstance.dispose();
        if (this.toastInstance) this.toastInstance.dispose();
    }
});

// Register global utility functions (assuming they are loaded from utils.js)
// Fallback basic versions are provided if utils.js functions are not found.
const utilityFunctionsDefaults = {
    formatDateTime: (datetimeStr) => {
        if (!datetimeStr) return 'N/A';
        try {
            const dt = new Date(datetimeStr);
            if (isNaN(dt.getTime())) return datetimeStr;
            return `${dt.getFullYear()}-${(dt.getMonth() + 1).toString().padStart(2, '0')}-${dt.getDate().toString().padStart(2, '0')} ${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}`;
        } catch (e) { return datetimeStr; }
    },
    getClientStatusClass: (status) => { // Uses Bootstrap 5.3 text-emphasis utility classes
        if (status === 'idle') return 'badge bg-success-subtle text-success-emphasis border border-success-subtle';
        if (status === 'processing_trade') return 'badge bg-primary-subtle text-primary-emphasis border border-primary-subtle';
        if (status === 'offline') return 'badge bg-secondary-subtle text-secondary-emphasis border border-secondary-subtle';
        return 'badge bg-warning-subtle text-warning-emphasis border border-warning-subtle';
    },
    getTradeStatusClass: (status, error_message) => { // Uses Bootstrap 5.3 text-emphasis utility classes
        if (error_message || status === 'trade_execution_failed' || status === 'internal_error') return 'badge bg-danger-subtle text-danger-emphasis border border-danger-subtle';
        if (status === 'trade_execution_succeeded') return 'badge bg-success-subtle text-success-emphasis border border-success-subtle';
        if (status === 'command_sent_to_client' || status === 'test_command_sent_to_client' || status === 'command_received' || status === 'test_triggered_execute_trade') return 'badge bg-info-subtle text-info-emphasis border border-info-subtle';
        return 'badge bg-light text-dark border';
    }
};

Object.keys(utilityFunctionsDefaults).forEach(funcName => {
    if (typeof window[funcName] === 'function') {
        app.config.globalProperties[funcName] = window[funcName];
    } else {
        console.warn(`utils.js: ${funcName} function not found globally. Using basic fallback.`);
        app.config.globalProperties[funcName] = utilityFunctionsDefaults[funcName];
    }
});

app.mount('#app');