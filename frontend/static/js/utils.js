// frontend/static/js/utils.js

/**
 * 格式化时间戳或日期字符串为本地化的日期时间字符串。
 * @param {number|string|Date} timestampOrDateString - 秒级时间戳, 可被Date解析的日期字符串, 或Date对象。
 * @param {string} locale - 区域设置，默认为 'zh-CN'。
 * @param {object} options - toLocaleString 的选项。
 * @returns {string} 格式化的日期时间字符串，或在无效输入时返回 '-' 或原始字符串。
 */
function formatDateTime(timestampOrDateString, locale = 'zh-CN', options = null) {
    if (!timestampOrDateString && timestampOrDateString !== 0) return '-';

    let date;
    if (timestampOrDateString instanceof Date) {
        date = timestampOrDateString;
    } else if (typeof timestampOrDateString === 'number') {
        // 假设是毫秒级时间戳 (来自 new Date().getTime() 或 Python datetime.timestamp() * 1000)
        // 如果后端传的是秒级时间戳 (time.time()), 需要乘以 1000
        // 先检查是否可能是秒级（例如，10位数通常是秒级）
        if (String(timestampOrDateString).length === 10) {
             date = new Date(timestampOrDateString * 1000);
        } else {
             date = new Date(timestampOrDateString); // 否则假设是毫秒
        }
    } else if (typeof timestampOrDateString === 'string') {
        // 尝试处理 ISO 8601 格式或其他标准格式
        // 如果字符串看起来像纯数字，则尝试作为时间戳处理
        if (/^\d+$/.test(timestampOrDateString)) {
            if (timestampOrDateString.length === 10) { // 秒级
                date = new Date(Number(timestampOrDateString) * 1000);
            } else if (timestampOrDateString.length === 13) { // 毫秒级
                date = new Date(Number(timestampOrDateString));
            } else {
                 date = new Date(timestampOrDateString); // 否则按字符串解析
            }
        } else {
            date = new Date(timestampOrDateString);
        }
    } else {
        return String(timestampOrDateString); // 类型未知
    }

    if (isNaN(date.getTime())) {
        return String(timestampOrDateString); // 无效日期
    }
    
    const defaultOptions = {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
        // timeZone: 'Asia/Shanghai' // 通常本地时间更好，除非有特定需求
    };

    try {
        // Safari 对于 / 的日期格式有时会有问题，确保替换
        let formattedDate = date.toLocaleString(locale, options || defaultOptions);
        if (formattedDate.includes('/')) { // 替换斜杠为连字符
            const parts = formattedDate.split(' ');
            if (parts[0]) {
                 parts[0] = parts[0].replace(/\//g, '-');
                 formattedDate = parts.join(' ');
            }
        }
        return formattedDate;
    } catch (e) {
        console.error("Error formatting date:", timestampOrDateString, e);
        // 回退到更简单的格式
        try {
            const YYYY = date.getFullYear();
            const MM = String(date.getMonth() + 1).padStart(2, '0');
            const DD = String(date.getDate()).padStart(2, '0');
            const hh = String(date.getHours()).padStart(2, '0');
            const mm = String(date.getMinutes()).padStart(2, '0');
            const ss = String(date.getSeconds()).padStart(2, '0');
            return `${YYYY}-${MM}-${DD} ${hh}:${mm}:${ss}`;
        } catch (fallbackError) {
            return String(timestampOrDateString); // 最终回退
        }
    }
}

/**
 * 根据 AutoX 客户端状态返回对应的 Bootstrap 背景色类名。
 * @param {string} status - 客户端状态。
 * @returns {string} Bootstrap 背景色类名。
 */
function getClientStatusClass(status) {
    if (!status) return 'bg-secondary';
    switch (String(status).toLowerCase()) {
        case 'idle':
            return 'bg-success'; // 空闲状态，绿色
        case 'processing_trade':
            return 'bg-info text-dark'; // 处理中，信息蓝
        case 'error':
            return 'bg-danger'; // 错误状态，红色
        case 'offline': // 假设的离线状态
            return 'bg-warning text-dark';
        case 'connected': // 早期版本可能的状态
             return 'bg-success';
        default:
            return 'bg-secondary'; // 其他未知状态，灰色
    }
}


/**
 * 根据 AutoX 交易日志条目的状态返回对应的 Bootstrap 背景色类名。
 * @param {string} logStatus - 交易日志状态。
 * @returns {string} Bootstrap 背景色类名。
 */
function getTradeStatusClass(logStatus) {
    if (!logStatus) return 'bg-light text-dark border'; // 默认或未知状态的样式
    const statusLower = String(logStatus).toLowerCase();

    if (statusLower.includes('success') || statusLower.includes('sent') || statusLower.includes('received') || statusLower.includes('ready')) {
        return 'bg-success';
    } else if (statusLower.includes('fail') || statusLower.includes('error')) {
        return 'bg-danger';
    } else if (statusLower.includes('pending') || statusLower.includes('processing')) {
        return 'bg-info text-dark';
    } else if (statusLower.includes('warn') || statusLower.includes('manual')) {
        return 'bg-warning text-dark';
    } else if (statusLower === 'status_from_client' || statusLower === 'command_sent_to_client' || statusLower === 'test_command_sent_to_client') {
        return 'bg-primary'; // 明确的指令发送或客户端状态更新
    } else {
        return 'bg-secondary'; // 其他状态
    }
}


/**
 * 根据日志级别返回对应的 Bootstrap 背景色类名。
 * (此函数在 autox-manager.html 中未使用，但保留以备将来之用或用于其他页面)
 * @param {string} level - 日志级别 (INFO, WARNING, ERROR, etc.)。
 * @returns {string} Bootstrap 背景色类名。
 */
function getLogLevelClass(level) {
    if (!level) return 'bg-secondary';
    switch (String(level).toUpperCase()) {
        case 'INFO':
            return 'bg-info text-dark';
        case 'WARNING':
            return 'bg-warning text-dark';
        case 'ERROR':
            return 'bg-danger';
        case 'DEBUG':
            return 'bg-success';
        case 'CRITICAL':
            return 'bg-danger fw-bold';
        default:
            return 'bg-secondary';
    }
}

/**
 * 根据胜率返回对应的文本颜色类名。
 * @param {number|null|undefined} rate - 胜率 (0-100)。
 * @returns {string} 文本颜色类名 (win-rate-high, win-rate-medium, win-rate-low) 或空字符串。
 */
function getWinRateClass(rate) {
    if (rate === null || typeof rate === 'undefined' || isNaN(parseFloat(rate))) {
        return 'win-rate-low';
    }
    const numericRate = parseFloat(rate);
    if (numericRate >= 70) return 'win-rate-high';
    if (numericRate >= 50) return 'win-rate-medium';
    return 'win-rate-low';
}

/**
 * 根据 PnL (盈亏) 值返回对应的文本颜色类名。
 * @param {number|null|undefined} pnl - PnL 值。
 * @returns {string} 文本颜色类名 (pnl-positive, pnl-negative, pnl-neutral) 或空字符串。
 */
function getPnlClass(pnl) {
    if (pnl === null || typeof pnl === 'undefined' || isNaN(parseFloat(pnl))) {
        return 'pnl-neutral';
    }
    const numericPnl = parseFloat(pnl);
    if (numericPnl > 0) return 'pnl-positive';
    if (numericPnl < 0) return 'pnl-negative';
    return 'pnl-neutral';
}

/**
 * 根据信号状态（是否已验证、结果如何）返回信号卡片的边框类名。
 * @param {object} signal - 信号对象，期望包含 verified 和 result 属性。
 * @returns {string} 边框类名。
 */
function getSignalStatusClass(signal) {
    if (!signal) return '';
    if (!signal.verified) return 'pending';
    return signal.result ? 'success' : 'failed';
}

/**
 * 格式化日期对象为 'YYYY-MM-DDTHH:mm' 格式，用于 datetime-local 输入框。
 * @param {Date|string|number} dateInput - 日期对象、可解析的日期字符串或时间戳。
 * @returns {string} 格式化的日期字符串，或在无效时返回空字符串。
 */
function formatDateForInput(dateInput) {
    let date;
    if (dateInput instanceof Date) {
        date = dateInput;
    } else if (typeof dateInput === 'string' || typeof dateInput === 'number') {
        date = new Date(dateInput);
    } else {
        return ''; //无法识别的输入类型
    }

    if (isNaN(date.getTime())) {
        return ''; // 无效日期
    }

    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${year}-${month}-${day}T${hours}:${minutes}`;
}