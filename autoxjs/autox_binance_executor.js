// autox_binance_executor.js (Aggressive Mode v3 - Integrated Optimized Find Logic & Delays)
// -------------------- 可配置参数 --------------------
var SERVER_WS_URL = "ws://192.168.1.2::8000/ws/autox_control";
var CLIENT_ID = "";
var SUPPORTED_SYMBOLS = ["ETHUSDT"]; // 您可以根据需要添加更多交易对

// --- 延迟与超时配置 (请根据您的设备性能和网络状况调整) ---
var FIND_TIMEOUT = 500;       // 控件查找超时(毫秒)
var OPERATION_DELAY = 100;    // 每次点击/设置文本后的短暂固定延迟(毫秒)
var CLICK_RETRY_TIMES = 1;    // 点击重试次数

var KEYBOARD_HIDE_DELAY = 300; // 隐藏键盘后的延迟(毫秒)
var DELAY_AFTER_KEYBOARD_HIDE_BEFORE_BUTTON_SEARCH = 300; // 隐藏键盘后，在搜索方向按钮前额外等待(毫秒)
var DELAY_BEFORE_CONFIRM_BUTTON_SEARCH = 400; // 点击方向按钮后，查找确认按钮前的额外延迟(毫秒)

var MAX_PARENT_SEARCH_DEPTH = 3; // 向上查找可点击父按钮的最大层数 (针对LinearLayout嵌套TextView的情况)

// -------------------- 全局变量 --------------------
var webSocketConnection = null;
var isWebSocketConnected = false;
var currentTradeSignalId = null;
var httpClient = null;
var explicitReconnectScheduled = false; // 用于控制特定情况下的重连逻辑

// -------------------- 辅助函数 --------------------
function logOnly(message) {
    console.log(message);
}

function initializeClientId() {
    var storedId = files.join(files.getSdcardPath(), "autox_client_id.txt");
    if (files.exists(storedId)) {
        CLIENT_ID = files.read(storedId);
        logOnly("从存储中读取到客户端ID: " + CLIENT_ID);
    } else {
        CLIENT_ID = java.util.UUID.randomUUID().toString();
        files.write(storedId, CLIENT_ID);
        logOnly("生成并存储了新的客户端ID: " + CLIENT_ID);
    }
    if (!CLIENT_ID) {
        logOnly("错误：未能初始化客户端ID！脚本退出。");
        exit();
    }
}

function findElement(selector, timeout, description) {
    timeout = timeout || FIND_TIMEOUT;
    description = description || selector.toString().substring(0, 50);
    var el = null;
    try {
        el = selector.findOne(timeout);
    } catch(e) {
        logOnly("查找控件 '" + description + "' 时发生选择器错误: " + e);
        return null;
    }
    if (el) {
        // logOnly("找到控件: " + description); // 正式脚本中可以注释掉，减少日志量
        return el;
    } else {
        logOnly("未找到控件 (超时" + timeout + "ms): " + description);
        return null;
    }
}

function clickElement(element, description, retries) {
    if (!element) {
        logOnly("无法点击空元素: " + description);
        return false;
    }
    retries = retries === undefined ? CLICK_RETRY_TIMES : retries; // 如果未提供retries，则使用全局默认值
    description = description || "某个控件";

    for (var i = 0; i <= retries; i++) {
        if (element.clickable()) { 
            if (element.click()) {
                // logOnly("通过 element.click() 点击了: " + description); // 正式脚本中可以注释掉
                sleep(OPERATION_DELAY);
                return true;
            }
        }
        var bounds = element.bounds();
        if (bounds && bounds.width() > 0 && bounds.height() > 0) {
            // logOnly("尝试通过坐标点击: " + description); // 正式脚本中可以注释掉
            if (click(bounds.centerX(), bounds.centerY())) {
                // logOnly("通过 click(x, y) 点击了: " + description); // 正式脚本中可以注释掉
                sleep(OPERATION_DELAY);
                return true;
            }
        }
        if (i < retries) {
            logOnly("点击 " + description + " 失败，重试 (" + (i + 1) + "/" + retries + ")");
            sleep(OPERATION_DELAY); 
        }
    }
    logOnly("多次尝试后仍无法点击 " + description);
    return false;
}

function hideKeyboard() {
    // logOnly("尝试隐藏键盘..."); // 正式脚本中可以注释掉
    if (currentActivity() !== null) { // 确保有活动窗口才执行back
        back();
        sleep(KEYBOARD_HIDE_DELAY);
        // logOnly("已执行隐藏键盘操作，并等待 " + KEYBOARD_HIDE_DELAY + "ms"); // 正式脚本中可以注释掉
    } else {
        logOnly("没有检测到当前活动窗口，跳过隐藏键盘操作。");
    }
}

// -------------------- WebSocket 相关函数 (保持不变)--------------------


function connectWebSocket() {
    logOnly(">>> connectWebSocket function called.");
    explicitReconnectScheduled = false; // 确保每次调用都重置此标志
    if (isWebSocketConnected && webSocketConnection) {
        logOnly("connectWebSocket: Already connected, returning.");
        return;
    }
    logOnly("尝试连接 WebSocket: " + SERVER_WS_URL);
    isWebSocketConnected = false; 

    if (!httpClient) {
        try {
            importPackage(Packages["okhttp3"]);
            httpClient = new OkHttpClient.Builder()
                .retryOnConnectionFailure(true)
                .readTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .writeTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
                .build();
            logOnly("OkHttpClient已初始化并配置了超时。");
        } catch(e) {
            logOnly("导入 OkHttp 包失败: " + e);
            console.error("OkHttp 初始化错误: ", e);
            return;
        }
    }

    var request = new Request.Builder().url(SERVER_WS_URL).build();
    var listener = {
        onOpen: function (webSocket, response) {
            logOnly(">>> onOpen triggered."); 
            isWebSocketConnected = true;
            webSocketConnection = webSocket;
            logOnly("WebSocket 连接成功!");
            sendRegisterMessage();
        },
        onMessage: function (webSocket, msg) {
            logOnly("收到 WebSocket 消息: " + msg);
            try {
                var message = JSON.parse(msg);
                if (message.type === "execute_trade") {
                    currentTradeSignalId = message.payload.signal_id;
                    logOnly("收到交易指令，信号ID: " + currentTradeSignalId);
                    sendStatusUpdate(currentTradeSignalId, "command_received", "已收到交易指令，准备执行。", null);

                    threads.start(function() {
                        executeBinanceTradeAggressive(
                            message.payload.symbol,
                            message.payload.direction,
                            message.payload.amount
                        );
                    });

                } else if (message.type === "registered"){
                    logOnly("服务器确认注册: " + (message.message || ""));
                } else if (message.type === "error"){
                     logOnly("服务器错误: " + (message.message || "未知错误"));
                }
                else if (message.type && message.type.startsWith("test_")) {
                    logOnly("收到测试指令: " + message.type);
                    sendStatusUpdate(null, "test_command_received", "已收到测试指令: " + message.type, {received_payload: message.payload});
                } else if (message.type === "server_shutting_down") {
                    logOnly("收到服务器关闭通知 (server_shutting_down)。");
                    var delaySeconds = 60; // 默认延迟
                    if (message.payload && typeof message.payload.reconnect_delay_seconds === 'number' && message.payload.reconnect_delay_seconds > 0) {
                        delaySeconds = message.payload.reconnect_delay_seconds;
                    }
                    logOnly("服务器建议在 " + delaySeconds + " 秒后重连。");

                    explicitReconnectScheduled = true; // 设置标志，避免 onClosed/onFailure 立即重连

                    if (webSocketConnection) {
                        try {
                            logOnly("主动关闭当前WebSocket连接 (因服务器关闭通知)...");
                            webSocketConnection.close(1000, "Server initiated shutdown");
                        } catch (e) {
                            logOnly("关闭WebSocket时发生错误: " + e);
                        }
                    }
                    isWebSocketConnected = false;
                    webSocketConnection = null; // 清理引用

                    logOnly("将在 " + delaySeconds + " 秒后尝试重新连接...");
                    setTimeout(function() {
                        logOnly("延迟时间已到，尝试重新连接 (因服务器关闭通知)...");
                        connectWebSocket();
                    }, delaySeconds * 1000);
                }
            } catch (e) {
                logOnly("处理WebSocket消息出错: " + e);
                console.error("消息处理错误:", e.stack ? e.stack : e);
                if (currentTradeSignalId) {
                    sendStatusUpdate(currentTradeSignalId, "internal_error", "客户端处理消息时发生错误。", e.toString());
                }
            }
        },
        onClosing: function (webSocket, code, reason) {
            logOnly(">>> onClosing triggered. Code: " + code + ", Reason: " + reason); 
            logOnly("WebSocket 正在关闭: " + code + ", " + reason);
            // 主动清理资源
            isWebSocketConnected = false;
            webSocketConnection = null;
        },
        onClosed: function (webSocket, code, reason) {
            logOnly(">>> onClosed triggered. Code: " + code + ", Reason: " + reason); 
            isWebSocketConnected = false;
            webSocketConnection = null;
            logOnly("WebSocket 已关闭 (internal state updated): " + code + ", " + reason);
            
            if (explicitReconnectScheduled) {
                logOnly("onClosed: 显式重连已安排 (server_shutting_down)，跳过本次自动重连。");
                return;
            }
            // 根据关闭代码决定是否重连
            if (code === 1008) { // 策略违规（如心跳超时）
                logOnly("由于策略违规关闭（可能是心跳超时），20秒后尝试重连...");
                setTimeout(connectWebSocket, 20000);
            } else if (code === 1000) { // 正常关闭
                logOnly("服务器正常关闭连接，30秒后尝试重连...");
                setTimeout(connectWebSocket, 30000);
            } else { // 其他异常情况
                logOnly("连接异常关闭，10秒后尝试重连...");
                setTimeout(connectWebSocket, 10000);
            }
        },
        onFailure: function (webSocket, t, response) {
            logOnly(">>> onFailure triggered. Error: " + (t ? t.getMessage() : "Unknown error object"));
            isWebSocketConnected = false;
            webSocketConnection = null;
            logOnly("WebSocket 连接失败: " + (t ? t.getMessage() : "N/A"));
            if (response) logOnly("失败响应: " + response.toString());
            if (t) console.error("WebSocket 失败详情:", t); else console.error("WebSocket 失败详情: Unknown error object");

            if (explicitReconnectScheduled) {
                logOnly("onFailure: 显式重连已安排 (server_shutting_down)，跳过本次自动重连。");
                return;
            }
            setTimeout(function() {
                logOnly(">>> onFailure: Connection failed, attempting reconnect in 20 seconds...");
                connectWebSocket();
            }, 20000);
        }
    };

     try {
        logOnly("httpClient.newWebSocket 调用中...");
        httpClient.newWebSocket(request, new WebSocketListener(listener));
        logOnly("httpClient.newWebSocket 调用完成。");
    } catch (e) {
        logOnly("创建 WebSocket 连接时出错: " + e);
        console.error("newWebSocket Error:", e);
        setTimeout(function() {
            logOnly(">>> setTimeout in newWebSocket CATCH block executing, calling connectWebSocket...");
            connectWebSocket();
        }, 10000); 
    }
}

function sendRegisterMessage() {
    if (!isWebSocketConnected || !webSocketConnection) {
        logOnly("WebSocket未连接，无法发送注册消息。");
        return;
    }
    lastMessageTime = new Date().getTime(); // 更新最后消息时间
    var registerMsg = {
        type: "register",
        payload: {
            client_id: CLIENT_ID,
            supported_symbols: SUPPORTED_SYMBOLS,
            device_name: device.brand + " " + device.model,
            app_version: "1.3.2-aggressive-v3-optimized" // 更新版本号
        }
    };
    try {
        webSocketConnection.send(JSON.stringify(registerMsg));
        logOnly("注册消息已发送: " + JSON.stringify(registerMsg));
    } catch (e) {
        logOnly("发送注册消息失败: " + e);
        isWebSocketConnected = false; 
        webSocketConnection = null;
        setTimeout(connectWebSocket, 5000); 
    }
}

function sendStatusUpdate(signalId, tradeStatus, details, additionalData) {
    if (!isWebSocketConnected || !webSocketConnection) {
        logOnly("WebSocket未连接，无法发送状态更新。");
        return;
    }
    var statusMsgPayload = {
        client_id: CLIENT_ID,
        signal_id: signalId || currentTradeSignalId,
        status: tradeStatus,
        details: details || null,
        timestamp: new Date().toISOString()
    };
    if (additionalData && typeof additionalData === 'object' && additionalData !== null) {
        if (additionalData instanceof Error) {
            statusMsgPayload.error_message = additionalData.message;
            statusMsgPayload.error_stack = additionalData.stack;
        } else {
            statusMsgPayload.general_data = additionalData;
        }
    } else if (typeof additionalData === 'string') {
         statusMsgPayload.error_message = additionalData;
    }

    var statusMsg = {
        type: "status_update",
        payload: statusMsgPayload
    };
    try {
        webSocketConnection.send(JSON.stringify(statusMsg));
        // logOnly("状态更新已发送: " + JSON.stringify(statusMsg)); 
    } catch (e) {
        logOnly("发送状态更新失败: " + e);
        isWebSocketConnected = false; 
        webSocketConnection = null;
        setTimeout(connectWebSocket, 5000); 
    }
}

// -------------------- 币安App自动化交易流程 (集成优化查找逻辑) --------------------
function executeBinanceTradeAggressive(symbol, direction, amountStr) {
    var localSignalId = currentTradeSignalId;
    logOnly("执行交易流程 (v3 Optimized): " + symbol + ", " + direction + ", " + amountStr + ", SignalID=" + localSignalId);
    sendStatusUpdate(localSignalId, "trade_execution_started", "交易流程开始执行。", {symbol: symbol, direction: direction, amount: amountStr});

    // 步骤 1: 输入投资额
    logOnly("步骤 1: 输入投资额...");
    // 首次查找金额输入框可以给更长的超时，例如 FIND_TIMEOUT * 2
    var amountInput = findElement(className("android.widget.EditText").depth(14), FIND_TIMEOUT * 2, "投资额输入框(按类名和深度)");
     if (!amountInput) { // 如果特定深度的找不到，尝试不带深度的通用查找
        logOnly("按深度查找投资额输入框失败，尝试通用查找...");
        amountInput = findElement(className("android.widget.EditText"), FIND_TIMEOUT * 2, "投资额输入框(通用按类名)");
    }

    if (!amountInput) {
        var errMsg = "未找到投资额输入框。";
        logOnly(errMsg);
        sendStatusUpdate(localSignalId, "trade_execution_failed", errMsg, { step: "find_amount_input" });
        currentTradeSignalId = null;
        return false;
    }

    if (!clickElement(amountInput, "投资额输入框(尝试激活)", 0)) { // 点击次数为0，因为setText前通常需要激活
         logOnly("警告: 点击投资额输入框以激活可能未成功，继续尝试setText...");
         sendStatusUpdate(localSignalId, "amount_input_activation_warning", "投资额输入框激活可能未成功。", { method: "clickElement" });
    }
    
    amountStr = parseInt(amountStr).toString(); // 确保是字符串整数
    logOnly("处理后投资额 (用于setText): " + amountStr);

    if (amountInput.setText(amountStr)) {
        logOnly("投资金额已尝试通过 setText 设置为: " + amountStr);
        sendStatusUpdate(localSignalId, "amount_set_attempted", "投资金额已尝试填写: " + amountStr, { method: "setText", success: true });
    } else {
        // setText 返回 false 不一定代表失败，有些情况下文本依然设置了
        logOnly("警告: setText 方法返回 false，投资额 '" + amountStr + "' 可能未完全按预期设置。继续流程但标记此警告。");
        sendStatusUpdate(localSignalId, "amount_set_warning", "setText方法返回false，投资额可能未设置成功。", { method: "setText", success_flag: false, amount_str: amountStr });
        // 不在此处立即返回失败，因为有时文本还是设置上了
    }
    hideKeyboard(); // hideKeyboard 内部包含 KEYBOARD_HIDE_DELAY
    
    logOnly("隐藏键盘后，等待 " + DELAY_AFTER_KEYBOARD_HIDE_BEFORE_BUTTON_SEARCH + "ms ...");
    sleep(DELAY_AFTER_KEYBOARD_HIDE_BEFORE_BUTTON_SEARCH);
    sendStatusUpdate(localSignalId, "after_keyboard_hide_delay_completed", "键盘隐藏后延迟完成。", null);


    // 步骤 2: 点击 "下跌" 或 "上涨" 按钮 (使用优化后的查找逻辑)
    logOnly("步骤 2: 点击方向按钮 (" + direction + ")...");
    var directionButtonRegex;
    var directionButtonDesc;

    if (direction.toLowerCase() === "up") {
        directionButtonRegex = /看涨|上涨|UP|BUY|^\s*买\s*$/i; // 根据实际按钮文本调整
        directionButtonDesc = "'上涨/看涨'按钮";
    } else { // "down"
        directionButtonRegex = /看跌|下跌|DOWN|SELL|^\s*卖\s*$/i; // 根据实际按钮文本调整
        directionButtonDesc = "'下跌/看跌'按钮";
    }

    var directionButton = null;
    // 主要策略: 查找可点击的LinearLayout，其内部有匹配文本的TextView
    var clickableLayouts = className("android.widget.LinearLayout").clickable(true).find();
    for (var i = 0; i < clickableLayouts.length; i++) {
        var layout = clickableLayouts[i];
        var textElementInside = layout.findOne(className("android.widget.TextView").textMatches(directionButtonRegex));
        if (textElementInside) {
            directionButton = layout; 
            logOnly("找到方向按钮 (LinearLayout策略): " + textElementInside.text());
            break;
        }
    }
    
    // 备选策略: 查找文本，然后向上查找其可点击的LinearLayout父控件
    if (!directionButton) {
        logOnly("LinearLayout策略未找到方向按钮，尝试TextView父控件策略...");
        var textElements = className("android.widget.TextView").textMatches(directionButtonRegex).find();
        for (var i = 0; i < textElements.length; i++) {
            var tv = textElements[i];
            var currentParent = tv.parent();
            var depth = 0;
            while(currentParent && depth < MAX_PARENT_SEARCH_DEPTH) {
                if (currentParent.className() === "android.widget.LinearLayout" && currentParent.clickable()) {
                    directionButton = currentParent;
                    logOnly("找到方向按钮 (TextView父控件策略): " + tv.text());
                    break; 
                }
                currentParent = currentParent.parent();
                depth++;
            }
            if (directionButton) break; 
        }
    }

    if (!directionButton) {
        var errMsgDir = "未找到" + directionButtonDesc + "。请检查按钮文字或UI结构。";
        logOnly(errMsgDir);
        sendStatusUpdate(localSignalId, "trade_execution_failed", errMsgDir, { step: "find_direction_button", regex: directionButtonRegex.source });
        currentTradeSignalId = null;
        return false;
    }

    if (!clickElement(directionButton, directionButtonDesc)) {
        var errMsgClickDir = "点击" + directionButtonDesc + "失败。";
        logOnly(errMsgClickDir);
        sendStatusUpdate(localSignalId, "trade_execution_failed", errMsgClickDir, { step: "click_direction_button", element_desc: directionButtonDesc });
        currentTradeSignalId = null;
        return false;
    }
    logOnly("已点击方向按钮: " + directionButtonDesc);
    sendStatusUpdate(localSignalId, "direction_selected", "方向已选择: " + direction, { button_description: directionButtonDesc });

    logOnly("点击方向按钮后，等待 " + DELAY_BEFORE_CONFIRM_BUTTON_SEARCH + "ms ...");
    sleep(DELAY_BEFORE_CONFIRM_BUTTON_SEARCH);
    sendStatusUpdate(localSignalId, "before_confirm_button_delay_completed", "确认按钮前延迟完成。", null);

    // 步骤 3: 查找并点击 "确认" 按钮 (使用优化后的查找逻辑)
    logOnly("步骤 3: 查找并点击'确认'按钮...");
    var confirmButtonRegex = /确认|CONFIRM|OK|确定|立即下单|PLACE ORDER/i; // 根据实际按钮文本调整
    var confirmButtonDesc = "'确认'按钮";
    var confirmButton = null;

    // 主要策略: 直接查找 className("android.widget.Button").text("确认")
    confirmButton = className("android.widget.Button").text("确认").findOne(FIND_TIMEOUT);

    if (!confirmButton) {
        logOnly("text('确认') 未找到，尝试 textMatches(confirmButtonRegex)...");
        confirmButton = className("android.widget.Button").textMatches(confirmButtonRegex).findOne(FIND_TIMEOUT);
    }
    
    // 备用（较低优先级，因为确认按钮通常是标准Button）: LinearLayout策略
    if (!confirmButton) {
        logOnly("标准Button未找到确认按钮，尝试LinearLayout策略...");
        var clickableConfirmLayouts = className("android.widget.LinearLayout").clickable(true).find();
        for (var i = 0; i < clickableConfirmLayouts.length; i++) {
            var layout = clickableConfirmLayouts[i];
            var textElementInside = layout.findOne(className("android.widget.TextView").textMatches(confirmButtonRegex));
            if (textElementInside) { 
                confirmButton = layout;
                logOnly("找到确认按钮 (LinearLayout策略): " + textElementInside.text());
                break;
            }
        }
    }

    if (!confirmButton) {
        var errMsgConfirm = "未找到最终的" + confirmButtonDesc + "！请检查按钮文字或UI结构。";
        logOnly(errMsgConfirm);
        // 在这里也打印一下当前可见的Button，帮助调试
        var allButtonsSnapshot = className("android.widget.Button").find();
        var buttonTexts = [];
        allButtonsSnapshot.forEach(function(b){ buttonTexts.push(b.text()); });
        logOnly("当前界面可见Button文本: " + buttonTexts.join(" | "));

        sendStatusUpdate(localSignalId, "trade_execution_failed", errMsgConfirm, { step: "find_confirm_button", regex: confirmButtonRegex.source, visible_buttons: buttonTexts });
        currentTradeSignalId = null;
        return false;
    }

    logOnly("找到" + confirmButtonDesc + " (文本: '" + (confirmButton.text() || "N/A") + "')，准备点击...");
    
    if (clickElement(confirmButton, confirmButtonDesc)) {
        logOnly("成功点击" + confirmButtonDesc + "。");
        sendStatusUpdate(localSignalId, "trade_confirmed_clicked", "已点击交易确认按钮。", { confirm_button_text: confirmButton.text() });
        // 可以在这里加一个短暂的sleep，等待可能的服务器响应或UI变化
        // sleep(500); 
    } else {
        var errMsgClickConfirm = "点击" + confirmButtonDesc + "失败。";
        logOnly(errMsgClickConfirm);
        sendStatusUpdate(localSignalId, "trade_execution_failed", errMsgClickConfirm, { step: "click_confirm_button", element_desc: confirmButtonDesc });
        currentTradeSignalId = null;
        return false;
    }
    
    logOnly("自动化交易流程执行完毕。SignalID: " + localSignalId);
    sendStatusUpdate(localSignalId, "trade_execution_completed", "自动化交易流程执行完毕。", null);
    currentTradeSignalId = null; // 清理当前信号ID
    return true;
}

// -------------------- 脚本主入口 (基本保持不变) --------------------
function main() {
    auto.waitFor();

    initializeClientId();

    logOnly("AutoX.js 币安事件合约执行脚本启动 (Aggressive Mode v3 - Optimized)。");
    logOnly("客户端ID: " + CLIENT_ID);
    logOnly("支持交易对: " + SUPPORTED_SYMBOLS.join(", "));
    logOnly("警告：此脚本依赖于特定UI结构和按钮文本。请务必已正确配置按钮文本的正则表达式。");
    logOnly("请确保币安App停留在事件合约交易页面，并且金额输入框可见。");

    connectWebSocket(); // 初始连接

    var keepAliveInterval = setInterval(() => {
        // logOnly("Keep-alive check. isWebSocketConnected: " + isWebSocketConnected); // 可以按需开启
        if (!isWebSocketConnected || !webSocketConnection) { 
            logOnly("Keep-alive: 检测到WebSocket未连接或对象丢失，尝试重新连接...");
            connectWebSocket();
        } else {
             if (!isWebSocketConnected) { // 理论上不应发生
                 logOnly("Keep-alive: isWebSocketConnected为false但连接对象存在，尝试强制重连...");
                 connectWebSocket();
             }
        }
    }, 15000); // 15秒检查一次

    events.on("exit", function() {
        logOnly("脚本即将退出，尝试关闭WebSocket连接...");
        if (webSocketConnection) {
            try {
                webSocketConnection.close(1000, "Script exiting gracefully");
            } catch (e) {
                logOnly("关闭WebSocket时发生错误: " + e);
            }
        }
        if (keepAliveInterval) {
            clearInterval(keepAliveInterval);
        }
        logOnly("脚本已退出。");
    });

    // Auto.js脚本在有事件监听或定时器时通常会保持运行
    logOnly("脚本主循环已启动，等待WebSocket指令...");
}

main();