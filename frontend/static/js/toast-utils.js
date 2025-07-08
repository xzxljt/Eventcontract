/**
 * 全局Toast通知工具
 * 提供统一的Toast通知功能，与Autox管理页面保持一致的样式
 */

class ToastManager {
    constructor() {
        this.toastInstance = null;
        this.init();
    }

    init() {
        // 创建Toast容器和元素
        this.createToastContainer();
        this.initBootstrapToast();
    }

    createToastContainer() {
        // 检查是否已存在Toast容器
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            // 创建Toast容器
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            toastContainer.style.zIndex = '1100';
            document.body.appendChild(toastContainer);
        }

        // 检查是否已存在Toast元素
        let toastElement = document.getElementById('globalToast');
        if (!toastElement) {
            // 创建Toast元素
            toastElement = document.createElement('div');
            toastElement.id = 'globalToast';
            toastElement.className = 'toast';
            toastElement.setAttribute('role', 'alert');
            toastElement.setAttribute('aria-live', 'assertive');
            toastElement.setAttribute('aria-atomic', 'true');
            toastElement.setAttribute('data-bs-delay', '5000');

            toastElement.innerHTML = `
                <div class="toast-header">
                    <i id="toastIcon" class="bi me-2"></i>
                    <strong id="toastTitle" class="me-auto"></strong>
                    <small id="toastTime" class="text-body-secondary ms-2"></small>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div id="toastBody" class="toast-body"></div>
            `;

            toastContainer.appendChild(toastElement);
        }
    }

    initBootstrapToast() {
        const toastEl = document.getElementById('globalToast');
        if (toastEl && typeof bootstrap !== 'undefined') {
            this.toastInstance = new bootstrap.Toast(toastEl, { delay: 5000 });
        }
    }

    /**
     * 显示Toast通知
     * @param {string} title - 通知标题
     * @param {string} message - 通知消息（支持HTML）
     * @param {string} type - 通知类型：'success', 'danger', 'warning', 'info'
     * @param {number} delay - 显示时长（毫秒），默认5000
     */
    show(title, message, type = 'success', delay = 5000) {
        // 确保Toast元素存在
        if (!document.getElementById('globalToast')) {
            this.createToastContainer();
            this.initBootstrapToast();
        }

        // 更新Toast内容
        const titleEl = document.getElementById('toastTitle');
        const messageEl = document.getElementById('toastBody');
        const timeEl = document.getElementById('toastTime');
        const iconEl = document.getElementById('toastIcon');

        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.innerHTML = message;
        if (timeEl) {
            timeEl.textContent = new Date().toLocaleTimeString('zh-CN', { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit' 
            });
        }

        // 设置图标和样式
        if (iconEl) {
            iconEl.className = this.getIconClass(type);
        }

        // 更新延迟时间
        const toastEl = document.getElementById('globalToast');
        if (toastEl) {
            toastEl.setAttribute('data-bs-delay', delay.toString());
        }

        // 重新初始化Toast实例（如果延迟时间改变）
        if (this.toastInstance) {
            this.toastInstance.dispose();
        }
        this.initBootstrapToast();

        // 显示Toast
        if (this.toastInstance) {
            this.toastInstance.show();
        }
    }

    /**
     * 根据类型获取对应的图标类名
     * @param {string} type - 通知类型
     * @returns {string} 图标类名
     */
    getIconClass(type) {
        const iconMap = {
            'success': 'bi-check-circle-fill text-success',
            'danger': 'bi-x-octagon-fill text-danger',
            'warning': 'bi-exclamation-triangle-fill text-warning',
            'info': 'bi-info-circle-fill text-info'
        };
        return `bi me-2 ${iconMap[type] || iconMap['info']}`;
    }

    /**
     * 隐藏Toast
     */
    hide() {
        if (this.toastInstance) {
            this.toastInstance.hide();
        }
    }

    /**
     * 销毁Toast实例
     */
    dispose() {
        if (this.toastInstance) {
            this.toastInstance.dispose();
            this.toastInstance = null;
        }
    }
}

// 创建全局Toast管理器实例
window.globalToast = new ToastManager();

// 提供便捷的全局函数
window.showToast = function(title, message, type = 'success', delay = 5000) {
    window.globalToast.show(title, message, type, delay);
};

// 提供特定类型的便捷函数
window.showSuccessToast = function(title, message, delay = 5000) {
    window.globalToast.show(title, message, 'success', delay);
};

window.showErrorToast = function(title, message, delay = 8000) {
    window.globalToast.show(title, message, 'danger', delay);
};

window.showWarningToast = function(title, message, delay = 6000) {
    window.globalToast.show(title, message, 'warning', delay);
};

window.showInfoToast = function(title, message, delay = 5000) {
    window.globalToast.show(title, message, 'info', delay);
};
