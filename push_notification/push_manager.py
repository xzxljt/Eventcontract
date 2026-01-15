import json
import os
import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
import requests

logger = logging.getLogger(__name__)

class PushChannel(ABC):
    """推送渠道基类"""
    
    @abstractmethod
    def send(self, message: str, **kwargs) -> Dict[str, Any]:
        """发送推送消息
        
        Args:
            message: 推送消息内容
            **kwargs: 额外参数
            
        Returns:
            Dict: 推送结果，包含status和message字段
        """
        pass
    
    @abstractmethod
    def test(self) -> Dict[str, Any]:
        """测试推送渠道
        
        Returns:
            Dict: 测试结果，包含status和message字段
        """
        pass

class TelegramChannel(PushChannel):
    """Telegram推送渠道"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Telegram推送渠道
        
        Args:
            config: 配置信息，包含bot_token和chat_id
        """
        self.bot_token = config.get('bot_token', '')
        self.chat_id = config.get('chat_id', '')
    
    def send(self, message: str, **kwargs) -> Dict[str, Any]:
        """发送Telegram消息
        
        Args:
            message: 消息内容
            **kwargs: 额外参数
            
        Returns:
            Dict: 推送结果
        """
        if not self.bot_token or not self.chat_id:
            return {
                "status": "error",
                "message": "Telegram配置不完整"
            }
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": kwargs.get('parse_mode', 'Markdown')
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            return {
                "status": "success",
                "message": "Telegram消息发送成功"
            }
        except Exception as e:
            logger.error(f"Telegram消息发送失败: {e}")
            return {
                "status": "error",
                "message": f"Telegram消息发送失败: {str(e)}"
            }
    
    def test(self) -> Dict[str, Any]:
        """测试Telegram推送渠道
        
        Returns:
            Dict: 测试结果
        """
        return self.send("这是一条测试消息，用于验证Telegram推送功能是否正常。")

class WechatChannel(PushChannel):
    """企业微信推送渠道（通过Webhook）"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化企业微信推送渠道
        
        Args:
            config: 配置信息，包含webhook_url
        """
        self.webhook_url = config.get('webhook_url', '')
    
    def send(self, message: str, **kwargs) -> Dict[str, Any]:
        """发送企业微信消息
        
        Args:
            message: 消息内容
            **kwargs: 额外参数
            
        Returns:
            Dict: 推送结果
        """
        if not self.webhook_url:
            return {
                "status": "error",
                "message": "企业微信Webhook URL未配置"
            }
        
        try:
            payload = {
                "msgtype": "text",
                "text": {
                    "content": message
                }
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('errcode') == 0:
                return {
                    "status": "success",
                    "message": "企业微信消息发送成功"
                }
            else:
                return {
                    "status": "error",
                    "message": f"企业微信消息发送失败: {result.get('errmsg', '未知错误')}"
                }
        except Exception as e:
            logger.error(f"企业微信消息发送失败: {e}")
            return {
                "status": "error",
                "message": f"企业微信消息发送失败: {str(e)}"
            }
    
    def test(self) -> Dict[str, Any]:
        """测试企业微信推送渠道
        
        Returns:
            Dict: 测试结果
        """
        return self.send("这是一条测试消息，用于验证企业微信推送功能是否正常。")

class PushManager:
    """推送管理器"""
    
    def __init__(self, config_path: str = 'push_notification/config.json'):
        """初始化推送管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.channels = self._init_channels()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            Dict: 配置信息
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {
                "channels": {
                    "telegram": [],
                    "wechat": []
                }
            }
        except Exception as e:
            logger.error(f"加载推送配置失败: {e}")
            return {
                "channels": {
                    "telegram": [],
                    "wechat": []
                }
            }
    
    def _save_config(self):
        """保存配置文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存推送配置失败: {e}")
    
    def _init_channels(self) -> Dict[str, List[PushChannel]]:
        """初始化推送渠道
        
        Returns:
            Dict: 推送渠道列表
        """
        channels = {
            "telegram": [],
            "wechat": []
        }
        
        # 初始化Telegram渠道
        for tg_config in self.config.get('channels', {}).get('telegram', []):
            try:
                channel = TelegramChannel(tg_config)
                channels["telegram"].append(channel)
            except Exception as e:
                logger.error(f"初始化Telegram渠道失败: {e}")
        
        # 初始化企业微信渠道
        for wc_config in self.config.get('channels', {}).get('wechat', []):
            try:
                channel = WechatChannel(wc_config)
                channels["wechat"].append(channel)
            except Exception as e:
                logger.error(f"初始化企业微信渠道失败: {e}")
        
        return channels
    
    def send_all(self, message: str, **kwargs) -> List[Dict[str, Any]]:
        """向所有配置的渠道发送消息
        
        Args:
            message: 消息内容
            **kwargs: 额外参数
            
        Returns:
            List[Dict]: 所有渠道的推送结果
        """
        results = []
        
        # 向所有Telegram渠道发送
        for i, channel in enumerate(self.channels.get("telegram", [])):
            result = channel.send(message, **kwargs)
            result["channel"] = "telegram"
            result["index"] = i
            results.append(result)
        
        # 向所有企业微信渠道发送
        for i, channel in enumerate(self.channels.get("wechat", [])):
            result = channel.send(message, **kwargs)
            result["channel"] = "wechat"
            result["index"] = i
            results.append(result)
        
        return results
    
    def send_to_channel(self, channel_type: str, channel_index: int, message: str, **kwargs) -> Dict[str, Any]:
        """向指定渠道发送消息
        
        Args:
            channel_type: 渠道类型，telegram或wechat
            channel_index: 渠道索引
            message: 消息内容
            **kwargs: 额外参数
            
        Returns:
            Dict: 推送结果
        """
        channels = self.channels.get(channel_type, [])
        if 0 <= channel_index < len(channels):
            result = channels[channel_index].send(message, **kwargs)
            result["channel"] = channel_type
            result["index"] = channel_index
            return result
        else:
            return {
                "status": "error",
                "message": f"渠道不存在: {channel_type}[{channel_index}]",
                "channel": channel_type,
                "index": channel_index
            }
    
    def test_channel(self, channel_type: str, channel_index: int) -> Dict[str, Any]:
        """测试指定渠道
        
        Args:
            channel_type: 渠道类型，telegram或wechat
            channel_index: 渠道索引
            
        Returns:
            Dict: 测试结果
        """
        channels = self.channels.get(channel_type, [])
        if 0 <= channel_index < len(channels):
            result = channels[channel_index].test()
            result["channel"] = channel_type
            result["index"] = channel_index
            return result
        else:
            return {
                "status": "error",
                "message": f"渠道不存在: {channel_type}[{channel_index}]",
                "channel": channel_type,
                "index": channel_index
            }
    
    def add_channel(self, channel_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """添加推送渠道
        
        Args:
            channel_type: 渠道类型，telegram或wechat
            config: 渠道配置
            
        Returns:
            Dict: 添加结果
        """
        try:
            if channel_type not in ["telegram", "wechat"]:
                return {
                    "status": "error",
                    "message": f"不支持的渠道类型: {channel_type}"
                }
            
            # 添加到配置
            if "channels" not in self.config:
                self.config["channels"] = {
                    "telegram": [],
                    "wechat": []
                }
            
            if channel_type not in self.config["channels"]:
                self.config["channels"][channel_type] = []
            
            self.config["channels"][channel_type].append(config)
            self._save_config()
            
            # 重新初始化渠道
            self.channels = self._init_channels()
            
            return {
                "status": "success",
                "message": f"{channel_type}渠道添加成功"
            }
        except Exception as e:
            logger.error(f"添加渠道失败: {e}")
            return {
                "status": "error",
                "message": f"添加渠道失败: {str(e)}"
            }
    
    def update_channel(self, channel_type: str, channel_index: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """更新推送渠道配置
        
        Args:
            channel_type: 渠道类型，telegram或wechat
            channel_index: 渠道索引
            config: 新的渠道配置
            
        Returns:
            Dict: 更新结果
        """
        try:
            channels = self.config.get("channels", {}).get(channel_type, [])
            if 0 <= channel_index < len(channels):
                channels[channel_index] = config
                self._save_config()
                
                # 重新初始化渠道
                self.channels = self._init_channels()
                
                return {
                    "status": "success",
                    "message": f"{channel_type}渠道更新成功"
                }
            else:
                return {
                    "status": "error",
                    "message": f"渠道不存在: {channel_type}[{channel_index}]"
                }
        except Exception as e:
            logger.error(f"更新渠道失败: {e}")
            return {
                "status": "error",
                "message": f"更新渠道失败: {str(e)}"
            }
    
    def delete_channel(self, channel_type: str, channel_index: int) -> Dict[str, Any]:
        """删除推送渠道
        
        Args:
            channel_type: 渠道类型，telegram或wechat
            channel_index: 渠道索引
            
        Returns:
            Dict: 删除结果
        """
        try:
            channels = self.config.get("channels", {}).get(channel_type, [])
            if 0 <= channel_index < len(channels):
                channels.pop(channel_index)
                self._save_config()
                
                # 重新初始化渠道
                self.channels = self._init_channels()
                
                return {
                    "status": "success",
                    "message": f"{channel_type}渠道删除成功"
                }
            else:
                return {
                    "status": "error",
                    "message": f"渠道不存在: {channel_type}[{channel_index}]"
                }
        except Exception as e:
            logger.error(f"删除渠道失败: {e}")
            return {
                "status": "error",
                "message": f"删除渠道失败: {str(e)}"
            }
    
    def get_config(self) -> Dict[str, Any]:
        """获取推送配置
        
        Returns:
            Dict: 推送配置
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置推送配置
        
        Args:
            config: 新的推送配置
            
        Returns:
            Dict: 设置结果
        """
        try:
            self.config = config
            self._save_config()
            self.channels = self._init_channels()
            return {
                "status": "success",
                "message": "推送配置更新成功"
            }
        except Exception as e:
            logger.error(f"设置配置失败: {e}")
            return {
                "status": "error",
                "message": f"设置配置失败: {str(e)}"
            }
