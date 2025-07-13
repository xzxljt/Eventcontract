"""
策略优化记录数据库管理模块
提供优化记录的持久化存储、查询和管理功能
"""

import sqlite3
import json
import logging
import asyncio
import aiosqlite
import os
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class OptimizationRecord:
    """优化记录数据类"""
    id: str
    symbol: str
    interval: str
    strategy_id: str
    strategy_name: str
    start_date: str
    end_date: str
    status: str  # running, completed, error, stopped
    progress: Dict[str, Any]
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

class OptimizationDatabase:
    """优化记录数据库管理器"""
    
    def __init__(self, db_path: str = None):
        """初始化数据库管理器"""
        if db_path is None:
            # 使用项目根目录下的data文件夹，确保使用绝对路径
            project_root = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(project_root, "data")
            db_path = os.path.join(data_dir, "optimization_records.db")

            # 确保使用绝对路径
            db_path = os.path.abspath(db_path)

        self.db_path = db_path
        self.max_records = 7  # 最多保留7条记录

        # 确保数据目录存在
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)

        # 记录数据库路径用于调试
        logger.info(f"优化记录数据库路径: {self.db_path}")
        
    async def initialize(self):
        """初始化数据库表"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS optimization_records (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress TEXT NOT NULL,
                    config TEXT NOT NULL,
                    results TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            
            # 创建索引
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON optimization_records(created_at DESC)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON optimization_records(status)
            """)
            
            await db.commit()
            logger.info("优化记录数据库初始化完成")
    
    async def save_record(self, record: OptimizationRecord) -> bool:
        """保存优化记录"""
        try:
            now = datetime.now().isoformat()
            if not record.created_at:
                record.created_at = now
            record.updated_at = now
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO optimization_records 
                    (id, symbol, interval, strategy_id, strategy_name, start_date, end_date,
                     status, progress, config, results, error_message, created_at, updated_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.symbol,
                    record.interval,
                    record.strategy_id,
                    record.strategy_name,
                    record.start_date,
                    record.end_date,
                    record.status,
                    json.dumps(record.progress, ensure_ascii=False),
                    json.dumps(record.config, ensure_ascii=False),
                    json.dumps(record.results, ensure_ascii=False) if record.results else None,
                    record.error_message,
                    record.created_at,
                    record.updated_at,
                    record.completed_at
                ))
                await db.commit()
                
            # 清理旧记录
            await self._cleanup_old_records()
            
            logger.info(f"优化记录已保存: {record.id}")
            return True
            
        except Exception as e:
            logger.error(f"保存优化记录失败: {e}")
            return False
    
    async def update_record_status(self, record_id: str, status: str, 
                                 progress: Optional[Dict[str, Any]] = None,
                                 results: Optional[Dict[str, Any]] = None,
                                 error_message: Optional[str] = None) -> bool:
        """更新记录状态"""
        try:
            now = datetime.now().isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                # 构建更新语句
                update_fields = ["status = ?", "updated_at = ?"]
                params = [status, now]
                
                if progress is not None:
                    update_fields.append("progress = ?")
                    params.append(json.dumps(progress, ensure_ascii=False))
                
                if results is not None:
                    update_fields.append("results = ?")
                    params.append(json.dumps(results, ensure_ascii=False))
                
                if error_message is not None:
                    update_fields.append("error_message = ?")
                    params.append(error_message)
                
                if status in ['completed', 'error', 'stopped']:
                    update_fields.append("completed_at = ?")
                    params.append(now)
                
                params.append(record_id)
                
                query = f"""
                    UPDATE optimization_records 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                """
                
                await db.execute(query, params)
                await db.commit()
                
            logger.info(f"记录状态已更新: {record_id} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"更新记录状态失败: {e}")
            return False
    
    async def get_record(self, record_id: str) -> Optional[OptimizationRecord]:
        """获取单个记录"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM optimization_records WHERE id = ?
                """, (record_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                if row:
                    return self._row_to_record(row)
                return None
                
        except Exception as e:
            logger.error(f"获取记录失败: {e}")
            return None
    
    async def get_all_records(self, limit: int = 10) -> List[OptimizationRecord]:
        """获取所有记录，按创建时间倒序"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM optimization_records 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    
                return [self._row_to_record(row) for row in rows]
                
        except Exception as e:
            logger.error(f"获取记录列表失败: {e}")
            return []
    
    async def get_running_record(self) -> Optional[OptimizationRecord]:
        """获取正在运行的记录"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM optimization_records 
                    WHERE status = 'running'
                    ORDER BY created_at DESC 
                    LIMIT 1
                """) as cursor:
                    row = await cursor.fetchone()
                    
                if row:
                    return self._row_to_record(row)
                return None
                
        except Exception as e:
            logger.error(f"获取运行中记录失败: {e}")
            return None
    
    async def delete_record(self, record_id: str) -> bool:
        """删除记录"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    DELETE FROM optimization_records WHERE id = ?
                """, (record_id,))
                await db.commit()
                
            logger.info(f"记录已删除: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除记录失败: {e}")
            return False
    
    async def _cleanup_old_records(self):
        """清理旧记录，保留最近的记录"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 获取需要删除的记录ID
                async with db.execute("""
                    SELECT id FROM optimization_records 
                    ORDER BY created_at DESC 
                    LIMIT -1 OFFSET ?
                """, (self.max_records,)) as cursor:
                    old_records = await cursor.fetchall()
                
                # 删除旧记录
                for record in old_records:
                    await db.execute("""
                        DELETE FROM optimization_records WHERE id = ?
                    """, (record[0],))
                
                await db.commit()
                
                if old_records:
                    logger.info(f"已清理 {len(old_records)} 条旧记录")
                    
        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")

    async def recover_interrupted_tasks(self):
        """恢复服务器重启后中断的任务状态"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 查找所有状态为running的记录
                async with db.execute("""
                    SELECT id, symbol, strategy_name FROM optimization_records
                    WHERE status = 'running'
                """) as cursor:
                    running_records = await cursor.fetchall()

                if running_records:
                    logger.info(f"发现 {len(running_records)} 个中断的优化任务，正在恢复状态...")

                    # 将所有running状态的记录更新为stopped
                    current_time = datetime.now().isoformat()
                    await db.execute("""
                        UPDATE optimization_records
                        SET status = 'stopped',
                            updated_at = ?,
                            completed_at = ?
                        WHERE status = 'running'
                    """, (current_time, current_time))
                    await db.commit()

                    # 记录恢复的任务
                    for record_id, symbol, strategy_name in running_records:
                        logger.info(f"已恢复中断任务: {record_id[:8]}... ({symbol} - {strategy_name})")

                    logger.info("所有中断任务状态已恢复")
                else:
                    logger.info("未发现中断的优化任务")

        except Exception as e:
            logger.error(f"恢复中断任务状态失败: {e}")

    async def create_backup(self) -> bool:
        """创建数据库备份"""
        try:
            backup_dir = Path(self.db_path).parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            # 生成备份文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"optimization_records_backup_{timestamp}.db"

            # 复制数据库文件
            shutil.copy2(self.db_path, backup_path)

            # 清理旧备份（保留最近5个）
            await self._cleanup_old_backups(backup_dir)

            logger.info(f"数据库备份已创建: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"创建数据库备份失败: {e}")
            return False

    async def _cleanup_old_backups(self, backup_dir: Path):
        """清理旧备份文件"""
        try:
            backup_files = list(backup_dir.glob("optimization_records_backup_*.db"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # 保留最近5个备份
            for old_backup in backup_files[5:]:
                old_backup.unlink()
                logger.info(f"已删除旧备份: {old_backup}")

        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")

    async def restore_from_backup(self, backup_path: str) -> bool:
        """从备份恢复数据库"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"备份文件不存在: {backup_path}")
                return False

            # 创建当前数据库的备份
            current_backup = f"{self.db_path}.before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.db_path, current_backup)

            # 恢复备份
            shutil.copy2(backup_path, self.db_path)

            logger.info(f"数据库已从备份恢复: {backup_path}")
            logger.info(f"原数据库已备份为: {current_backup}")
            return True

        except Exception as e:
            logger.error(f"从备份恢复数据库失败: {e}")
            return False

    async def get_backup_list(self) -> List[Dict[str, Any]]:
        """获取备份文件列表"""
        try:
            backup_dir = Path(self.db_path).parent / "backups"
            if not backup_dir.exists():
                return []

            backup_files = list(backup_dir.glob("optimization_records_backup_*.db"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            backups = []
            for backup_file in backup_files:
                stat = backup_file.stat()
                backups.append({
                    'path': str(backup_file),
                    'name': backup_file.name,
                    'size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

            return backups

        except Exception as e:
            logger.error(f"获取备份列表失败: {e}")
            return []

    def _row_to_record(self, row) -> OptimizationRecord:
        """将数据库行转换为记录对象"""
        return OptimizationRecord(
            id=row[0],
            symbol=row[1],
            interval=row[2],
            strategy_id=row[3],
            strategy_name=row[4],
            start_date=row[5],
            end_date=row[6],
            status=row[7],
            progress=json.loads(row[8]) if row[8] else {},
            config=json.loads(row[9]) if row[9] else {},
            results=json.loads(row[10]) if row[10] else None,
            error_message=row[11],
            created_at=row[12],
            updated_at=row[13],
            completed_at=row[14]
        )

# 全局数据库实例
_db_instance: Optional[OptimizationDatabase] = None

async def get_optimization_db() -> OptimizationDatabase:
    """获取数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = OptimizationDatabase()
        await _db_instance.initialize()
        # 服务器启动时恢复中断的任务状态
        await _db_instance.recover_interrupted_tasks()
        # 创建启动备份
        await _db_instance.create_backup()
    return _db_instance
