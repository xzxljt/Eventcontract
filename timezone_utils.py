# timezone_utils.py
# 时区处理工具，用于统一处理UTC和中国时区(UTC+8)的转换

from datetime import datetime, timedelta, timezone

# 定义中国时区(UTC+8)
CHINA_TIMEZONE = timezone(timedelta(hours=8))

def to_china_timezone(dt):
    """
    将UTC时间转换为中国时区(UTC+8)时间
    
    参数:
        dt: datetime对象或ISO格式字符串
    返回:
        中国时区的datetime对象
    """
    if isinstance(dt, str):
        # 处理ISO格式字符串，移除Z后缀
        dt = dt.replace('Z', '+00:00')
        dt = datetime.fromisoformat(dt)
    
    # 如果dt没有时区信息，假定为UTC时间
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        
    # 转换到中国时区
    return dt.astimezone(CHINA_TIMEZONE)

def to_utc(dt):
    """
    将中国时区时间转换为UTC时间
    
    参数:
        dt: datetime对象或ISO格式字符串
    返回:
        UTC时区的datetime对象
    """
    if isinstance(dt, str):
        # 尝试解析字符串为datetime对象
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            # 如果不是ISO格式，尝试其他格式
            try:
                dt = datetime.strptime(dt, '%Y-%m-%dT%H:%M')
            except ValueError:
                # 前端datetime-local输入可能没有秒
                dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    
    # 如果dt没有时区信息，假定为中国时区时间
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CHINA_TIMEZONE)
        
    # 转换到UTC时区
    return dt.astimezone(timezone.utc)

def now_china():
    """
    获取当前中国时区时间
    
    返回:
        当前的中国时区datetime对象
    """
    return datetime.now(CHINA_TIMEZONE)

def now_utc():
    """
    获取当前UTC时间
    
    返回:
        当前的UTC时区datetime对象
    """
    return datetime.now(timezone.utc)

def format_for_display(dt):
    """
    将datetime对象格式化为适合前端显示的ISO格式字符串(中国时区)
    
    参数:
        dt: datetime对象或ISO格式字符串
    返回:
        ISO格式的字符串，带有时区信息
    """
    if isinstance(dt, str):
        dt = dt.replace('Z', '+00:00')
        dt = datetime.fromisoformat(dt)
    
    # 确保dt有时区信息
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # 转换到中国时区
    china_time = dt.astimezone(CHINA_TIMEZONE)
    
    # 返回ISO格式字符串，包含时区信息
    return china_time.isoformat()

def parse_frontend_datetime(dt_str):
    """
    解析前端传入的日期时间字符串
    
    参数:
        dt_str: 前端传入的日期时间字符串
    返回:
        UTC时区的datetime对象
    """
    try:
        # 尝试解析ISO格式
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            # 尝试解析datetime-local输入格式 (YYYY-MM-DDTHH:MM)
            dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            # 尝试解析其他可能的格式
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    
    # 如果没有时区信息，假定为中国时区
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CHINA_TIMEZONE)
    
    # 转换为UTC并返回
    return dt.astimezone(timezone.utc)