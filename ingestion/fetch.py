"""资料获取层入口——委托给 fetch_backends 多 API 编排。

保持原 API（fetch_many / FetchedPage）以兼容上层 pipeline.py。
"""
from .fetch_backends import FetchedPage, fetch_many  # noqa: F401
