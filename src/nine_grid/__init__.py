"""九宫格检索增强规划，接口版本 1.0。"""
import os

# Windows 下本地特征与数学库可能加载不同 OpenMP 实现。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
VERSION = "1.0"
