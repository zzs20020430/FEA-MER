import os
import importlib

# # 获取当前目录
# package_dir = os.path.dirname(__file__)

# # 遍历目录中的所有文件
# for filename in os.listdir(package_dir):
#     # 检查是否为 .py 文件且不是 __init__.py
#     if filename.endswith('.py') and filename != '__init__.py':
#         # 获取模块名（去掉 .py 后缀）
#         module_name = filename[:-3]
#         # 动态导入模块
#         module = importlib.import_module(f'.{module_name}', package=__name__)
#         # 将模块添加到当前包的命名空间中
#         globals()[module_name] = module