import os
import sys
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# 设置环境变量
def setup_environment():
    # 获取可执行文件所在目录
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    # 添加必要的环境变量
    os.environ['PATH'] = application_path + os.pathsep + os.environ.get('PATH', '')
    
    # 设置工作目录
    os.chdir(application_path)

# 运行时钩子入口
def pre_init():
    setup_environment() 