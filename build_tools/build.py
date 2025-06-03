import os
import sys
import platform
import PyInstaller.__main__
import shutil
import brainflow

def check_required_files():
    """检查必要的文件和目录是否存在"""
    required_paths = [
        'src/main.py',
        'src/EEGThread.py',  # 添加EEG线程文件
        'resources/images/icon.png',
        'attention_classifier/model.py',
    ]
    
    missing_files = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("错误：以下必要文件不存在：")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True

def prepare_build_environment():
    """准备构建环境"""
    # 清理旧的构建文件
    build_dirs = ['build_tools/build', 'build_tools/dist']
    for dir_path in build_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # 创建资源目录
    resource_dirs = [
        'build_tools/build/resources/images',
        'build_tools/build/resources/videos',
        'build_tools/build/attention_classifier',
    ]
    for dir_path in resource_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 复制资源文件
    try:
        # 复制resources目录
        if os.path.exists('resources'):
            shutil.copytree('resources', 'build_tools/build/resources', dirs_exist_ok=True)
        
        # 复制attention_classifier目录
        if os.path.exists('attention_classifier'):
            shutil.copytree('attention_classifier', 'build_tools/build/attention_classifier', dirs_exist_ok=True)
        
        return True
    except Exception as e:
        print(f"错误：准备构建环境时出错：{str(e)}")
        return False

def build():
    """执行构建过程"""
    # 获取当前操作系统
    system = platform.system()
    
    # 检查必要文件
    if not check_required_files():
        print("构建终止：缺少必要文件")
        return False
    
    # 准备构建环境
    if not prepare_build_environment():
        print("构建终止：环境准备失败")
        return False
    
    print("开始构建...")
    
    # 基本的打包参数
    args = [
        'src/main.py',  # 主程序文件
        '--name=BrainView',  # 可执行文件名称
        '--windowed',  # 不显示控制台窗口
        '--noconfirm',  # 覆盖输出目录
        '--clean',  # 清理临时文件
        
        # 添加图标
        '--icon=resources/images/icon.png',
        
        # 添加数据文件
        '--add-data=resources:resources',  # 所有资源文件
        '--add-data=attention_classifier:attention_classifier',  # 注意力分类器模型
        
        # 添加 brainflow 动态库
        f'--add-binary={os.path.dirname(brainflow.__file__)}/lib/*:brainflow/lib',
        
        # 指定工作目录
        '--workpath=build_tools/build',
        '--distpath=build_tools/dist',
        '--specpath=build_tools/build',
        
        # 单文件模式
        '--onefile',
        
        # 添加必要的隐式导入
        '--hidden-import=numpy',
        '--hidden-import=scipy',
        '--hidden-import=sklearn',
        '--hidden-import=pandas',
        '--hidden-import=pyqtgraph',
        '--hidden-import=brainflow',
        '--hidden-import=nibabel',
        '--hidden-import=torch',
        '--hidden-import=PIL',
        '--hidden-import=serial',  # 添加串口通信库
        '--hidden-import=serial.tools.list_ports',  # 添加串口工具
        '--hidden-import=logging',  # 添加日志模块
        '--hidden-import=src.EEGThread',  # 添加 EEGThread 模块
    ]
    
    # 根据操作系统添加特定参数
    if system == 'Darwin':  # macOS
        # 检测是否为Apple Silicon
        is_arm = platform.machine() == 'arm64'
        args.extend([
            f'--target-arch={"arm64" if is_arm else "x86_64"}',  # 根据实际架构选择
            '--codesign-identity=',  # 跳过代码签名
            '--osx-bundle-identifier=com.imoonlab.brainview'
        ])
        print(f"正在为 {'Apple Silicon' if is_arm else 'Intel'} Mac 构建...")
    elif system == 'Windows':
        args.extend([
            '--runtime-hook=build_tools/windows_runtime_hook.py'
        ])
    
    try:
        # 运行PyInstaller
        PyInstaller.__main__.run(args)
        print("构建完成！")
        
        # 如果是 macOS，创建 DMG 安装包
        # if system == 'Darwin':
        #     print("正在创建 DMG 安装包...")
        #     try:
        #         import dmgbuild
        #         os.chdir('build_tools')
        #         dmgbuild.build_dmg(
        #             filename='dist/BrainView-Installer.dmg',
        #             volume_name='BrainView 安装包',
        #             settings_file='dmg_settings.py'
        #         )
        #         print("DMG 安装包创建成功！")
        #     except Exception as e:
        #         print(f"创建 DMG 安装包失败：{str(e)}")
        #         return False
        #     finally:
        #         os.chdir('..')
        
        return True
    except Exception as e:
        print(f"构建失败：{str(e)}")
        return False

if __name__ == '__main__':
    # 确保在项目根目录下运行
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 执行构建
    success = build()
    sys.exit(0 if success else 1) 