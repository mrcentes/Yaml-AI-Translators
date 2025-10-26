# yaml_translator_pro.py - YAML批量AI本地化工具 v1.0

import os
import sys
import json
import threading
import time
import shutil
import requests
import subprocess
import webbrowser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Menu
import queue

# ==================== 修复 Windows DPI 模糊问题 ====================
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ==================== 检测拖拽支持 ====================
HAS_DND = False
DND_INSTALL_AVAILABLE = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    # 检测是否可以安装
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        DND_INSTALL_AVAILABLE = True
    except:
        pass

VERSION = "1.0"
APP_TITLE = f"YAML批量AI本地化工具 v{VERSION}"


# ==================== 核心翻译器 ====================
class DeepSeekTranslator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.lock = threading.Lock()
        
    def clean_translated_text(self, text):
        """智能清理翻译后的文本"""
        double_quote_count = text.count('"')
        single_quote_count = text.count("'")
        
        if double_quote_count >= 2 or single_quote_count >= 2:
            quote_positions = []
            for i, char in enumerate(text):
                if char in ['"', "'"]:
                    quote_positions.append(i)
            
            if len(quote_positions) >= 2:
                second_quote_pos = quote_positions[1]
                before = text[:second_quote_pos + 1]
                after = text[second_quote_pos + 1:]
                after_cleaned = after.replace('"', '').replace("'", '').replace(':', '：')
                return before + after_cleaned
        
        return text
    
    def escape_yaml_value(self, text):
        """转义YAML特殊字符"""
        special_chars = [':', '{', '}', '[', ']', ',', '&', '*', '#', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`']
        
        if any(char in text for char in special_chars) or '"' in text or "'" in text:
            escaped = text.replace("'", "''")
            return f"'{escaped}'"
        return text
        
    def translate(self, text, context_info=None, timeout=30):
        """使用DeepSeek API翻译文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        base_prompt = """请将以下英文翻译为中文,如果已经为中文则不翻译。

重要规则：
1. 只返回翻译结果，不要包含其他内容
2. 翻译结果中尽量避免使用双引号和单引号
3. 如果必须使用引号，用中文引号「」『』代替
4. 避免在翻译结果中使用英文冒号:，使用中文冒号：代替"""

        if context_info:
            context_parts = []
            if context_info.get('name'):
                context_parts.append(f"对象名称: {context_info['name']}")
            if context_info.get('description'):
                context_parts.append(f"对象描述: {context_info['description']}")

            if context_parts:
                context_str = "\n".join(context_parts)
                prompt = f"{base_prompt}\n\n上下文信息：\n{context_str}\n\n待翻译文本：{text}"
            else:
                prompt = f"{base_prompt}\n\n待翻译文本：{text}"
        else:
            prompt = f"{base_prompt}\n\n待翻译文本：{text}"

        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            with self.lock:
                time.sleep(0.1)

            response = requests.post(self.base_url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()

            translated_text = result['choices'][0]['message']['content'].strip()
            translated_text = self.clean_translated_text(translated_text)
            
            return translated_text, None

        except Exception as e:
            return text, str(e)


class YamlTranslatorCore:
    def __init__(self, api_key, max_threads=4, progress_callback=None, log_callback=None, config=None):
        self.translator = DeepSeekTranslator(api_key)
        self.max_threads = max_threads
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.config = config or {}
        self.stop_flag = False
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'start_time': None,
            'end_time': None
        }
        
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        if self.log_callback:
            self.log_callback(formatted_msg)
        print(formatted_msg)
    
    def update_progress(self, current, total, status=""):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(current, total, status)
    
    def find_yaml_files(self, path):
        """查找YAML文件"""
        yaml_files = []
        if os.path.isfile(path):
            if path.lower().endswith(('.yml', '.yaml')):
                yaml_files.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.yml', '.yaml')):
                        yaml_files.append(os.path.join(root, file))
        return yaml_files
    
    def backup_file(self, file_path):
        """备份文件"""
        if not self.config.get('auto_backup', True):
            return True
            
        backup_path = file_path + '.backup'
        try:
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            self.log(f"备份失败 {file_path}: {e}", "WARNING")
            return False
    
    def contains_chinese(self, text):
        """检查是否包含中文"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def find_context_value(self, lines, current_index, field_name, search_direction, search_range=3):
        """查找上下文值"""
        if search_direction == "down":
            start = current_index + 1
            end = min(len(lines), current_index + 1 + search_range)
            search_lines = lines[start:end]
        else:
            start = max(0, current_index - search_range)
            end = current_index
            search_lines = lines[start:end]
        
        for line in search_lines:
            stripped = line.lstrip()
            if stripped.startswith(f"{field_name}:"):
                _, value = stripped.split(":", 1)
                value = value.strip().strip('"').strip("'")
                return value
        return None
    
    def process_yaml_file(self, file_path):
        """处理单个YAML文件"""
        if self.stop_flag:
            return
        
        self.log(f"处理文件: {os.path.basename(file_path)}")
        
        try:
            self.backup_file(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            translated_lines = []
            file_translations = 0
            max_retries = self.config.get('max_retries', 3)
            retry_delay = self.config.get('retry_delay', 5)
            timeout = self.config.get('api_timeout', 30)
            
            for i, line in enumerate(lines):
                if self.stop_flag:
                    self.log("用户停止翻译", "WARNING")
                    return
                
                stripped_line = line.lstrip()
                leading_spaces = len(line) - len(stripped_line)

                if stripped_line.startswith("name:") or stripped_line.startswith("description:"):
                    key, value = stripped_line.split(":", 1)
                    
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    if self.config.get('skip_chinese', True) and self.contains_chinese(value):
                        translated_lines.append(line)
                        continue
                    
                    context_info = {}
                    
                    if key == "name":
                        description_value = self.find_context_value(lines, i, "description", "down")
                        if description_value:
                            context_info['description'] = description_value
                    elif key == "description":
                        name_value = self.find_context_value(lines, i, "name", "up")
                        if name_value:
                            context_info['name'] = name_value
                    
                    # 重试逻辑
                    translated_value = None
                    error = None
                    
                    for attempt in range(max_retries):
                        if self.stop_flag:
                            break
                            
                        translated_value, error = self.translator.translate(
                            value, context_info if context_info else None, timeout=timeout
                        )
                        
                        if not error:
                            break
                        
                        if attempt < max_retries - 1:
                            self.log(f"翻译失败，{retry_delay}秒后重试 ({attempt + 1}/{max_retries})...", "WARNING")
                            time.sleep(retry_delay)
                    
                    if error:
                        self.log(f"翻译失败: {value[:30]}... - {error}", "ERROR")
                        self.stats['failed_translations'] += 1
                        translated_lines.append(line)
                    else:
                        if translated_value != value:
                            file_translations += 1
                        
                        escaped_value = self.translator.escape_yaml_value(translated_value)
                        
                        if escaped_value.startswith("'") or escaped_value.startswith('"'):
                            translated_line = f"{' ' * leading_spaces}{key}: {escaped_value}\n"
                        else:
                            translated_line = f"{' ' * leading_spaces}{key}: {escaped_value}\n"
                        
                        translated_lines.append(translated_line)
                        self.stats['total_translations'] += 1
                        self.stats['successful_translations'] += 1
                else:
                    translated_lines.append(line)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(translated_lines)

            self.stats['processed_files'] += 1
            self.log(f"✓ 完成: {os.path.basename(file_path)} (翻译 {file_translations} 项)", "SUCCESS")
            
            # 自动删除备份
            if self.config.get('auto_delete_backup', False):
                backup_path = file_path + '.backup'
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            
        except Exception as e:
            self.log(f"✗ 处理失败 {os.path.basename(file_path)}: {e}", "ERROR")
            backup_path = file_path + '.backup'
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, file_path)
                    self.log(f"已从备份恢复: {os.path.basename(file_path)}", "INFO")
                except:
                    pass
    
    def translate_files(self, file_paths):
        """翻译文件列表"""
        self.stop_flag = False
        self.stats = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        self.log(f"开始翻译 {len(file_paths)} 个文件")
        self.log(f"线程数: {self.max_threads}")
        self.log("=" * 60)
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for file_path in file_paths:
                if self.stop_flag:
                    break
                future = executor.submit(self.process_yaml_file, file_path)
                futures.append(future)
            
            for i, future in enumerate(futures):
                if self.stop_flag:
                    break
                future.result()
                self.update_progress(i + 1, len(file_paths), 
                                    f"处理中: {i + 1}/{len(file_paths)}")
        
        self.stats['end_time'] = datetime.now()
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.log("=" * 60)
        self.log(f"翻译完成！", "SUCCESS")
        self.log(f"处理文件: {self.stats['processed_files']}/{self.stats['total_files']}")
        self.log(f"翻译成功: {self.stats['successful_translations']}")
        self.log(f"翻译失败: {self.stats['failed_translations']}")
        self.log(f"耗时: {elapsed:.2f}秒")
        
        return self.stats
    
    def stop(self):
        """停止翻译"""
        self.stop_flag = True


# ==================== 配置管理器 ====================
class ConfigManager:
    def __init__(self, config_file="translator_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置"""
        default_config = {
            'api_keys': [],
            'current_key': '',
            'max_threads': 4,
            'auto_backup': True,
            'skip_chinese': True,
            'auto_delete_backup': False,
            'api_timeout': 30,
            'enable_retry': True,
            'max_retries': 3,
            'retry_delay': 5,
            'theme': 'light',
            'display_mode': 'simple',
            'sort_mode': 'add_order',
            'log_level': 'standard',
            'auto_save_log': False,
            'log_path': '',
            'save_history': True,
            'max_history': 100,
            'history': [],
            'shortcuts': {
                'add_files': 'Ctrl+O',
                'add_folder': 'Ctrl+D',
                'start': 'F5',
                'stop': 'Escape',
                'clear_log': 'Ctrl+L',
                'remove': 'Delete',
                'settings': 'Ctrl+comma'
            },
            'proxy_enabled': False,
            'proxy_host': '',
            'proxy_port': 8080
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
            except:
                pass
        
        return default_config
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, indent=2, fp=f, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def add_api_key(self, key, name=""):
        """添加API Key"""
        if not name:
            name = f"Key {len(self.config['api_keys']) + 1}"
        
        key_data = {
            'key': key,
            'name': name,
            'created': datetime.now().isoformat()
        }
        
        for k in self.config['api_keys']:
            if k['key'] == key:
                return False
        
        self.config['api_keys'].append(key_data)
        self.save_config()
        return True
    
    def remove_api_key(self, key):
        """删除API Key"""
        self.config['api_keys'] = [k for k in self.config['api_keys'] if k['key'] != key]
        if self.config['current_key'] == key:
            self.config['current_key'] = ''
        self.save_config()
    
    def add_history(self, stats, files):
        """添加历史记录"""
        if not self.config.get('save_history', True):
            return
        
        history_item = {
            'timestamp': datetime.now().isoformat(),
            'total_files': stats['total_files'],
            'processed_files': stats['processed_files'],
            'successful_translations': stats['successful_translations'],
            'failed_translations': stats['failed_translations'],
            'duration': (stats['end_time'] - stats['start_time']).total_seconds() if stats['end_time'] else 0,
            'files': [os.path.basename(f) for f in files[:10]]  # 只保存前10个文件名
        }
        
        if 'history' not in self.config:
            self.config['history'] = []
        
        self.config['history'].insert(0, history_item)
        
        # 限制历史记录数量
        max_history = self.config.get('max_history', 100)
        self.config['history'] = self.config['history'][:max_history]
        
        self.save_config()


# ==================== GUI 主界面 ====================
class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)
        
        self.config_manager = ConfigManager()
        self.translator_core = None
        self.is_translating = False
        self.file_queue = []
        self.file_info = {}  # 存储文件详细信息
        
        # 配置样式
        self.setup_styles()
        
        # 配置网格权重
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # 创建UI
        self.create_menu_bar()
        self.create_toolbar()
        self.create_statusbar()
        self.create_main_content()
        self.create_bottom_bar()
        
        # 加载设置
        self.load_settings()
        self.apply_theme()
        
        # 绑定快捷键
        self.bind_shortcuts()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styles(self):
        """配置UI样式"""
        self.style = ttk.Style()
        
        available_themes = self.style.theme_names()
        if 'vista' in available_themes:
            self.style.theme_use('vista')
        elif 'clam' in available_themes:
            self.style.theme_use('clam')
        
        # 配置自定义样式
        self.style.configure('Accent.TButton', font=('Microsoft YaHei UI', 9, 'bold'))
        self.style.configure('Title.TLabel', font=('Microsoft YaHei UI', 11, 'bold'))
        self.style.configure('TLabelframe', font=('Microsoft YaHei UI', 9))
        self.style.configure('TLabelframe.Label', font=('Microsoft YaHei UI', 9, 'bold'))
        
    def apply_theme(self):
        """应用主题"""
        theme = self.config_manager.config.get('theme', 'light')
        
        if theme == 'dark':
            # 暗色主题配色
            bg = '#2b2b2b'
            fg = '#e0e0e0'
            select_bg = '#4a9eff'
            
            self.root.configure(bg=bg)
            self.style.configure('TFrame', background=bg)
            self.style.configure('TLabel', background=bg, foreground=fg)
            self.style.configure('TLabelframe', background=bg, foreground=fg)
            self.style.configure('TLabelframe.Label', background=bg, foreground=fg)
            
            # 更新文本控件颜色
            if hasattr(self, 'log_text'):
                self.log_text.configure(bg='#1e1e1e', fg=fg, insertbackground=fg)
            if hasattr(self, 'file_listbox'):
                self.file_listbox.configure(bg='#1e1e1e', fg=fg, selectbackground=select_bg)
        else:
            # 亮色主题（默认）
            self.root.configure(bg='SystemButtonFace')
            self.style.configure('TFrame', background='SystemButtonFace')
            self.style.configure('TLabel', background='SystemButtonFace', foreground='black')
            
            if hasattr(self, 'log_text'):
                self.log_text.configure(bg='white', fg='black', insertbackground='black')
            if hasattr(self, 'file_listbox'):
                self.file_listbox.configure(bg='white', fg='black', selectbackground='#0078d7')
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件(F)", menu=file_menu)
        file_menu.add_command(label="添加文件", command=self.add_files, accelerator="Ctrl+O")
        file_menu.add_command(label="添加文件夹", command=self.add_folder, accelerator="Ctrl+D")
        file_menu.add_command(label="清空列表", command=self.clear_files)
        file_menu.add_separator()
        file_menu.add_command(label="导出日志", command=self.export_log)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 备份菜单
        backup_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="备份(B)", menu=backup_menu)
        backup_menu.add_command(label="恢复所有备份", command=self.restore_all_backups)
        backup_menu.add_command(label="清理所有备份", command=self.cleanup_all_backups)
        backup_menu.add_command(label="查看备份文件...", command=self.view_backups)
        
        # 工具菜单
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具(T)", menu=tools_menu)
        tools_menu.add_command(label="管理 API Key", command=self.show_key_manager)
        tools_menu.add_command(label="设置...", command=self.show_settings, accelerator="Ctrl+,")
        tools_menu.add_separator()
        if not HAS_DND and DND_INSTALL_AVAILABLE:
            tools_menu.add_command(label="安装拖拽支持", command=self.install_dnd)
        
        # 视图菜单
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图(V)", menu=view_menu)
        
        self.theme_var = tk.StringVar(value=self.config_manager.config.get('theme', 'light'))
        view_menu.add_radiobutton(label="亮色主题", variable=self.theme_var, 
                                  value='light', command=self.change_theme)
        view_menu.add_radiobutton(label="暗色主题", variable=self.theme_var, 
                                  value='dark', command=self.change_theme)
        
        # 帮助菜单
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助(H)", menu=help_menu)
        help_menu.add_command(label="翻译历史记录", command=self.show_history)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_separator()
        help_menu.add_command(label="关于", command=self.show_about)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.root, padding="10 8")
        toolbar.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        toolbar.columnconfigure(1, weight=1)
        
        # API Key
        ttk.Label(toolbar, text="API Key:", font=('Microsoft YaHei UI', 9)).grid(
            row=0, column=0, padx=(0, 8), sticky='w')
        
        self.key_combo = ttk.Combobox(toolbar, state='readonly', font=('Consolas', 9))
        self.key_combo.grid(row=0, column=1, padx=(0, 8), sticky='ew')
        self.key_combo.bind('<<ComboboxSelected>>', self.on_key_selected)
        
        ttk.Button(toolbar, text="管理", command=self.show_key_manager, width=8).grid(
            row=0, column=2, padx=(0, 15))
        
        # 分隔符
        ttk.Separator(toolbar, orient=tk.VERTICAL).grid(row=0, column=3, sticky='ns', padx=10)
        
        # 线程数
        ttk.Label(toolbar, text="并发线程:", font=('Microsoft YaHei UI', 9)).grid(
            row=0, column=4, padx=(0, 8), sticky='w')
        
        self.thread_spin = ttk.Spinbox(toolbar, from_=1, to=200, width=8, font=('Consolas', 9))
        self.thread_spin.set(4)
        self.thread_spin.grid(row=0, column=5, padx=(0, 8))
        
        ttk.Label(toolbar, text="(建议: 1-50)", font=('Microsoft YaHei UI', 8), 
                 foreground='gray').grid(row=0, column=6, sticky='w')
    
    def create_statusbar(self):
        """创建状态栏"""
        statusbar = ttk.Frame(self.root, relief=tk.SUNKEN, padding="5 2")
        statusbar.grid(row=1, column=0, sticky='ew', padx=5)
        
        # 版本
        ttk.Label(statusbar, text=f"v{VERSION}", font=('Microsoft YaHei UI', 8)).pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # API 状态
        self.api_status_label = ttk.Label(statusbar, text="🔴 API未配置", 
                                         font=('Microsoft YaHei UI', 8))
        self.api_status_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 拖拽状态
        dnd_status = "🟢 拖拽: 可用" if HAS_DND else "🔴 拖拽: 不可用"
        ttk.Label(statusbar, text=dnd_status, font=('Microsoft YaHei UI', 8)).pack(
            side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 文件数
        self.file_count_status = ttk.Label(statusbar, text="文件: 0", 
                                          font=('Microsoft YaHei UI', 8))
        self.file_count_status.pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 状态文本
        self.status_text = ttk.Label(statusbar, text="就绪", font=('Microsoft YaHei UI', 8))
        self.status_text.pack(side=tk.LEFT, padx=5)
    
    def create_main_content(self):
        """创建主内容区域"""
        main_container = ttk.Frame(self.root)
        main_container.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)
        
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        # 左侧面板 - 文件管理
        left_panel = ttk.LabelFrame(main_container, text=" 📁 待翻译文件 ", padding="8")
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)
        
        # 文件列表控制栏
        list_control = ttk.Frame(left_panel)
        list_control.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        ttk.Label(list_control, text="查看:").pack(side=tk.LEFT, padx=(0, 5))
        self.display_mode = ttk.Combobox(list_control, width=10, state='readonly',
                                        values=['简洁模式', '详细模式', '超详细模式'])
        self.display_mode.set('简洁模式')
        self.display_mode.bind('<<ComboboxSelected>>', self.change_display_mode)
        self.display_mode.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(list_control, text="排序:").pack(side=tk.LEFT, padx=(0, 5))
        self.sort_mode = ttk.Combobox(list_control, width=12, state='readonly',
                                     values=['按添加顺序', '按名称(A-Z)', '按名称(Z-A)', 
                                            '按路径', '按大小', '按修改时间'])
        self.sort_mode.set('按添加顺序')
        self.sort_mode.bind('<<ComboboxSelected>>', self.change_sort_mode)
        self.sort_mode.pack(side=tk.LEFT)
        
        # 文件列表
        list_container = ttk.Frame(left_panel)
        list_container.grid(row=1, column=0, sticky='nsew', pady=(0, 8))
        list_container.rowconfigure(0, weight=1)
        list_container.columnconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        self.file_listbox = tk.Listbox(
            list_container,
            yscrollcommand=scrollbar.set,
            selectmode=tk.EXTENDED,
            font=('Microsoft YaHei UI', 9),
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1
        )
        self.file_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.file_listbox.yview)
        
        # 绑定右键菜单
        self.file_listbox.bind("<Button-3>", self.show_context_menu)
        
        # 启用拖拽
        if HAS_DND:
            self.file_listbox.drop_target_register(DND_FILES)
            self.file_listbox.dnd_bind('<<Drop>>', self.on_drop)
        
        # 文件操作按钮
        file_btn_frame = ttk.Frame(left_panel)
        file_btn_frame.grid(row=2, column=0, sticky='ew', pady=(0, 8))
        file_btn_frame.columnconfigure(0, weight=1)
        file_btn_frame.columnconfigure(1, weight=1)
        file_btn_frame.columnconfigure(2, weight=1)
        file_btn_frame.columnconfigure(3, weight=1)
        
        ttk.Button(file_btn_frame, text="📄 添加文件", command=self.add_files).grid(
            row=0, column=0, padx=2, sticky='ew')
        ttk.Button(file_btn_frame, text="📁 添加文件夹", command=self.add_folder).grid(
            row=0, column=1, padx=2, sticky='ew')
        ttk.Button(file_btn_frame, text="🗑️ 清空", command=self.clear_files).grid(
            row=0, column=2, padx=2, sticky='ew')
        ttk.Button(file_btn_frame, text="❌ 移除", command=self.remove_selected).grid(
            row=0, column=3, padx=2, sticky='ew')
        
        # 文件统计
        self.file_count_label = ttk.Label(left_panel, text="已选择: 0 个文件",
                                         font=('Microsoft YaHei UI', 9, 'bold'),
                                         foreground='#0066cc')
        self.file_count_label.grid(row=3, column=0, sticky='w')
        
        # 右侧面板 - 进度和日志
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        
        # 进度区域
        progress_frame = ttk.LabelFrame(right_panel, text=" 📊 翻译进度 ", padding="8")
        progress_frame.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        
        self.progress_label = ttk.Label(progress_frame, text="就绪",
                                       font=('Microsoft YaHei UI', 9))
        self.progress_label.grid(row=1, column=0)
        
        self.stats_label = ttk.Label(progress_frame, text="",
                                     font=('Microsoft YaHei UI', 9))
        self.stats_label.grid(row=2, column=0, pady=(5, 0))
        
        # 日志区域
        log_frame = ttk.LabelFrame(right_panel, text=" 📝 运行日志 ", padding="8")
        log_frame.grid(row=1, column=0, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1
        )
        self.log_text.grid(row=0, column=0, sticky='nsew', pady=(0, 8))
        
        # 配置日志颜色标签
        self.log_text.tag_config("INFO", foreground="#333333")
        self.log_text.tag_config("SUCCESS", foreground="#008000", font=('Consolas', 9, 'bold'))
        self.log_text.tag_config("WARNING", foreground="#FF8C00", font=('Consolas', 9, 'bold'))
        self.log_text.tag_config("ERROR", foreground="#DC143C", font=('Consolas', 9, 'bold'))
        
        # 日志操作按钮
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.grid(row=1, column=0, sticky='ew')
        log_btn_frame.columnconfigure(0, weight=1)
        log_btn_frame.columnconfigure(1, weight=1)
        
        ttk.Button(log_btn_frame, text="🗑️ 清空日志", command=self.clear_log).grid(
            row=0, column=0, padx=(0, 4), sticky='ew')
        ttk.Button(log_btn_frame, text="💾 导出日志", command=self.export_log).grid(
            row=0, column=1, sticky='ew')
    
    def create_bottom_bar(self):
        """创建底部控制栏"""
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        bottom_frame.columnconfigure(1, weight=1)
        
        # 左侧按钮组
        left_btns = ttk.Frame(bottom_frame)
        left_btns.grid(row=0, column=0, sticky='w')
        
        self.start_btn = ttk.Button(
            left_btns,
            text="▶️  开始翻译",
            command=self.start_translation,
            style='Accent.TButton',
            width=15
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 8))
        
        self.stop_btn = ttk.Button(
            left_btns,
            text="⏸️  停止翻译",
            command=self.stop_translation,
            state=tk.DISABLED,
            width=15
        )
        self.stop_btn.grid(row=0, column=1)
        
        # 右侧提示
        self.hint_frame = ttk.Frame(bottom_frame)
        self.hint_frame.grid(row=0, column=2, sticky='e', padx=10)
        
        self.update_hint_text()
    
    def update_hint_text(self):
        """更新提示文本"""
        for widget in self.hint_frame.winfo_children():
            widget.destroy()
        
        if HAS_DND:
            ttk.Label(self.hint_frame, text="💡 支持拖拽文件/文件夹到列表",
                     font=('Microsoft YaHei UI', 9),
                     foreground='gray').pack()
        else:
            ttk.Label(self.hint_frame, text="💡 请使用按钮添加文件 | ",
                     font=('Microsoft YaHei UI', 9),
                     foreground='gray').pack(side=tk.LEFT)
            
            if DND_INSTALL_AVAILABLE:
                install_link = ttk.Label(self.hint_frame, text="点击安装拖拽功能",
                                        font=('Microsoft YaHei UI', 9),
                                        foreground='red',
                                        cursor='hand2')
                install_link.pack(side=tk.LEFT)
                install_link.bind('<Button-1>', lambda e: self.install_dnd())
    
    def bind_shortcuts(self):
        """绑定快捷键"""
        shortcuts = self.config_manager.config.get('shortcuts', {})
        
        self.root.bind('<Control-o>', lambda e: self.add_files())
        self.root.bind('<Control-d>', lambda e: self.add_folder())
        self.root.bind('<F5>', lambda e: self.start_translation())
        self.root.bind('<Escape>', lambda e: self.stop_translation())
        self.root.bind('<Control-l>', lambda e: self.clear_log())
        self.root.bind('<Delete>', lambda e: self.remove_selected())
        self.root.bind('<Control-comma>', lambda e: self.show_settings())
    
    def load_settings(self):
        """加载设置"""
        keys = self.config_manager.config.get('api_keys', [])
        if keys:
            key_names = [f"{k['name']} ({k['key'][:10]}...)" for k in keys]
            self.key_combo['values'] = key_names
            
            current_key = self.config_manager.config.get('current_key', '')
            if current_key:
                for i, k in enumerate(keys):
                    if k['key'] == current_key:
                        self.key_combo.current(i)
                        self.api_status_label.config(text="🟢 API已连接")
                        break
        
        thread_count = self.config_manager.config.get('max_threads', 4)
        self.thread_spin.set(thread_count)
        
        display_mode = self.config_manager.config.get('display_mode', 'simple')
        mode_map = {'simple': '简洁模式', 'detail': '详细模式', 'ultra': '超详细模式'}
        self.display_mode.set(mode_map.get(display_mode, '简洁模式'))
        
        sort_mode = self.config_manager.config.get('sort_mode', 'add_order')
        sort_map = {
            'add_order': '按添加顺序',
            'name_asc': '按名称(A-Z)',
            'name_desc': '按名称(Z-A)',
            'path': '按路径',
            'size': '按大小',
            'time': '按修改时间'
        }
        self.sort_mode.set(sort_map.get(sort_mode, '按添加顺序'))
    
    # ==================== 事件处理函数 ====================
    
    def on_key_selected(self, event):
        """选择API Key"""
        index = self.key_combo.current()
        if index >= 0:
            keys = self.config_manager.config.get('api_keys', [])
            self.config_manager.config['current_key'] = keys[index]['key']
            self.config_manager.save_config()
            self.api_status_label.config(text="🟢 API已连接")
    
    def on_drop(self, event):
        """处理拖拽事件"""
        files = self.root.tk.splitlist(event.data)
        for file_path in files:
            file_path = file_path.strip('{}')
            self.add_path(file_path)
    
    def show_context_menu(self, event):
        """显示右键菜单"""
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        menu = Menu(self.root, tearoff=0)
        
        if len(selection) == 1:
            # 单选
            menu.add_command(label="📂 打开文件位置", command=self.open_file_location)
            menu.add_command(label="📝 用编辑器打开", command=self.open_with_editor)
            menu.add_command(label="📋 复制文件路径", command=self.copy_file_path)
            menu.add_command(label="📋 复制文件名", command=self.copy_file_name)
            menu.add_separator()
            menu.add_command(label="❌ 从列表移除", command=self.remove_selected)
        else:
            # 多选
            menu.add_command(label="📂 打开文件位置", command=self.open_file_location)
            menu.add_command(label=f"📋 复制路径({len(selection)}个)", command=self.copy_file_path)
            menu.add_separator()
            menu.add_command(label=f"❌ 从列表移除({len(selection)}个)", command=self.remove_selected)
        
        menu.post(event.x_root, event.y_root)
    
    def open_file_location(self):
        """打开文件位置"""
        selection = self.file_listbox.curselection()
        if selection:
            idx = selection[0]
            file_path = self.file_queue[idx]
            folder = os.path.dirname(file_path)
            
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder])
            else:
                subprocess.run(['xdg-open', folder])
    
    def open_with_editor(self):
        """用编辑器打开"""
        selection = self.file_listbox.curselection()
        if selection:
            idx = selection[0]
            file_path = self.file_queue[idx]
            
            if sys.platform == 'win32':
                os.startfile(file_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', file_path])
            else:
                subprocess.run(['xdg-open', file_path])
    
    def copy_file_path(self):
        """复制文件路径"""
        selection = self.file_listbox.curselection()
        if selection:
            paths = [self.file_queue[idx] for idx in selection]
            self.root.clipboard_clear()
            self.root.clipboard_append('\n'.join(paths))
            self.log_message(f"[INFO] 已复制 {len(paths)} 个文件路径")
    
    def copy_file_name(self):
        """复制文件名"""
        selection = self.file_listbox.curselection()
        if selection:
            idx = selection[0]
            file_name = os.path.basename(self.file_queue[idx])
            self.root.clipboard_clear()
            self.root.clipboard_append(file_name)
            self.log_message(f"[INFO] 已复制文件名: {file_name}")
    
    def change_display_mode(self, event=None):
        """切换查看模式"""
        mode = self.display_mode.get()
        mode_map = {'简洁模式': 'simple', '详细模式': 'detail', '超详细模式': 'ultra'}
        self.config_manager.config['display_mode'] = mode_map.get(mode, 'simple')
        self.config_manager.save_config()
        self.refresh_file_list()
    
    def change_sort_mode(self, event=None):
        """切换排序模式"""
        mode = self.sort_mode.get()
        mode_map = {
            '按添加顺序': 'add_order',
            '按名称(A-Z)': 'name_asc',
            '按名称(Z-A)': 'name_desc',
            '按路径': 'path',
            '按大小': 'size',
            '按修改时间': 'time'
        }
        self.config_manager.config['sort_mode'] = mode_map.get(mode, 'add_order')
        self.config_manager.save_config()
        self.sort_files()
        self.refresh_file_list()
    
    def sort_files(self):
        """排序文件列表"""
        sort_mode = self.config_manager.config.get('sort_mode', 'add_order')
        
        if sort_mode == 'add_order':
            return  # 保持原顺序
        
        if sort_mode == 'name_asc':
            self.file_queue.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_mode == 'name_desc':
            self.file_queue.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)
        elif sort_mode == 'path':
            self.file_queue.sort()
        elif sort_mode == 'size':
            self.file_queue.sort(key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0)
        elif sort_mode == 'time':
            self.file_queue.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    
    def refresh_file_list(self):
        """刷新文件列表显示"""
        self.file_listbox.delete(0, tk.END)
        
        display_mode = self.config_manager.config.get('display_mode', 'simple')
        
        for file_path in self.file_queue:
            if display_mode == 'simple':
                text = os.path.basename(file_path)
            elif display_mode == 'detail':
                name = os.path.basename(file_path)
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                size_str = self.format_size(size)
                text = f"{name}  |  {size_str}"
            else:  # ultra
                name = os.path.basename(file_path)
                size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                size_str = self.format_size(size)
                mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                text = f"{name}  |  {size_str}  |  {time_str}  |  ⏳等待"
            
            self.file_listbox.insert(tk.END, text)
    
    def format_size(self, size):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def add_path(self, path):
        """添加路径"""
        if os.path.isfile(path):
            if path.lower().endswith(('.yml', '.yaml')) and path not in self.file_queue:
                self.file_queue.append(path)
                self.refresh_file_list()
        elif os.path.isdir(path):
            core = YamlTranslatorCore("", 1)
            yaml_files = core.find_yaml_files(path)
            added = 0
            for f in yaml_files:
                if f not in self.file_queue:
                    self.file_queue.append(f)
                    added += 1
            if added > 0:
                self.refresh_file_list()
                self.log_message(f"[INFO] 从文件夹添加了 {added} 个 YAML 文件")
        
        self.update_file_count()
    
    def add_files(self):
        """添加文件"""
        files = filedialog.askopenfilenames(
            title="选择 YAML 文件",
            filetypes=[("YAML 文件", "*.yml *.yaml"), ("所有文件", "*.*")]
        )
        for file in files:
            self.add_path(file)
    
    def add_folder(self):
        """添加文件夹"""
        folder = filedialog.askdirectory(title="选择文件夹")
        if folder:
            self.add_path(folder)
    
    def clear_files(self):
        """清空文件列表"""
        if self.is_translating:
            messagebox.showwarning("警告", "翻译进行中，无法清空列表")
            return
        
        if self.file_queue and not messagebox.askyesno("确认", "确定要清空文件列表吗？"):
            return
        
        self.file_listbox.delete(0, tk.END)
        self.file_queue.clear()
        self.file_info.clear()
        self.update_file_count()
        self.log_message("[INFO] 已清空文件列表")
    
    def remove_selected(self):
        """移除选中的文件"""
        selected = self.file_listbox.curselection()
        for index in reversed(selected):
            self.file_listbox.delete(index)
            self.file_queue.pop(index)
        self.update_file_count()
    
    def update_file_count(self):
        """更新文件计数"""
        count = len(self.file_queue)
        self.file_count_label.config(text=f"已选择: {count} 个文件")
        self.file_count_status.config(text=f"文件: {count}")
    
    def log_message(self, message):
        """显示日志"""
        self.log_text.insert(tk.END, message + "\n")
        
        # 根据日志级别设置颜色
        if "[ERROR]" in message:
            tag = "ERROR"
        elif "[WARNING]" in message:
            tag = "WARNING"
        elif "[SUCCESS]" in message or "✓" in message:
            tag = "SUCCESS"
        else:
            tag = "INFO"
        
        # 获取最后一行
        last_line = self.log_text.index("end-1c linestart")
        self.log_text.tag_add(tag, last_line, "end-1c")
        
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
        # 自动保存日志
        if self.config_manager.config.get('auto_save_log', False):
            log_path = self.config_manager.config.get('log_path', '')
            if log_path:
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(message + '\n')
                except:
                    pass
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
    
    def export_log(self):
        """导出日志"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialfile=f"translation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("成功", f"日志已导出到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败:\n{e}")
    
    def update_progress_ui(self, current, total, status):
        """更新进度UI"""
        if total > 0:
            progress = (current / total) * 100
            self.progress_bar['value'] = progress
        
        self.progress_label.config(text=status)
        self.status_text.config(text=status)
        self.root.update_idletasks()
    
    def update_stats(self, stats):
        """更新统计信息"""
        if stats:
            text = (f"总翻译: {stats['total_translations']} | "
                   f"成功: {stats['successful_translations']} | "
                   f"失败: {stats['failed_translations']}")
            self.stats_label.config(text=text)
    
    def start_translation(self):
        """开始翻译"""
        if not self.file_queue:
            messagebox.showwarning("警告", "请先添加要翻译的文件")
            return
        
        current_key = self.config_manager.config.get('current_key', '')
        if not current_key:
            messagebox.showwarning("警告", "请先选择或添加 API Key")
            return
        
        if self.is_translating:
            return
        
        # 保存设置
        try:
            thread_count = int(self.thread_spin.get())
            self.config_manager.config['max_threads'] = thread_count
            self.config_manager.save_config()
        except:
            thread_count = 4
        
        self.is_translating = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.status_text.config(text="翻译中...")
        
        # 在新线程中执行翻译
        def translate_thread():
            try:
                # 获取配置
                config = {
                    'auto_backup': self.config_manager.config.get('auto_backup', True),
                    'skip_chinese': self.config_manager.config.get('skip_chinese', True),
                    'auto_delete_backup': self.config_manager.config.get('auto_delete_backup', False),
                    'api_timeout': self.config_manager.config.get('api_timeout', 30),
                    'max_retries': self.config_manager.config.get('max_retries', 3) if self.config_manager.config.get('enable_retry', True) else 1,
                    'retry_delay': self.config_manager.config.get('retry_delay', 5)
                }
                
                self.translator_core = YamlTranslatorCore(
                    current_key,
                    max_threads=thread_count,
                    progress_callback=self.update_progress_ui,
                    log_callback=self.log_message,
                    config=config
                )
                
                stats = self.translator_core.translate_files(self.file_queue)
                self.update_stats(stats)
                
                # 保存历史记录
                self.config_manager.add_history(stats, self.file_queue)
                
                # 显示完成消息
                self.root.after(0, lambda: messagebox.showinfo(
                    "翻译完成",
                    f"翻译完成！\n\n"
                    f"处理文件: {stats['processed_files']}/{stats['total_files']}\n"
                    f"翻译成功: {stats['successful_translations']}\n"
                    f"翻译失败: {stats['failed_translations']}\n"
                    f"耗时: {stats.get('duration', 0):.1f}秒"
                ))
                
            except Exception as e:
                self.log_message(f"[ERROR] 翻译过程出错: {e}")
                self.root.after(0, lambda: messagebox.showerror("错误", f"翻译失败:\n{e}"))
            finally:
                self.is_translating = False
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.status_text.config(text="就绪")
        
        thread = threading.Thread(target=translate_thread, daemon=True)
        thread.start()
    
    def stop_translation(self):
        """停止翻译"""
        if self.translator_core and messagebox.askyesno("确认", "确定要停止翻译吗？"):
            self.translator_core.stop()
            self.log_message("[WARNING] 正在停止翻译...")
            self.status_text.config(text="正在停止...")
    
    # ==================== 对话框和窗口 ====================
    
    def show_key_manager(self):
        """显示API Key管理窗口"""
        manager_window = tk.Toplevel(self.root)
        manager_window.title("API Key 管理")
        manager_window.geometry("700x450")
        manager_window.minsize(600, 400)
        manager_window.transient(self.root)
        manager_window.grab_set()
        
        manager_window.rowconfigure(1, weight=1)
        manager_window.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(manager_window, padding="15 15 15 10")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text="API Key 管理", style='Title.TLabel').pack(anchor=tk.W)
        
        # 列表
        list_frame = ttk.Frame(manager_window, padding="0 0 15 10")
        list_frame.grid(row=1, column=0, sticky='nsew', padx=15)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        columns = ('name', 'key', 'created')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        tree.heading('name', text='名称')
        tree.heading('key', text='API Key')
        tree.heading('created', text='创建时间')
        
        tree.column('name', width=150)
        tree.column('key', width=300)
        tree.column('created', width=180)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        def refresh_tree():
            tree.delete(*tree.get_children())
            for k in self.config_manager.config.get('api_keys', []):
                masked_key = k['key'][:10] + "..." + k['key'][-8:]
                created = k.get('created', 'N/A')[:19] if 'created' in k else "未知"
                tree.insert('', tk.END, values=(k['name'], masked_key, created))
        
        refresh_tree()
        
        # 按钮
        btn_frame = ttk.Frame(manager_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def add_key():
            add_window = tk.Toplevel(manager_window)
            add_window.title("添加 API Key")
            add_window.geometry("500x200")
            add_window.resizable(False, False)
            add_window.transient(manager_window)
            add_window.grab_set()
            
            add_window.rowconfigure(0, weight=1)
            add_window.columnconfigure(0, weight=1)
            
            form = ttk.Frame(add_window, padding="20")
            form.grid(row=0, column=0, sticky='nsew')
            
            ttk.Label(form, text="名称:", font=('Microsoft YaHei UI', 9)).grid(
                row=0, column=0, padx=10, pady=15, sticky=tk.W)
            name_entry = ttk.Entry(form, width=45, font=('Microsoft YaHei UI', 9))
            name_entry.grid(row=0, column=1, padx=10, pady=15, sticky='ew')
            name_entry.focus()
            
            ttk.Label(form, text="API Key:", font=('Microsoft YaHei UI', 9)).grid(
                row=1, column=0, padx=10, pady=15, sticky=tk.W)
            key_entry = ttk.Entry(form, width=45, show='*', font=('Consolas', 9))
            key_entry.grid(row=1, column=1, padx=10, pady=15, sticky='ew')
            
            form.columnconfigure(1, weight=1)
            
            def save_key():
                name = name_entry.get().strip()
                key = key_entry.get().strip()
                
                if not key:
                    messagebox.showwarning("警告", "API Key 不能为空", parent=add_window)
                    return
                
                if self.config_manager.add_api_key(key, name if name else None):
                    messagebox.showinfo("成功", "API Key 已添加", parent=add_window)
                    refresh_tree()
                    self.load_settings()
                    add_window.destroy()
                else:
                    messagebox.showwarning("警告", "该 API Key 已存在", parent=add_window)
            
            btn_frame_add = ttk.Frame(form)
            btn_frame_add.grid(row=2, column=1, pady=20, sticky=tk.E)
            
            ttk.Button(btn_frame_add, text="保存", command=save_key, width=12).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame_add, text="取消", command=add_window.destroy, width=12).pack(side=tk.LEFT)
        
        def remove_key():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("警告", "请选择要删除的 API Key", parent=manager_window)
                return
            
            if messagebox.askyesno("确认", "确定要删除选中的 API Key 吗？", parent=manager_window):
                for item in selection:
                    values = tree.item(item)['values']
                    for k in self.config_manager.config.get('api_keys', []):
                        if k['name'] == values[0]:
                            self.config_manager.remove_api_key(k['key'])
                            break
                
                refresh_tree()
                self.load_settings()
        
        ttk.Button(btn_frame, text="➕ 添加", command=add_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑️ 删除", command=remove_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=manager_window.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    def show_settings(self):
        """显示设置对话框"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("600x550")
        settings_window.minsize(550, 500)
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        settings_window.rowconfigure(1, weight=1)
        settings_window.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(settings_window, padding="15 15 15 10")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text="⚙️ 设置", style='Title.TLabel').pack(anchor=tk.W)
        
        # 选项卡
        notebook = ttk.Notebook(settings_window)
        notebook.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 10))
        
        # ===== 基本设置选项卡 =====
        basic_tab = ttk.Frame(notebook, padding="15")
        notebook.add(basic_tab, text="基本设置")
        
        # 翻译设置
        trans_frame = ttk.LabelFrame(basic_tab, text="翻译设置", padding="10")
        trans_frame.pack(fill=tk.X, pady=(0, 10))
        
        auto_backup_var = tk.BooleanVar(value=self.config_manager.config.get('auto_backup', True))
        ttk.Checkbutton(trans_frame, text="自动创建备份文件", variable=auto_backup_var).pack(anchor=tk.W, pady=2)
        
        skip_chinese_var = tk.BooleanVar(value=self.config_manager.config.get('skip_chinese', True))
        ttk.Checkbutton(trans_frame, text="跳过已包含中文的字段", variable=skip_chinese_var).pack(anchor=tk.W, pady=2)
        
        auto_del_backup_var = tk.BooleanVar(value=self.config_manager.config.get('auto_delete_backup', False))
        ttk.Checkbutton(trans_frame, text="翻译成功后自动删除备份", variable=auto_del_backup_var).pack(anchor=tk.W, pady=2)
        
        thread_frame = ttk.Frame(trans_frame)
        thread_frame.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(thread_frame, text="默认并发线程数:").pack(side=tk.LEFT, padx=(0, 8))
        thread_var = tk.IntVar(value=self.config_manager.config.get('max_threads', 4))
        ttk.Spinbox(thread_frame, from_=1, to=200, textvariable=thread_var, width=10).pack(side=tk.LEFT)
        ttk.Label(thread_frame, text="(1-200)", foreground='gray').pack(side=tk.LEFT, padx=(8, 0))
        
        timeout_frame = ttk.Frame(trans_frame)
        timeout_frame.pack(fill=tk.X, pady=2)
        ttk.Label(timeout_frame, text="API 请求超时:").pack(side=tk.LEFT, padx=(0, 8))
        timeout_var = tk.IntVar(value=self.config_manager.config.get('api_timeout', 30))
        ttk.Spinbox(timeout_frame, from_=5, to=300, textvariable=timeout_var, width=10).pack(side=tk.LEFT)
        ttk.Label(timeout_frame, text="秒", foreground='gray').pack(side=tk.LEFT, padx=(8, 0))
        
        # 界面设置
        ui_frame = ttk.LabelFrame(basic_tab, text="界面设置", padding="10")
        ui_frame.pack(fill=tk.X)
        
        # ===== 高级设置选项卡 =====
        advanced_tab = ttk.Frame(notebook, padding="15")
        notebook.add(advanced_tab, text="高级设置")
        
        # 网络设置
        net_frame = ttk.LabelFrame(advanced_tab, text="网络设置", padding="10")
        net_frame.pack(fill=tk.X, pady=(0, 10))
        
        proxy_var = tk.BooleanVar(value=self.config_manager.config.get('proxy_enabled', False))
        ttk.Checkbutton(net_frame, text="使用代理", variable=proxy_var).pack(anchor=tk.W, pady=2)
        
        proxy_frame = ttk.Frame(net_frame)
        proxy_frame.pack(fill=tk.X, pady=2)
        ttk.Label(proxy_frame, text="HTTP代理:").pack(side=tk.LEFT, padx=(20, 8))
        proxy_host_var = tk.StringVar(value=self.config_manager.config.get('proxy_host', ''))
        ttk.Entry(proxy_frame, textvariable=proxy_host_var, width=30).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(proxy_frame, text="端口:").pack(side=tk.LEFT, padx=(0, 8))
        proxy_port_var = tk.IntVar(value=self.config_manager.config.get('proxy_port', 8080))
        ttk.Entry(proxy_frame, textvariable=proxy_port_var, width=8).pack(side=tk.LEFT)
        
        # 重试设置
        retry_frame = ttk.LabelFrame(advanced_tab, text="失败重试", padding="10")
        retry_frame.pack(fill=tk.X, pady=(0, 10))
        
        retry_var = tk.BooleanVar(value=self.config_manager.config.get('enable_retry', True))
        ttk.Checkbutton(retry_frame, text="失败自动重试", variable=retry_var).pack(anchor=tk.W, pady=2)
        
        retry_count_frame = ttk.Frame(retry_frame)
        retry_count_frame.pack(fill=tk.X, pady=2)
        ttk.Label(retry_count_frame, text="重试次数:").pack(side=tk.LEFT, padx=(20, 8))
        retry_count_var = tk.IntVar(value=self.config_manager.config.get('max_retries', 3))
        ttk.Spinbox(retry_count_frame, from_=1, to=10, textvariable=retry_count_var, width=8).pack(side=tk.LEFT)
        
        retry_delay_frame = ttk.Frame(retry_frame)
        retry_delay_frame.pack(fill=tk.X, pady=2)
        ttk.Label(retry_delay_frame, text="重试延迟:").pack(side=tk.LEFT, padx=(20, 8))
        retry_delay_var = tk.IntVar(value=self.config_manager.config.get('retry_delay', 5))
        ttk.Spinbox(retry_delay_frame, from_=1, to=60, textvariable=retry_delay_var, width=8).pack(side=tk.LEFT)
        ttk.Label(retry_delay_frame, text="秒").pack(side=tk.LEFT, padx=(8, 0))
        
        # 日志设置
        log_frame = ttk.LabelFrame(advanced_tab, text="日志设置", padding="10")
        log_frame.pack(fill=tk.X)
        
        auto_save_log_var = tk.BooleanVar(value=self.config_manager.config.get('auto_save_log', False))
        ttk.Checkbutton(log_frame, text="自动保存日志", variable=auto_save_log_var).pack(anchor=tk.W, pady=2)
        
        save_history_var = tk.BooleanVar(value=self.config_manager.config.get('save_history', True))
        ttk.Checkbutton(log_frame, text="保存翻译历史记录", variable=save_history_var).pack(anchor=tk.W, pady=2)
        
        history_count_frame = ttk.Frame(log_frame)
        history_count_frame.pack(fill=tk.X, pady=2)
        ttk.Label(history_count_frame, text="最多保留:").pack(side=tk.LEFT, padx=(20, 8))
        history_count_var = tk.IntVar(value=self.config_manager.config.get('max_history', 100))
        ttk.Entry(history_count_frame, textvariable=history_count_var, width=8).pack(side=tk.LEFT)
        ttk.Label(history_count_frame, text="条").pack(side=tk.LEFT, padx=(8, 0))
        
        # ===== 快捷键选项卡 =====
        shortcuts_tab = ttk.Frame(notebook, padding="15")
        notebook.add(shortcuts_tab, text="快捷键")
        
        shortcuts_info = ttk.Label(shortcuts_tab, 
            text="快捷键配置（当前版本使用默认快捷键）",
            foreground='gray')
        shortcuts_info.pack(pady=10)
        
        shortcuts_list = [
            ("添加文件", "Ctrl+O"),
            ("添加文件夹", "Ctrl+D"),
            ("开始翻译", "F5"),
            ("停止翻译", "Esc"),
            ("清空日志", "Ctrl+L"),
            ("移除选中", "Delete"),
            ("设置", "Ctrl+,")
        ]
        
        for name, key in shortcuts_list:
            frame = ttk.Frame(shortcuts_tab)
            frame.pack(fill=tk.X, pady=3)
            ttk.Label(frame, text=name + ":", width=15).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(frame, text=key, foreground='blue').pack(side=tk.LEFT)
        
        # ===== 关于选项卡 =====
        about_tab = ttk.Frame(notebook, padding="15")
        notebook.add(about_tab, text="关于")
        
        about_text = f"""
{APP_TITLE}

一个YAML文件批量AI翻译工具

特性:
• 支持多线程并发翻译
• 智能上下文翻译
• 自动备份和恢复
• 翻译历史记录
• 丰富的配置选项

作者: Mr.Centes，Claude
版本: {VERSION}
        """
        
        ttk.Label(about_tab, text=about_text, justify=tk.LEFT).pack(pady=20)
        
        # 底部按钮
        btn_frame = ttk.Frame(settings_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def save_settings():
            # 保存所有设置
            self.config_manager.config['auto_backup'] = auto_backup_var.get()
            self.config_manager.config['skip_chinese'] = skip_chinese_var.get()
            self.config_manager.config['auto_delete_backup'] = auto_del_backup_var.get()
            self.config_manager.config['max_threads'] = thread_var.get()
            self.config_manager.config['api_timeout'] = timeout_var.get()
            self.config_manager.config['proxy_enabled'] = proxy_var.get()
            self.config_manager.config['proxy_host'] = proxy_host_var.get()
            self.config_manager.config['proxy_port'] = proxy_port_var.get()
            self.config_manager.config['enable_retry'] = retry_var.get()
            self.config_manager.config['max_retries'] = retry_count_var.get()
            self.config_manager.config['retry_delay'] = retry_delay_var.get()
            self.config_manager.config['auto_save_log'] = auto_save_log_var.get()
            self.config_manager.config['save_history'] = save_history_var.get()
            self.config_manager.config['max_history'] = history_count_var.get()
            
            self.config_manager.save_config()
            
            # 应用线程数设置
            self.thread_spin.set(thread_var.get())
            
            messagebox.showinfo("成功", "设置已保存", parent=settings_window)
            settings_window.destroy()
        
        def reset_defaults():
            if messagebox.askyesno("确认", "确定要恢复默认设置吗？", parent=settings_window):
                auto_backup_var.set(True)
                skip_chinese_var.set(True)
                auto_del_backup_var.set(False)
                thread_var.set(4)
                timeout_var.set(30)
                proxy_var.set(False)
                retry_var.set(True)
                retry_count_var.set(3)
                retry_delay_var.set(5)
                auto_save_log_var.set(False)
                save_history_var.set(True)
                history_count_var.set(100)
        
        ttk.Button(btn_frame, text="保存", command=save_settings, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=settings_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="恢复默认", command=reset_defaults, width=12).pack(side=tk.RIGHT, padx=5)
    
    def show_history(self):
        """显示翻译历史记录"""
        history_window = tk.Toplevel(self.root)
        history_window.title("翻译历史记录")
        history_window.geometry("800x500")
        history_window.minsize(700, 400)
        history_window.transient(self.root)
        history_window.grab_set()
        
        history_window.rowconfigure(1, weight=1)
        history_window.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(history_window, padding="15 15 15 5")
        title_frame.grid(row=0, column=0, sticky='ew')
        
        ttk.Label(title_frame, text="📊 翻译历史记录", style='Title.TLabel').pack(side=tk.LEFT)
        
        max_history = self.config_manager.config.get('max_history', 100)
        current_count = len(self.config_manager.config.get('history', []))
        ttk.Label(title_frame, text=f"(最多保留{max_history}条，当前{current_count}条)",
                 font=('Microsoft YaHei UI', 8),
                 foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # 列表
        list_frame = ttk.Frame(history_window, padding="0 0 15 10")
        list_frame.grid(row=1, column=0, sticky='nsew', padx=15)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        columns = ('time', 'files', 'success', 'failed', 'duration')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        tree.heading('time', text='时间')
        tree.heading('files', text='文件数')
        tree.heading('success', text='成功')
        tree.heading('failed', text='失败')
        tree.heading('duration', text='耗时')
        
        tree.column('time', width=180)
        tree.column('files', width=80)
        tree.column('success', width=80)
        tree.column('failed', width=80)
        tree.column('duration', width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # 加载历史记录
        for item in self.config_manager.config.get('history', []):
            time_str = item['timestamp'][:19].replace('T', ' ')
            duration_str = f"{item.get('duration', 0):.1f}秒"
            
            tree.insert('', tk.END, values=(
                time_str,
                item['processed_files'],
                item['successful_translations'],
                item['failed_translations'],
                duration_str
            ))
        
        # 按钮
        btn_frame = ttk.Frame(history_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def clear_history():
            if messagebox.askyesno("确认", "确定要清除所有历史记录吗？", parent=history_window):
                self.config_manager.config['history'] = []
                self.config_manager.save_config()
                tree.delete(*tree.get_children())
                messagebox.showinfo("成功", "历史记录已清除", parent=history_window)
        
        ttk.Button(btn_frame, text="清除全部历史", command=clear_history, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=history_window.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    def show_help(self):
        """显示使用说明"""
        help_window = tk.Toplevel(self.root)
        help_window.title("使用说明")
        help_window.geometry("700x600")
        help_window.transient(self.root)
        
        help_window.rowconfigure(0, weight=1)
        help_window.columnconfigure(0, weight=1)
        
        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, 
                                             font=('Microsoft YaHei UI', 10),
                                             padding="20")
        help_text.grid(row=0, column=0, sticky='nsew')
        
        content = f"""
{APP_TITLE} - 使用说明

一、快速开始
1. 配置 API Key：点击"工具" → "管理 API Key"，添加你的 DeepSeek API Key
2. 添加文件：通过按钮或拖拽添加需要翻译的 YAML 文件
3. 调整线程数：根据需要调整并发线程数（建议 1-50）
4. 开始翻译：点击"开始翻译"按钮

二、主要功能

1. 文件管理
   • 支持添加单个文件或整个文件夹
   • 支持拖拽文件/文件夹（需安装 tkinterdnd2）
   • 右键菜单：打开位置、复制路径等
   • 多种显示模式和排序方式

2. 翻译设置
   • 自动创建备份：翻译前自动备份原文件（.backup）
   • 跳过中文字段：已包含中文的字段不翻译
   • 并发控制：调整线程数提高翻译效率
   • 失败重试：网络错误时自动重试

3. 备份管理
   • 恢复所有备份：一键恢复所有 .backup 文件
   • 清理所有备份：批量删除备份文件
   • 查看备份文件：列出所有备份

4. 翻译历史
   • 自动记录每次翻译任务
   • 查看详细统计信息
   • 支持导出报告

三、快捷键
• Ctrl+O     - 添加文件
• Ctrl+D     - 添加文件夹
• F5         - 开始翻译
• Esc        - 停止翻译
• Ctrl+L     - 清空日志
• Delete     - 移除选中文件
• Ctrl+,     - 打开设置

四、高级功能

1. 代理设置
   在"设置" → "高级设置"中配置 HTTP 代理

2. 自动重试
   网络错误时自动重试，可配置重试次数和延迟

3. 日志管理
   支持自动保存日志到文件

4. 主题切换
   支持亮色和暗色两种主题

五、注意事项
• 首次使用需要配置 DeepSeek API Key
• 翻译前会自动创建备份文件
• 大文件建议适当降低线程数
• 翻译过程中不要关闭程序

六、常见问题

Q: 无法拖拽文件怎么办？
A: 需要安装 tkinterdnd2，点击底部提示链接一键安装

Q: 翻译失败怎么办？
A: 检查 API Key 是否正确，网络是否正常，可开启自动重试

Q: 如何恢复翻译前的文件？
A: 使用"备份" → "恢复所有备份"功能

Q: 历史记录太多怎么办？
A: 在设置中调整"最多保留"数量，或清除历史记录
        """
        
        help_text.insert('1.0', content)
        help_text.config(state='disabled')
        
        btn_frame = ttk.Frame(help_window, padding="15")
        btn_frame.grid(row=1, column=0, sticky='ew')
        ttk.Button(btn_frame, text="关闭", command=help_window.destroy, width=12).pack(side=tk.RIGHT)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = f"""
{APP_TITLE}

一个专业的 YAML 文件批量翻译工具

特性:
• 支持多线程并发翻译
• 智能上下文翻译
• 自动备份和恢复
• 翻译历史记录
• 丰富的配置选项
• 快捷键支持
• 主题切换

技术栈:
• Python 3
• Tkinter GUI
• DeepSeek AI API

版本: {VERSION}
        """
        messagebox.showinfo("关于", about_text)
    
    # ==================== 备份管理 ====================
    
    def restore_all_backups(self):
        """恢复所有备份"""
        if not messagebox.askyesno("确认", "确定要恢复所有备份文件吗？\n这将覆盖当前的文件。"):
            return
        
        folder = filedialog.askdirectory(title="选择要恢复备份的文件夹")
        if not folder:
            return
        
        restored = 0
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.backup'):
                    backup_path = os.path.join(root, file)
                    original_path = backup_path[:-7]  # 移除 .backup
                    
                    try:
                        shutil.copy2(backup_path, original_path)
                        restored += 1
                    except Exception as e:
                        self.log_message(f"[ERROR] 恢复失败: {file} - {e}")
        
        if restored > 0:
            messagebox.showinfo("完成", f"成功恢复 {restored} 个文件")
            self.log_message(f"[SUCCESS] 恢复了 {restored} 个备份文件")
        else:
            messagebox.showinfo("提示", "未找到备份文件")
    
    def cleanup_all_backups(self):
        """清理所有备份"""
        if not messagebox.askyesno("确认", "确定要删除所有备份文件吗？\n此操作不可恢复！"):
            return
        
        folder = filedialog.askdirectory(title="选择要清理备份的文件夹")
        if not folder:
            return
        
        deleted = 0
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.backup'):
                    backup_path = os.path.join(root, file)
                    try:
                        os.remove(backup_path)
                        deleted += 1
                    except Exception as e:
                        self.log_message(f"[ERROR] 删除失败: {file} - {e}")
        
        if deleted > 0:
            messagebox.showinfo("完成", f"成功删除 {deleted} 个备份文件")
            self.log_message(f"[SUCCESS] 删除了 {deleted} 个备份文件")
        else:
            messagebox.showinfo("提示", "未找到备份文件")
    
    def view_backups(self):
        """查看备份文件"""
        folder = filedialog.askdirectory(title="选择要查看备份的文件夹")
        if not folder:
            return
        
        backups = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.backup'):
                    backup_path = os.path.join(root, file)
                    backups.append(backup_path)
        
        if not backups:
            messagebox.showinfo("提示", "未找到备份文件")
            return
        
        # 显示备份列表窗口
        backup_window = tk.Toplevel(self.root)
        backup_window.title("备份文件列表")
        backup_window.geometry("700x500")
        backup_window.transient(self.root)
        
        backup_window.rowconfigure(1, weight=1)
        backup_window.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(backup_window, padding="15 15 15 10")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text=f"找到 {len(backups)} 个备份文件", 
                 style='Title.TLabel').pack(anchor=tk.W)
        
        # 列表
        list_frame = ttk.Frame(backup_window, padding="0 0 15 10")
        list_frame.grid(row=1, column=0, sticky='nsew', padx=15)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        backup_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        backup_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=backup_listbox.yview)
        
        for backup in backups:
            backup_listbox.insert(tk.END, os.path.relpath(backup, folder))
        
        # 按钮
        btn_frame = ttk.Frame(backup_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def restore_selected():
            selection = backup_listbox.curselection()
            if not selection:
                messagebox.showwarning("警告", "请选择要恢复的备份", parent=backup_window)
                return
            
            restored = 0
            for idx in selection:
                backup_path = backups[idx]
                original_path = backup_path[:-7]
                
                try:
                    shutil.copy2(backup_path, original_path)
                    restored += 1
                except Exception as e:
                    self.log_message(f"[ERROR] 恢复失败: {os.path.basename(backup_path)} - {e}")
            
            messagebox.showinfo("完成", f"成功恢复 {restored} 个文件", parent=backup_window)
        
        ttk.Button(btn_frame, text="恢复选中", command=restore_selected, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=backup_window.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    # ==================== 其他功能 ====================
    
    def install_dnd(self):
        """安装 tkinterdnd2"""
        install_window = tk.Toplevel(self.root)
        install_window.title("安装拖拽支持")
        install_window.geometry("400x200")
        install_window.resizable(False, False)
        install_window.transient(self.root)
        install_window.grab_set()
        
        install_window.rowconfigure(1, weight=1)
        install_window.columnconfigure(0, weight=1)
        
        # 标题
        ttk.Label(install_window, text="安装 tkinterdnd2", 
                 style='Title.TLabel', padding="20 20 20 10").grid(row=0, column=0)
        
        # 进度
        progress_frame = ttk.Frame(install_window, padding="20")
        progress_frame.grid(row=1, column=0, sticky='nsew')
        
        progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        status_label = ttk.Label(progress_frame, text="正在安装，请稍候...")
        status_label.pack()
        
        progress_bar.start()
        
        def do_install():
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "tkinterdnd2"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    # 安装成功
                    install_window.after(0, lambda: on_success())
                else:
                    # 安装失败
                    install_window.after(0, lambda: on_failure(result.stderr))
                    
            except Exception as e:
                install_window.after(0, lambda: on_failure(str(e)))
        
        def on_success():
            progress_bar.stop()
            status_label.config(text="✅ 安装成功！")
            
            btn_frame = ttk.Frame(progress_frame)
            btn_frame.pack(pady=10)
            
            ttk.Label(btn_frame, text="请重启程序以启用拖拽功能", 
                     foreground='green').pack(pady=10)
            ttk.Button(btn_frame, text="确定", 
                      command=install_window.destroy, width=12).pack()
        
        def on_failure(error):
            progress_bar.stop()
            install_window.destroy()
            
            # 显示失败对话框
            fail_window = tk.Toplevel(self.root)
            fail_window.title("安装失败")
            fail_window.geometry("450x250")
            fail_window.transient(self.root)
            fail_window.grab_set()
            
            fail_window.rowconfigure(1, weight=1)
            fail_window.columnconfigure(0, weight=1)
            
            ttk.Label(fail_window, text="⚠️ 安装失败", 
                     style='Title.TLabel', padding="20").grid(row=0, column=0)
            
            msg_frame = ttk.Frame(fail_window, padding="20")
            msg_frame.grid(row=1, column=0, sticky='nsew')
            
            ttk.Label(msg_frame, text="无法自动安装 tkinterdnd2").pack(pady=(0, 10))
            ttk.Label(msg_frame, text="✅ 翻译功能仍可正常使用", 
                     foreground='green').pack(pady=(0, 10))
            ttk.Label(msg_frame, text="（可通过按钮添加文件）", 
                     foreground='gray').pack(pady=(0, 20))
            
            btn_frame = ttk.Frame(msg_frame)
            btn_frame.pack()
            
            ttk.Button(btn_frame, text="查看手动安装指南", 
                      command=lambda: self.show_manual_install(fail_window), 
                      width=18).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="知道了", 
                      command=fail_window.destroy, 
                      width=12).pack(side=tk.LEFT, padx=5)
        
        # 在新线程中执行安装
        thread = threading.Thread(target=do_install, daemon=True)
        thread.start()
    
    def show_manual_install(self, parent=None):
        """显示手动安装指南"""
        guide_window = tk.Toplevel(parent or self.root)
        guide_window.title("手动安装指南")
        guide_window.geometry("500x350")
        guide_window.transient(parent or self.root)
        
        guide_window.rowconfigure(1, weight=1)
        guide_window.columnconfigure(0, weight=1)
        
        ttk.Label(guide_window, text="手动安装 tkinterdnd2", 
                 style='Title.TLabel', padding="20").grid(row=0, column=0)
        
        text_frame = ttk.Frame(guide_window, padding="20")
        text_frame.grid(row=1, column=0, sticky='nsew')
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        guide_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, 
                                               font=('Microsoft YaHei UI', 9))
        guide_text.grid(row=0, column=0, sticky='nsew')
        
        content = """方法1: 使用 pip (推荐)

1. 打开命令提示符 (Windows) 或终端 (Mac/Linux)
2. 输入以下命令:

   pip install tkinterdnd2

3. 等待安装完成
4. 重启本程序


方法2: 如果 pip 失败

1. 以管理员身份运行命令提示符
2. 重新执行上述命令


方法3: 使用 Python 模块

1. 打开命令提示符
2. 输入:

   python -m pip install tkinterdnd2


常见问题:

Q: 提示"pip不是内部或外部命令"
A: 需要先安装 Python 并将其添加到系统路径

Q: 安装后仍无法使用
A: 确保重启了程序

Q: 权限错误
A: 以管理员身份运行命令提示符

如有其他问题，请查看 tkinterdnd2 官方文档
        """
        
        guide_text.insert('1.0', content)
        guide_text.config(state='disabled')
        
        btn_frame = ttk.Frame(guide_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def copy_command():
            self.root.clipboard_clear()
            self.root.clipboard_append("pip install tkinterdnd2")
            messagebox.showinfo("成功", "命令已复制到剪贴板", parent=guide_window)
        
        ttk.Button(btn_frame, text="复制安装命令", command=copy_command, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=guide_window.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    def change_theme(self):
        """切换主题"""
        theme = self.theme_var.get()
        self.config_manager.config['theme'] = theme
        self.config_manager.save_config()
        self.apply_theme()
        messagebox.showinfo("提示", "主题已切换")
    
    def on_closing(self):
        """关闭窗口"""
        if self.is_translating:
            if not messagebox.askyesno("确认", "翻译正在进行中，确定要退出吗？"):
                return
        
        self.root.destroy()


# ==================== 主程序入口 ====================
def main():
    # 创建主窗口
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = TranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()