# YAML批量AI本地化工具 v1.10

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
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        DND_INSTALL_AVAILABLE = True
    except:
        pass

VERSION = "1.1"
APP_TITLE = f"YAML批量AI本地化工具 v{VERSION}"

# ==================== 平台预设库 ====================
PLATFORM_PRESETS = {
    'openai': {
        'name': 'OpenAI',
        'display_name': '🧠 OpenAI (GPT系列)',
        'url': 'https://api.openai.com/v1/chat/completions',
        'models': ['gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo'],
        'default_model': 'gpt-3.5-turbo',
        'docs_url': 'https://platform.openai.com/docs'
    },
    'deepseek': {
        'name': 'DeepSeek',
        'display_name': '🤖 DeepSeek (推荐)',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'models': ['deepseek-chat', 'deepseek-coder'],
        'default_model': 'deepseek-chat',
        'docs_url': 'https://platform.deepseek.com/docs'
    },
    'moonshot': {
        'name': 'Moonshot',
        'display_name': '🌙 Moonshot (Kimi)',
        'url': 'https://api.moonshot.cn/v1/chat/completions',
        'models': ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k'],
        'default_model': 'moonshot-v1-8k',
        'docs_url': 'https://platform.moonshot.cn/docs'
    },
    'zhipu': {
        'name': 'ZhipuAI',
        'display_name': '🧩 智谱AI (GLM)',
        'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
        'models': ['glm-4', 'glm-4v', 'glm-3-turbo'],
        'default_model': 'glm-4',
        'docs_url': 'https://open.bigmodel.cn/dev/api'
    },
    'qwen': {
        'name': 'Qwen',
        'display_name': '☁️ 通义千问',
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        'models': ['qwen-turbo', 'qwen-plus', 'qwen-max'],
        'default_model': 'qwen-turbo',
        'docs_url': 'https://help.aliyun.com/zh/dashscope/'
    },
    'claude': {
        'name': 'Claude',
        'display_name': '🧠 Claude (Anthropic)',
        'url': 'https://api.anthropic.com/v1/messages',
        'models': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
        'default_model': 'claude-3-sonnet',
        'docs_url': 'https://docs.anthropic.com/claude/reference/getting-started-with-the-api'
    },
    'mistral': {
        'name': 'Mistral',
        'display_name': '🎯 Mistral AI',
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'models': ['mistral-large', 'mistral-medium', 'mistral-small'],
        'default_model': 'mistral-medium',
        'docs_url': 'https://docs.mistral.ai/'
    },
    'groq': {
        'name': 'Groq',
        'display_name': '⚡ Groq',
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'models': ['mixtral-8x7b-32768', 'llama2-70b-4096', 'gemma-7b-it'],
        'default_model': 'mixtral-8x7b-32768',
        'docs_url': 'https://console.groq.com/docs/speech-text'
    },
    'perplexity': {
        'name': 'Perplexity',
        'display_name': '🔍 Perplexity AI',
        'url': 'https://api.perplexity.ai/chat/completions',
        'models': ['pplx-7b-online', 'pplx-70b-online', 'pplx-7b', 'pplx-70b'],
        'default_model': 'pplx-7b-online',
        'docs_url': 'https://docs.perplexity.ai/'
    },
    'cohere': {
        'name': 'Cohere',
        'display_name': '📝 Cohere',
        'url': 'https://api.cohere.ai/v1/chat',
        'models': ['command-r-plus', 'command-r', 'command-light'],
        'default_model': 'command-r',
        'docs_url': 'https://docs.cohere.com/docs/chat-api'
    },
    'xai': {
        'name': 'xAI',
        'display_name': '✨ xAI (Grok)',
        'url': 'https://api.x.ai/v1/chat/completions',
        'models': ['grok-beta'],
        'default_model': 'grok-beta',
        'docs_url': 'https://docs.x.ai/'
    },
    'fireworks': {
        'name': 'Fireworks',
        'display_name': '🔥 Fireworks AI',
        'url': 'https://api.fireworks.ai/inference/v1/chat/completions',
        'models': ['llama-v2-7b-chat', 'llama-v2-13b-chat', 'mistral-7b-instruct'],
        'default_model': 'llama-v2-13b-chat',
        'docs_url': 'https://docs.fireworks.ai/'
    },
    'ai21': {
        'name': 'AI21',
        'display_name': '🎨 AI21 Labs',
        'url': 'https://api.ai21.com/studio/v1/chat/completions',
        'models': ['j2-ultra', 'j2-mid', 'j2-light'],
        'default_model': 'j2-mid',
        'docs_url': 'https://docs.ai21.com/'
    },
    'makersuite': {
        'name': 'Google Makersuite',
        'display_name': '🔮 Google Makersuite',
        'url': 'https://generativelanguage.googleapis.com/v1beta/models/generateContent',
        'models': ['gemini-pro', 'gemini-pro-vision'],
        'default_model': 'gemini-pro',
        'docs_url': 'https://ai.google.dev/'
    },
    'nanogpt': {
        'name': 'NanoGPT',
        'display_name': '⚙️ NanoGPT',
        'url': 'https://nano-gpt.com/api/v1/chat/completions',
        'models': ['nano-gpt', 'nano-gpt-large'],
        'default_model': 'nano-gpt',
        'docs_url': 'https://nano-gpt.com/docs'
    },
    'electronhub': {
        'name': 'ElectronHub',
        'display_name': '⚛️ ElectronHub',
        'url': 'https://api.electronhub.ai/v1/chat/completions',
        'models': ['electron-v1', 'electron-turbo'],
        'default_model': 'electron-v1',
        'docs_url': 'https://electronhub.ai/docs'
    },
    'aimlapi': {
        'name': 'AIML API',
        'display_name': '🤖 AIML API',
        'url': 'https://api.aimlapi.com/v1/chat/completions',
        'models': ['gpt-4', 'gpt-3.5-turbo', 'claude-2'],
        'default_model': 'gpt-3.5-turbo',
        'docs_url': 'https://aimlapi.com/docs'
    },
    'pollinations': {
        'name': 'Pollinations',
        'display_name': '🌸 Pollinations',
        'url': 'https://text.pollinations.ai/openai/v1/chat/completions',
        'models': ['openai', 'mistral', 'neural-chat'],
        'default_model': 'openai',
        'docs_url': 'https://pollinations.ai/'
    },
    'custom': {
        'name': 'Custom',
        'display_name': '⚙️ 自定义API',
        'url': '',
        'models': [],
        'default_model': '',
        'docs_url': ''
    }
}

# 默认提示词
DEFAULT_PROMPT = """请将以下英文翻译为中文,如果已经为中文则不翻译。

重要规则：
1. 只返回翻译结果，不要包含其他内容
2. 翻译结果中尽量避免使用双引号和单引号
3. 如果必须使用引号，用中文引号「」『』代替
4. 避免在翻译结果中使用英文冒号:，使用中文冒号：代替"""

# ==================== 核心翻译器 ====================
class UniversalTranslator:
    """通用翻译器 - 支持多平台API"""
    
    def __init__(self, api_config):
        """
        Args:
            api_config: {
                'platform': 'deepseek',
                'api_key': 'sk-xxx',
                'model': 'deepseek-chat',
                'url': 'https://...',
                'temperature': 0.3,
                'max_tokens': 1000,
                'custom_prompt': '...'
            }
        """
        self.config = api_config
        self.platform = api_config.get('platform', 'deepseek')
        self.api_key = api_config['api_key']
        self.model = api_config.get('model', 'deepseek-chat')
        self.base_url = api_config.get('url', PLATFORM_PRESETS.get(self.platform, {}).get('url', ''))
        self.temperature = api_config.get('temperature', 0.3)
        self.max_tokens = api_config.get('max_tokens', 1000)
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
        """翻译文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 从配置中获取自定义提示词，如果没有则使用默认提示词
        base_prompt = self.config.get('custom_prompt', DEFAULT_PROMPT)

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
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
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

    
    def test_connection(self):
        """测试API连接"""
        try:
            test_text = "Hello"
            start_time = time.time()
            result, error = self.translate(test_text, timeout=10)
            elapsed = time.time() - start_time
            
            if error:
                return False, f"翻译失败: {error}"
            
            if result and result != test_text:
                # 检查响应时间
                if elapsed > 5:
                    return True, f"测试成功但响应较慢: \"{test_text}\" → \"{result}\" (耗时 {elapsed:.2f}秒)"
                else:
                    return True, f"测试成功: \"{test_text}\" → \"{result}\" (耗时 {elapsed:.2f}秒)"
            else:
                return False, "API响应异常"
                
        except Exception as e:
            return False, f"连接失败: {str(e)}"


class YamlTranslatorCore:
    """YAML翻译核心"""
    
    def __init__(self, api_config, max_threads=4, progress_callback=None, 
                 log_callback=None, translation_callback=None, config=None):
        self.translator = UniversalTranslator(api_config)
        self.max_threads = max_threads
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.translation_callback = translation_callback
        self.config = config or {}
        self.stop_flag = False
        self.translation_records = []  # 记录翻译详情
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'skipped_translations': 0,
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
    
    def record_translation(self, file_path, field_type, original, translated, status):
        """记录翻译详情"""
        self.translation_records.append({
            'file': file_path,
            'field': field_type,
            'original': original,
            'translated': translated,
            'status': status,  # 'success', 'failed', 'skipped'
            'timestamp': datetime.now().isoformat()
        })
    
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
    
    def get_output_path(self, original_path, base_folder):
        """获取输出文件路径"""
        output_mode = self.config.get('output_mode', 'export')
        
        if output_mode == 'overwrite':
            return original_path
        
        # 导出模式
        output_folder = self.config.get('output_folder', '')
        if not output_folder:
            output_folder = os.path.join(os.path.dirname(original_path), 'translated')
        
        keep_structure = self.config.get('keep_structure', True)
        add_tag = self.config.get('add_language_tag', True)
        tag = self.config.get('language_tag', '_zh_CN')
        tag_position = self.config.get('tag_position', 'end')
        
        if keep_structure:
            # 保持目录结构
            rel_path = os.path.relpath(original_path, base_folder)
            output_path = os.path.join(output_folder, rel_path)
        else:
            # 平铺
            filename = os.path.basename(original_path)
            output_path = os.path.join(output_folder, filename)
        
        # 添加语言标识
        if add_tag and tag:
            dir_name = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            name, ext = os.path.splitext(filename)
            
            if tag_position == 'before_ext':
                # 扩展名前: config.zh_CN.yml
                new_filename = f"{name}.{tag.lstrip('_')}{ext}"
            else:
                # 文件名末尾: config_zh_CN.yml
                new_filename = f"{name}{tag}{ext}"
            
            output_path = os.path.join(dir_name, new_filename)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        return output_path
    
    def process_yaml_file(self, file_path, base_folder):
        """处理单个YAML文件"""
        if self.stop_flag:
            return
        
        file_name = os.path.basename(file_path)
        self.log(f"处理文件: {file_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            translated_lines = []
            file_translations = 0
            file_skipped = 0
            max_retries = self.config.get('max_retries', 3) if self.config.get('enable_retry', True) else 1
            retry_delay = self.config.get('retry_delay', 5)
            timeout = self.config.get('api_timeout', 30)
            
            # 双语输出配置
            enable_bilingual = self.config.get('enable_bilingual', False)
            bilingual_separator = self.config.get('bilingual_separator', ' | ')
            bilingual_order = self.config.get('bilingual_order', 'cn_first')
            
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
                    
                    # 跳过中文
                    if self.config.get('skip_chinese', True) and self.contains_chinese(value):
                        translated_lines.append(line)
                        file_skipped += 1
                        self.record_translation(file_path, key, value, value, 'skipped')
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
                        self.record_translation(file_path, key, value, value, 'failed')
                    else:
                        if translated_value != value:
                            file_translations += 1
                            
                            # 实时回调翻译结果
                            if self.translation_callback:
                                self.translation_callback(value, translated_value)
                        
                        # === 双语输出处理 ===
                        if enable_bilingual and translated_value != value:
                            if bilingual_order == 'cn_first':
                                final_value = f"{translated_value}{bilingual_separator}{value}"
                            else:
                                final_value = f"{value}{bilingual_separator}{translated_value}"
                        else:
                            final_value = translated_value
                        # === 双语输出结束 ===
                        
                        escaped_value = self.translator.escape_yaml_value(final_value)
                        
                        if escaped_value.startswith("'") or escaped_value.startswith('"'):
                            translated_line = f"{' ' * leading_spaces}{key}: {escaped_value}\n"
                        else:
                            translated_line = f"{' ' * leading_spaces}{key}: {escaped_value}\n"
                        
                        translated_lines.append(translated_line)
                        self.stats['total_translations'] += 1
                        self.stats['successful_translations'] += 1
                        self.record_translation(file_path, key, value, final_value, 'success')
                else:
                    translated_lines.append(line)

            # 获取输出路径
            output_path = self.get_output_path(file_path, base_folder)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(translated_lines)

            self.stats['processed_files'] += 1
            self.stats['skipped_translations'] += file_skipped
            
            if output_path != file_path:
                self.log(f"✓ 完成: {file_name} → {os.path.basename(output_path)} (翻译 {file_translations} 项)", "SUCCESS")
            else:
                self.log(f"✓ 完成: {file_name} (翻译 {file_translations} 项)", "SUCCESS")
            
        except Exception as e:
            self.log(f"✗ 处理失败 {file_name}: {e}", "ERROR")
    
    def translate_files(self, file_paths, base_folder=None):
        """翻译文件列表"""
        self.stop_flag = False
        self.translation_records = []
        self.stats = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'skipped_translations': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        # 确定基准文件夹
        if not base_folder:
            if len(file_paths) == 1:
                base_folder = os.path.dirname(file_paths[0])
            else:
                base_folder = os.path.commonpath(file_paths)
        
        self.log(f"开始翻译 {len(file_paths)} 个文件")
        self.log(f"线程数: {self.max_threads}")
        self.log(f"输出模式: {'覆盖源文件' if self.config.get('output_mode') == 'overwrite' else '导出到新文件夹'}")
        if self.config.get('enable_bilingual', False):
            order_text = "中文在前" if self.config.get('bilingual_order') == 'cn_first' else "原文在前"
            sep = self.config.get('bilingual_separator', ' | ')
            self.log(f"双语输出: 已启用 ({order_text}，分隔符: '{sep}')")
        self.log("=" * 60)
        
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = []
            for file_path in file_paths:
                if self.stop_flag:
                    break
                future = executor.submit(self.process_yaml_file, file_path, base_folder)
                futures.append(future)
            
            for i, future in enumerate(futures):
                if self.stop_flag:
                    break
                future.result()
                self.update_progress(i + 1, len(file_paths), 
                                    f"处理中: {i + 1}/{len(file_paths)}")
        
        self.stats['end_time'] = datetime.now()
        elapsed = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        self.stats['duration'] = elapsed
        
        self.log("=" * 60)
        self.log(f"翻译完成！", "SUCCESS")
        self.log(f"处理文件: {self.stats['processed_files']}/{self.stats['total_files']}")
        self.log(f"翻译成功: {self.stats['successful_translations']}")
        self.log(f"跳过项: {self.stats['skipped_translations']}")
        self.log(f"翻译失败: {self.stats['failed_translations']}")
        self.log(f"耗时: {elapsed:.2f}秒")
        
        return self.stats
    
    def stop(self):
        """停止翻译"""
        self.stop_flag = True


# ==================== 配置管理器 ====================
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file="translator_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置"""
        default_config = {
            # API Keys
            'api_keys': [],
            'current_key_id': None,
            
            # 翻译设置
            'max_threads': 4,
            'skip_chinese': True,
            'api_timeout': 30,
            'enable_retry': True,
            'max_retries': 3,
            'retry_delay': 5,
            
            # 输出设置
            'output_mode': 'export',
            'output_folder': '',
            'keep_structure': True,
            'add_language_tag': True,
            'language_tag': '_zh_CN',
            'tag_position': 'end',
            'generate_report': True,
            'report_path': 'auto',
            'conflict_handling': 'ask',
            
            # 双语输出设置
            'enable_bilingual': False,
            'bilingual_separator': ' | ',
            'bilingual_order': 'cn_first',  # 'cn_first' 或 'en_first'
            
            # 语言标识预设
            'preset_tags': [
                {'tag': '_zh_CN', 'label': '简体中文'},
                {'tag': '_zh_TW', 'label': '繁体中文'},
                {'tag': '_cn', 'label': '中文简写'},
                {'tag': '_chinese', 'label': '英文标识'},
                {'tag': '_translated', 'label': '已翻译'}
            ],
            'tag_history': [],
            'max_tag_history': 10,
            
            # UI设置
            'theme': 'light',
            'display_mode': 'simple',
            'sort_mode': 'add_order',
            
            # 日志设置
            'log_level': 'standard',
            'auto_save_log': False,
            'log_path': '',
            
            # 历史记录
            'save_history': True,
            'max_history': 100,
            'history': [],
            
            # 快捷键
            'shortcuts': {
                'add_files': 'Ctrl+O',
                'add_folder': 'Ctrl+D',
                'start': 'F5',
                'stop': 'Escape',
                'clear_log': 'Ctrl+L',
                'remove': 'Delete',
                'settings': 'Ctrl+comma'
            },
            
            # 代理设置
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
    
    def add_api_key(self, key_data):
        """添加API Key"""
        key_id = str(int(time.time() * 1000))
        key_data['id'] = key_id
        key_data['created'] = datetime.now().isoformat()
        key_data['last_used'] = None
        key_data['use_count'] = 0
        
        self.config['api_keys'].append(key_data)
        self.save_config()
        return key_id
    
    def update_api_key(self, key_id, key_data):
        """更新API Key"""
        for i, key in enumerate(self.config['api_keys']):
            if key['id'] == key_id:
                key_data['id'] = key_id
                key_data['created'] = key.get('created', datetime.now().isoformat())
                self.config['api_keys'][i] = key_data
                self.save_config()
                return True
        return False
    
    def remove_api_key(self, key_id):
        """删除API Key"""
        self.config['api_keys'] = [k for k in self.config['api_keys'] if k['id'] != key_id]
        if self.config['current_key_id'] == key_id:
            self.config['current_key_id'] = None
        self.save_config()
    
    def get_api_keys(self):
        """获取所有API Keys"""
        return self.config['api_keys']
    
    def get_current_key(self):
        """获取当前使用的Key"""
        key_id = self.config.get('current_key_id')
        if key_id:
            for key in self.config['api_keys']:
                if key['id'] == key_id:
                    return key
        return None
    
    def set_current_key(self, key_id):
        """设置当前使用的Key"""
        self.config['current_key_id'] = key_id
        self.save_config()
    
    def add_tag_to_history(self, tag):
        """添加语言标识到历史"""
        # 移除已存在的
        self.config['tag_history'] = [
            item for item in self.config.get('tag_history', [])
            if item['tag'] != tag
        ]
        
        # 添加到开头
        self.config['tag_history'].insert(0, {
            'tag': tag,
            'last_used': datetime.now().isoformat()
        })
        
        # 限制数量
        max_history = self.config.get('max_tag_history', 10)
        self.config['tag_history'] = self.config['tag_history'][:max_history]
        
        self.save_config()
    
    def add_history(self, stats, files):
        """添加翻译历史"""
        if not self.config.get('save_history', True):
            return
        
        history_item = {
            'timestamp': datetime.now().isoformat(),
            'total_files': stats['total_files'],
            'processed_files': stats['processed_files'],
            'successful_translations': stats['successful_translations'],
            'failed_translations': stats['failed_translations'],
            'skipped_translations': stats.get('skipped_translations', 0),
            'duration': stats.get('duration', 0),
            'files': [os.path.basename(f) for f in files[:10]]
        }
        
        if 'history' not in self.config:
            self.config['history'] = []
        
        self.config['history'].insert(0, history_item)
        
        max_history = self.config.get('max_history', 100)
        self.config['history'] = self.config['history'][:max_history]
        
        self.save_config()


# ==================== 报告生成器 ====================
class ReportGenerator:
    """翻译报告生成器"""
    
    @staticmethod
    def generate_html_report(stats, translation_records, output_path, api_config):
        """生成HTML对比报告"""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>翻译对比报告</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
            margin: 5px 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
        }
        .file-section {
            background: white;
            margin-bottom: 15px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .file-header {
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .file-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .file-info {
            color: #666;
            font-size: 13px;
        }
        .translation-item {
            border-left: 3px solid #4CAF50;
            padding: 12px;
            margin: 10px 0;
            background: #fafafa;
            border-radius: 4px;
        }
        .translation-item.failed {
            border-left-color: #f44336;
            background: #ffebee;
        }
        .translation-item.skipped {
            border-left-color: #FF9800;
            background: #fff3e0;
        }
        .original {
            color: #666;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .translated {
            color: #000;
            font-weight: 500;
            font-size: 14px;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }
        .status-success {
            background: #4CAF50;
            color: white;
        }
        .status-failed {
            background: #f44336;
            color: white;
        }
        .status-skipped {
            background: #FF9800;
            color: white;
        }
        .footer {
            text-align: center;
            color: #999;
            margin-top: 30px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌐 YAML翻译对比报告</h1>
        <p>生成时间: {timestamp}</p>
        <p>使用平台: {platform} ({model})</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{total_files}</div>
            <div class="stat-label">处理文件</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{successful}</div>
            <div class="stat-label">翻译成功</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{skipped}</div>
            <div class="stat-label">跳过项</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{failed}</div>
            <div class="stat-label">失败项</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{duration}</div>
            <div class="stat-label">总耗时</div>
        </div>
    </div>
    
    {file_sections}
    
    <div class="footer">
        <p>由 {app_name} 生成</p>
    </div>
</body>
</html>"""
        
        # 按文件分组翻译记录
        files_data = {}
        for record in translation_records:
            file_path = record['file']
            if file_path not in files_data:
                files_data[file_path] = []
            files_data[file_path].append(record)
        
        # 生成文件区块
        file_sections_html = ""
        for file_path, records in files_data.items():
            file_name = os.path.basename(file_path)
            
            success_count = len([r for r in records if r['status'] == 'success'])
            skipped_count = len([r for r in records if r['status'] == 'skipped'])
            failed_count = len([r for r in records if r['status'] == 'failed'])
            
            items_html = ""
            for record in records:
                status_class = record['status']
                status_text = {'success': '成功', 'failed': '失败', 'skipped': '跳过'}[status_class]
                status_badge = f'<span class="status-badge status-{status_class}">{status_text}</span>'
                
                items_html += f"""
                <div class="translation-item {status_class}">
                    <div class="original">{record['field']}: "{record['original']}" {status_badge}</div>
                    <div class="translated">→ "{record['translated']}"</div>
                </div>
                """
            
            file_sections_html += f"""
            <div class="file-section">
                <div class="file-header">
                    <div class="file-title">📄 {file_name}</div>
                    <div class="file-info">路径: {file_path}</div>
                    <div class="file-info">统计: 成功 {success_count} | 跳过 {skipped_count} | 失败 {failed_count}</div>
                </div>
                {items_html}
            </div>
            """
        
        # 填充模板
        duration_str = f"{stats.get('duration', 0):.1f}秒"
        
        html_content = html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            platform=PLATFORM_PRESETS.get(api_config.get('platform', 'deepseek'), {}).get('name', '未知'),
            model=api_config.get('model', ''),
            total_files=stats['total_files'],
            successful=stats['successful_translations'],
            skipped=stats.get('skipped_translations', 0),
            failed=stats['failed_translations'],
            duration=duration_str,
            file_sections=file_sections_html,
            app_name=APP_TITLE
        )
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


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
        self.current_base_folder = None
        
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
        
        self.style.configure('Accent.TButton', font=('Microsoft YaHei UI', 9, 'bold'))
        self.style.configure('Title.TLabel', font=('Microsoft YaHei UI', 11, 'bold'))
        self.style.configure('TLabelframe', font=('Microsoft YaHei UI', 9))
        self.style.configure('TLabelframe.Label', font=('Microsoft YaHei UI', 9, 'bold'))
        
    def apply_theme(self):
        """应用主题"""
        theme = self.config_manager.config.get('theme', 'light')
        
        if theme == 'dark':
            bg = '#2b2b2b'
            fg = '#e0e0e0'
            select_bg = '#4a9eff'
            
            self.root.configure(bg=bg)
            self.style.configure('TFrame', background=bg)
            self.style.configure('TLabel', background=bg, foreground=fg)
            self.style.configure('TLabelframe', background=bg, foreground=fg)
            self.style.configure('TLabelframe.Label', background=bg, foreground=fg)
            
            if hasattr(self, 'log_text'):
                self.log_text.configure(bg='#1e1e1e', fg=fg, insertbackground=fg)
            if hasattr(self, 'file_listbox'):
                self.file_listbox.configure(bg='#1e1e1e', fg=fg, selectbackground=select_bg)
        else:
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
        
        ttk.Label(statusbar, text=f"v{VERSION}", font=('Microsoft YaHei UI', 8)).pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.api_status_label = ttk.Label(statusbar, text="🔴 API未配置", 
                                         font=('Microsoft YaHei UI', 8))
        self.api_status_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        dnd_status = "🟢 拖拽: 可用" if HAS_DND else "🔴 拖拽: 不可用"
        ttk.Label(statusbar, text=dnd_status, font=('Microsoft YaHei UI', 8)).pack(
            side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.file_count_status = ttk.Label(statusbar, text="文件: 0", 
                                          font=('Microsoft YaHei UI', 8))
        self.file_count_status.pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.status_text = ttk.Label(statusbar, text="就绪", font=('Microsoft YaHei UI', 8))
        self.status_text.pack(side=tk.LEFT, padx=5)
    
    def create_main_content(self):
        """创建主内容区域"""
        main_container = ttk.Frame(self.root)
        main_container.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)
        
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        # 左侧面板
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
        
        self.file_listbox.bind("<Button-3>", self.show_context_menu)
        
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
        
        self.file_count_label = ttk.Label(left_panel, text="已选择: 0 个文件",
                                         font=('Microsoft YaHei UI', 9, 'bold'),
                                         foreground='#0066cc')
        self.file_count_label.grid(row=3, column=0, sticky='w')
        
        # 右侧面板
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
        
        self.log_text.tag_config("INFO", foreground="#333333")
        self.log_text.tag_config("SUCCESS", foreground="#008000", font=('Consolas', 9, 'bold'))
        self.log_text.tag_config("WARNING", foreground="#FF8C00", font=('Consolas', 9, 'bold'))
        self.log_text.tag_config("ERROR", foreground="#DC143C", font=('Consolas', 9, 'bold'))
        
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
        bottom_frame.columnconfigure(2, weight=1)
        
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
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))
        
        self.output_btn = ttk.Button(
            left_btns,
            text="📂 输出到...",
            command=self.show_output_quick_settings,
            width=15
        )
        self.output_btn.grid(row=0, column=2)
        
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
        self.root.bind('<Control-o>', lambda e: self.add_files())
        self.root.bind('<Control-d>', lambda e: self.add_folder())
        self.root.bind('<F5>', lambda e: self.start_translation())
        self.root.bind('<Escape>', lambda e: self.stop_translation())
        self.root.bind('<Control-l>', lambda e: self.clear_log())
        self.root.bind('<Delete>', lambda e: self.remove_selected())
        self.root.bind('<Control-comma>', lambda e: self.show_settings())
    
    def load_settings(self):
        """加载设置"""
        keys = self.config_manager.get_api_keys()
        if keys:
            key_names = []
            for k in keys:
                platform_name = PLATFORM_PRESETS.get(k.get('platform', 'custom'), {}).get('name', '自定义')
                key_names.append(f"{k['name']} ({platform_name})")
            
            self.key_combo['values'] = key_names
            
            current_key = self.config_manager.get_current_key()
            if current_key:
                for i, k in enumerate(keys):
                    if k['id'] == current_key['id']:
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
    
    # ==================== 事件处理 ====================
    
    def on_key_selected(self, event):
        """选择API Key"""
        index = self.key_combo.current()
        if index >= 0:
            keys = self.config_manager.get_api_keys()
            self.config_manager.set_current_key(keys[index]['id'])
            self.api_status_label.config(text="🟢 API已连接")
    
    def on_drop(self, event):
        """处理拖拽"""
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
            menu.add_command(label="📂 打开文件位置", command=self.open_file_location)
            menu.add_command(label="📝 用编辑器打开", command=self.open_with_editor)
            menu.add_command(label="📋 复制文件路径", command=self.copy_file_path)
            menu.add_command(label="📋 复制文件名", command=self.copy_file_name)
            menu.add_separator()
            menu.add_command(label="❌ 从列表移除", command=self.remove_selected)
        else:
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
            return
        
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
            else:
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
            core = YamlTranslatorCore({'api_key': '', 'platform': 'deepseek'}, 1)
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
        
        # 更新基准文件夹
        if self.file_queue:
            if len(self.file_queue) == 1:
                self.current_base_folder = os.path.dirname(self.file_queue[0])
            else:
                self.current_base_folder = os.path.commonpath(self.file_queue)
    
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
        self.current_base_folder = None
        self.update_file_count()
        self.log_message("[INFO] 已清空文件列表")
    
    def remove_selected(self):
        """移除选中的文件"""
        selected = self.file_listbox.curselection()
        for index in reversed(selected):
            self.file_listbox.delete(index)
            self.file_queue.pop(index)
        self.update_file_count()
        
        # 更新基准文件夹
        if self.file_queue:
            if len(self.file_queue) == 1:
                self.current_base_folder = os.path.dirname(self.file_queue[0])
            else:
                self.current_base_folder = os.path.commonpath(self.file_queue)
        else:
            self.current_base_folder = None
    
    def update_file_count(self):
        """更新文件计数"""
        count = len(self.file_queue)
        self.file_count_label.config(text=f"已选择: {count} 个文件")
        self.file_count_status.config(text=f"文件: {count}")
    
    def log_message(self, message):
        """显示日志"""
        self.log_text.insert(tk.END, message + "\n")
        
        if "[ERROR]" in message:
            tag = "ERROR"
        elif "[WARNING]" in message:
            tag = "WARNING"
        elif "[SUCCESS]" in message or "✓" in message:
            tag = "SUCCESS"
        else:
            tag = "INFO"
        
        last_line = self.log_text.index("end-1c linestart")
        self.log_text.tag_add(tag, last_line, "end-1c")
        
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
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
                   f"跳过: {stats.get('skipped_translations', 0)} | "
                   f"失败: {stats['failed_translations']}")
            self.stats_label.config(text=text)
    
    def on_translation(self, original, translated):
        """翻译回调 - 显示实时翻译"""
        self.log_message(f'[INFO] "{original[:30]}..." → "{translated[:30]}..."')
    
    def start_translation(self):
        """开始翻译"""
        if not self.file_queue:
            messagebox.showwarning("警告", "请先添加要翻译的文件")
            return
        
        current_key = self.config_manager.get_current_key()
        if not current_key:
            messagebox.showwarning("警告", "请先选择或添加 API Key")
            return
        
        # 检查输出设置
        output_mode = self.config_manager.config.get('output_mode', 'export')
        if output_mode == 'export':
            output_folder = self.config_manager.config.get('output_folder', '')
            if not output_folder:
                # 询问输出文件夹
                if not messagebox.askyesno("提示", 
                    "未设置输出文件夹，将使用默认位置\n（源文件夹下的 'translated' 文件夹）\n\n是否继续？"):
                    return
        elif output_mode == 'overwrite':
            if not messagebox.askyesno("⚠️ 警告", 
                "覆盖模式将直接修改源文件！\n虽然会创建备份，但仍有风险。\n\n确定要继续吗？"):
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
        
        # 保存语言标识到历史
        if self.config_manager.config.get('add_language_tag', False):
            tag = self.config_manager.config.get('language_tag', '_zh_CN')
            if tag:
                self.config_manager.add_tag_to_history(tag)
        
        self.is_translating = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.status_text.config(text="翻译中...")
        
        # 在新线程中执行翻译
        def translate_thread():
            try:
                # 构建API配置
                api_config = {
                    'platform': current_key.get('platform', 'deepseek'),
                    'api_key': current_key['api_key'],
                    'model': current_key.get('model', 'deepseek-chat'),
                    'url': current_key.get('url', ''),
                    'temperature': current_key.get('temperature', 0.3),
                    'max_tokens': current_key.get('max_tokens', 1000),
                    'custom_prompt': current_key.get('custom_prompt', DEFAULT_PROMPT)
                }
                
                # 构建翻译配置
                translate_config = {
                    'skip_chinese': self.config_manager.config.get('skip_chinese', True),
                    'api_timeout': self.config_manager.config.get('api_timeout', 30),
                    'enable_retry': self.config_manager.config.get('enable_retry', True),
                    'max_retries': self.config_manager.config.get('max_retries', 3),
                    'retry_delay': self.config_manager.config.get('retry_delay', 5),
                    'output_mode': self.config_manager.config.get('output_mode', 'export'),
                    'output_folder': self.config_manager.config.get('output_folder', ''),
                    'keep_structure': self.config_manager.config.get('keep_structure', True),
                    'add_language_tag': self.config_manager.config.get('add_language_tag', True),
                    'language_tag': self.config_manager.config.get('language_tag', '_zh_CN'),
                    'tag_position': self.config_manager.config.get('tag_position', 'end'),
                    'enable_bilingual': self.config_manager.config.get('enable_bilingual', False),
                    'bilingual_separator': self.config_manager.config.get('bilingual_separator', ' | '),
                    'bilingual_order': self.config_manager.config.get('bilingual_order', 'cn_first')
                }
                
                self.translator_core = YamlTranslatorCore(
                    api_config,
                    max_threads=thread_count,
                    progress_callback=self.update_progress_ui,
                    log_callback=self.log_message,
                    translation_callback=self.on_translation,
                    config=translate_config
                )
                
                stats = self.translator_core.translate_files(self.file_queue, self.current_base_folder)
                self.update_stats(stats)
                
                # 保存历史记录
                self.config_manager.add_history(stats, self.file_queue)
                
                # 生成报告
                if self.config_manager.config.get('generate_report', True):
                    try:
                        report_path = self.config_manager.config.get('report_path', 'auto')
                        if report_path == 'auto':
                            output_folder = self.config_manager.config.get('output_folder', '')
                            if not output_folder:
                                output_folder = os.path.join(os.path.dirname(self.file_queue[0]), 'translated')
                            report_path = os.path.join(output_folder, 
                                f"translation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                        
                        report_path = ReportGenerator.generate_html_report(
                            stats, 
                            self.translator_core.translation_records,
                            report_path,
                            api_config
                        )
                        self.log_message(f"[SUCCESS] 报告已生成: {report_path}")
                        
                        # 询问是否打开报告
                        if messagebox.askyesno("完成", 
                            f"翻译完成！\n\n"
                            f"处理文件: {stats['processed_files']}/{stats['total_files']}\n"
                            f"翻译成功: {stats['successful_translations']}\n"
                            f"跳过: {stats.get('skipped_translations', 0)}\n"
                            f"失败: {stats['failed_translations']}\n"
                            f"耗时: {stats.get('duration', 0):.1f}秒\n\n"
                            f"是否打开对比报告？"):
                            webbrowser.open(f"file://{os.path.abspath(report_path)}")
                    except Exception as e:
                        self.log_message(f"[WARNING] 报告生成失败: {e}")
                else:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "翻译完成",
                        f"翻译完成！\n\n"
                        f"处理文件: {stats['processed_files']}/{stats['total_files']}\n"
                        f"翻译成功: {stats['successful_translations']}\n"
                        f"跳过: {stats.get('skipped_translations', 0)}\n"
                        f"失败: {stats['failed_translations']}\n"
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
        # 创建管理窗口
        manager_window = tk.Toplevel(self.root)
        manager_window.title("API Key 管理")
        manager_window.geometry("800x500")
        manager_window.minsize(700, 450)
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
        
        columns = ('name', 'platform', 'model', 'status')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        tree.heading('name', text='名称')
        tree.heading('platform', text='平台')
        tree.heading('model', text='模型')
        tree.heading('status', text='状态')
        
        tree.column('name', width=150)
        tree.column('platform', width=150)
        tree.column('model', width=200)
        tree.column('status', width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        def refresh_tree():
            tree.delete(*tree.get_children())
            for k in self.config_manager.get_api_keys():
                platform_name = PLATFORM_PRESETS.get(k.get('platform', 'custom'), {}).get('name', '自定义')
                tree.insert('', tk.END, values=(
                    k['name'], 
                    platform_name, 
                    k.get('model', 'N/A'),
                    '⚠ 未测试'
                ))
        
        refresh_tree()
        
        # 按钮
        btn_frame = ttk.Frame(manager_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def add_key():
            self.show_add_edit_key_dialog(manager_window, refresh_tree)
        
        def edit_key():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("警告", "请选择要编辑的 API Key", parent=manager_window)
                return
            
            item = tree.item(selection[0])
            key_name = item['values'][0]
            
            for k in self.config_manager.get_api_keys():
                if k['name'] == key_name:
                    self.show_add_edit_key_dialog(manager_window, refresh_tree, k)
                    break
        
        def remove_key():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("警告", "请选择要删除的 API Key", parent=manager_window)
                return
            
            if messagebox.askyesno("确认", "确定要删除选中的 API Key 吗？", parent=manager_window):
                for item in selection:
                    values = tree.item(item)['values']
                    for k in self.config_manager.get_api_keys():
                        if k['name'] == values[0]:
                            self.config_manager.remove_api_key(k['id'])
                            break
                
                refresh_tree()
                self.load_settings()
        
        def test_key():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("警告", "请选择要测试的 API Key", parent=manager_window)
                return
            
            item = tree.item(selection[0])
            key_name = item['values'][0]
            
            for k in self.config_manager.get_api_keys():
                if k['name'] == key_name:
                    self.test_api_key(k, manager_window, tree, selection[0])
                    break
        
        ttk.Button(btn_frame, text="➕ 添加", command=add_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✏️ 编辑", command=edit_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🧪 测试", command=test_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑️ 删除", command=remove_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=manager_window.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    def show_add_edit_key_dialog(self, parent, refresh_callback, key_data=None):
        """显示添加/编辑API Key对话框"""
        is_edit = key_data is not None
        
        dialog = tk.Toplevel(parent)
        dialog.title("编辑 API Key" if is_edit else "添加 API Key")
        dialog.geometry("550x600")
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()
        
        dialog.rowconfigure(1, weight=1)
        dialog.columnconfigure(0, weight=1)
        
        # 标题
        ttk.Label(dialog, text="编辑 API Key" if is_edit else "添加 API Key", 
                 style='Title.TLabel', padding="20 20 20 10").grid(row=0, column=0)
        
        # 表单
        form = ttk.Frame(dialog, padding="20")
        form.grid(row=1, column=0, sticky='nsew')
        
        # 名称
        ttk.Label(form, text="名称:").grid(row=0, column=0, sticky=tk.W, pady=10)
        name_var = tk.StringVar(value=key_data['name'] if is_edit else '')
        name_entry = ttk.Entry(form, textvariable=name_var, width=40)
        name_entry.grid(row=0, column=1, pady=10, sticky='ew')
        
        # 平台
        ttk.Label(form, text="平台:").grid(row=1, column=0, sticky=tk.W, pady=10)
        platform_var = tk.StringVar(value=key_data.get('platform', 'deepseek') if is_edit else 'deepseek')
        platform_combo = ttk.Combobox(form, textvariable=platform_var, state='readonly', width=38)
        platform_combo['values'] = [preset['display_name'] for preset in PLATFORM_PRESETS.values()]
        
        # 设置当前值
        if is_edit:
            current_platform = key_data.get('platform', 'deepseek')
            display_name = PLATFORM_PRESETS.get(current_platform, {}).get('display_name', '')
            if display_name:
                platform_combo.set(display_name)
        else:
            platform_combo.current(0)
        
        platform_combo.grid(row=1, column=1, pady=10, sticky='ew')
        
        # 模型
        ttk.Label(form, text="模型:").grid(row=2, column=0, sticky=tk.W, pady=10)
        model_var = tk.StringVar(value=key_data.get('model', 'deepseek-chat') if is_edit else 'deepseek-chat')
        model_combo = ttk.Combobox(form, textvariable=model_var, width=38)
        model_combo.grid(row=2, column=1, pady=10, sticky='ew')
        
        def update_models(event=None):
            """根据平台更新模型列表"""
            selected_display_name = platform_combo.get()
            
            # 找到对应的平台ID
            platform_id = None
            for pid, preset in PLATFORM_PRESETS.items():
                if preset['display_name'] == selected_display_name:
                    platform_id = pid
                    break
            
            if platform_id:
                models = PLATFORM_PRESETS[platform_id]['models']
                model_combo['values'] = models
                if models:
                    model_combo.set(models[0])
                
                # 自定义平台允许手动输入
                if platform_id == 'custom':
                    model_combo.config(state='normal')
                    url_entry.config(state='normal')
                else:
                    model_combo.config(state='readonly')
                    url_entry.config(state='disabled')
                    url_var.set(PLATFORM_PRESETS[platform_id]['url'])
        
        platform_combo.bind('<<ComboboxSelected>>', update_models)
        
        # API Key
        ttk.Label(form, text="API Key:").grid(row=3, column=0, sticky=tk.W, pady=10)
        key_frame = ttk.Frame(form)
        key_frame.grid(row=3, column=1, pady=10, sticky='ew')
        
        show_key = tk.BooleanVar(value=False)
        key_var = tk.StringVar(value=key_data.get('api_key', '') if is_edit else '')
        key_entry = ttk.Entry(key_frame, textvariable=key_var, show='*', width=32)
        key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def toggle_show():
            if show_key.get():
                key_entry.config(show='')
            else:
                key_entry.config(show='*')
        
        show_btn = ttk.Checkbutton(key_frame, text="👁️", variable=show_key, command=toggle_show, width=3)
        show_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # API URL
        ttk.Label(form, text="API URL:").grid(row=4, column=0, sticky=tk.W, pady=10)
        url_var = tk.StringVar(value=key_data.get('url', '') if is_edit else PLATFORM_PRESETS['deepseek']['url'])
        url_entry = ttk.Entry(form, textvariable=url_var, width=40, state='disabled')
        url_entry.grid(row=4, column=1, pady=10, sticky='ew')
        
        # 高级选项
        advanced_frame = ttk.LabelFrame(form, text="高级选项", padding="10")
        advanced_frame.grid(row=5, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(advanced_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W, pady=5)
        temp_var = tk.DoubleVar(value=key_data.get('temperature', 0.3) if is_edit else 0.3)
        ttk.Entry(advanced_frame, textvariable=temp_var, width=10).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Label(advanced_frame, text="(0.0-2.0)", foreground='gray').grid(row=0, column=2, sticky=tk.W, padx=(5,0))
        
        ttk.Label(advanced_frame, text="Max Tokens:").grid(row=1, column=0, sticky=tk.W, pady=5)
        tokens_var = tk.IntVar(value=key_data.get('max_tokens', 1000) if is_edit else 1000)
        ttk.Entry(advanced_frame, textvariable=tokens_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # === 新增：自定义提示词 ===
        ttk.Label(advanced_frame, text="自定义提示词:", font=('Microsoft YaHei UI', 9, 'bold')).grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        prompt_frame = ttk.Frame(advanced_frame)
        prompt_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=5)
        
        prompt_text = scrolledtext.ScrolledText(prompt_frame, height=6, width=50, 
                                               font=('Consolas', 9), wrap=tk.WORD)
        prompt_text.pack(fill=tk.BOTH, expand=True)
        
        # 填充默认提示词
        current_prompt = key_data.get('custom_prompt', DEFAULT_PROMPT) if is_edit else DEFAULT_PROMPT
        prompt_text.insert('1.0', current_prompt)
        
        # 恢复默认按钮
        def restore_default_prompt():
            prompt_text.delete('1.0', tk.END)
            prompt_text.insert('1.0', DEFAULT_PROMPT)
        
        button_frame = ttk.Frame(advanced_frame)
        button_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        ttk.Button(button_frame, text="🔄 恢复默认提示词", command=restore_default_prompt, width=20).pack(side=tk.LEFT)
        # === 新增结束 ===
        
        form.columnconfigure(1, weight=1)

        
        # 初始化模型列表
        update_models()
        
        # 按钮
        btn_frame = ttk.Frame(dialog, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def save():
            name = name_var.get().strip()
            api_key = key_var.get().strip()
            
            if not name:
                messagebox.showwarning("警告", "请输入名称", parent=dialog)
                return
            
            if not api_key:
                messagebox.showwarning("警告", "请输入 API Key", parent=dialog)
                return
            
            # 找到平台ID
            selected_display_name = platform_combo.get()
            platform_id = None
            for pid, preset in PLATFORM_PRESETS.items():
                if preset['display_name'] == selected_display_name:
                    platform_id = pid
                    break
            
            # === 新增：获取自定义提示词 ===
            custom_prompt = prompt_text.get('1.0', tk.END).strip()
            if not custom_prompt:
                custom_prompt = DEFAULT_PROMPT
            # === 新增结束 ===
            
            new_key_data = {
                'name': name,
                'platform': platform_id,
                'api_key': api_key,
                'model': model_var.get(),
                'url': url_var.get(),
                'temperature': temp_var.get(),
                'max_tokens': tokens_var.get(),
                'custom_prompt': custom_prompt
            }
            
            if is_edit:
                self.config_manager.update_api_key(key_data['id'], new_key_data)
            else:
                self.config_manager.add_api_key(new_key_data)
            
            messagebox.showinfo("成功", "API Key 已保存", parent=dialog)
            refresh_callback()
            self.load_settings()
            dialog.destroy()
        
        ttk.Button(btn_frame, text="💾 保存", command=save, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="❌ 取消", command=dialog.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    def test_api_key(self, key_data, parent, tree, tree_item):
        """测试API Key"""
        test_window = tk.Toplevel(parent)
        test_window.title("测试 API 连接")
        test_window.geometry("400x250")
        test_window.resizable(False, False)
        test_window.transient(parent)
        test_window.grab_set()
        
        test_window.rowconfigure(1, weight=1)
        test_window.columnconfigure(0, weight=1)
        
        ttk.Label(test_window, text="测试 API 连接", 
                 style='Title.TLabel', padding="20").grid(row=0, column=0)
        
        content = ttk.Frame(test_window, padding="20")
        content.grid(row=1, column=0, sticky='nsew')
        
        platform_name = PLATFORM_PRESETS.get(key_data.get('platform', 'custom'), {}).get('name', '自定义')
        ttk.Label(content, text=f"平台: {platform_name}").pack(pady=5)
        ttk.Label(content, text=f"模型: {key_data.get('model', 'N/A')}").pack(pady=5)
        
        progress = ttk.Progressbar(content, mode='indeterminate')
        progress.pack(fill=tk.X, pady=15)
        
        status_label = ttk.Label(content, text="正在测试...")
        status_label.pack(pady=10)
        
        progress.start()
        
        def do_test():
            try:
                api_config = {
                    'platform': key_data.get('platform', 'deepseek'),
                    'api_key': key_data['api_key'],
                    'model': key_data.get('model', 'deepseek-chat'),
                    'url': key_data.get('url', ''),
                    'temperature': key_data.get('temperature', 0.3),
                    'max_tokens': key_data.get('max_tokens', 1000)
                }
                
                translator = UniversalTranslator(api_config)
                success, message = translator.test_connection()
                
                test_window.after(0, lambda: on_result(success, message))
                
            except Exception as e:
                test_window.after(0, lambda: on_result(False, str(e)))
        
        def on_result(success, message):
            progress.stop()
            
            if success:
                status_label.config(text="✅ 测试成功！")
                ttk.Label(content, text=message, foreground='green', wraplength=350).pack(pady=10)
                
                # 更新树视图
                values = list(tree.item(tree_item)['values'])
                values[3] = '✓ 成功'
                tree.item(tree_item, values=values)
            else:
                status_label.config(text="❌ 测试失败")
                ttk.Label(content, text=message, foreground='red', wraplength=350).pack(pady=10)
            
            ttk.Button(content, text="确定", command=test_window.destroy, width=12).pack(pady=10)
        
        thread = threading.Thread(target=do_test, daemon=True)
        thread.start()
    
    def show_settings(self):
        """显示设置对话框"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("650x700")
        settings_window.minsize(600, 650)
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
        
        # ===== 输出设置选项卡 =====
        output_tab = ttk.Frame(notebook, padding="15")
        notebook.add(output_tab, text="输出设置")
        
        # 输出模式
        mode_frame = ttk.LabelFrame(output_tab, text="输出模式", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        output_mode_var = tk.StringVar(value=self.config_manager.config.get('output_mode', 'export'))
        
        ttk.Radiobutton(mode_frame, text="导出到指定文件夹（推荐）", 
                       variable=output_mode_var, value='export').pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(mode_frame, text="覆盖原文件（危险，但会创建备份）", 
                       variable=output_mode_var, value='overwrite').pack(anchor=tk.W, pady=2)
        
        # 导出选项
        export_frame = ttk.LabelFrame(output_tab, text="导出选项", padding="10")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        folder_frame = ttk.Frame(export_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Label(folder_frame, text="输出文件夹:").pack(side=tk.LEFT)
        output_folder_var = tk.StringVar(value=self.config_manager.config.get('output_folder', ''))
        ttk.Entry(folder_frame, textvariable=output_folder_var, width=35).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_folder():
            folder = filedialog.askdirectory(title="选择输出文件夹")
            if folder:
                output_folder_var.set(folder)
        
        ttk.Button(folder_frame, text="浏览...", command=browse_folder, width=10).pack(side=tk.LEFT)
        
        keep_structure_var = tk.BooleanVar(value=self.config_manager.config.get('keep_structure', True))
        ttk.Checkbutton(export_frame, text="保持原目录结构", variable=keep_structure_var).pack(anchor=tk.W, pady=2)
        
        # 语言标识
        tag_frame = ttk.Frame(export_frame)
        tag_frame.pack(fill=tk.X, pady=5)
        
        add_tag_var = tk.BooleanVar(value=self.config_manager.config.get('add_language_tag', True))
        ttk.Checkbutton(tag_frame, text="添加语言标识:", variable=add_tag_var).pack(side=tk.LEFT)
        
        language_tag_var = tk.StringVar(value=self.config_manager.config.get('language_tag', '_zh_CN'))
        
        # 标识输入框和历史
        tag_combo = ttk.Combobox(tag_frame, textvariable=language_tag_var, width=15)
        tag_combo.pack(side=tk.LEFT, padx=5)
        
        # 填充预设和历史
        tag_values = []
        for preset in self.config_manager.config.get('preset_tags', []):
            tag_values.append(preset['tag'])
        for history in self.config_manager.config.get('tag_history', []):
            if history['tag'] not in tag_values:
                tag_values.append(history['tag'])
        tag_combo['values'] = tag_values
        
        # 位置选择
        position_frame = ttk.Frame(export_frame)
        position_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        tag_position_var = tk.StringVar(value=self.config_manager.config.get('tag_position', 'end'))
        
        ttk.Label(position_frame, text="位置:").pack(side=tk.LEFT)
        ttk.Radiobutton(position_frame, text="文件名末尾 (file_zh_CN.yml)", 
                       variable=tag_position_var, value='end').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(position_frame, text="扩展名前 (file.zh_CN.yml)", 
                       variable=tag_position_var, value='before_ext').pack(side=tk.LEFT)
        
        # 实时预览
        preview_frame = ttk.Frame(export_frame)
        preview_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        preview_label = ttk.Label(preview_frame, text="预览: ", foreground='gray')
        preview_label.pack(side=tk.LEFT)
        
        preview_text = ttk.Label(preview_frame, text="config.yml → config_zh_CN.yml", foreground='blue')
        preview_text.pack(side=tk.LEFT)
        
        def update_preview(*args):
            tag = language_tag_var.get()
            position = tag_position_var.get()
            
            if position == 'before_ext':
                result = f"config.{tag.lstrip('_')}.yml"
            else:
                result = f"config{tag}.yml"
            
            preview_text.config(text=f"config.yml → {result}")
        
        language_tag_var.trace('w', update_preview)
        tag_position_var.trace('w', update_preview)
        
        # 双语输出选项
        bilingual_frame = ttk.LabelFrame(output_tab, text="双语输出", padding="10")
        bilingual_frame.pack(fill=tk.X, pady=(0, 10))
        
        bilingual_var = tk.BooleanVar(value=self.config_manager.config.get('enable_bilingual', False))
        ttk.Checkbutton(bilingual_frame, text="启用双语输出（中文 | 原文）", 
                       variable=bilingual_var).pack(anchor=tk.W, pady=2)
        
        # 分隔符选择
        separator_frame = ttk.Frame(bilingual_frame)
        separator_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        ttk.Label(separator_frame, text="分隔符:").pack(side=tk.LEFT, padx=(0, 8))
        separator_var = tk.StringVar(value=self.config_manager.config.get('bilingual_separator', ' | '))
        
        separators = [' | ', ' / ', ' - ', ' · ', ' • ']
        separator_combo = ttk.Combobox(separator_frame, textvariable=separator_var, 
                                      values=separators, width=10)
        separator_combo.pack(side=tk.LEFT)
        
        # 顺序选择
        order_frame = ttk.Frame(bilingual_frame)
        order_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        bilingual_order_var = tk.StringVar(value=self.config_manager.config.get('bilingual_order', 'cn_first'))
        
        ttk.Label(order_frame, text="显示顺序:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(order_frame, text="中文在前", 
                       variable=bilingual_order_var, value='cn_first').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(order_frame, text="原文在前", 
                       variable=bilingual_order_var, value='en_first').pack(side=tk.LEFT)
        
        # 预览
        preview_bilingual = ttk.Frame(bilingual_frame)
        preview_bilingual.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        preview_bilingual_label = ttk.Label(preview_bilingual, text="", foreground='blue')
        preview_bilingual_label.pack(side=tk.LEFT)
        
        def update_bilingual_preview(*args):
            if not bilingual_var.get():
                preview_bilingual_label.config(text='预览: "Items" → "物品"')
                return
            
            sep = separator_var.get()
            order = bilingual_order_var.get()
            
            if order == 'cn_first':
                result = f'"Items" → "物品{sep}Items"'
            else:
                result = f'"Items" → "Items{sep}物品"'
            
            preview_bilingual_label.config(text=f'预览: {result}')
        
        bilingual_var.trace('w', update_bilingual_preview)
        separator_var.trace('w', update_bilingual_preview)
        bilingual_order_var.trace('w', update_bilingual_preview)
        
        update_bilingual_preview()
        
        # 高级选项
        advanced_frame = ttk.LabelFrame(output_tab, text="高级选项", padding="10")
        advanced_frame.pack(fill=tk.X)
        
        generate_report_var = tk.BooleanVar(value=self.config_manager.config.get('generate_report', True))
        ttk.Checkbutton(advanced_frame, text="生成对比报告 (HTML)", variable=generate_report_var).pack(anchor=tk.W, pady=2)
        
        # ===== 翻译设置选项卡 =====
        trans_tab = ttk.Frame(notebook, padding="15")
        notebook.add(trans_tab, text="翻译设置")
        
        basic_frame = ttk.LabelFrame(trans_tab, text="基本设置", padding="10")
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        skip_chinese_var = tk.BooleanVar(value=self.config_manager.config.get('skip_chinese', True))
        ttk.Checkbutton(basic_frame, text="跳过已包含中文的字段", variable=skip_chinese_var).pack(anchor=tk.W, pady=2)
        
        thread_frame = ttk.Frame(basic_frame)
        thread_frame.pack(fill=tk.X, pady=5)
        ttk.Label(thread_frame, text="默认并发线程数:").pack(side=tk.LEFT, padx=(0, 8))
        thread_var = tk.IntVar(value=self.config_manager.config.get('max_threads', 4))
        ttk.Spinbox(thread_frame, from_=1, to=200, textvariable=thread_var, width=10).pack(side=tk.LEFT)
        ttk.Label(thread_frame, text="(1-200)", foreground='gray').pack(side=tk.LEFT, padx=(8, 0))
        
        timeout_frame = ttk.Frame(basic_frame)
        timeout_frame.pack(fill=tk.X, pady=5)
        ttk.Label(timeout_frame, text="API 请求超时:").pack(side=tk.LEFT, padx=(0, 8))
        timeout_var = tk.IntVar(value=self.config_manager.config.get('api_timeout', 30))
        ttk.Spinbox(timeout_frame, from_=5, to=300, textvariable=timeout_var, width=10).pack(side=tk.LEFT)
        ttk.Label(timeout_frame, text="秒", foreground='gray').pack(side=tk.LEFT, padx=(8, 0))
        
        # 重试设置
        retry_frame = ttk.LabelFrame(trans_tab, text="失败重试", padding="10")
        retry_frame.pack(fill=tk.X)
        
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
        
        # ===== 关于选项卡 =====
        about_tab = ttk.Frame(notebook, padding="15")
        notebook.add(about_tab, text="关于")
        
        about_text = f"""
{APP_TITLE}

一个专业的 YAML 文件批量翻译工具

主要特性:
• 支持20+个AI平台API (OpenAI, Claude, Mistral, Groq等)
• 多线程并发翻译
• 智能上下文翻译
• 文件导出功能（不覆盖源文件）
• 双语输出功能（中文 | 原文）
• 自定义翻译提示词
• 自动生成对比报告
• 翻译历史记录
• 丰富的配置选项

版本: {VERSION}

支持的AI平台:
🧠 OpenAI, Claude, Mistral, Groq, Perplexity
🤖 DeepSeek, xAI, Cohere, AI21
🌙 Moonshot, Google Makersuite, Fireworks
☁️ 通义千问, 智谱AI, ElectronHub, NanoGPT
🎯 AIML API, Pollinations, 自定义API
        """
        
        ttk.Label(about_tab, text=about_text, justify=tk.LEFT).pack(pady=20)
        
        # 底部按钮
        btn_frame = ttk.Frame(settings_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def save_settings():
            # 保存所有设置
            self.config_manager.config['output_mode'] = output_mode_var.get()
            self.config_manager.config['output_folder'] = output_folder_var.get()
            self.config_manager.config['keep_structure'] = keep_structure_var.get()
            self.config_manager.config['add_language_tag'] = add_tag_var.get()
            self.config_manager.config['language_tag'] = language_tag_var.get()
            self.config_manager.config['tag_position'] = tag_position_var.get()
            self.config_manager.config['generate_report'] = generate_report_var.get()
            
            # 双语输出设置
            self.config_manager.config['enable_bilingual'] = bilingual_var.get()
            self.config_manager.config['bilingual_separator'] = separator_var.get()
            self.config_manager.config['bilingual_order'] = bilingual_order_var.get()
            
            self.config_manager.config['skip_chinese'] = skip_chinese_var.get()
            self.config_manager.config['max_threads'] = thread_var.get()
            self.config_manager.config['api_timeout'] = timeout_var.get()
            self.config_manager.config['enable_retry'] = retry_var.get()
            self.config_manager.config['max_retries'] = retry_count_var.get()
            self.config_manager.config['retry_delay'] = retry_delay_var.get()
            
            self.config_manager.save_config()
            
            # 应用线程数设置
            self.thread_spin.set(thread_var.get())
            
            messagebox.showinfo("成功", "设置已保存", parent=settings_window)
            settings_window.destroy()
        
        ttk.Button(btn_frame, text="保存", command=save_settings, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=settings_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    def show_output_quick_settings(self):
        """显示输出快速设置对话框"""
        quick_window = tk.Toplevel(self.root)
        quick_window.title("输出设置")
        quick_window.geometry("550x600")
        quick_window.minsize(500, 500)
        quick_window.transient(self.root)
        quick_window.grab_set()
        
        quick_window.rowconfigure(1, weight=1)
        quick_window.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(quick_window, padding="15 15 15 10")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text="📂 输出设置", style='Title.TLabel').pack(anchor=tk.W)
        
        # 创建可滚动的主内容区域
        scroll_container = ttk.Frame(quick_window)
        scroll_container.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        scroll_container.rowconfigure(0, weight=1)
        scroll_container.columnconfigure(0, weight=1)
        
        # 创建Canvas
        canvas = tk.Canvas(scroll_container, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky='nsew')
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 在Canvas中创建内容框架
        content = ttk.Frame(canvas, padding="15")
        canvas_window = canvas.create_window((0, 0), window=content, anchor='nw')
        
        # 绑定Canvas大小变化事件
        def on_canvas_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind('<Configure>', on_canvas_configure)
        
        # 绑定鼠标滚轮事件
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # 当内容框架大小变化时更新滚动区域
        def on_content_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        content.bind('<Configure>', on_content_configure)
        
        # 输出模式
        mode_frame = ttk.LabelFrame(content, text="输出模式", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        output_mode_var = tk.StringVar(value=self.config_manager.config.get('output_mode', 'export'))
        
        ttk.Radiobutton(mode_frame, text="📁 导出到指定文件夹（推荐，不修改源文件）", 
                       variable=output_mode_var, value='export').pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(mode_frame, text="⚠️  覆盖原文件（危险，但会创建备份）", 
                       variable=output_mode_var, value='overwrite').pack(anchor=tk.W, pady=3)
        
        # 输出文件夹选择
        folder_frame = ttk.LabelFrame(content, text="输出位置", padding="10")
        folder_frame.pack(fill=tk.X, pady=(0, 15))
        
        path_frame = ttk.Frame(folder_frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(path_frame, text="输出文件夹:").pack(side=tk.LEFT)
        output_folder_var = tk.StringVar(value=self.config_manager.config.get('output_folder', ''))
        
        path_entry = ttk.Entry(path_frame, textvariable=output_folder_var, width=30)
        path_entry.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        
        def browse_folder():
            folder = filedialog.askdirectory(
                title="选择输出文件夹",
                parent=quick_window
            )
            if folder:
                output_folder_var.set(folder)
        
        ttk.Button(path_frame, text="浏览...", command=browse_folder, width=10).pack(side=tk.LEFT)
        
        # 提示信息
        hint_label = ttk.Label(folder_frame, 
                              text="💡 留空则使用源文件夹下的 'translated' 子文件夹",
                              font=('Microsoft YaHei UI', 8),
                              foreground='gray')
        hint_label.pack(anchor=tk.W, pady=(5, 0))
        
        keep_structure_var = tk.BooleanVar(value=self.config_manager.config.get('keep_structure', True))
        ttk.Checkbutton(folder_frame, text="保持原目录结构", 
                       variable=keep_structure_var).pack(anchor=tk.W, pady=(10, 0))
        
        # 语言标识设置
        tag_frame = ttk.LabelFrame(content, text="语言标识", padding="10")
        tag_frame.pack(fill=tk.X, pady=(0, 15))
        
        add_tag_var = tk.BooleanVar(value=self.config_manager.config.get('add_language_tag', True))
        ttk.Checkbutton(tag_frame, text="添加语言标识到文件名", 
                       variable=add_tag_var).pack(anchor=tk.W, pady=5)
        
        # 标识输入
        tag_input_frame = ttk.Frame(tag_frame)
        tag_input_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        ttk.Label(tag_input_frame, text="标识:").pack(side=tk.LEFT)
        
        language_tag_var = tk.StringVar(value=self.config_manager.config.get('language_tag', '_zh_CN'))
        
        tag_combo = ttk.Combobox(tag_input_frame, textvariable=language_tag_var, width=15)
        tag_combo.pack(side=tk.LEFT, padx=8)
        
        # 填充预设和历史
        tag_values = []
        for preset in self.config_manager.config.get('preset_tags', []):
            tag_values.append(preset['tag'])
        for history in self.config_manager.config.get('tag_history', []):
            if history['tag'] not in tag_values:
                tag_values.append(history['tag'])
        tag_combo['values'] = tag_values
        
        # 位置选择
        position_frame = ttk.Frame(tag_frame)
        position_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        tag_position_var = tk.StringVar(value=self.config_manager.config.get('tag_position', 'end'))
        
        ttk.Label(position_frame, text="位置:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(position_frame, text="末尾", 
                       variable=tag_position_var, value='end').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(position_frame, text="扩展名前", 
                       variable=tag_position_var, value='before_ext').pack(side=tk.LEFT)
        
        # 预览
        preview_frame = ttk.Frame(tag_frame)
        preview_frame.pack(fill=tk.X, pady=(10, 5), padx=(20, 0))
        
        preview_label = ttk.Label(preview_frame, text="", foreground='blue', font=('Consolas', 9))
        preview_label.pack(side=tk.LEFT)
        
        def update_preview(*args):
            if not add_tag_var.get():
                preview_label.config(text="预览: config.yml → config.yml")
                return
            
            tag = language_tag_var.get()
            position = tag_position_var.get()
            
            if position == 'before_ext':
                result = f"config.{tag.lstrip('_')}.yml"
            else:
                result = f"config{tag}.yml"
            
            preview_label.config(text=f"预览: config.yml → {result}")
        
        language_tag_var.trace('w', update_preview)
        tag_position_var.trace('w', update_preview)
        add_tag_var.trace('w', update_preview)
        
        update_preview()
        
        # 双语输出
        bilingual_frame = ttk.LabelFrame(content, text="双语输出", padding="10")
        bilingual_frame.pack(fill=tk.X)
        
        bilingual_var = tk.BooleanVar(value=self.config_manager.config.get('enable_bilingual', False))
        ttk.Checkbutton(bilingual_frame, text="✨ 启用双语输出（同时显示中文和原文）", 
                       variable=bilingual_var).pack(anchor=tk.W, pady=5)
        
        sep_frame = ttk.Frame(bilingual_frame)
        sep_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        ttk.Label(sep_frame, text="分隔符:").pack(side=tk.LEFT)
        separator_var = tk.StringVar(value=self.config_manager.config.get('bilingual_separator', ' | '))
        ttk.Combobox(sep_frame, textvariable=separator_var, 
                    values=[' | ', ' / ', ' - ', ' · ', ' • '], width=8).pack(side=tk.LEFT, padx=8)
        
        order_frame = ttk.Frame(bilingual_frame)
        order_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        bilingual_order_var = tk.StringVar(value=self.config_manager.config.get('bilingual_order', 'cn_first'))
        
        ttk.Radiobutton(order_frame, text="中文 | 原文", 
                       variable=bilingual_order_var, value='cn_first').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(order_frame, text="原文 | 中文", 
                       variable=bilingual_order_var, value='en_first').pack(side=tk.LEFT)
        
        # 预览
        preview_bi = ttk.Label(bilingual_frame, text="", foreground='blue', font=('Consolas', 9))
        preview_bi.pack(anchor=tk.W, pady=(10, 0), padx=(20, 0))
        
        def update_bi_preview(*args):
            if not bilingual_var.get():
                preview_bi.config(text='预览: "Items" → "物品"')
            else:
                sep = separator_var.get()
                order = bilingual_order_var.get()
                if order == 'cn_first':
                    preview_bi.config(text=f'预览: "Items" → "物品{sep}Items"')
                else:
                    preview_bi.config(text=f'预览: "Items" → "Items{sep}物品"')
        
        bilingual_var.trace('w', update_bi_preview)
        separator_var.trace('w', update_bi_preview)
        bilingual_order_var.trace('w', update_bi_preview)
        update_bi_preview()
        
        # 底部按钮
        btn_frame = ttk.Frame(quick_window, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def save_and_close():
            # 保存设置
            self.config_manager.config['output_mode'] = output_mode_var.get()
            self.config_manager.config['output_folder'] = output_folder_var.get()
            self.config_manager.config['keep_structure'] = keep_structure_var.get()
            self.config_manager.config['add_language_tag'] = add_tag_var.get()
            self.config_manager.config['language_tag'] = language_tag_var.get()
            self.config_manager.config['tag_position'] = tag_position_var.get()
            self.config_manager.config['enable_bilingual'] = bilingual_var.get()
            self.config_manager.config['bilingual_separator'] = separator_var.get()
            self.config_manager.config['bilingual_order'] = bilingual_order_var.get()
            
            self.config_manager.save_config()
            
            # 保存语言标识到历史
            if add_tag_var.get() and language_tag_var.get():
                self.config_manager.add_tag_to_history(language_tag_var.get())
            
            messagebox.showinfo("成功", "输出设置已保存", parent=quick_window)
            quick_window.destroy()
        
        def open_full_settings():
            """打开完整设置窗口"""
            quick_window.destroy()
            self.show_settings()
        
        def on_close():
            """关闭窗口时解绑鼠标滚轮事件"""
            canvas.unbind_all("<MouseWheel>")
            quick_window.destroy()
        
        quick_window.protocol("WM_DELETE_WINDOW", on_close)
        
        ttk.Button(btn_frame, text="💾 保存", command=save_and_close, 
                  width=12, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⚙️ 更多设置...", command=open_full_settings, 
                  width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=on_close, 
                  width=12).pack(side=tk.RIGHT, padx=5)

    def show_history(self):
        """显示翻译历史"""
        history_window = tk.Toplevel(self.root)
        history_window.title("翻译历史记录")
        history_window.geometry("900x550")
        history_window.minsize(800, 500)
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
        
        columns = ('time', 'files', 'success', 'skipped', 'failed', 'duration')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        tree.heading('time', text='时间')
        tree.heading('files', text='文件数')
        tree.heading('success', text='成功')
        tree.heading('skipped', text='跳过')
        tree.heading('failed', text='失败')
        tree.heading('duration', text='耗时')
        
        tree.column('time', width=180)
        tree.column('files', width=80)
        tree.column('success', width=80)
        tree.column('skipped', width=80)
        tree.column('failed', width=80)
        tree.column('duration', width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # 加载历史
        for item in self.config_manager.config.get('history', []):
            time_str = item['timestamp'][:19].replace('T', ' ')
            duration_str = f"{item.get('duration', 0):.1f}秒"
            
            tree.insert('', tk.END, values=(
                time_str,
                item['processed_files'],
                item['successful_translations'],
                item.get('skipped_translations', 0),
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
        help_window.geometry("750x650")
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
1. 配置 API Key
   • 点击"工具" → "管理 API Key"
   • 选择平台（DeepSeek、OpenAI、Moonshot等）
   • 输入 API Key 并保存
   • 测试连接确保可用

2. 添加文件
   • 通过按钮添加单个文件或文件夹
   • 支持拖拽文件/文件夹（需安装 tkinterdnd2）

3. 配置输出
   • 点击"📂 输出到..."按钮
   • 选择输出模式（导出或覆盖）
   • 设置输出文件夹
   • 配置语言标识
   • 启用双语输出（可选）

4. 开始翻译
   • 点击"开始翻译"按钮
   • 等待完成并查看对比报告

二、输出模式说明

1. 导出模式（推荐）
   • 不修改源文件
   • 翻译结果保存到指定文件夹
   • 可选保持原目录结构
   • 可添加语言标识

2. 覆盖模式
   • 直接替换源文件内容
   • 自动创建 .backup 备份
   • 适合直接更新项目文件

三、双语输出功能（新功能）

启用后，翻译结果会同时包含中文和原文：

• 中文在前: "物品 | Items"
• 原文在前: "Items | 物品"
• 可选分隔符: | / - · •

适用场景：
• 游戏Mod翻译（玩家可对照理解）
• 文档翻译（保留原文参考）
• 学习用途（中英对照）

四、语言标识功能

• 自动记住最近使用的标识
• 预设常用标识（_zh_CN、_zh_TW等）
• 支持自定义标识
• 两种位置：文件名末尾 或 扩展名前

示例：
  文件名末尾: config.yml → config_zh_CN.yml
  扩展名前: config.yml → config.zh_CN.yml

五、多平台 API 支持

支持平台：
• 🤖 DeepSeek - 推荐，性价比高
• 🧠 OpenAI - GPT系列，效果好
• 🌙 Moonshot - Kimi，上下文长
• 🧩 智谱AI - GLM系列
• ☁️ 通义千问 - 阿里云
• ⚙️ 自定义 - 支持任何OpenAI兼容API

六、对比报告

翻译完成后自动生成 HTML 报告：
• 详细的翻译对比
• 文件级别的统计
• 成功/跳过/失败分类
• 美观的网页界面

七、快捷键

• Ctrl+O     - 添加文件
• Ctrl+D     - 添加文件夹
• F5         - 开始翻译
• Esc        - 停止翻译
• Ctrl+L     - 清空日志
• Delete     - 移除选中文件
• Ctrl+,     - 打开设置

八、注意事项

• 建议线程数设置为 1-50
• 首次使用建议使用导出模式
• 大批量翻译建议分批进行
• 注意 API 调用限流
• 定期查看翻译历史记录

九、常见问题

Q: 无法拖拽文件怎么办？
A: 点击底部提示链接一键安装 tkinterdnd2

Q: 翻译失败怎么办？
A: 检查 API Key、网络连接，开启自动重试

Q: 如何恢复源文件？
A: 导出模式源文件未修改；覆盖模式可用 .backup 文件

Q: 双语输出会影响游戏运行吗？
A: 不会，只是文本变长，游戏会正常显示

Q: 支持哪些翻译方向？
A: 目前主要支持英文→中文
        """
        
        help_text.insert('1.0', content)
        help_text.config(state='disabled')
        
        btn_frame = ttk.Frame(help_window, padding="15")
        btn_frame.grid(row=1, column=0, sticky='ew')
        ttk.Button(btn_frame, text="关闭", command=help_window.destroy, width=12).pack(side=tk.RIGHT)
    
    def show_about(self):
        """显示关于"""
        about_text = f"""
{APP_TITLE}

一个YAML文件批量AI翻译工具

主要特性:
• 支持多平台 API (DeepSeek, OpenAI, Moonshot等)
• 文件导出功能，不覆盖源文件
• 双语输出功能（中文 | 原文）
• 自动生成精美的对比报告
• 多线程并发翻译
• 智能上下文翻译
• 翻译历史记录
• 丰富的配置选项

作者: Mr.Centes，Claude
版本: {VERSION}

更新日志:
v1.1 - 新增双语输出功能
     - 优化输出设置界面
     - 改进用户体验
        """
        messagebox.showinfo("关于", about_text)
    
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
        
        ttk.Label(install_window, text="安装 tkinterdnd2", 
                 style='Title.TLabel', padding="20 20 20 10").grid(row=0, column=0)
        
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
                    install_window.after(0, lambda: on_success())
                else:
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
            
            messagebox.showerror("安装失败", 
                f"无法自动安装 tkinterdnd2\n\n"
                f"✅ 翻译功能仍可正常使用\n"
                f"（可通过按钮添加文件）\n\n"
                f"错误信息: {error[:100]}")
        
        thread = threading.Thread(target=do_install, daemon=True)
        thread.start()
    
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
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = TranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()