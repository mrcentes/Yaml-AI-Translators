# YAML批量AI本地化工具 v1.30 (重构优化版)
# 更新日志:
# v1.30 - 重大重构版本
#       - 拆分大类，模块化设计
#       - 智能上下文提取（相邻字段关联）
#       - 可配置字段系统（不再硬编码）
#       - 字段配置管理界面
#       - 优化性能和线程安全
#       - 改进错误处理
#       - 提取HTML模板
#       - 修复所有已知问题

import os
import sys
import json
import threading
import time
import shutil
import requests
import subprocess
import webbrowser
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any, Set
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Menu
import yaml

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

VERSION = "1.30"
APP_TITLE = f"YAML批量AI本地化工具 v{VERSION}"

# ==================== 统一说明文本 ====================
APP_DESCRIPTION = """一个专业的 YAML 文件批量翻译工具

主要特性:
✓ 支持7个主流AI平台API
✓ 多线程并发翻译，速度快
✓ 智能上下文翻译（相邻字段关联）
✓ 可配置翻译字段（不再硬编码）
✓ 完整的重试机制和速率限制
✓ 文件导出功能（不覆盖源文件）
✓ 双语输出功能（中文 | 原文）
✓ 自定义翻译提示词
✓ API模型自动更新
✓ 自动生成对比报告
✓ 翻译历史记录
✓ 丰富的配置选项"""

# ==================== 平台预设库 ====================
PLATFORM_PRESETS = {
    'openai': {
        'name': 'OpenAI',
        'url': 'https://api.openai.com/v1/chat/completions',
        'models': ['gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo'],
        'default_model': 'gpt-3.5-turbo',
        'docs_url': 'https://platform.openai.com/docs'
    },
    'deepseek': {
        'name': 'DeepSeek',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'models': ['deepseek-chat', 'deepseek-coder'],
        'default_model': 'deepseek-chat',
        'docs_url': 'https://platform.deepseek.com/docs'
    },
    'moonshot': {
        'name': 'Moonshot',
        'url': 'https://api.moonshot.cn/v1/chat/completions',
        'models': ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k'],
        'default_model': 'moonshot-v1-8k',
        'docs_url': 'https://platform.moonshot.cn/docs'
    },
    'zhipu': {
        'name': 'ZhipuAI',
        'url': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
        'models': ['glm-4', 'glm-4v', 'glm-3-turbo'],
        'default_model': 'glm-4',
        'docs_url': 'https://open.bigmodel.cn/dev/api'
    },
    'qwen': {
        'name': 'Qwen',
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        'models': ['qwen-turbo', 'qwen-plus', 'qwen-max'],
        'default_model': 'qwen-turbo',
        'docs_url': 'https://help.aliyun.com/zh/dashscope/'
    },
    'claude': {
        'name': 'Claude',
        'url': 'https://api.anthropic.com/v1/messages',
        'models': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
        'default_model': 'claude-3-sonnet',
        'docs_url': 'https://docs.anthropic.com/claude/reference/getting-started-with-the-api'
    },
    'xai': {
        'name': 'xAI',
        'url': 'https://api.x.ai/v1/chat/completions',
        'models': ['grok-beta'],
        'default_model': 'grok-beta',
        'docs_url': 'https://docs.x.ai/'
    },
    'custom': {
        'name': 'Custom',
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

# ==================== 数据类定义 ====================

@dataclass
class FieldRule:
    """字段翻译规则"""
    field_name: str
    enabled: bool = True
    priority: int = 0
    context_fields: List[str] = None  # 相关联的上下文字段
    
    def __post_init__(self):
        if self.context_fields is None:
            self.context_fields = []

@dataclass
class TranslationConfig:
    """翻译配置"""
    skip_chinese: bool = True
    api_timeout: int = 30
    enable_retry: bool = True
    max_retries: int = 3
    retry_delay: int = 5
    output_mode: str = 'export'
    output_folder: str = ''
    keep_structure: bool = True
    add_language_tag: bool = True
    language_tag: str = '_zh_CN'
    tag_position: str = 'end'
    enable_bilingual: bool = False
    bilingual_separator: str = ' | '
    bilingual_order: str = 'cn_first'
    field_rules: List[FieldRule] = None
    
    def __post_init__(self):
        if self.field_rules is None:
            # 默认字段规则
            self.field_rules = [
                FieldRule('name', True, 1, ['description', 'tooltip']),
                FieldRule('description', True, 2, ['name']),
                FieldRule('tooltip', True, 3, ['name', 'description']),
                FieldRule('title', True, 4, []),
                FieldRule('label', True, 5, []),
            ]

# ==================== 优化的速率限制器 ====================

class RateLimiter:
    """优化的速率限制器 - 使用 deque 提高性能"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()  # 使用 deque 替代 list
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """如果超出限制则等待"""
        with self.lock:
            now = time.time()
            
            # 移除过期的请求记录
            while self.requests and now - self.requests[0] >= self.time_window:
                self.requests.popleft()
            
            # 如果达到限制，等待
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # 清理过期记录
                    while self.requests and time.time() - self.requests[0] >= self.time_window:
                        self.requests.popleft()
            
            self.requests.append(now)

# ==================== YAML 加载器（模块级别定义，只定义一次）====================

class CustomYAMLLoader(yaml.SafeLoader):
    """自定义 YAML 加载器，支持未知标签"""
    pass

def multi_constructor(loader, tag_suffix, node):
    """处理所有未知标签"""
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    else:
        return loader.construct_scalar(node)

CustomYAMLLoader.add_multi_constructor('!', multi_constructor)

# ==================== 核心翻译器 ====================

class UniversalTranslator:
    """通用翻译器 - 支持多平台API"""
    
    # 特殊字符集合（使用 set 提高查找效率）
    SPECIAL_CHARS = set(':{}[],&*#?|-<>=!%@`')
    
    def __init__(self, api_config: Dict[str, Any], rate_limiter: RateLimiter):
        self.config = api_config
        self.platform = api_config.get('platform', 'deepseek')
        self.api_key = api_config['api_key']
        self.model = api_config.get('model', 'deepseek-chat')
        self.base_url = api_config.get('url', PLATFORM_PRESETS.get(self.platform, {}).get('url', ''))
        self.temperature = api_config.get('temperature', 0.3)
        self.max_tokens = api_config.get('max_tokens', 1000)
        self.rate_limiter = rate_limiter
        self.lock = threading.Lock()
        self.retry_config = {
            'max_retries': api_config.get('max_retries', 3),
            'retry_delay': api_config.get('retry_delay', 5)
        }
    
    def _build_prompt(self, text: str, context_info: Optional[Dict] = None) -> str:
        """构建翻译提示词"""
        base_prompt = self.config.get('custom_prompt', DEFAULT_PROMPT)
        
        if not context_info:
            return f"{base_prompt}\n\n待翻译文本：{text}"
        
        # 构建上下文信息
        context_parts = []
        if context_info.get('field_name'):
            context_parts.append(f"字段名: {context_info['field_name']}")
        
        if context_info.get('related_fields'):
            related = context_info['related_fields']
            if related:
                context_parts.append("相关字段:")
                for field_name, field_value in related.items():
                    context_parts.append(f"  - {field_name}: {field_value}")
        
        if context_parts:
            context_str = "\n".join(context_parts)
            return f"{base_prompt}\n\n上下文信息：\n{context_str}\n\n待翻译文本：{text}"
        
        return f"{base_prompt}\n\n待翻译文本：{text}"
    
    def _make_request(self, prompt: str, timeout: int) -> Tuple[Optional[str], Optional[str]]:
        """发送API请求"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            with self.lock:
                time.sleep(0.1)  # 防止过快请求
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            translated_text = result['choices'][0]['message']['content'].strip()
            return translated_text, None
            
        except requests.exceptions.Timeout:
            return None, "请求超时"
        except requests.exceptions.ConnectionError as e:
            return None, f"连接错误: {str(e)}"
        except requests.exceptions.HTTPError as e:
            return None, f"HTTP错误: {str(e)}"
        except (KeyError, IndexError) as e:
            return None, f"响应格式错误: {str(e)}"
        except Exception as e:
            return None, f"未知错误: {str(e)}"
    
    def clean_translated_text(self, text: str) -> str:
        """智能清理翻译后的文本"""
        if not text:
            return text
        
        # 统计引号数量
        double_quotes = text.count('"')
        single_quotes = text.count("'")
        
        # 如果引号很少，不需要处理
        if double_quotes < 2 and single_quotes < 2:
            return text
        
        # 找到第二个引号的位置
        quote_positions = []
        for i, char in enumerate(text):
            if char in ['"', "'"]:
                quote_positions.append(i)
                if len(quote_positions) >= 2:
                    break
        
        # 如果找到至少两个引号，保留前两个引号之间的内容，清理后面的
        if len(quote_positions) >= 2:
            second_quote_pos = quote_positions[1]
            before = text[:second_quote_pos + 1]
            after = text[second_quote_pos + 1:]
            # 清理后面部分的引号和冒号
            after = after.replace('"', '').replace("'", '').replace(':', '：')
            return before + after
        
        return text
    
    def escape_yaml_value(self, text: str) -> str:
        """转义YAML特殊字符"""
        if not isinstance(text, str):
            text = str(text)
        
        if not text:
            return text
        
        # 检查是否需要引号
        needs_quotes = False
        
        # 检查特殊字符
        if any(char in self.SPECIAL_CHARS for char in text):
            needs_quotes = True
        
        # 检查是否包含引号
        if '"' in text or "'" in text:
            needs_quotes = True
        
        # 检查开头是否是特殊字符
        if text and text[0] in self.SPECIAL_CHARS:
            needs_quotes = True
        
        if needs_quotes:
            # 转义单引号
            escaped = text.replace("'", "''")
            return f"'{escaped}'"
        
        return text
    
    def translate(self, text: str, context_info: Optional[Dict] = None, timeout: int = 30) -> Tuple[str, Optional[str]]:
        """翻译文本 - 带重试机制"""
        # 速率限制
        self.rate_limiter.wait_if_needed()
        
        # 构建提示词
        prompt = self._build_prompt(text, context_info)
        
        # 重试逻辑
        max_retries = self.retry_config['max_retries']
        retry_delay = self.retry_config['retry_delay']
        
        for attempt in range(max_retries):
            translated_text, error = self._make_request(prompt, timeout)
            
            if error is None and translated_text:
                # 清理文本
                translated_text = self.clean_translated_text(translated_text)
                return translated_text, None
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            
            # 最后一次尝试失败
            return text, error or "翻译失败"
        
        return text, "翻译失败"
    
    def test_connection(self) -> Tuple[bool, str]:
        """测试API连接"""
        try:
            test_text = "Hello"
            start_time = time.time()
            result, error = self.translate(test_text, timeout=10)
            elapsed = time.time() - start_time
            
            if error:
                return False, f"翻译失败: {error}"
            
            if result and result != test_text:
                if elapsed > 5:
                    return True, f"测试成功但响应较慢: \"{test_text}\" → \"{result}\" (耗时 {elapsed:.2f}秒)"
                else:
                    return True, f"测试成功: \"{test_text}\" → \"{result}\" (耗时 {elapsed:.2f}秒)"
            else:
                return False, "API响应异常"
                
        except Exception as e:
            return False, f"连接失败: {str(e)}"
    
    @staticmethod
    def fetch_available_models(platform_id: str, api_key: str, base_url: str, 
                               timeout: int = 15) -> Tuple[Optional[List[str]], Optional[str]]:
        """从API获取可用模型列表"""
        try:
            if not api_key or not base_url:
                return None, "API Key 或 URL 不能为空"
            
            # 构建模型列表URL
            if '/chat/completions' in base_url:
                models_url = base_url.replace('/chat/completions', '/models')
            elif '/v1/' in base_url:
                base = base_url.rsplit('/v1/', 1)[0]
                models_url = f"{base}/v1/models"
            else:
                models_url = base_url.rstrip('/') + '/models'
            
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(models_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            model_objects = data.get('data', [])
            if not model_objects and isinstance(data, list):
                model_objects = data
            
            model_ids = sorted([model['id'] for model in model_objects if 'id' in model])
            
            if not model_ids:
                return None, "未能从API响应中找到模型列表"
            
            return model_ids, None
            
        except requests.exceptions.RequestException as e:
            return None, f"网络错误: {str(e)}"
        except Exception as e:
            return None, f"获取模型失败: {str(e)}"

# ==================== 字段提取器 ====================

class FieldExtractor:
    """字段提取器 - 负责从YAML数据中提取需要翻译的字段"""
    
    def __init__(self, field_rules: List[FieldRule]):
        self.field_rules = {rule.field_name: rule for rule in field_rules if rule.enabled}
        self.field_priorities = {name: rule.priority for name, rule in self.field_rules.items()}
    
    def should_translate(self, field_name: str) -> bool:
        """判断字段是否需要翻译"""
        return field_name in self.field_rules
    
    def get_context_fields(self, field_name: str) -> List[str]:
        """获取字段的上下文字段列表"""
        rule = self.field_rules.get(field_name)
        return rule.context_fields if rule else []
    
    def extract_fields(self, data: Any, path_prefix: str = '') -> List[Dict[str, Any]]:
        """
        提取所有需要翻译的字段
        返回: [{'path': '字段路径', 'field_name': '字段名', 'value': '值', 'parent': 父节点}]
        """
        fields = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path_prefix}.{key}" if path_prefix else key
                
                # 如果是需要翻译的字段
                if self.should_translate(key) and isinstance(value, str) and value.strip():
                    fields.append({
                        'path': current_path,
                        'field_name': key,
                        'value': value,
                        'parent': data,
                        'priority': self.field_priorities.get(key, 999)
                    })
                
                # 递归处理嵌套结构
                elif isinstance(value, (dict, list)):
                    fields.extend(self.extract_fields(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path_prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    fields.extend(self.extract_fields(item, current_path))
        
        # 按优先级排序
        fields.sort(key=lambda x: x['priority'])
        return fields
    
    def build_context(self, field_info: Dict[str, Any], all_fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        为字段构建上下文信息
        只包含相邻的相关字段
        """
        context = {
            'field_name': field_info['field_name'],
            'related_fields': {}
        }
        
        # 获取需要关联的字段名列表
        context_field_names = self.get_context_fields(field_info['field_name'])
        if not context_field_names:
            return context
        
        # 获取父节点
        parent = field_info['parent']
        if not isinstance(parent, dict):
            return context
        
        # 只提取同一父节点下的相关字段
        for context_field_name in context_field_names:
            if context_field_name in parent and isinstance(parent[context_field_name], str):
                value = parent[context_field_name].strip()
                if value:
                    context['related_fields'][context_field_name] = value
        
        return context

# ==================== YAML翻译核心 ====================

class YamlTranslatorCore:
    """YAML翻译核心 - 重构版"""
    
    def __init__(self, api_config: Dict[str, Any], translation_config: TranslationConfig,
                 max_threads: int = 4, progress_callback=None, log_callback=None, 
                 translation_callback=None):
        
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)
        self.translator = UniversalTranslator(api_config, self.rate_limiter)
        self.config = translation_config
        self.field_extractor = FieldExtractor(translation_config.field_rules)
        
        self.max_threads = max_threads
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.translation_callback = translation_callback
        
        self.stop_flag = threading.Event()
        self.translation_records = []
        self.stats_lock = threading.Lock()
        
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
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        if self.log_callback:
            self.log_callback(formatted_msg)
        print(formatted_msg)
    
    def update_progress(self, current: int, total: int, status: str = ""):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(current, total, status)
    
    def record_translation(self, file_path: str, field_path: str, original: str, 
                          translated: str, status: str):
        """记录翻译详情"""
        self.translation_records.append({
            'file': file_path,
            'field': field_path,
            'original': original,
            'translated': translated,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
    
    def contains_chinese(self, text: str) -> bool:
        """检查是否包含中文"""
        return any('\u4e00' <= char <= '\u9fff' for char in str(text))
    
    def find_yaml_files(self, path: str) -> List[str]:
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
    
    def get_output_path(self, original_path: str, base_folder: str) -> str:
        """获取输出文件路径"""
        if self.config.output_mode == 'overwrite':
            backup_path = original_path + '.backup'
            if not os.path.exists(backup_path):
                try:
                    shutil.copy2(original_path, backup_path)
                    self.log(f"已创建备份文件: {os.path.basename(backup_path)}", "INFO")
                except Exception as e:
                    self.log(f"备份创建失败: {e}", "WARNING")
            return original_path
        
        output_folder = self.config.output_folder or os.path.join(os.path.dirname(original_path), 'translated')
        
        if self.config.keep_structure:
            rel_path = os.path.relpath(original_path, base_folder)
            output_path = os.path.join(output_folder, rel_path)
        else:
            filename = os.path.basename(original_path)
            output_path = os.path.join(output_folder, filename)
        
        if self.config.add_language_tag and self.config.language_tag:
            dir_name = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            name, ext = os.path.splitext(filename)
            
            tag = self.config.language_tag
            if self.config.tag_position == 'before_ext':
                new_filename = f"{name}.{tag.lstrip('_')}{ext}"
            else:
                new_filename = f"{name}{tag}{ext}"
            
            output_path = os.path.join(dir_name, new_filename)
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except Exception as e:
            self.log(f"输出目录创建失败: {e}", "ERROR")
        
        return output_path
    
    def translate_fields(self, fields: List[Dict[str, Any]], file_path: str) -> Tuple[int, int, int]:
        """
        翻译提取的字段列表
        返回: (成功数, 跳过数, 失败数)
        """
        successful = 0
        skipped = 0
        failed = 0
        
        for field_info in fields:
            if self.stop_flag.is_set():
                break
            
            field_name = field_info['field_name']
            field_path = field_info['path']
            original_value = field_info['value']
            parent = field_info['parent']
            
            # 检查是否跳过中文
            if self.config.skip_chinese and self.contains_chinese(original_value):
                skipped += 1
                self.record_translation(file_path, field_path, original_value, original_value, 'skipped')
                continue
            
            # 构建上下文
            context = self.field_extractor.build_context(field_info, fields)
            
            # 翻译
            translated_value, error = self.translator.translate(original_value, context)
            
            if error:
                failed += 1
                self.log(f"翻译失败: {field_path} - {error}", "ERROR")
                self.record_translation(file_path, field_path, original_value, original_value, 'failed')
            else:
                successful += 1
                if self.translation_callback:
                    self.translation_callback(original_value, translated_value)
                
                # 应用双语输出
                if self.config.enable_bilingual and translated_value != original_value:
                    sep = self.config.bilingual_separator
                    if self.config.bilingual_order == 'cn_first':
                        parent[field_name] = f"{translated_value}{sep}{original_value}"
                    else:
                        parent[field_name] = f"{original_value}{sep}{translated_value}"
                else:
                    parent[field_name] = translated_value
                
                self.record_translation(file_path, field_path, original_value, parent[field_name], 'success')
        
        return successful, skipped, failed
    
    def process_yaml_file(self, file_path: str, base_folder: str):
        """处理单个YAML文件"""
        if self.stop_flag.is_set():
            return
        
        file_name = os.path.basename(file_path)
        self.log(f"处理文件: {file_name}")
        
        try:
            # 加载YAML文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=CustomYAMLLoader)
            
            if data is None:
                self.log(f"文件为空或格式不正确: {file_name}", "WARNING")
                return
            
            # 提取需要翻译的字段
            fields = self.field_extractor.extract_fields(data)
            
            if not fields:
                self.log(f"未找到需要翻译的字段: {file_name}", "INFO")
                return
            
            self.log(f"找到 {len(fields)} 个待翻译字段", "INFO")
            
            # 翻译字段
            successful, skipped, failed = self.translate_fields(fields, file_path)
            
            # 保存文件
            output_path = self.get_output_path(file_path, base_folder)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False, 
                         default_flow_style=False, indent=2)
            
            # 更新统计
            with self.stats_lock:
                self.stats['processed_files'] += 1
                self.stats['successful_translations'] += successful
                self.stats['skipped_translations'] += skipped
                self.stats['failed_translations'] += failed
            
            total_actions = successful + skipped + failed
            if output_path != file_path:
                self.log(f"完成: {file_name} → {os.path.basename(output_path)} (处理 {total_actions} 项)", "SUCCESS")
            else:
                self.log(f"完成: {file_name} (处理 {total_actions} 项)", "SUCCESS")
            
        except yaml.YAMLError as e:
            self.log(f"YAML解析失败 {file_name}: {str(e)[:100]}", "ERROR")
        except IOError as e:
            self.log(f"文件IO错误 {file_name}: {str(e)[:100]}", "ERROR")
        except Exception as e:
            self.log(f"处理失败 {file_name}: {str(e)[:100]}", "ERROR")
    
    def translate_files(self, file_paths: List[str], base_folder: str = None) -> Dict[str, Any]:
        """翻译文件列表"""
        self.stop_flag.clear()
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
        
        if not base_folder:
            if len(file_paths) == 1:
                base_folder = os.path.dirname(file_paths[0])
            else:
                try:
                    base_folder = os.path.commonpath(file_paths)
                except ValueError:
                    base_folder = os.path.dirname(file_paths[0])
        
        self.log(f"开始翻译 {len(file_paths)} 个文件")
        self.log(f"线程数: {self.max_threads}")
        self.log(f"输出模式: {'覆盖源文件' if self.config.output_mode == 'overwrite' else '导出到新文件夹'}")
        if self.config.enable_bilingual:
            order_text = "中文在前" if self.config.bilingual_order == 'cn_first' else "原文在前"
            sep = self.config.bilingual_separator
            self.log(f"双语输出: 已启用 ({order_text}，分隔符: '{sep}')")
        
        # 显示字段配置
        enabled_fields = [rule.field_name for rule in self.config.field_rules if rule.enabled]
        self.log(f"翻译字段: {', '.join(enabled_fields)}")
        self.log("=" * 60)
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(self.process_yaml_file, file_path, base_folder) 
                      for file_path in file_paths]
            
            for i, future in enumerate(futures):
                if self.stop_flag.is_set():
                    break
                try:
                    future.result()
                except Exception as e:
                    self.log(f"线程执行错误: {e}", "ERROR")
                self.update_progress(i + 1, len(file_paths), f"处理中: {i + 1}/{len(file_paths)}")
        
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
        self.stop_flag.set()

# ==================== 配置管理器 ====================

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "translator_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'api_keys': [],
            'current_key_id': None,
            'max_threads': 4,
            'skip_chinese': True,
            'api_timeout': 30,
            'enable_retry': True,
            'max_retries': 3,
            'retry_delay': 5,
            'output_mode': 'export',
            'output_folder': '',
            'keep_structure': True,
            'add_language_tag': True,
            'language_tag': '_zh_CN',
            'tag_position': 'end',
            'generate_report': True,
            'report_path': 'auto',
            'enable_bilingual': False,
            'bilingual_separator': ' | ',
            'bilingual_order': 'cn_first',
            'preset_tags': [
                {'tag': '_zh_CN', 'label': '简体中文'},
                {'tag': '_zh_TW', 'label': '繁体中文'},
                {'tag': '_cn', 'label': '中文简写'},
                {'tag': '_chinese', 'label': '英文标识'},
                {'tag': '_translated', 'label': '已翻译'}
            ],
            'tag_history': [],
            'max_tag_history': 10,
            'theme': 'light',
            'display_mode': 'simple',
            'sort_mode': 'add_order',
            'save_history': True,
            'max_history': 100,
            'history': [],
            # 字段规则配置
            'field_rules': [
                {'field_name': 'name', 'enabled': True, 'priority': 1, 'context_fields': ['description', 'tooltip']},
                {'field_name': 'description', 'enabled': True, 'priority': 2, 'context_fields': ['name']},
                {'field_name': 'tooltip', 'enabled': True, 'priority': 3, 'context_fields': ['name', 'description']},
                {'field_name': 'title', 'enabled': True, 'priority': 4, 'context_fields': []},
                {'field_name': 'label', 'enabled': True, 'priority': 5, 'context_fields': []},
            ]
        }
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        default_config = self._get_default_config()
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    default_config.update(loaded)
            except Exception as e:
                print(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def save_config(self) -> bool:
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, indent=2, fp=f, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def get_translation_config(self) -> TranslationConfig:
        """获取翻译配置对象"""
        field_rules = []
        for rule_data in self.config.get('field_rules', []):
            field_rules.append(FieldRule(
                field_name=rule_data['field_name'],
                enabled=rule_data.get('enabled', True),
                priority=rule_data.get('priority', 0),
                context_fields=rule_data.get('context_fields', [])
            ))
        
        return TranslationConfig(
            skip_chinese=self.config.get('skip_chinese', True),
            api_timeout=self.config.get('api_timeout', 30),
            enable_retry=self.config.get('enable_retry', True),
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 5),
            output_mode=self.config.get('output_mode', 'export'),
            output_folder=self.config.get('output_folder', ''),
            keep_structure=self.config.get('keep_structure', True),
            add_language_tag=self.config.get('add_language_tag', True),
            language_tag=self.config.get('language_tag', '_zh_CN'),
            tag_position=self.config.get('tag_position', 'end'),
            enable_bilingual=self.config.get('enable_bilingual', False),
            bilingual_separator=self.config.get('bilingual_separator', ' | '),
            bilingual_order=self.config.get('bilingual_order', 'cn_first'),
            field_rules=field_rules
        )
    
    def add_api_key(self, key_data: Dict[str, Any]) -> str:
        """添加 API Key"""
        key_id = str(int(time.time() * 1000))
        key_data['id'] = key_id
        key_data['created'] = datetime.now().isoformat()
        key_data['last_used'] = None
        key_data['use_count'] = 0
        
        self.config['api_keys'].append(key_data)
        self.save_config()
        return key_id
    
    def update_api_key(self, key_id: str, key_data: Dict[str, Any]) -> bool:
        """更新API Key"""
        for i, key in enumerate(self.config['api_keys']):
            if key['id'] == key_id:
                key_data['id'] = key_id
                key_data['created'] = key.get('created', datetime.now().isoformat())
                self.config['api_keys'][i] = key_data
                self.save_config()
                return True
        return False
    
    def remove_api_key(self, key_id: str):
        """删除API Key"""
        self.config['api_keys'] = [k for k in self.config['api_keys'] if k['id'] != key_id]
        if self.config['current_key_id'] == key_id:
            self.config['current_key_id'] = None
        self.save_config()
    
    def get_api_keys(self) -> List[Dict[str, Any]]:
        """获取所有API Keys"""
        return self.config['api_keys']
    
    def get_current_key(self) -> Optional[Dict[str, Any]]:
        """获取当前使用的Key"""
        key_id = self.config.get('current_key_id')
        if key_id:
            for key in self.config['api_keys']:
                if key['id'] == key_id:
                    return key
        return None
    
    def set_current_key(self, key_id: str):
        """设置当前使用的Key"""
        self.config['current_key_id'] = key_id
        self.save_config()
    
    def add_history(self, stats: Dict[str, Any], files: List[str]):
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

# ==================== 报告生成器（HTML模板提取）====================

class ReportGenerator:
    """翻译报告生成器"""
    
    # HTML模板常量
    HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>翻译对比报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Microsoft YaHei', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            margin: 5px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-number {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
        .file-section {{
            background: white;
            margin-bottom: 15px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .file-header {{
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .file-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .file-info {{
            color: #666;
            font-size: 13px;
        }}
        .translation-item {{
            border-left: 3px solid #4CAF50;
            padding: 12px;
            margin: 10px 0;
            background: #fafafa;
            border-radius: 4px;
        }}
        .translation-item.failed {{
            border-left-color: #f44336;
            background: #ffebee;
        }}
        .translation-item.skipped {{
            border-left-color: #FF9800;
            background: #fff3e0;
        }}
        .original {{
            color: #666;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .translated {{
            color: #000;
            font-weight: 500;
            font-size: 14px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .status-success {{
            background: #4CAF50;
            color: white;
        }}
        .status-failed {{
            background: #f44336;
            color: white;
        }}
        .status-skipped {{
            background: #FF9800;
            color: white;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>YAML翻译对比报告</h1>
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
    
    @staticmethod
    def _build_file_section(file_path: str, records: List[Dict[str, Any]]) -> str:
        """构建单个文件的HTML部分"""
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
        
        return f"""
        <div class="file-section">
            <div class="file-header">
                <div class="file-title">{file_name}</div>
                <div class="file-info">路径: {file_path}</div>
                <div class="file-info">统计: 成功 {success_count} | 跳过 {skipped_count} | 失败 {failed_count}</div>
            </div>
            {items_html}
        </div>
        """
    
    @staticmethod
    def generate_html_report(stats: Dict[str, Any], translation_records: List[Dict[str, Any]], 
                            output_path: str, api_config: Dict[str, Any]) -> Optional[str]:
        """生成HTML对比报告"""
        # 按文件分组记录
        files_data = {}
        for record in translation_records:
            file_path = record['file']
            if file_path not in files_data:
                files_data[file_path] = []
            files_data[file_path].append(record)
        
        # 构建所有文件部分的HTML
        file_sections_html = ""
        for file_path, records in files_data.items():
            file_sections_html += ReportGenerator._build_file_section(file_path, records)
        
        # 填充模板
        duration_str = f"{stats.get('duration', 0):.1f}秒"
        platform_name = PLATFORM_PRESETS.get(api_config.get('platform', 'deepseek'), {}).get('name', '未知')
        
        html_content = ReportGenerator.HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            platform=platform_name,
            model=api_config.get('model', ''),
            total_files=stats['total_files'],
            successful=stats['successful_translations'],
            skipped=stats.get('skipped_translations', 0),
            failed=stats['failed_translations'],
            duration=duration_str,
            file_sections=file_sections_html,
            app_name=APP_TITLE
        )
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return output_path
        except Exception as e:
            print(f"报告生成失败: {e}")
            return None

# ==================== GUI基类和工具 ====================

class BaseDialog:
    """对话框基类"""
    
    def __init__(self, parent: tk.Tk, title: str, width: int = 600, height: int = 500):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry(f"{width}x{height}")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.parent = parent
    
    def center_window(self):
        """居中显示窗口"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def destroy(self):
        """关闭对话框"""
        self.dialog.destroy()

# ==================== 字段配置对话框 ====================

class FieldConfigDialog(BaseDialog):
    """字段配置管理对话框"""
    
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager, callback=None):
        super().__init__(parent, "字段配置管理", 800, 600)
        self.config_manager = config_manager
        self.callback = callback
        self.field_rules = self.config_manager.config.get('field_rules', []).copy()
        
        self.setup_ui()
        self.load_field_rules()
        self.center_window()
    
    def setup_ui(self):
        """设置UI"""
        self.dialog.rowconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(self.dialog, padding="15")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text="字段配置管理", 
                 font=('Microsoft YaHei UI', 11, 'bold')).pack(side=tk.LEFT)
        
        ttk.Label(title_frame, text="配置需要翻译的字段及其上下文关联", 
                 font=('Microsoft YaHei UI', 9),
                 foreground='gray').pack(side=tk.LEFT, padx=(15, 0))
        
        # 主区域
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.grid(row=1, column=0, sticky='nsew')
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # 创建Treeview
        columns = ('enabled', 'field', 'priority', 'context')
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        self.tree.heading('enabled', text='启用')
        self.tree.heading('field', text='字段名')
        self.tree.heading('priority', text='优先级')
        self.tree.heading('context', text='上下文字段')
        
        self.tree.column('enabled', width=60, anchor='center')
        self.tree.column('field', width=150)
        self.tree.column('priority', width=80, anchor='center')
        self.tree.column('context', width=400)
        
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        self.tree.bind('<Double-1>', self.on_double_click)
        
        # 按钮区域
        btn_frame = ttk.Frame(self.dialog, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        ttk.Button(btn_frame, text="添加字段", command=self.add_field, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="编辑", command=self.edit_field, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除", command=self.delete_field, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="上移", command=self.move_up, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="下移", command=self.move_down, width=8).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="保存", command=self.save_config, width=12,
                  style='Accent.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy, width=12).pack(side=tk.RIGHT)
    
    def load_field_rules(self):
        """加载字段规则到列表"""
        self.tree.delete(*self.tree.get_children())
        
        for rule in self.field_rules:
            enabled_text = '✓' if rule.get('enabled', True) else ''
            context_text = ', '.join(rule.get('context_fields', []))
            
            self.tree.insert('', tk.END, values=(
                enabled_text,
                rule['field_name'],
                rule.get('priority', 0),
                context_text
            ))
    
    def on_double_click(self, event):
        """双击编辑"""
        self.edit_field()
    
    def add_field(self):
        """添加新字段"""
        self.show_field_editor()
    
    def edit_field(self):
        """编辑字段"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要编辑的字段", parent=self.dialog)
            return
        
        item = self.tree.item(selection[0])
        field_name = item['values'][1]
        
        for rule in self.field_rules:
            if rule['field_name'] == field_name:
                self.show_field_editor(rule)
                break
    
    def delete_field(self):
        """删除字段"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的字段", parent=self.dialog)
            return
        
        if not messagebox.askyesno("确认", "确定要删除选中的字段配置吗？", parent=self.dialog):
            return
        
        item = self.tree.item(selection[0])
        field_name = item['values'][1]
        
        self.field_rules = [r for r in self.field_rules if r['field_name'] != field_name]
        self.load_field_rules()
    
    def move_up(self):
        """上移"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        field_name = item['values'][1]
        
        for i, rule in enumerate(self.field_rules):
            if rule['field_name'] == field_name and i > 0:
                self.field_rules[i], self.field_rules[i-1] = self.field_rules[i-1], self.field_rules[i]
                # 更新优先级
                self.field_rules[i]['priority'] = i
                self.field_rules[i-1]['priority'] = i-1
                self.load_field_rules()
                break
    
    def move_down(self):
        """下移"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        field_name = item['values'][1]
        
        for i, rule in enumerate(self.field_rules):
            if rule['field_name'] == field_name and i < len(self.field_rules) - 1:
                self.field_rules[i], self.field_rules[i+1] = self.field_rules[i+1], self.field_rules[i]
                # 更新优先级
                self.field_rules[i]['priority'] = i
                self.field_rules[i+1]['priority'] = i+1
                self.load_field_rules()
                break
    
    def show_field_editor(self, rule_data=None):
        """显示字段编辑器"""
        is_edit = rule_data is not None
        
        editor = tk.Toplevel(self.dialog)
        editor.title("编辑字段" if is_edit else "添加字段")
        editor.geometry("500x400")
        editor.transient(self.dialog)
        editor.grab_set()
        
        editor.rowconfigure(1, weight=1)
        editor.columnconfigure(0, weight=1)
        
        # 标题
        ttk.Label(editor, text="编辑字段配置" if is_edit else "添加字段配置",
                 font=('Microsoft YaHei UI', 10, 'bold'),
                 padding="15").grid(row=0, column=0, sticky='ew')
        
        # 表单
        form = ttk.Frame(editor, padding="20")
        form.grid(row=1, column=0, sticky='nsew')
        
        # 字段名
        ttk.Label(form, text="字段名:").grid(row=0, column=0, sticky=tk.W, pady=8)
        field_name_var = tk.StringVar(value=rule_data['field_name'] if is_edit else '')
        field_entry = ttk.Entry(form, textvariable=field_name_var, width=40)
        field_entry.grid(row=0, column=1, pady=8, sticky='ew')
        if is_edit:
            field_entry.config(state='disabled')
        
        # 启用
        enabled_var = tk.BooleanVar(value=rule_data.get('enabled', True) if is_edit else True)
        ttk.Checkbutton(form, text="启用此字段", variable=enabled_var).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=8)
        
        # 优先级
        ttk.Label(form, text="优先级:").grid(row=2, column=0, sticky=tk.W, pady=8)
        priority_var = tk.IntVar(value=rule_data.get('priority', 0) if is_edit else len(self.field_rules))
        ttk.Spinbox(form, from_=0, to=100, textvariable=priority_var, width=10).grid(
            row=2, column=1, sticky=tk.W, pady=8)
        ttk.Label(form, text="(数字越小优先级越高)", 
                 foreground='gray', font=('Microsoft YaHei UI', 8)).grid(
            row=2, column=1, sticky=tk.E, pady=8)
        
        # 上下文字段
        ttk.Label(form, text="上下文字段:").grid(row=3, column=0, sticky=tk.NW, pady=8)
        
        context_frame = ttk.Frame(form)
        context_frame.grid(row=3, column=1, sticky='ew', pady=8)
        
        context_text = tk.Text(context_frame, height=6, width=40, font=('Consolas', 9))
        context_text.pack(fill=tk.BOTH, expand=True)
        
        if is_edit and rule_data.get('context_fields'):
            context_text.insert('1.0', '\n'.join(rule_data['context_fields']))
        
        ttk.Label(form, text="每行一个字段名\n这些字段的值会作为翻译上下文", 
                 foreground='gray', font=('Microsoft YaHei UI', 8),
                 justify=tk.LEFT).grid(row=4, column=1, sticky=tk.W, pady=(0, 8))
        
        form.columnconfigure(1, weight=1)
        
        # 按钮
        btn_frame = ttk.Frame(editor, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        def save():
            field_name = field_name_var.get().strip()
            if not field_name:
                messagebox.showwarning("警告", "请输入字段名", parent=editor)
                return
            
            # 检查重复
            if not is_edit:
                for rule in self.field_rules:
                    if rule['field_name'] == field_name:
                        messagebox.showwarning("警告", "字段名已存在", parent=editor)
                        return
            
            # 解析上下文字段
            context_fields = []
            for line in context_text.get('1.0', tk.END).strip().split('\n'):
                line = line.strip()
                if line:
                    context_fields.append(line)
            
            new_rule = {
                'field_name': field_name,
                'enabled': enabled_var.get(),
                'priority': priority_var.get(),
                'context_fields': context_fields
            }
            
            if is_edit:
                # 更新现有规则
                for i, rule in enumerate(self.field_rules):
                    if rule['field_name'] == field_name:
                        self.field_rules[i] = new_rule
                        break
            else:
                # 添加新规则
                self.field_rules.append(new_rule)
            
            self.load_field_rules()
            editor.destroy()
        
        ttk.Button(btn_frame, text="保存", command=save, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=editor.destroy, width=12).pack(side=tk.LEFT)
    
    def save_config(self):
        """保存配置"""
        self.config_manager.config['field_rules'] = self.field_rules
        self.config_manager.save_config()
        
        if self.callback:
            self.callback()
        
        messagebox.showinfo("成功", "字段配置已保存", parent=self.dialog)
        self.destroy()

# ==================== API Key管理对话框 ====================

class KeyManagerDialog(BaseDialog):
    """API Key管理对话框"""
    
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager, callback=None):
        super().__init__(parent, "API Key 管理", 800, 500)
        self.config_manager = config_manager
        self.callback = callback
        
        self.setup_ui()
        self.refresh_tree()
        self.center_window()
    
    def setup_ui(self):
        """设置UI"""
        self.dialog.rowconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(self.dialog, padding="15 15 15 10")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text="API Key 管理", 
                 font=('Microsoft YaHei UI', 11, 'bold')).pack(anchor=tk.W)
        
        # 列表
        list_frame = ttk.Frame(self.dialog, padding="0 0 15 10")
        list_frame.grid(row=1, column=0, sticky='nsew', padx=15)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        columns = ('name', 'platform', 'model', 'status')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        self.tree.heading('name', text='名称')
        self.tree.heading('platform', text='平台')
        self.tree.heading('model', text='模型')
        self.tree.heading('status', text='状态')
        
        self.tree.column('name', width=150)
        self.tree.column('platform', width=150)
        self.tree.column('model', width=200)
        self.tree.column('status', width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # 按钮
        btn_frame = ttk.Frame(self.dialog, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        ttk.Button(btn_frame, text="添加", command=self.add_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="编辑", command=self.edit_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="测试", command=self.test_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除", command=self.remove_key, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=self.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    def refresh_tree(self):
        """刷新列表"""
        self.tree.delete(*self.tree.get_children())
        for k in self.config_manager.get_api_keys():
            platform_name = PLATFORM_PRESETS.get(k.get('platform', 'custom'), {}).get('name', '自定义')
            self.tree.insert('', tk.END, values=(
                k['name'], 
                platform_name, 
                k.get('model', 'N/A'),
                '未测试'
            ))
    
    def add_key(self):
        """添加Key"""
        KeyEditorDialog(self.dialog, self.config_manager, None, self.refresh_tree)
    
    def edit_key(self):
        """编辑Key"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要编辑的 API Key", parent=self.dialog)
            return
        
        item = self.tree.item(selection[0])
        key_name = item['values'][0]
        
        for k in self.config_manager.get_api_keys():
            if k['name'] == key_name:
                KeyEditorDialog(self.dialog, self.config_manager, k, self.refresh_tree)
                break
    
    def test_key(self):
        """测试Key"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要测试的 API Key", parent=self.dialog)
            return
        
        item = self.tree.item(selection[0])
        key_name = item['values'][0]
        
        for k in self.config_manager.get_api_keys():
            if k['name'] == key_name:
                self.test_api_key(k, selection[0])
                break
    
    def test_api_key(self, key_data: Dict[str, Any], tree_item):
        """测试API Key"""
        test_window = tk.Toplevel(self.dialog)
        test_window.title("测试 API 连接")
        test_window.geometry("400x250")
        test_window.resizable(False, False)
        test_window.transient(self.dialog)
        test_window.grab_set()
        
        test_window.rowconfigure(1, weight=1)
        test_window.columnconfigure(0, weight=1)
        
        ttk.Label(test_window, text="测试 API 连接", 
                 font=('Microsoft YaHei UI', 11, 'bold'),
                 padding="20").grid(row=0, column=0)
        
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
                
                rate_limiter = RateLimiter()
                translator = UniversalTranslator(api_config, rate_limiter)
                success, message = translator.test_connection()
                
                test_window.after(0, lambda: on_result(success, message))
                
            except Exception as e:
                test_window.after(0, lambda: on_result(False, str(e)))
        
        def on_result(success, message):
            progress.stop()
            
            if success:
                status_label.config(text="测试成功！")
                ttk.Label(content, text=message, foreground='green', wraplength=350).pack(pady=10)
                
                values = list(self.tree.item(tree_item)['values'])
                values[3] = '成功'
                self.tree.item(tree_item, values=values)
            else:
                status_label.config(text="测试失败")
                ttk.Label(content, text=message, foreground='red', wraplength=350).pack(pady=10)
            
            ttk.Button(content, text="确定", command=test_window.destroy, width=12).pack(pady=10)
        
        thread = threading.Thread(target=do_test, daemon=True)
        thread.start()
    
    def remove_key(self):
        """删除Key"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的 API Key", parent=self.dialog)
            return
        
        if messagebox.askyesno("确认", "确定要删除选中的 API Key 吗？", parent=self.dialog):
            item = self.tree.item(selection[0])
            key_name = item['values'][0]
            
            for k in self.config_manager.get_api_keys():
                if k['name'] == key_name:
                    self.config_manager.remove_api_key(k['id'])
                    break
            
            self.refresh_tree()
            if self.callback:
                self.callback()

# ==================== API Key编辑对话框 ====================

class KeyEditorDialog(BaseDialog):
    """API Key编辑对话框"""
    
    def __init__(self, parent, config_manager: ConfigManager, key_data: Optional[Dict] = None, callback=None):
        is_edit = key_data is not None
        title = "编辑 API Key" if is_edit else "添加 API Key"
        super().__init__(parent, title, 600, 750)
        
        self.config_manager = config_manager
        self.key_data = key_data
        self.callback = callback
        self.is_edit = is_edit
        
        self.setup_ui()
        self.center_window()
    
    def setup_ui(self):
        """设置UI"""
        self.dialog.rowconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        
        # 标题
        ttk.Label(self.dialog, 
                 text="编辑 API Key" if self.is_edit else "添加 API Key",
                 font=('Microsoft YaHei UI', 11, 'bold'),
                 padding="20 20 20 10").grid(row=0, column=0)
        
        # 创建滚动区域
        canvas = tk.Canvas(self.dialog, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky='nsew', padx=15, pady=10)
        
        scrollbar = ttk.Scrollbar(self.dialog, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        form = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=form, anchor='nw')
        
        def on_canvas_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind('<Configure>', on_canvas_configure)
        
        # 表单字段
        row = 0
        
        # 名称
        ttk.Label(form, text="名称:").grid(row=row, column=0, sticky=tk.W, pady=8, padx=(10, 5))
        self.name_var = tk.StringVar(value=self.key_data['name'] if self.is_edit else '')
        ttk.Entry(form, textvariable=self.name_var, width=40).grid(
            row=row, column=1, columnspan=2, pady=8, sticky='ew', padx=(0, 10))
        row += 1
        
        # 平台
        ttk.Label(form, text="平台:").grid(row=row, column=0, sticky=tk.W, pady=8, padx=(10, 5))
        self.platform_combo = ttk.Combobox(form, state='readonly', width=37)
        self.platform_combo['values'] = [preset['name'] for preset in PLATFORM_PRESETS.values()]
        self.platform_combo.grid(row=row, column=1, columnspan=2, pady=8, sticky='ew', padx=(0, 10))
        self.platform_combo.bind('<<ComboboxSelected>>', self.on_platform_change)
        row += 1
        
        # API Key
        ttk.Label(form, text="API Key:").grid(row=row, column=0, sticky=tk.W, pady=8, padx=(10, 5))
        self.key_var = tk.StringVar(value=self.key_data.get('api_key', '') if self.is_edit else '')
        self.key_entry = ttk.Entry(form, textvariable=self.key_var, show='*', width=40)
        self.key_entry.grid(row=row, column=1, columnspan=2, pady=8, sticky='ew', padx=(0, 10))
        row += 1
        
        # API URL
        ttk.Label(form, text="API URL:").grid(row=row, column=0, sticky=tk.W, pady=8, padx=(10, 5))
        self.url_var = tk.StringVar(value=self.key_data.get('url', '') if self.is_edit else '')
        self.url_entry = ttk.Entry(form, textvariable=self.url_var, width=40)
        self.url_entry.grid(row=row, column=1, columnspan=2, pady=8, sticky='ew', padx=(0, 10))
        row += 1
        
        # 模型
        ttk.Label(form, text="模型:").grid(row=row, column=0, sticky=tk.W, pady=8, padx=(10, 5))
        self.model_var = tk.StringVar(value=self.key_data.get('model', '') if self.is_edit else '')
        self.model_combo = ttk.Combobox(form, textvariable=self.model_var, width=30)
        self.model_combo.grid(row=row, column=1, pady=8, sticky='ew')
        
        self.fetch_btn = ttk.Button(form, text="获取模型", width=12, command=self.fetch_models)
        self.fetch_btn.grid(row=row, column=2, pady=8, padx=(5, 10), sticky='ew')
        row += 1
        
        # 高级选项
        advanced_frame = ttk.LabelFrame(form, text="高级选项", padding="10")
        advanced_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=10, padx=10)
        row += 1
        
        ttk.Label(advanced_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.temp_var = tk.DoubleVar(value=self.key_data.get('temperature', 0.3) if self.is_edit else 0.3)
        ttk.Entry(advanced_frame, textvariable=self.temp_var, width=10).grid(
            row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(advanced_frame, text="Max Tokens:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.tokens_var = tk.IntVar(value=self.key_data.get('max_tokens', 1000) if self.is_edit else 1000)
        ttk.Entry(advanced_frame, textvariable=self.tokens_var, width=10).grid(
            row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(advanced_frame, text="自定义提示词:", 
                 font=('Microsoft YaHei UI', 9, 'bold')).grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        self.prompt_text = scrolledtext.ScrolledText(advanced_frame, height=6, 
                                                     font=('Consolas', 9), wrap=tk.WORD)
        self.prompt_text.grid(row=3, column=0, columnspan=3, sticky='ew', pady=5)
        self.prompt_text.insert('1.0', self.key_data.get('custom_prompt', DEFAULT_PROMPT) 
                               if self.is_edit else DEFAULT_PROMPT)
        
        form.columnconfigure(1, weight=1)
        
        # 初始化平台选择
        if self.is_edit:
            platform_id = self.key_data.get('platform', 'deepseek')
            name = PLATFORM_PRESETS.get(platform_id, {}).get('name', '')
            self.platform_combo.set(name)
        else:
            self.platform_combo.current(1)  # 默认 DeepSeek
        
        self.on_platform_change()
        
        # 按钮
        btn_frame = ttk.Frame(self.dialog, padding="15")
        btn_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        
        ttk.Button(btn_frame, text="保存", command=self.save, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    def get_platform_id(self, name: str) -> Optional[str]:
        """从名称获取平台ID"""
        for pid, preset in PLATFORM_PRESETS.items():
            if preset['name'] == name:
                return pid
        return None
    
    def on_platform_change(self, event=None):
        """平台改变时更新表单"""
        platform_id = self.get_platform_id(self.platform_combo.get())
        if not platform_id:
            return
        
        preset = PLATFORM_PRESETS[platform_id]
        self.model_combo['values'] = preset['models']
        
        if not self.model_var.get() or self.model_var.get() not in preset['models']:
            self.model_var.set(preset.get('default_model', ''))
        
        self.url_var.set(preset['url'])
        
        if platform_id == 'custom':
            self.url_entry.config(state='normal')
            self.model_combo.config(state='normal')
        else:
            self.url_entry.config(state='disabled')
            self.model_combo.config(state='readonly')
    
    def fetch_models(self):
        """获取模型列表"""
        platform_id = self.get_platform_id(self.platform_combo.get())
        api_key = self.key_var.get().strip()
        base_url = self.url_var.get().strip()
        
        if not api_key:
            messagebox.showwarning("警告", "请先输入 API Key", parent=self.dialog)
            return
        
        if not base_url:
            messagebox.showwarning("警告", "请先输入 API URL", parent=self.dialog)
            return
        
        self.fetch_btn.config(state='disabled', text="获取中...")
        self.model_combo.config(state='disabled')
        
        def fetch_worker():
            try:
                models, error = UniversalTranslator.fetch_available_models(
                    platform_id, api_key, base_url, timeout=15
                )
                self.dialog.after(0, lambda: self.on_fetch_complete(models, error))
            except Exception as e:
                self.dialog.after(0, lambda: self.on_fetch_complete(None, str(e)))
        
        thread = threading.Thread(target=fetch_worker, daemon=True)
        thread.start()
    
    def on_fetch_complete(self, models: Optional[List[str]], error: Optional[str]):
        """获取模型完成"""
        self.fetch_btn.config(state='normal', text="获取模型")
        self.model_combo.config(state='normal')
        
        if error:
            messagebox.showerror("获取失败", error, parent=self.dialog)
        elif models:
            self.model_combo['values'] = models
            self.model_var.set(models[0] if models else '')
            messagebox.showinfo("成功", f"成功获取 {len(models)} 个模型！", parent=self.dialog)
    
    def save(self):
        """保存"""
        name = self.name_var.get().strip()
        api_key = self.key_var.get().strip()
        
        if not name:
            messagebox.showwarning("警告", "请输入名称", parent=self.dialog)
            return
        
        if not api_key:
            messagebox.showwarning("警告", "请输入 API Key", parent=self.dialog)
            return
        
        platform_id = self.get_platform_id(self.platform_combo.get())
        if not platform_id:
            messagebox.showwarning("警告", "请选择平台", parent=self.dialog)
            return
        
        custom_prompt = self.prompt_text.get('1.0', tk.END).strip()
        if not custom_prompt:
            custom_prompt = DEFAULT_PROMPT
        
        new_key_data = {
            'name': name,
            'platform': platform_id,
            'api_key': api_key,
            'model': self.model_var.get().strip(),
            'url': self.url_var.get().strip(),
            'temperature': self.temp_var.get(),
            'max_tokens': self.tokens_var.get(),
            'custom_prompt': custom_prompt
        }
        
        if self.is_edit:
            self.config_manager.update_api_key(self.key_data['id'], new_key_data)
        else:
            self.config_manager.add_api_key(new_key_data)
        
        messagebox.showinfo("成功", "API Key 已保存", parent=self.dialog)
        
        if self.callback:
            self.callback()
        
        self.destroy()

# ==================== 设置对话框 ====================

class SettingsDialog(BaseDialog):
    """设置对话框"""
    
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager, callback=None):
        super().__init__(parent, "设置", 700, 700)
        self.config_manager = config_manager
        self.callback = callback
        
        self.setup_ui()
        self.load_settings()
        self.center_window()
    
    def setup_ui(self):
        """设置UI"""
        self.dialog.rowconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(self.dialog, padding="15 15 15 10")
        title_frame.grid(row=0, column=0, sticky='ew')
        ttk.Label(title_frame, text="设置", 
                 font=('Microsoft YaHei UI', 11, 'bold')).pack(anchor=tk.W)
        
        # Notebook
        notebook = ttk.Notebook(self.dialog)
        notebook.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 10))
        
        # 输出设置选项卡
        self.create_output_tab(notebook)
        
        # 翻译设置选项卡
        self.create_translation_tab(notebook)
        
        # 界面设置选项卡
        self.create_ui_tab(notebook)
        
        # 按钮
        btn_frame = ttk.Frame(self.dialog, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        ttk.Button(btn_frame, text="保存", command=self.save_settings, 
                  width=12, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    def create_output_tab(self, notebook):
        """创建输出设置选项卡"""
        output_tab = ttk.Frame(notebook, padding="15")
        notebook.add(output_tab, text="输出设置")
        
        # 输出模式
        mode_frame = ttk.LabelFrame(output_tab, text="输出模式", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.output_mode_var = tk.StringVar()
        
        ttk.Radiobutton(mode_frame, text="导出到指定文件夹（推荐）", 
                       variable=self.output_mode_var, value='export').pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(mode_frame, text="覆盖原文件（危险，但会创建备份）", 
                       variable=self.output_mode_var, value='overwrite').pack(anchor=tk.W, pady=2)
        
        # 导出选项
        export_frame = ttk.LabelFrame(output_tab, text="导出选项", padding="10")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        folder_frame = ttk.Frame(export_frame)
        folder_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(folder_frame, text="输出文件夹:").pack(side=tk.LEFT)
        self.output_folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=35).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(folder_frame, text="浏览...", 
                  command=self.browse_output_folder, width=10).pack(side=tk.LEFT)
        
        self.keep_structure_var = tk.BooleanVar()
        ttk.Checkbutton(export_frame, text="保持原目录结构", 
                       variable=self.keep_structure_var).pack(anchor=tk.W, pady=5)
        
        # 语言标识
        tag_frame = ttk.LabelFrame(output_tab, text="语言标识", padding="10")
        tag_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.add_tag_var = tk.BooleanVar()
        ttk.Checkbutton(tag_frame, text="添加语言标识到文件名", 
                       variable=self.add_tag_var).pack(anchor=tk.W, pady=5)
        
        tag_input_frame = ttk.Frame(tag_frame)
        tag_input_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        ttk.Label(tag_input_frame, text="标识:").pack(side=tk.LEFT)
        self.language_tag_var = tk.StringVar()
        self.tag_combo = ttk.Combobox(tag_input_frame, textvariable=self.language_tag_var, width=15)
        self.tag_combo.pack(side=tk.LEFT, padx=8)
        
        position_frame = ttk.Frame(tag_frame)
        position_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        self.tag_position_var = tk.StringVar()
        
        ttk.Label(position_frame, text="位置:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(position_frame, text="末尾", 
                       variable=self.tag_position_var, value='end').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(position_frame, text="扩展名前", 
                       variable=self.tag_position_var, value='before_ext').pack(side=tk.LEFT)
        
        # 双语输出
        bilingual_frame = ttk.LabelFrame(output_tab, text="双语输出", padding="10")
        bilingual_frame.pack(fill=tk.X)
        
        self.bilingual_var = tk.BooleanVar()
        ttk.Checkbutton(bilingual_frame, text="启用双语输出（同时显示中文和原文）", 
                       variable=self.bilingual_var).pack(anchor=tk.W, pady=5)
        
        sep_frame = ttk.Frame(bilingual_frame)
        sep_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        ttk.Label(sep_frame, text="分隔符:").pack(side=tk.LEFT)
        self.separator_var = tk.StringVar()
        ttk.Combobox(sep_frame, textvariable=self.separator_var, 
                    values=[' | ', ' / ', ' - ', ' · ', ' '], width=8).pack(side=tk.LEFT, padx=8)
        
        order_frame = ttk.Frame(bilingual_frame)
        order_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        self.bilingual_order_var = tk.StringVar()
        
        ttk.Radiobutton(order_frame, text="中文 | 原文", 
                       variable=self.bilingual_order_var, value='cn_first').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(order_frame, text="原文 | 中文", 
                       variable=self.bilingual_order_var, value='en_first').pack(side=tk.LEFT)
    
    def create_translation_tab(self, notebook):
        """创建翻译设置选项卡"""
        trans_tab = ttk.Frame(notebook, padding="15")
        notebook.add(trans_tab, text="翻译设置")
        
        # 基本设置
        basic_frame = ttk.LabelFrame(trans_tab, text="基本设置", padding="10")
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.skip_chinese_var = tk.BooleanVar()
        ttk.Checkbutton(basic_frame, text="跳过已包含中文的字段", 
                       variable=self.skip_chinese_var).pack(anchor=tk.W, pady=2)
        
        thread_frame = ttk.Frame(basic_frame)
        thread_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(thread_frame, text="默认并发线程数:").pack(side=tk.LEFT, padx=(0, 8))
        self.thread_var = tk.IntVar()
        ttk.Spinbox(thread_frame, from_=1, to=200, textvariable=self.thread_var, width=10).pack(side=tk.LEFT)
        ttk.Label(thread_frame, text="(1-200)", foreground='gray').pack(side=tk.LEFT, padx=(8, 0))
        
        timeout_frame = ttk.Frame(basic_frame)
        timeout_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(timeout_frame, text="API 请求超时:").pack(side=tk.LEFT, padx=(0, 8))
        self.timeout_var = tk.IntVar()
        ttk.Spinbox(timeout_frame, from_=5, to=300, textvariable=self.timeout_var, width=10).pack(side=tk.LEFT)
        ttk.Label(timeout_frame, text="秒", foreground='gray').pack(side=tk.LEFT, padx=(8, 0))
        
        # 失败重试
        retry_frame = ttk.LabelFrame(trans_tab, text="失败重试", padding="10")
        retry_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.retry_var = tk.BooleanVar()
        ttk.Checkbutton(retry_frame, text="失败自动重试", 
                       variable=self.retry_var).pack(anchor=tk.W, pady=2)
        
        retry_count_frame = ttk.Frame(retry_frame)
        retry_count_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(retry_count_frame, text="重试次数:").pack(side=tk.LEFT, padx=(20, 8))
        self.retry_count_var = tk.IntVar()
        ttk.Spinbox(retry_count_frame, from_=1, to=10, 
                   textvariable=self.retry_count_var, width=8).pack(side=tk.LEFT)
        
        retry_delay_frame = ttk.Frame(retry_frame)
        retry_delay_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(retry_delay_frame, text="重试延迟:").pack(side=tk.LEFT, padx=(20, 8))
        self.retry_delay_var = tk.IntVar()
        ttk.Spinbox(retry_delay_frame, from_=1, to=60, 
                   textvariable=self.retry_delay_var, width=8).pack(side=tk.LEFT)
        ttk.Label(retry_delay_frame, text="秒").pack(side=tk.LEFT, padx=(8, 0))
        
        # 报告设置
        report_frame = ttk.LabelFrame(trans_tab, text="报告设置", padding="10")
        report_frame.pack(fill=tk.X)
        
        self.generate_report_var = tk.BooleanVar()
        ttk.Checkbutton(report_frame, text="生成对比报告 (HTML)", 
                       variable=self.generate_report_var).pack(anchor=tk.W, pady=2)
    
    def create_ui_tab(self, notebook):
        """创建界面设置选项卡"""
        ui_tab = ttk.Frame(notebook, padding="15")
        notebook.add(ui_tab, text="界面设置")
        
        # 主题设置
        theme_frame = ttk.LabelFrame(ui_tab, text="主题", padding="10")
        theme_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.theme_var = tk.StringVar()
        
        ttk.Radiobutton(theme_frame, text="亮色主题", 
                       variable=self.theme_var, value='light').pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(theme_frame, text="暗色主题", 
                       variable=self.theme_var, value='dark').pack(anchor=tk.W, pady=2)
        
        # 历史设置
        history_frame = ttk.LabelFrame(ui_tab, text="历史记录", padding="10")
        history_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.save_history_var = tk.BooleanVar()
        ttk.Checkbutton(history_frame, text="保存翻译历史", 
                       variable=self.save_history_var).pack(anchor=tk.W, pady=5)
        
        history_count_frame = ttk.Frame(history_frame)
        history_count_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(history_count_frame, text="最多保存:").pack(side=tk.LEFT, padx=(20, 8))
        self.max_history_var = tk.IntVar()
        ttk.Spinbox(history_count_frame, from_=10, to=1000, 
                   textvariable=self.max_history_var, width=10).pack(side=tk.LEFT)
        ttk.Label(history_count_frame, text="条记录").pack(side=tk.LEFT, padx=(8, 0))
        
        # 显示设置
        display_frame = ttk.LabelFrame(ui_tab, text="显示设置", padding="10")
        display_frame.pack(fill=tk.X)
        
        mode_frame = ttk.Frame(display_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="文件列表显示:").pack(side=tk.LEFT, padx=(0, 8))
        self.display_mode_var = tk.StringVar()
        ttk.Combobox(mode_frame, textvariable=self.display_mode_var, width=15, state='readonly',
                    values=['简洁模式', '详细模式', '超详细模式']).pack(side=tk.LEFT)
    
    def browse_output_folder(self):
        """浏览输出文件夹"""
        folder = filedialog.askdirectory(title="选择输出文件夹", parent=self.dialog)
        if folder:
            self.output_folder_var.set(folder)
    
    def load_settings(self):
        """加载设置"""
        config = self.config_manager.config
        
        # 输出设置
        self.output_mode_var.set(config.get('output_mode', 'export'))
        self.output_folder_var.set(config.get('output_folder', ''))
        self.keep_structure_var.set(config.get('keep_structure', True))
        self.add_tag_var.set(config.get('add_language_tag', True))
        self.language_tag_var.set(config.get('language_tag', '_zh_CN'))
        self.tag_position_var.set(config.get('tag_position', 'end'))
        
        # 设置标签下拉框
        tag_values = []
        for preset in config.get('preset_tags', []):
            tag_values.append(preset['tag'])
        self.tag_combo['values'] = tag_values
        
        self.bilingual_var.set(config.get('enable_bilingual', False))
        self.separator_var.set(config.get('bilingual_separator', ' | '))
        self.bilingual_order_var.set(config.get('bilingual_order', 'cn_first'))
        
        # 翻译设置
        self.skip_chinese_var.set(config.get('skip_chinese', True))
        self.thread_var.set(config.get('max_threads', 4))
        self.timeout_var.set(config.get('api_timeout', 30))
        self.retry_var.set(config.get('enable_retry', True))
        self.retry_count_var.set(config.get('max_retries', 3))
        self.retry_delay_var.set(config.get('retry_delay', 5))
        self.generate_report_var.set(config.get('generate_report', True))
        
        # 界面设置
        self.theme_var.set(config.get('theme', 'light'))
        self.save_history_var.set(config.get('save_history', True))
        self.max_history_var.set(config.get('max_history', 100))
        
        display_mode = config.get('display_mode', 'simple')
        mode_map = {'simple': '简洁模式', 'detail': '详细模式', 'ultra': '超详细模式'}
        self.display_mode_var.set(mode_map.get(display_mode, '简洁模式'))
    
    def save_settings(self):
        """保存设置"""
        config = self.config_manager.config
        
        # 输出设置
        config['output_mode'] = self.output_mode_var.get()
        config['output_folder'] = self.output_folder_var.get()
        config['keep_structure'] = self.keep_structure_var.get()
        config['add_language_tag'] = self.add_tag_var.get()
        config['language_tag'] = self.language_tag_var.get()
        config['tag_position'] = self.tag_position_var.get()
        config['enable_bilingual'] = self.bilingual_var.get()
        config['bilingual_separator'] = self.separator_var.get()
        config['bilingual_order'] = self.bilingual_order_var.get()
        
        # 翻译设置
        config['skip_chinese'] = self.skip_chinese_var.get()
        config['max_threads'] = self.thread_var.get()
        config['api_timeout'] = self.timeout_var.get()
        config['enable_retry'] = self.retry_var.get()
        config['max_retries'] = self.retry_count_var.get()
        config['retry_delay'] = self.retry_delay_var.get()
        config['generate_report'] = self.generate_report_var.get()
        
        # 界面设置
        config['theme'] = self.theme_var.get()
        config['save_history'] = self.save_history_var.get()
        config['max_history'] = self.max_history_var.get()
        
        mode_map = {'简洁模式': 'simple', '详细模式': 'detail', '超详细模式': 'ultra'}
        config['display_mode'] = mode_map.get(self.display_mode_var.get(), 'simple')
        
        self.config_manager.save_config()
        
        if self.callback:
            self.callback()
        
        messagebox.showinfo("成功", "设置已保存", parent=self.dialog)
        self.destroy()

# ==================== 历史记录对话框 ====================

class HistoryDialog(BaseDialog):
    """历史记录对话框"""
    
    def __init__(self, parent: tk.Tk, config_manager: ConfigManager):
        super().__init__(parent, "翻译历史记录", 900, 550)
        self.config_manager = config_manager
        
        self.setup_ui()
        self.load_history()
        self.center_window()
    
    def setup_ui(self):
        """设置UI"""
        self.dialog.rowconfigure(1, weight=1)
        self.dialog.columnconfigure(0, weight=1)
        
        # 标题
        title_frame = ttk.Frame(self.dialog, padding="15 15 15 5")
        title_frame.grid(row=0, column=0, sticky='ew')
        
        ttk.Label(title_frame, text="翻译历史记录", 
                 font=('Microsoft YaHei UI', 11, 'bold')).pack(side=tk.LEFT)
        
        max_history = self.config_manager.config.get('max_history', 100)
        current_count = len(self.config_manager.config.get('history', []))
        ttk.Label(title_frame, text=f"(最多保留{max_history}条，当前{current_count}条)",
                 font=('Microsoft YaHei UI', 8),
                 foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # 列表
        list_frame = ttk.Frame(self.dialog, padding="0 0 15 10")
        list_frame.grid(row=1, column=0, sticky='nsew', padx=15)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        columns = ('time', 'files', 'success', 'skipped', 'failed', 'duration')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        self.tree.heading('time', text='时间')
        self.tree.heading('files', text='文件数')
        self.tree.heading('success', text='成功')
        self.tree.heading('skipped', text='跳过')
        self.tree.heading('failed', text='失败')
        self.tree.heading('duration', text='耗时')
        
        self.tree.column('time', width=180)
        self.tree.column('files', width=80)
        self.tree.column('success', width=80)
        self.tree.column('skipped', width=80)
        self.tree.column('failed', width=80)
        self.tree.column('duration', width=100)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # 按钮
        btn_frame = ttk.Frame(self.dialog, padding="15")
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        ttk.Button(btn_frame, text="清除全部历史", command=self.clear_history, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=self.destroy, width=12).pack(side=tk.RIGHT, padx=5)
    
    def load_history(self):
        """加载历史"""
        for item in self.config_manager.config.get('history', []):
            time_str = item['timestamp'][:19].replace('T', ' ')
            duration_str = f"{item.get('duration', 0):.1f}秒"
            
            self.tree.insert('', tk.END, values=(
                time_str,
                item['processed_files'],
                item['successful_translations'],
                item.get('skipped_translations', 0),
                item['failed_translations'],
                duration_str
            ))
    
    def clear_history(self):
        """清除历史"""
        if messagebox.askyesno("确认", "确定要清除所有历史记录吗？", parent=self.dialog):
            self.config_manager.config['history'] = []
            self.config_manager.save_config()
            self.tree.delete(*self.tree.get_children())
            messagebox.showinfo("成功", "历史记录已清除", parent=self.dialog)

# ==================== 主GUI类（精简版）====================

class TranslatorGUI:
    """主GUI类 - 重构精简版"""
    
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
        
        self.setup_styles()
        self.create_ui()
        self.load_settings()
        self.apply_theme()
        self.bind_shortcuts()
        
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
    
    def create_ui(self):
        """创建UI"""
        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        self.create_menu_bar()
        self.create_toolbar()
        self.create_statusbar()
        self.create_main_content()
        self.create_bottom_bar()
    
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
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 工具菜单
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具(T)", menu=tools_menu)
        tools_menu.add_command(label="管理 API Key", command=self.show_key_manager)
        tools_menu.add_command(label="字段配置", command=self.show_field_config)
        tools_menu.add_command(label="设置...", command=self.show_settings, accelerator="Ctrl+,")
        
        # 帮助菜单
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助(H)", menu=help_menu)
        help_menu.add_command(label="翻译历史记录", command=self.show_history)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.root, padding="10 8")
        toolbar.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        toolbar.columnconfigure(1, weight=1)
        
        ttk.Label(toolbar, text="API Key:", font=('Microsoft YaHei UI', 9)).grid(
            row=0, column=0, padx=(0, 8), sticky='w')
        
        self.key_combo = ttk.Combobox(toolbar, state='readonly', font=('Consolas', 9))
        self.key_combo.grid(row=0, column=1, padx=(0, 8), sticky='ew')
        self.key_combo.bind('<<ComboboxSelected>>', self.on_key_selected)
        
        ttk.Button(toolbar, text="管理", command=self.show_key_manager, width=8).grid(
            row=0, column=2, padx=(0, 15))
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).grid(row=0, column=3, sticky='ns', padx=10)
        
        ttk.Label(toolbar, text="并发线程:", font=('Microsoft YaHei UI', 9)).grid(
            row=0, column=4, padx=(0, 8), sticky='w')
        
        self.thread_spin = ttk.Spinbox(toolbar, from_=1, to=200, width=8, font=('Consolas', 9))
        self.thread_spin.set(4)
        self.thread_spin.grid(row=0, column=5, padx=(0, 8))
    
    def create_statusbar(self):
        """创建状态栏"""
        statusbar = ttk.Frame(self.root, relief=tk.SUNKEN, padding="5 2")
        statusbar.grid(row=1, column=0, sticky='ew', padx=5)
        
        ttk.Label(statusbar, text=f"v{VERSION}", font=('Microsoft YaHei UI', 8)).pack(side=tk.LEFT, padx=5)
        ttk.Separator(statusbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.api_status_label = ttk.Label(statusbar, text="API未配置", font=('Microsoft YaHei UI', 8))
        self.api_status_label.pack(side=tk.LEFT, padx=5)
        
        self.file_count_status = ttk.Label(statusbar, text="文件: 0", font=('Microsoft YaHei UI', 8))
        self.file_count_status.pack(side=tk.LEFT, padx=5)
        
        self.status_text = ttk.Label(statusbar, text="就绪", font=('Microsoft YaHei UI', 8))
        self.status_text.pack(side=tk.LEFT, padx=5)
    
    def create_main_content(self):
        """创建主内容区域"""
        main_container = ttk.Frame(self.root)
        main_container.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)
        
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        
        # 左侧文件列表
        self.create_file_list_panel(main_container)
        
        # 右侧日志和进度
        self.create_log_panel(main_container)
    
    def create_file_list_panel(self, parent):
        """创建文件列表面板"""
        left_panel = ttk.LabelFrame(parent, text=" 待翻译文件 ", padding="8")
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)
        
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
            font=('Microsoft YaHei UI', 9)
        )
        self.file_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar.config(command=self.file_listbox.yview)
        
        if HAS_DND:
            self.file_listbox.drop_target_register(DND_FILES)
            self.file_listbox.dnd_bind('<<Drop>>', self.on_drop)
        
        # 按钮
        btn_frame = ttk.Frame(left_panel)
        btn_frame.grid(row=2, column=0, sticky='ew')
        
        ttk.Button(btn_frame, text="添加文件", command=self.add_files).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="添加文件夹", command=self.add_folder).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="清空", command=self.clear_files).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.file_count_label = ttk.Label(left_panel, text="已选择: 0 个文件",
                                         font=('Microsoft YaHei UI', 9, 'bold'))
        self.file_count_label.grid(row=3, column=0, sticky='w', pady=(8, 0))
    
    def create_log_panel(self, parent):
        """创建日志面板"""
        right_panel = ttk.Frame(parent)
        right_panel.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        
        # 进度
        progress_frame = ttk.LabelFrame(right_panel, text=" 翻译进度 ", padding="8")
        progress_frame.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="就绪")
        self.progress_label.grid(row=1, column=0, sticky='w')
        
        # 日志
        log_frame = ttk.LabelFrame(right_panel, text=" 运行日志 ", padding="8")
        log_frame.grid(row=1, column=0, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        self.log_text.grid(row=0, column=0, sticky='nsew', pady=(0, 8))
        
        self.log_text.tag_config("INFO", foreground="#333333")
        self.log_text.tag_config("SUCCESS", foreground="#008000", font=('Consolas', 9, 'bold'))
        self.log_text.tag_config("WARNING", foreground="#FF8C00", font=('Consolas', 9, 'bold'))
        self.log_text.tag_config("ERROR", foreground="#DC143C", font=('Consolas', 9, 'bold'))
        
        ttk.Button(log_frame, text="清空日志", command=self.clear_log).grid(row=1, column=0, sticky='ew')
    
    def create_bottom_bar(self):
        """创建底部控制栏"""
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        
        self.start_btn = ttk.Button(
            bottom_frame,
            text="开始翻译",
            command=self.start_translation,
            style='Accent.TButton',
            width=15
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            bottom_frame,
            text="停止翻译",
            command=self.stop_translation,
            state=tk.DISABLED,
            width=15
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(bottom_frame, text="字段配置", command=self.show_field_config, width=12).pack(side=tk.LEFT, padx=5)
    
    # ==================== 事件处理 ====================
    
    def on_key_selected(self, event):
        """选择API Key"""
        index = self.key_combo.current()
        if index >= 0:
            keys = self.config_manager.get_api_keys()
            self.config_manager.set_current_key(keys[index]['id'])
            self.api_status_label.config(text="API已连接")
    
    def on_drop(self, event):
        """处理拖拽"""
        files = self.root.tk.splitlist(event.data)
        for file_path in files:
            file_path = file_path.strip('{}')
            self.add_path(file_path)
    
    def add_path(self, path):
        """添加路径"""
        if os.path.isfile(path):
            if path.lower().endswith(('.yml', '.yaml')) and path not in self.file_queue:
                self.file_queue.append(path)
                self.file_listbox.insert(tk.END, os.path.basename(path))
        elif os.path.isdir(path):
            core = YamlTranslatorCore(
                {'api_key': '', 'platform': 'deepseek'}, 
                TranslationConfig(), 
                1
            )
            yaml_files = core.find_yaml_files(path)
            added = 0
            for f in yaml_files:
                if f not in self.file_queue:
                    self.file_queue.append(f)
                    self.file_listbox.insert(tk.END, os.path.basename(f))
                    added += 1
            if added > 0:
                self.log_message(f"[INFO] 从文件夹添加了 {added} 个 YAML 文件")
        