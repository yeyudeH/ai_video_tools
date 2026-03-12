# ==================== 导入模块 ====================
import os
import shutil
from pathlib import Path
from PIL import Image
import gradio as gr
from datetime import datetime
import re
import json
import requests
from urllib.parse import urlparse
import openai  # 确保已安装 openai 库
# 用于图像生成的第三方库
from dashscope import VideoSynthesis
from http import HTTPStatus
import time
import subprocess
import tempfile
import ast
import dashscope
from dashscope import MultiModalConversation
from ffmpy import FFmpeg
try:
    from ark import Ark  # 确保已安装 ark 库
except ImportError:
    Ark = None

# ==================== 全局配置 ====================
# 素材存储根目录
MATERIAL_BASE_PATH = "."
ROLE_SUB_PATH = "roles"
ENV_SUB_PATH = "envs"
TMP_PATH = "./tmp"  # 临时文件存储路径

# 默认分类
DEFAULT_CLASS_NAME = "default"

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']

# LLM 配置 (请根据实际配置修改)
llm_key = ""  # 替换为实际的 API key
llm_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 通常不需要修改
llm_model = "qwen3.5-plus"  # 或其他模型

# 图像生成配置 (根据实际服务修改)
IMAGE_SOURCE = "wan"  # 可选: "wan" (万象) 或 "seedance" (Seedance)
VIDEO_SOURCE = "wan"  # 可选: "wan" (万象) 或 "seedance" (Seedance)
default = {"image_source":IMAGE_SOURCE,"video_source":VIDEO_SOURCE}

# 万象图像配置 (如果使用)
WAN_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1"
WAN_IMAGE_KEY = ""
WAN_IMAGE_MODEL = "qwen-image-plus-2026-01-09"
WAN_IMAGE_SIZE = "1664*928"

# 万象视频配置 (如果使用)
WAN_VIDEO_URL = "https://dashscope.aliyuncs.com/api/v1"
WAN_VIDEO_KEY = ""
WAN_VIDEO_MODEL = "wan2.6-i2v-flash"
WAN_VIDEO_SIZE = "720P"
WAN_VIDEO_SIZE_MANY = "1280*720"#参考图版本
WAN_VIDEO_MODEL_MANY = "wan2.6-r2v-flash"#参考图版本
WAN_VIDEO_URL_MANY = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"#参考图版本

# Seedance图像配置 (如果使用)
SEEDANCE_IMAGE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SEEDANCE_IMAGE_KEY = ""
SEEDANCE_IMAGE_MODEL = "doubao-seedream-5-0-260128"
SEEDANCE_IMAGE_SIZE = "2k"

# Seedance配置 (如果使用)
SEEDANCE_VIDEO_URL = "https://api.seedance.com/v1"
SEEDANCE_VIDEO_KEY = ""
SEEDANCE_VIDEO_MODEL = "doubao-seedance-1-0-pro-250528"
SEEDANCE_VIDEO_SIZE = "16:9"
# ==================== 补充工具函数 ====================
def get_url(data):
    """
    通用提取URL(用于处理万相视频生成返回的视频下载信息)
    支持：JSON字符串 / 字典
    安全处理所有 null/None 情况
    """
    # 字符串先转字典
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return None
    
    # 确保是字典
    if not isinstance(data, dict):
        return None
    
    # 链式安全获取，任何一步为None都返回None
    try:
        return data.get("output") \
                   and data["output"].get("choices") \
                   and len(data["output"]["choices"]) > 0 \
                   and data["output"]["choices"][0].get("message") \
                   and data["output"]["choices"][0]["message"].get("content") \
                   and len(data["output"]["choices"][0]["message"]["content"]) > 0 \
                   and data["output"]["choices"][0]["message"]["content"][0].get("image") \
                   or None
    except:
        return None

def image_to_url(image_path):
    """
    将图片上传到图床，返回图片URL
    """
    # 确保图床URL已配置
    if not default.get('image_upload_url'):
        return None
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(default['image_upload_url'], files=files)
        response.raise_for_status()
        data = response.json()
        
        # 从响应中提取URL
        if 'url' in data:
            return data['url']
        elif 'files' in data and len(data['files']) > 0:
            return data['files'][0].get('url')
        return None

# ==================== 工具函数 ====================
def sanitize_name(name):
    """清理名称中的非法字符"""
    if not name:
        return None
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    return safe_name if safe_name else None


def ensure_dir(path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, save_folder, filename=None):
    """
    下载文件到指定文件夹，返回文件路径
    
    参数:
        url: 文件下载链接
        save_folder: 保存文件的文件夹路径
        filename: 可选，自定义文件名，如果不提供则从URL中提取
    
    返回:
        保存后的文件完整路径
    """
    # 确保保存文件夹存在
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # 如果未指定文件名，从URL中提取
    if not filename:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = 'downloaded_file'
    
    # 构建完整文件路径
    file_path = os.path.join(save_folder, filename)
    
    # 发送HTTP请求（使用stream=True流式下载大文件）
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查请求是否成功
    
    # 写入文件
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # 过滤掉保持连接的chunk
                f.write(chunk)
    
    # 获取绝对路径
    abs_path = os.path.abspath(file_path)
    
    print(f"文件已下载到: {abs_path}")
    return abs_path


# ==================== 角色创建函数 ====================
def create_role(
        name, 
        class_name, 
        role_description, 
        role_image, 
        role_audio, 
        audio_content, 
        is_confirmed):
    '''
    角色档案创建函数
    '''
    # 必填字段检查
    if name is None or name.strip() == "":
        return "❌ 错误：角色名称不能为空"
    
    if role_image is None:
        return "❌ 错误：角色图片为必填项"
    
    if role_audio is None:
        return "❌ 错误：角色音频为必填项"
    
    if audio_content is None or audio_content.strip() == "":
        return "❌ 错误：音频内容描述不能为空"
    
    # 清理名称
    safe_name = sanitize_name(name)
    if not safe_name:
        return "❌ 错误：角色名称包含非法字符"
    
    # 处理分类名称
    if class_name is None or class_name.strip() == "":
        class_name = DEFAULT_CLASS_NAME
    safe_class = sanitize_name(class_name) or DEFAULT_CLASS_NAME
    
    # 构建角色目录路径
    base_path = Path(MATERIAL_BASE_PATH) / ROLE_SUB_PATH
    role_dir = base_path / safe_class / safe_name
    
    # 检查角色是否已存在
    if role_dir.exists():
        if not is_confirmed:
            return f"⚠️ 警告：角色档案 '{safe_name}' 已存在，请勾选「强制覆盖」后重试！"
        else:
            shutil.rmtree(role_dir)
    
    # 创建目录
    ensure_dir(role_dir)
    
    # 保存角色图片
    try:
        img = Image.open(role_image)
        img_path = role_dir / f"{safe_name}.png"
        img.save(img_path, 'PNG')
    except Exception as e:
        return f"❌ 错误：图片保存失败 - {str(e)}"
    
    # 保存角色描述（可选）
    if role_description and role_description.strip():
        desc_path = role_dir / f"{safe_name}_description.txt"
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write(role_description)
    
    # 保存音频文件
    try:
        audio_ext = os.path.splitext(role_audio)[1].lower()
        if audio_ext not in ['.wav', '.mp3', '.ogg', '.flac']:
            audio_ext = '.mp3'
        audio_path = role_dir / f"{safe_name}{audio_ext}"
        shutil.copy2(role_audio, audio_path)
    except Exception as e:
        return f"❌ 错误：音频保存失败 - {str(e)}"
    
    # 保存音频内容描述
    audio_text_path = role_dir / f"{safe_name}_audio_content.txt"
    with open(audio_text_path, 'w', encoding='utf-8') as f:
        f.write(audio_content)
    
    # 生成创建时间记录
    meta_path = role_dir / f"{safe_name}_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f'{{"created_at": "{datetime.now().isoformat()}", "type": "role"}}')
    
    return f"✅ 成功：角色档案 '{safe_name}' 已创建！\n📁 存储路径：{role_dir}"

# ==================== 环境创建函数 ====================
def create_env(
        name, 
        class_name,
        env_description, 
        env_image, 
        is_confirmed,

        ):
    '''
    环境素材创建函数
    '''
    # 必填字段检查
    if name is None or name.strip() == "":
        return "❌ 错误：环境名称不能为空"
    
    if env_image is None:
        return "❌ 错误：环境图片为必填项"
    
    # 清理名称
    safe_name = sanitize_name(name)
    if not safe_name:
        return "❌ 错误：环境名称包含非法字符"
    
    # 处理分类名称
    if class_name is None or class_name.strip() == "":
        class_name = DEFAULT_CLASS_NAME
    safe_class = sanitize_name(class_name) or DEFAULT_CLASS_NAME
    
    # 构建环境目录路径
    base_path = Path(MATERIAL_BASE_PATH) / ENV_SUB_PATH
    env_dir = base_path / safe_class / safe_name
    
    # 检查环境是否已存在
    if env_dir.exists():
        if not is_confirmed:
            return f"⚠️ 警告：环境素材 '{safe_name}' 已存在，请勾选「强制覆盖」后重试！"
        else:
            shutil.rmtree(env_dir)
    
    # 创建目录
    ensure_dir(env_dir)
    
    # 保存环境图片
    try:
        img = Image.open(env_image)
        # 检查图片格式
        img_ext = os.path.splitext(env_image)[1].lower()
        save_ext = '.png' if img_ext not in SUPPORTED_IMAGE_FORMATS else img_ext
        img_path = env_dir / f"{safe_name}{save_ext}"
        img.save(img_path)
    except Exception as e:
        return f"❌ 错误：图片保存失败 - {str(e)}"
    
    # 保存环境描述（可选）
    if env_description and env_description.strip():
        desc_path = env_dir / f"{safe_name}_description.txt"
        with open(desc_path, 'w', encoding='utf-8') as f:
            f.write(env_description)
    
    # 生成创建时间记录
    meta_path = env_dir / f"{safe_name}_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f'{{"created_at": "{datetime.now().isoformat()}", "type": "environment"}}')
    
    # 创建环境配置文件（可选，用于后续扩展）
    config_path = env_dir / f"{safe_name}_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(f"""# 环境配置文件
name: {safe_name}
created: {datetime.now().isoformat()}
image: {safe_name}{save_ext}
description: |
  {env_description if env_description else '无描述'}
""")
    
    return f"✅ 成功：环境素材 '{safe_name}' 已创建！\n📁 存储路径：{env_dir}"


# ==================== 剧本创作函数 ====================
def script_expand(simple_text):
    '''
    剧本扩写函数
    '''
    user_content = f"""请根据以下简易文本描述，扩充成详细的分镜剧本。

【简易描述】
{simple_text}

【输出要求】
1. 按时间顺序组织内容，以时间为主键
2. 每个场景必须包含以下字段：
   - 时间：场景发生的时间点
   - 环境：场景的环境描述
   - 镜头：镜头运动和拍摄方式
   - 人物：场景中的人物
   - 事件/动作/情节：具体发生的事件和动作
   - 备注：时间必须是整数单位,每个场景时间最低为2秒,最高为10秒,时间单位为秒

3. 输出为纯文本格式，不要使用 JSON
4. 确保内容连贯、逻辑清晰
"""

    client = openai.OpenAI(
        api_key=llm_key,
        base_url=llm_url,
    )

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "你是一名专业的剧本创作助手，擅长根据简要描述扩写成详细的分镜剧本。"},
            {"role": "user", "content": user_content},
        ]
    )
    json_string = completion.choices[0].message.content
    
    return json_string

def script_update(script, suggestions):
    user_content = f"""请根据以下改良建议，对现有剧本进行优化和改进。

【现有剧本】
{script}

【改良建议】
{suggestions}

【工作要求】
1. 保留原剧本的核心结构和格式
2. 根据建议进行针对性修改
3. 确保修改后的内容连贯、逻辑清晰
4. 输出为纯文本格式，不要使用 JSON
5. 如果建议与剧本内容冲突，请合理调整
"""

    client = openai.OpenAI(
        api_key=llm_key,
        base_url=llm_url,
    )

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "你是一名专业的剧本创作助手，擅长根据修改建议对剧本进行优化和改进。"},
            {"role": "user", "content": user_content},
        ]
    )
    json_string = completion.choices[0].message.content
    
    return json_string

def script_json(script):
    user_content = f"""请将以下剧本转换为结构化的 JSON 数据。

【剧本内容】
{script}

【输出要求】
1. 必须输出有效的 JSON 格式
2. 输出为一个列表，每个场景是列表中的一个对象
3. 每个场景对象必须包含以下字段：
   - 消耗时间：场景的时间长度，单位为秒（数字类型，如 5 表示 5 秒）
   - 环境：场景的环境描述（字符串）
   - 镜头：镜头运动和拍摄方式（字符串）
   - 人物：场景中的人物（字符串或数组）
   - 事件/动作/情节：具体发生的事件和动作（字符串）
   - 备注：其他需要说明的信息（字符串，可为空）

4. 不要输出任何解释文字，只输出 JSON
5. 确保 JSON 格式正确，可以被直接解析
6. "消耗时间" 必须是数字，用于后续视频生成参数

【JSON 模板示例】
[
  {{
    "消耗时间": 5,
    "环境": "城市街道，下雨天",
    "镜头": "远景，缓慢推进",
    "人物": "女孩",
    "事件/动作/情节": "女孩在雨中奔跑，表情焦急",
    "备注": "营造紧张氛围"
  }},
  {{
    "消耗时间": 5,
    "环境": "街角咖啡店门口",
    "镜头": "中景，跟随拍摄",
    "人物": "女孩，店员",
    "事件/动作/情节": "女孩跑到咖啡店门口躲雨，抬头看向店内",
    "备注": ""
  }}
]
"""
    
    client = openai.OpenAI(
        api_key=llm_key,
        base_url=llm_url,
    )

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "你是一名专业的剧本创作助手，擅长将剧本转换为结构化 JSON。"},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"}
    )
    text = completion.choices[0].message.content
    
    # print(text)

    # 清理 Markdown 代码块标记
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
        
        # 如果是字典，检查是否是列表包装
        if isinstance(data, dict):
            # 如果字典只有一个键，且值是列表，返回该列表
            if len(data) == 1:
                value = list(data.values())[0]
                if isinstance(value, list):
                    return value
            return [data]
        elif isinstance(data, list):
            return data
        else:
            return [{"error": "返回数据格式不正确"}]
    except json.JSONDecodeError as e:
        return [{"error": f"无法解析 JSON 数据：{str(e)}", "raw_text": text[:500]}]

# ==================== 图片生成函数 ====================
def script_json_init(script_json_output,script_index,script_index_max):
    '''
    给重定义剧本片段总数量，返回当前序号剧本片段和剧本片段总数量
    '''
    script_index_max = len(script_json_output)

    script_json_output = ast.literal_eval(script_json_output)

    return script_json_output[script_index],script_index_max

def image_make(script):
    '''
    图像生成函数
    '''
    # 确保临时目录存在
    Path(TMP_PATH).mkdir(parents=True, exist_ok=True)
    
    if default['image_source'] == "wan" and dashscope is not None:
        # 调用万象图像生成接口
        dashscope.base_http_api_url = WAN_IMAGE_URL

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": script}
                ]
            }
        ]

        response = MultiModalConversation.call(
            api_key=WAN_IMAGE_KEY,
            model=WAN_IMAGE_MODEL,
            messages=messages,
            result_format='message',
            stream=False,
            watermark=False,
            prompt_extend=True,
            negative_prompt="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。",
            size=WAN_IMAGE_SIZE
        )

        if response.status_code == 200:
            # 从响应中提取图片URL
            data = response.output.choices[0].message.content
            if isinstance(data, list) and data:
                # 从列表中提取第一个图片URL
                image_url = data[0]['image']
                return download_file(image_url, TMP_PATH)
            else:
                return "❌ 错误：生成图片失败，响应格式不正确"
        else:
            return f"❌ 错误：万象API请求失败，状态码: {response.status_code}"
    
    elif default['image_source'] == "seedance" and Ark is not None:
        # 调用Seedance图像生成接口
        client = Ark(
            base_url=SEEDANCE_IMAGE_URL, 
            api_key=SEEDANCE_IMAGE_KEY, 
        )

        imagesResponse = client.images.generate( 
            model=SEEDANCE_IMAGE_MODEL,
            prompt=script,
            size=SEEDANCE_IMAGE_SIZE,
            response_format="url",
            watermark=False
        )

        if imagesResponse.data:
            return download_file(imagesResponse.data[0].url, TMP_PATH)
        else:
            return "❌ 错误：Seedance API返回空图片数据"
    
    else:
        return "❌ 错误：未配置图像生成源或缺少必要的库。请检查配置并安装依赖。"
# ==================== 视频生成函数 ====================

def video_make_init(script_json_output,script_image,image_script=None):
    '''
    生成视频初始化
    返回：
        剧本片段
        序号总数
        首帧图
    '''
    script_json_output = ast.literal_eval(script_json_output)

    script_index_max = len(script_json_output)

    return script_json_output[0],script_index_max,script_image

def video_make(script_start,script_images,image_script=None):
    '''
    控制选择逻辑（无图/首帧/参考图）
    参数：
        生成的剧本
        首帧图
    返回：
        视频
    '''
    script_start = ast.literal_eval(script_start)
    viedo_make_time = int(script_start["消耗时间"])

    png_url = []

    # 关键修复：确保 script_images 是列表
    if isinstance(script_images, str):
        script_images = [script_images]  # 如果是字符串，包成列表
    elif not script_images:
        script_images = []               # 如果是 None 或空，设为空列表

    # 现在可以安全遍历了
    for script_image in script_images:
        if script_image:  # 确保路径不为空
            try:
                url = upload_img(script_image)
                png_url.append(url)
            except Exception as e:
                print(f"上传失败 {script_image}: {e}")

    script_images = png_url  # 更新为 URL 列表

    if script_images is not None:#有首帧
        if len(script_images) > 1:
            return video_make_many_image(script_start,image_script,script_images,viedo_make_time)
        
        
        return video_make_one_image(script_start,script_images,viedo_make_time)

    return video_make_no_image(script_start,viedo_make_time)

def video_make_no_ro_one_image(script_start,script_image):
    '''
    基础的视频生成（无图/单图）
    '''
    # print(script_start)
    # print(type(script_start))
    try:
        script_start = ast.literal_eval(script_start)
        # print("script_start:\n",script_start)
        # print(type(script_start))
        video_make_time = script_start["消耗时间"]
    except:
        try:
            video_make_time = script_start["消耗时间"]
        except:
            video_make_time = -1

    print("消耗时间",video_make_time)

    if script_image:
        #单图
        script_image = upload_img(script_image)
        return video_make_one_image(script_start,script_image,video_make_time)
        
    return video_make_no_image(script_start,video_make_time)

def video_make_no_image(script, video_make_time=-1, shot_type="multi"):
    """
    生成视频（无图片输入）
    
    参数:
        script: 剧本片段内容
        video_time: 视频时长（秒），-1表示使用剧本中的消耗时间
        shot_type: 镜头类型（"single"单镜头或"multi"多镜头）
    
    返回:
        生成的视频文件路径
    """
    # 如果指定了视频时长，使用指定时长
    if video_make_time != -1:
        video_make_time = int(video_make_time)
    
    if default['video_source'] == "wan":
        # 调用万象视频生成接口
        video_url = video_make_wan_no_image(
            prompt=script,
            api_key=WAN_VIDEO_KEY,
            duration=video_make_time,
            model=WAN_VIDEO_MODEL,
            size=WAN_VIDEO_SIZE,
            shot_type=shot_type,
            base_url=WAN_VIDEO_URL,
            watermark=False
            )
        return download_file(video_url, TMP_PATH)
       
    elif default['video_source'] == "seedance":
        # 调用Seedance视频生成接口
        video_url = video_make_seedance_no_image(
            prompt=script,
            api_key=SEEDANCE_VIDEO_KEY,
            base_url=SEEDANCE_VIDEO_URL,
            duration=video_make_time,
            ratio=SEEDANCE_VIDEO_SIZE,#"16:9",
            watermark=False
            )
        return download_file(video_url, TMP_PATH)
    
    else:
        return "❌ 未配置视频生成源"

def video_make_one_image(script,image_url, video_make_time=-1, shot_type="multi"):
    """
    生成视频（首帧版本）
    
    参数:
        script: 剧本片段内容
        image_url:参考图外链
        video_make_time: 视频时长（秒），-1表示使用剧本中的消耗时间
        shot_type: 镜头类型（"single"单镜头或"multi"多镜头）
    
    返回:
        生成的视频文件路径
    """
    # 如果指定了视频时长，使用指定时长
    if video_make_time != -1:
        video_make_time = int(video_make_time)
    
    if default['video_source'] == "wan":
        # 调用万象视频生成接口
        video_url = video_make_wan_one_image(
            api_key=WAN_VIDEO_KEY,
            prompt=script,
            img_url=image_url,
            duration=video_make_time,
            model=WAN_VIDEO_MODEL,
            resolution=WAN_VIDEO_SIZE,#"720P",
            shot_type=shot_type,
            base_url=WAN_VIDEO_URL
            )

        # print(video_url)

        return download_file(video_url, TMP_PATH)
      
    elif default['video_source'] == "seedance":
        # 调用Seedance视频生成接口
        video_url = video_make_seedance_one_image(
            prompt=script,
            api_key=SEEDANCE_VIDEO_KEY,
            image_url=image_url,
            model=SEEDANCE_VIDEO_MODEL,
            base_url=SEEDANCE_VIDEO_URL,
            duration=video_make_time,
            ratio=SEEDANCE_VIDEO_SIZE,#"16:9",
            watermark=False
            )
        return download_file(video_url, TMP_PATH)
    
    else:
        return "❌ 未配置视频生成源"

def video_make_many_image(script,image_script,image_urls, video_make_time=-1, shot_type="multi"):
    '''
    生成视频（参考图版本）
    参数：
        script:剧本
        script_image:剧本辅助提示词
        image_urls:参考图列表
        video_make_time=-1:消耗时间
        shot_type="multi":镜头
    返回：
        生成视频文件路径
    ''' 
    # 如果指定了视频时长，使用指定时长
    if video_make_time != -1:
        video_make_time = int(video_make_time)
    
    if default['video_source'] == "wan":
        # 调用万象视频生成接口
        video_url = video_make_wan_many_image(
            api_key=WAN_VIDEO_KEY,
            prompt=script,
            image_prompt=image_script,
            reference_urls=image_urls,
            duration=video_make_time,
            model=WAN_VIDEO_MODEL_MANY,
            size=WAN_VIDEO_SIZE_MANY,#"1280*720",
            shot_type=shot_type,
            video_url=WAN_VIDEO_URL_MANY
            )

        return download_file(video_url, TMP_PATH)
      
    elif default['video_source'] == "seedance":
        # 调用Seedance视频生成接口
        video_url = video_make_seedance_many_image(
            image_prompt=image_script,
            prompt=script,
            api_key=SEEDANCE_VIDEO_KEY,
            image_urls=image_urls,
            model=SEEDANCE_VIDEO_MODEL,
            base_url=SEEDANCE_VIDEO_URL,
            duration=video_make_time,
            ratio=SEEDANCE_VIDEO_SIZE,#"16:9",
            watermark=False
            )
        return download_file(video_url, TMP_PATH)
    
    else:
        return "❌ 未配置视频生成源"

def video_make_wan_no_image(prompt, 
                   api_key,
                   duration,  
                   model, 
                   base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis",
                   size="1280*720", 
                   shot_type="multi", 
                   watermark=False):
    """
    wan无图生成视频。
    
    参数:
        prompt (str): 视频生成的提示词。
        api_key (str, optional): API Key。如果未提供，则从环境变量 DASHSCOPE_API_KEY 获取。
        duration (int): 视频时长（秒），默认 15。
        model (str): 模型名称，默认 'wan2.6-t2v'。
        size (str): 视频分辨率，默认 "1280*720"。
        shot_type (str): 镜头类型，默认 "multi"。
        prompt_extend (bool): 是否自动扩展提示词，默认 True。
        watermark (bool): 是否添加水印，默认 False。
        
    返回:
        视频下载链接。
    """

    video_headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    video_payload = {
        "model": model,
        "input": {
            "prompt": str(prompt),
            "reference_urls": ["https://cdn.wanx.aliyuncs.com/static/demo-wan26/vace.mp4"]
        },
        "parameters": {
            "size": size,
            "duration": duration,
            "shot_type": shot_type,
            "watermark": watermark
        }
    }

    response = requests.post(base_url, json=video_payload, headers=video_headers)

    task_id = response.json()["output"]["task_id"]
    view_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    view_headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        time.sleep(2)
        response = requests.get(view_url, headers=view_headers)
        data = response.json()["output"]["task_status"]
        if data == "SUCCEEDED":
            break
        elif data == "FAILED" or data == "CANCELED" or data == "UNKNOWN":
            return "生成任务失败"
    
    return response.json()["output"]["video_url"]

def video_make_wan_one_image(
    api_key: str,
    prompt: str,
    img_url: str,
    duration: int,
    model: str = 'wan2.6-i2v-flash',
    resolution: str = "720P",
    shot_type: str = "multi",
    prompt_extend: bool = True,
    watermark: bool = False,
    base_url: str = 'https://dashscope.aliyuncs.com/api/v1'
):
    """
    wan首帧生成视频。

    参数:
        api_key (str): DashScope API Key，若未提供则从环境变量 DASHSCOPE_API_KEY 获取。
        prompt (str): 视频生成的提示词。
        img_url (str): 输入图片的 URL。
        model (str): 使用的模型名称，默认为 'wan2.6-i2v-flash'。
        resolution (str): 输出视频分辨率，默认为 "720P"。
        duration (int): 视频时长（秒），默认为 10。
        shot_type (str): 镜头类型，默认为 "multi"。
        prompt_extend (bool): 是否启用提示词扩展，默认为 True。
        watermark (bool): 是否添加水印，默认为 False。
        base_url (str): DashScope API 的基础 URL，默认为北京地域地址。

    返回:
        下载视频链接
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")
    
    if api_key == "YOUR_API_KEY":
        return {
            "success": False,
            "error": "未提供有效的 API Key，请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数。"
        }

    dashscope.base_http_api_url = base_url

    print('正在生成视频，请稍候...')
    rsp = VideoSynthesis.call(
        api_key=api_key,
        model=model,
        prompt=str(prompt),
        img_url=img_url,
        resolution=resolution,
        duration=duration,
        shot_type=shot_type,
        prompt_extend=prompt_extend,
        watermark=watermark
    )

    # print(duration)

    if rsp.status_code == HTTPStatus.OK:
        # print(rsp)
        # print("")
        video_url = rsp.output.video_url
        # print(rsp)
        # print(f"视频生成成功！URL: {video_url}")
        return video_url

def video_make_wan_many_image(
        api_key,
        model,
        image_prompt,
        prompt,
        reference_urls:list,
        duration,
        size = "1280*720",
        shot_type="multi",
        video_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        ):
    
    """
    根据提示词和辅助提示词，图片生成视频
    返回下载链接
    """

    video_headers = {
        "X-DashScope-Async": "enable",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    video_payload = {
        "model": model,
        "input": {
            "prompt": str(image_prompt) + str(prompt),
            "reference_urls": reference_urls
        },
        "parameters": {
            "size": size,
            "duration": int(duration),
            "audio": True,  # Python 中使用布尔值 True，而非 JSON 的 true
            "shot_type": shot_type,
            "watermark": False
        }
    }
    
    response = requests.post(video_url, headers=video_headers, json=video_payload)

    print(response)
    print(type(response))

    # 检查响应状态码
    if response.status_code == 200:
        data = response.json()
        
    task_id = data["output"]["task_id"]

    view_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    view_headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        time.sleep(2)
        response = requests.get(view_url, headers=view_headers)
        data = response.json()["output"]["task_status"]
        if data == "SUCCEEDED":
            break
        elif data == "FAILED" or data == "CANCELED" or data == "UNKNOWN":
            return "生成任务失败"
    
    return response.json()["output"]["video_url"]

def video_make_seedance_no_image(prompt,
        api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        duration=-1,
        ratio="16:9",
        watermark=False):
    '''
    火山无图生成视频
    参数：
        prompt：提示词
        api_key：密钥
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        duration=-1,#自动分配时间
        ratio="16:9",#分辨率
        watermark=False#默认无水印
    返回：
        视频下载链接
    '''
    client = Ark(
        base_url=base_url,
        api_key=api_key,
    )

    create_result = client.content_generation.tasks.create(
        model="doubao-seedance-1-0-pro-250528", # Replace with Model ID 
        content=[
            {
                # Combination of text prompt and parameters
                "type": "text",
                "text": prompt
            }
        ],
        ratio=ratio,
        duration=duration,
        watermark=watermark,
    )

    resp = client.content_generation.tasks.get(
        task_id=create_result["id"],
    )
    

    return resp["content"]["video_url"]

def video_make_seedance_one_image(prompt,
        api_key,
        image_url,
        model,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        duration=-1,
        ratio="16:9",
        watermark=False):
    '''
    火山首帧生成视频
    参数：
        prompt=提示词
        api_key=密钥
        image_url=首帧图片外链
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        duration=-1,#自动分配时间
        ratio="16:9",#分辨率
        watermark=False#默认无水印
    返回：
        视频下载链接
    '''
    client = Ark(
        base_url=base_url,
        api_key=api_key,
    )

    create_result = client.content_generation.tasks.create(
        model=model, # Replace with Model ID 
        content=[
            {
                # Combination of text prompt and parameters
                "type": "text",
                "text": prompt
            },
            {
                # The URL of the first frame image
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ],
        ratio=ratio,
        duration=duration,
        watermark=watermark,
    )

    resp = client.content_generation.tasks.get(
        task_id=create_result["id"],
    )
    

    return resp["content"]["video_url"]

def video_make_seedance_many_image(
        image_prompt,
        prompt,
        api_key,
        image_urls,
        model,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        duration=-1,
        ratio="16:9",
        watermark=False):
    '''
    火山参考图生成视频，图片外链列表+提示词+辅助提示词
    image_prompt=辅助提示词
    prompt=提示词
    api_key=密钥
    image_url=首帧图片外链
    model=模型
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    duration=-1,#自动分配时间
    ratio="16:9",#分辨率
    watermark=False#默认无水印
    '''
    client = Ark(
        base_url=base_url,
        api_key=api_key,
    )

    content = [{
                "type": "text",
                "text": image_prompt + prompt
            }]

    for image_url in image_urls:
        content.append({ 
                "type": "image_url", 
                "image_url": { 
                    "url": image_url
                },
                "role": "reference_image"  
            })

    create_result = client.content_generation.tasks.create(
        model=model, # Replace with Model ID 
        content=content,
        ratio=ratio,
        duration=duration,
        watermark=watermark,
    )
    
    resp = client.content_generation.tasks.get(
        task_id=create_result["id"],
    )
    

    return resp["content"]["video_url"]

def video_make_pass_no_or_one(
        video_list,
        script_video,
        script_json_output,
        script_index,
        script_index_max,
        ):
    '''
    视频通过，放入提供列表存储，序号+1(无图/单图)
    参数：
        通过视频存储列表
        通过的视频
        总剧本
        当前序号
        最大序号

    返回：
        通过视频存储列表
        新片段
        新首帧
        新序号
        新视频
    '''
    # print("test")
    # print(type(video_list))
    # print(type(script_index))
    # print(type(script_index_max))
    video_list.append(script_video)
    
    script_index+=1

    image = get_last_frame_path(script_video)

    # print(script_index)
    # print(script_index_max)

    if script_index == script_index_max:
        # script_index=0
        return video_list,"已全部生成完成,请勿继续，新剧本请重启",image,script_index,script_video

    script_new = ast.literal_eval(script_json_output)

    script_new = script_new[script_index]

    return video_list,script_new,image,script_index,video_make_no_ro_one_image(script_new,image)

def get_last_frame_path(video_path):
    """
    提取视频最后一帧并保存到 TMP_PATH 目录下，返回图片路径。
    依赖系统 ffmpeg 命令。
    """
    # 1. 确保目标目录存在
    os.makedirs(TMP_PATH, exist_ok=True)
    
    # 2. 生成输出文件名 (基于视频文件名)
    basename = os.path.basename(video_path)
    name, _ = os.path.splitext(basename)
    # 文件名标记改为 _last_frame
    output_filename = f"{name}_last_frame.jpg"
    output_path = os.path.join(TMP_PATH, output_filename)
    
    # 3. 构建 FFmpeg 命令 (关键修改在这里)
    # -sseof -3: 从文件末尾向前倒数 3 秒开始读取 (防止最后一帧是黑屏或损坏，留一点余量)
    # -i video_path: 输入文件
    # -vframes 1: 只读取 1 帧
    # -y: 覆盖已存在的文件
    # 注意：-sseof 必须放在 -i 之前
    cmd = [
        'ffmpeg', '-v', 'error', 
        '-sseof', '-0.08',  # 【关键】从结尾倒数 n 秒处开始(wan 0.08,seedance 0.06)
        '-i', video_path,
        '-vframes', '1', 
        '-y', 
        output_path
    ]
    
    try:
        # 执行命令
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 额外检查：如果生成的文件大小为 0，说明提取失败（可能视频太短或损坏）
        if os.path.getsize(output_path) == 0:
            print(f"警告：提取的尾帧文件大小为 0，可能视频过短或损坏: {video_path}")
            return None
            
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 错误: {e.stderr.decode() if e.stderr else '未知错误'}")
        return None
    except FileNotFoundError:
        print("错误：未找到 ffmpeg 命令，请确保已安装并添加到环境变量。")
        return None
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return None

def upload_img(path):
    '''
    图床返回外链
    '''
    with open(path, 'rb') as f:
        # 直接返回解析后的 URL
        return requests.post('图床链接', files={'files[]': f}).json()['files'][0]['url'] 

# ==================== 进阶视频生成函数 ====================
def video_make_init_advanced(
        script_json_output,
        image_confirm_state,
        script_image,
        role_ex,
        env_ex,
        script_index
        ):
    '''
    生成视频初始化(进阶),
    返回：
        剧本片段
        序号总数
        首帧图
        辅助提示词
        首帧辅助提示词
    '''
    script_json_output = ast.literal_eval(script_json_output)

    script_index_max = len(script_json_output)

    index = 1
    image_name = ""#首帧图描述
    role_name = ""#人物辅助提示词
    env_name = ""#环境辅助提示词

    if script_image:
        if default["video_source"] == "seedance":
            image_name += f"图{index}是首帧图。"
        else:
            image_name += f"Character{index} 是首帧图。"
        index +=1

    for role_path in role_ex:
        # print(role_path)
        # print(type(role_path))
        if default["video_source"] == "seedance":
            role_name += f"图{index}是人物{Path(role_path[0]).stem}。"
        else:
            role_name += f"Character{index} 是人物{Path(role_path[0]).stem}。"
        index +=1
    
    for env_path in env_ex:
        if default["video_source"] == "seedance":
            env_name += f"图{index}是环境{Path(env_path[0]).stem}。"
        else:
            env_name += f"Character{index} 是环境{Path(env_path[0]).stem}。"
        index +=1

    image_prompt = role_name + env_name

    #首帧
    try:
        script_image = script_image if script_image else image_confirm_state
    except:
        script_image = None

    return script_json_output[script_index],script_index_max,script_image,image_prompt,image_name

def get_subfolders(target_path: str) -> list[str]:
    """获取目标路径下的所有直接子文件夹名称列表"""
    path = Path(target_path)
    return [d.name for d in path.iterdir() if d.is_dir()]

def get_subfolders_role(target_path: str) -> list[str]:
    """
    获取目标路径下的所有直接子文件夹名称列表(role)
    返回：
        类更新
        名更新
    """
    path = Path(ROLE_SUB_PATH) / target_path

    list_ = [d.name for d in path.iterdir() if d.is_dir()]

    try:
        sult = list_[0]
    except:
        sult = None
    return gr.Dropdown(value=target_path),gr.Dropdown(choices=list_,value=sult)

def updata_name_role(role_name_new):
    return gr.Dropdown(value=role_name_new)

def updata_name_env(env_name_new):
    return gr.Dropdown(value=env_name_new)

def get_subfolders_env(target_path: str) -> list[str]:
    """获取目标路径下的所有直接子文件夹名称列表(env)"""
    path = Path(ENV_SUB_PATH) / target_path

    list_ = [d.name for d in path.iterdir() if d.is_dir()]

    try:
        sult = list_[0]
    except:
        sult = None

    return gr.Dropdown(value=target_path),gr.Dropdown(choices=list_,value=sult)

def role_work_add(role_ex,role_class_new,role_name_new):
    '''
    添加人物到工作区
    '''
    if not role_ex:
        role_ex = []
    role_ex.append(f"{ROLE_SUB_PATH}/{role_class_new}/{role_name_new}/{role_name_new}.png")
    return role_ex

def env_work_add(env_ex,env_class_new,env_name_new):
    '''
    添加环境到工作区
    '''
    if not env_ex:
        env_ex = []
    env_ex.append(f"{ENV_SUB_PATH}/{env_class_new}/{env_name_new}/{env_name_new}.png")
    return env_ex

def new_role_btn():
    '''
    人物刷新按钮
    '''
    return gr.Dropdown(
                            choices=get_subfolders(ROLE_SUB_PATH),  # 🔑 可选列表
                            label="请选择分类",
                            value=get_subfolders(ROLE_SUB_PATH)[0],  # 默认值
                            interactive=True
                        )

def new_env_btn():
    return gr.Dropdown(
                            choices=get_subfolders(ENV_SUB_PATH),  # 🔑 可选列表
                            label="请选择分类",
                            value=get_subfolders(ENV_SUB_PATH)[0],  # 默认值
                            interactive=True
                        )

def video_make_many(
        script_start,
        image_one_script,
        image_script,
        script_image,
        role_ex,
        env_ex,
        ):
    '''
    生成视频（进阶）
    参数：
        剧本提示词
        首帧提示词
        辅助提示词
        首帧图
        人物图
        环境图
    返回：
        生成视频
        新首帧
        新剧本提示词
        辅助提示词
        剧本序号
    '''
    promat = image_one_script + image_script + script_start

    image_list = []

    if script_image:
        image_list.append(upload_img(script_image))

    for i in role_ex:
        image_list.append(upload_img(i[0]))

    for i in env_ex:
        image_list.append(upload_img(i[0]))

    try:
        script_start = ast.literal_eval(script_start)
        # print("script_start:\n",script_start)
        # print(type(script_start))
        video_make_time = script_start["消耗时间"]
    except:
        video_make_time = -1

    view_video = video_make_many_image(script_start,promat,image_list,video_make_time)

    return view_video

def video_make_many_pass(
        script_json_output,
        script_index,
        script_index_max,
        video_view,
        video_list,
        image_one_script,
        image_script,
        role_ex,
        env_ex,
    ):
    '''
    视频生成通过（进阶）
    参数：
        总剧本
        剧本序号
        剧本总序号
        原视频
        视频储存列表
        首帧辅助提示词
        辅助提示词
        人物列表
        环境列表
    返回：
        新序号
        新片段
        视频储存列表
        新视频
        新首帧提示词
        新辅助提示词
    '''
    script_json_output = ast.literal_eval(script_json_output)

    script_index+=1

    video_list.append(video_view)

    if script_index == script_index_max:
        script_index-=1
        return script_index,"已全部完成，请勿再按",video_list,video_view,image_one_script,image_script

    #剧本片段
    script_strat = script_json_output[script_index]

    #原视频首帧
    video_input_one_image = get_last_frame_path(video_view)

    #新构建辅助提示词
    index = 1
    image_name = ""#首帧图描述
    role_name = ""#人物辅助提示词
    env_name = ""#环境辅助提示词
    if image_one_script:#（全局已经首帧提示词，所以只更新首帧的提示词）
        if default["video_source"] == "seedance":
            image_name += f"图{index}是首帧图。"
        else:
            image_name += f"Character{index} 是首帧图。"

        image_prompt = image_script

    else:#没有首帧，全部构建一次
        if video_input_one_image:
            if default["video_source"] == "seedance":
                image_name += f"图{index}是首帧图。"
            else:
                image_name += f"Character{index} 是首帧图。"
            index +=1

        for role_path in role_ex:
            # print(role_path)
            # print(type(role_path))
            if default["video_source"] == "seedance":
                role_name += f"图{index}是人物{Path(role_path[0]).stem}。"
            else:
                role_name += f"Character{index} 是人物{Path(role_path[0]).stem}。"
            index +=1
        
        for env_path in env_ex:
            if default["video_source"] == "seedance":
                env_name += f"图{index}是环境{Path(env_path[0]).stem}。"
            else:
                env_name += f"Character{index} 是环境{Path(env_path[0]).stem}。"
            index +=1

        image_prompt = role_name + env_name

    #视频
    video_view = video_make_many(script_strat,image_name,image_prompt,video_input_one_image,role_ex,env_ex)

    return script_index,script_strat,video_list,video_view,image_name,image_prompt

# ==================== 视频拼接函数 ====================

def video_new_view(video_list,video_index):
    '''
    视频下一个预览函数
    参数：
        储存视频列表
        视频序号
    返回：
        新的预览视频
        新的序号
    '''
    if len(video_list) == video_index:
        video_index = 0
        return video_list[video_index],video_index,str(video_index)

    video_index+=1

    return video_list[video_index-1],video_index,str(video_index)

def video_splicing(video_list):
    '''
    视频拼接函数
    参数：
        储存视频列表
    返回：
        拼接完成的视频
    '''
    video_merge_preview = merge_videos(video_list)
    return video_merge_preview,video_merge_preview

def merge_videos(video_paths):
    '''
    视频拼接函数
    参数：
        视频路径列表
    返回：
        合成视频路径
    '''
    if not video_paths:
        return None
    
    # 检查文件是否存在，有缺失直接返回 None
    for p in video_paths:
        if not os.path.isfile(p):
            return None

    os.makedirs("output", exist_ok=True)

    # 生成文件名：原名_时间.mp4
    name, _ = os.path.splitext(os.path.basename(video_paths[0]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.abspath(os.path.join("output", f"{name}_{timestamp}.mp4"))

    # 创建临时列表文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        list_path = f.name
        for p in video_paths:
            f.write(f"file '{os.path.abspath(p).replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'\n")

    try:
        # 核心命令：-c copy 实现最轻量级无损合并
        ff = FFmpeg(inputs={list_path: None}, outputs={out_path: ["-c", "copy"]})
        ff.run()
        return out_path
    except:
        # 失败则清理可能产生的空文件并返回 None
        if os.path.exists(out_path):
            os.remove(out_path)
        return None
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
# ==================== Gradio 界面 ====================
def create_gradio_interface():
    '''创建 Gradio 多页面界面'''
    
    with gr.Blocks(title="ai视频生成系统") as demo:

        script_json_output = gr.State(value=None)#结构化剧本
        script_index = gr.State(value=0) # 结构化剧本序号
        script_index_max = gr.State(value=0)  # 序号总数量

        image_confirm_state = gr.State(value=None)#首帧图
        video_list = gr.State(value=[])#通过视频存储

        image_script = gr.State(value=None)#辅助提示词
        image_one_script = gr.State(value=None)#首帧辅助提示词

        video_index = gr.State(value=0)# 视频拼接页面序号
        video_splicing_end = gr.State(value=None)#拼接完成的视频路径

        gr.Markdown("# 🎬 素材导入管理系统")
        gr.Markdown("统一管理角色、环境、剧本与图片素材的导入工具")
        
        with gr.Tabs() as tabs:
            # ==================== Tab 1: 角色素材导入 ====================
            with gr.Tab("👤 角色素材导入"):
                gr.Markdown("### 创建角色档案")
                gr.Markdown("请完整填写以下信息以创建角色素材")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 基本信息")
                        role_name = gr.Textbox(
                            label="角色名称 *", 
                            placeholder="请输入角色名称",
                            info="必填项，将用于创建文件夹名称"
                        )
                        role_class = gr.Textbox(
                            label="角色分类", 
                            placeholder="请输入角色分类（可选）",
                            info="留空则使用默认分类 'default'"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 描述信息")
                        role_description = gr.Textbox(
                            label="角色描述", 
                            placeholder="请输入角色背景描述（可选）",
                            lines=3
                        )
                        role_audio_content = gr.Textbox(
                            label="音频内容描述 *", 
                            placeholder="请输入音频内容/台词描述",
                            lines=2,
                            info="必填项，将保存为文本文件"
                        )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 素材上传")
                        role_audio = gr.Audio(
                            label="上传音频 *", 
                            sources=["upload", "microphone"], 
                            type="filepath",
                            format="mp3"
                        )
                    
                    with gr.Column(scale=1):
                        role_image = gr.Image(
                            label="人物图片 *", 
                            type="filepath",
                            sources=["upload", "clipboard"]
                        )
                
                with gr.Row():
                    role_is_confirmed = gr.Checkbox(
                        label="⚠️ 强制覆盖已存在的角色档案", 
                        value=False
                    )
                
                with gr.Row():
                    role_btn = gr.Button("🚀 提交创建", variant="primary", scale=1)
                    role_output = gr.Textbox(
                        label="操作结果", 
                        lines=3, 
                        scale=2,
                        interactive=False
                    )
                
                # 绑定事件
                role_btn.click(
                    fn=create_role,
                    inputs=[
                        role_name, 
                        role_class, 
                        role_description, 
                        role_image, 
                        role_audio, 
                        role_audio_content, 
                        role_is_confirmed
                    ],
                    outputs=role_output
                )
                
                # 使用提示
                gr.Markdown("""
                ### 💡 角色导入提示
                1. **角色名称**、**音频** 和 **音频内容描述** 为必填项
                2. 图片支持 JPG、PNG、WEBP 等格式，将自动转换为 PNG 保存
                3. 音频支持 MP3、WAV、OGG、FLAC 格式
                4. 如需覆盖已存在的角色，请勾选「强制覆盖」选项
                """)
            
            # ==================== Tab 2: 环境素材导入 ====================
            with gr.Tab("🌍 环境素材导入"):
                gr.Markdown("### 创建环境素材")
                gr.Markdown("请完整填写以下信息以创建环境素材")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 基本信息")
                        env_name = gr.Textbox(
                            label="环境名称 *", 
                            placeholder="请输入环境名称",
                            info="必填项，将用于创建文件夹名称"
                        )
                        
                        env_class = gr.Textbox(
                            label="环境分类", 
                            placeholder="请输入环境分类（可选）",
                            info="留空则使用默认分类 'default'"
                        )

                        env_description = gr.Textbox(
                            label="环境描述", 
                            placeholder="请输入环境描述（可选）",
                            lines=4,
                            info="可描述环境特征、氛围、用途等"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 素材上传")
                        env_image = gr.Image(
                            label="环境图片 *", 
                            type="filepath",
                            sources=["upload", "clipboard"]
                        )
                        
                        env_is_confirmed = gr.Checkbox(
                            label="⚠️ 强制覆盖已存在的环境素材", 
                            value=False
                        )
                
                with gr.Row():
                    env_btn = gr.Button("🚀 提交创建", variant="primary", scale=1)
                    env_output = gr.Textbox(
                        label="操作结果", 
                        lines=5, 
                        scale=2,
                        interactive=False
                    )
                
                # 绑定事件
                env_btn.click(
                    fn=create_env,
                    inputs=[
                        env_name, 
                        env_class,
                        env_description, 
                        env_image, 
                        env_is_confirmed
                    ],
                    outputs=env_output
                )
                
                # 使用提示
                gr.Markdown("""
                ### 💡 环境导入提示
                1. **环境名称** 和 **环境图片** 为必填项
                2. 图片支持 JPG、PNG、WEBP、BMP 格式
                3. 环境描述为可选，建议填写以便后续管理
                4. 如需覆盖已存在的环境，请勾选「强制覆盖」选项
                5. 系统会自动生成 YAML 配置文件便于后续扩展
                """)
            
            # ==================== Tab 3: 剧本创作 ====================
            with gr.Tab("📝 剧本创作"):
                gr.Markdown("### 剧本创作工具")
                gr.Markdown("通过AI扩写和结构化剧本，用于视频生成")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 基础输入")
                        script_summary = gr.Textbox(
                            label="剧本概括 *", 
                            placeholder="请输入剧本的简要概括",
                            info="必填项，用于扩写剧本"
                        )
                        script_updata = gr.Textbox(
                            label="改良建议", 
                            placeholder="请输入对剧本的改良建议（可选）",
                            info="可选，用于优化现有剧本"
                        )
                        script_expand_btn = gr.Button("1. 扩写剧本", variant="primary")
                        script_update_btn = gr.Button("2. 改良剧本", variant="primary")
                    
                    with gr.Column(scale=1):
                        script_output = gr.Textbox(
                            label="加工后的剧本", 
                            placeholder="这里将显示扩写或改良后的剧本内容",
                            lines=15,
                            interactive=False
                        )
                
                gr.Markdown("### 结构化输出")
                with gr.Row():
                    script_json_btn = gr.Button("3. 转换为结构化JSON", variant="primary")
                    script_json_output = gr.Textbox(
                        label="结构化剧本", 
                        placeholder="这里将显示结构化后的剧本内容",
                        lines=15,
                        interactive=False
                    )
                
                # 绑定事件
                script_expand_btn.click(
                    fn=script_expand,
                    inputs=[script_summary],
                    outputs=script_output
                )
                script_update_btn.click(
                    fn=script_update,
                    inputs=[script_output, script_updata],
                    outputs=script_output
                )
                script_json_btn.click(
                    fn=script_json,
                    inputs=[script_output],
                    outputs=script_json_output
                )
                
                # 使用提示
                gr.Markdown("""
                ### 💡 剧本创作提示
                1. **剧本概括** 为必填项，用于扩写剧本
                2. **改良建议** 为可选，用于优化现有剧本
                3. 扩写后，可以使用「改良剧本」按钮进一步调整
                4. 最后使用「转换为结构化JSON」按钮，将剧本转换为视频生成所需的格式
                5. 结构化JSON包含：消耗时间、环境、镜头、人物、事件、备注
                """)
            
            # ==================== Tab 4: 图片生成 ====================
            with gr.Tab("🖼️ 图片生成"):
                gr.Markdown("### AI图片生成工具")
                gr.Markdown("根据剧本内容生成场景图片")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 剧本输入")
                        script_init = gr.Textbox(
                            label="初始剧本 *", 
                            placeholder="请输入剧本内容（可从剧本创作页面复制）",
                            lines=4,
                            info="必填项，用于生成图片"
                        )
                        script_custom = gr.Textbox(
                            label="自定义提示词", 
                            placeholder="请输入自定义图片生成提示词（可选）",
                            lines=2,
                            info="留空则使用初始剧本内容"
                        )
                        script_init_btn = gr.Button("1. 初始化", variant="primary")
                        image_init_btn = gr.Button("2. 生成初始图", variant="primary")
                        image_custom_btn = gr.Button("3. 生成自定义图", variant="primary")
                    
                    with gr.Column(scale=1):
                        image_tmp_view = gr.Image(
                            label="生成图片预览", 
                            interactive=False,
                            height=400
                        )
                        image_confirm_btn = gr.Button("4. 确认为首帧", variant="primary")
                        image_view = gr.Image(
                            label="首帧图 (确认后显示)", 
                            interactive=False,
                            height=400
                        )
                
                # 绑定事件
                script_init_btn.click(
                    fn = script_json_init,
                    inputs=[script_json_output,script_index,script_index_max],
                    outputs=[script_init,script_index_max]
                )

                image_init_btn.click(
                    fn=image_make,
                    inputs=[script_init],
                    outputs=image_tmp_view
                )
                
                image_custom_btn.click(
                    fn=image_make,
                    inputs=[script_custom],
                    outputs=image_tmp_view
                )
                
                image_confirm_btn.click(
                    fn=lambda img: (img, img),
                    inputs=[image_tmp_view],
                    outputs=[image_view, image_confirm_state]
                )
                
                # 使用提示
                gr.Markdown("""
                ### 💡 图片生成提示
                1. **初始剧本** 为剧本提供，用于生成场景图片
                2. **自定义提示词** 为可选，可覆盖初始剧本内容
                3. 点击「生成初始图」使用初始剧本内容生成图片
                4. 点击「生成自定义图」使用自定义提示词生成图片
                5. 点击「确认为首帧」将生成的图片设为视频首帧
                6. 系统使用 {IMAGE_SOURCE} 服务进行图片生成
                """)
            
            # ==================== Tab 5: 视频生成 ====================
            with gr.Tab("🎬 视频生成"):
                gr.Markdown("### 视频生成工具")
                gr.Markdown("使用剧本结构化内容和首帧图片生成完整视频")
                
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 当前剧本片段")
                        script_start = gr.Textbox(label="本段剧本内容", placeholder="剧本内容片段", lines=10)#, interactive=False)
                        
                        gr.Markdown("#### 生成控制")
                        with gr.Row():
                            script_init_btn = gr.Button("1. 初始化", variant="primary")
                            script_video_btn = gr.Button("2. 生成视频", variant="primary")
                        
                        gr.Markdown("#### 结果确认")
                        with gr.Row():
                            script_yes_btn = gr.Button("3-1 视频通过", variant="primary")
                            script_no_btn = gr.Button("3-2 视频不通过", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("#### 生成视频预览")
                        script_video = gr.Video(label="生成视频预览", interactive=False)
                        
                        gr.Markdown("#### 首帧图片")
                        script_image = gr.Image(label="首帧图", type="filepath")
                
               
                # 初始化视频生成
                script_init_btn.click(
                    fn = video_make_init,
                    inputs=[script_json_output,image_confirm_state],
                    outputs=[script_start,script_index_max,script_image]
                )
                
                # 生成视频
                script_video_btn.click(
                    fn=video_make_no_ro_one_image,
                    inputs=[
                        script_start,
                        script_image
                    ],
                    outputs=[
                        script_video
                    ]
                )
                
                # 视频通过
                script_yes_btn.click(
                    fn=video_make_pass_no_or_one,
                    inputs=[
                        video_list,
                        script_video,
                        script_json_output,
                        script_index,
                        script_index_max
                    ],
                    outputs=[
                        video_list,
                        script_start,
                        script_image,
                        script_index,
                        script_video
                    ]
                )
                
                # 视频不通过
                script_no_btn.click(
                    fn=video_make_no_ro_one_image,
                    inputs=[
                        script_start,
                        script_image,
                    ],
                    outputs=[
                        script_video
                    ]
                )
                
                # 添加使用提示
                gr.Markdown("""
                ### 💡 视频生成使用指南
                1. **初始化**：点击「1. 初始化」加载剧本结构化内容
                2. **生成视频**：点击「2. 生成视频」使用当前剧本片段生成视频
                3. **确认结果**：
                - 点击「3-1 视频通过」：确认当前视频，自动进入下一段生成
                - 点击「3-2 视频不通过」：重新生成当前段视频
                4. **首帧图**：显示当前使用的首帧图片（来自图片生成页面）
                5. **进度显示**：顶部显示当前进度（当前/总段数）
                
                > 💡 提示：视频生成需要时间，系统会自动等待API响应完成
                """)
            
            # ==================== Tab 6: 进阶生成 ====================
            with gr.Tab("💡视频生成进阶版"):
                gr.Markdown("视频生成功能")
                gr.Markdown(f"当前进度: {script_index}/{script_index_max} ") # 这里的进度会在后续函数中更新
                script_progress = gr.Markdown("")
                gr.Markdown("要求：首帧+人物+环境<=4")
                gr.Markdown(f"当前进度: ")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("选择人物素材")
                        role_class_new = gr.Dropdown(
                            choices=get_subfolders(ROLE_SUB_PATH),  # 🔑 可选列表
                            label="请选择分类",
                            value=get_subfolders(ROLE_SUB_PATH)[0],  # 默认值
                            interactive=True
                        )
                        role_name_new = gr.Dropdown(
                            choices=["选项1", "选项2", "选项3"],  # 🔑 可选列表
                            label="请选择人物",
                            value="default",  # 默认值
                            interactive=True
                        )
                        role_work_add_btn = gr.Button("确认添加", variant="primary")
                        role_new_btn = gr.Button("刷新人物列表", variant="primary")

                    with gr.Column():
                        gr.Markdown("选择环境素材")
                        env_class_new = gr.Dropdown(
                            choices=get_subfolders(ENV_SUB_PATH),  # 🔑 可选列表
                            label="请选择分类",
                            value=get_subfolders(ENV_SUB_PATH)[0],  # 默认值
                            interactive=True
                        )
                        env_name_new = gr.Dropdown(
                            choices=["选项1", "选项2", "选项3"],  # 🔑 可选列表
                            label="请选择环境",
                            value="default",  # 默认值
                            interactive=True
                        )
                        env_work_add_btn = gr.Button("确认添加", variant="primary")
                        env_new_btn = gr.Button("刷新环境列表", variant="primary")
                    

                with gr.Row():
                    with gr.Column():
                        role_ex = gr.Gallery(
                            label="导入人物图片预览",
                            columns=3,
                            height=300,
                            allow_preview=True,
                            interactive=False,
                            value=[]
                        )
                    with gr.Column():
                        env_ex = gr.Gallery(
                            label="导入环境图片预览",
                            columns=3,
                            height=300,
                            allow_preview=True,
                            interactive=False,
                            value=[]
                        )

                with gr.Row():
                    with gr.Column():
                        script_start = gr.Textbox(label="本段剧本内容", placeholder="剧本内容片段",lines=10, interactive=True)
                    with gr.Column():
                        video_view = gr.Video(label="生成视频预览", interactive=False,)
                    with gr.Column():
                        script_image = gr.Image(label="首帧图", type="filepath")

                with gr.Row():
                    with gr.Column():
                        script_init_btn = gr.Button("1.初始化", variant="primary")
                        script_video_btn = gr.Button("2.生成视频", variant="primary")
                    with gr.Column():
                        script_yes_btn = gr.Button("3-1视频通过", variant="primary")
                        script_no_btn = gr.Button("3-2视频不通过", variant="primary")

                #动态更新人物类下具体名
                role_class_new.change(
                    fn=get_subfolders_role,
                    inputs=[role_class_new,],
                    outputs=[role_class_new,role_name_new]
                )

                #动态更新人物名下具体名
                role_name_new.change(
                    fn=updata_name_role,
                    inputs=[role_name_new,],
                    outputs=[role_name_new]
                )

                #动态更新环境类下具体名
                env_class_new.change(
                    fn=get_subfolders_env,
                    inputs=[env_class_new,],
                    outputs=[env_class_new,env_name_new]
                )

                #动态更新环境下具体名
                env_name_new.change(
                    fn=updata_name_env,
                    inputs=[env_name_new,],
                    outputs=[env_name_new]
                )

                #人物确认添加按钮
                role_work_add_btn.click(
                    fn=role_work_add,
                    inputs=[role_ex,role_class_new,role_name_new],
                    outputs=role_ex
                )

                #人物刷新按钮
                role_new_btn.click(
                    fn=new_role_btn,
                    inputs=[],
                    outputs=role_class_new    
                )

                #环境确认提交按钮
                env_work_add_btn.click(
                    fn=env_work_add,
                    inputs=[env_ex,env_class_new,env_name_new],
                    outputs=env_ex
                )

                #环境刷新按钮
                env_new_btn.click(
                    fn=new_env_btn,
                    inputs=[],
                    outputs=env_class_new    
                )

                # 初始化视频生成
                script_init_btn.click(
                    fn = video_make_init_advanced,
                    inputs=[
                        script_json_output,
                        image_confirm_state,
                        script_image,
                        role_ex,
                        env_ex,
                        script_index
                        ],
                    outputs=[
                        script_start,
                        script_index_max,
                        script_image,#首帧图
                        image_script,
                        image_one_script#首帧提示词
                        ]
                )
                
                # 生成视频
                script_video_btn.click(
                    fn=video_make_many,
                    inputs=[
                        script_start,
                        image_one_script,
                        image_script,
                        script_image,
                        role_ex,
                        env_ex,
                    ],
                    outputs=[
                        video_view,
                    ]
                )
                
                # 视频通过
                script_yes_btn.click(
                    fn=video_make_many_pass,
                    inputs=[
                        script_json_output,
                        script_index,
                        script_index_max,
                        video_view,
                        video_list,
                        image_one_script,
                        image_script,
                        role_ex,
                        env_ex,
                    ],
                    outputs=[
                        script_index,
                        script_start,
                        video_list,
                        video_view,
                        image_one_script,
                        image_script
                    ]
                )
                
                # 视频不通过
                script_no_btn.click(
                    fn=video_make_many,
                    inputs=[
                        script_start,
                        image_one_script,
                        image_script,
                        script_image,
                        role_ex,
                        env_ex,
                    ],
                    outputs=[
                        video_view
                    ]
                )
                
                # 添加使用提示
                gr.Markdown("""
                ### 💡 视频生成使用指南
                1. **初始化**：点击「1. 初始化」加载剧本结构化内容
                2. **生成视频**：点击「2. 生成视频」使用当前剧本片段生成视频
                3. **确认结果**：
                - 点击「3-1 视频通过」：确认当前视频，自动进入下一段生成
                - 点击「3-2 视频不通过」：重新生成当前段视频
                4. **首帧图**：显示当前使用的首帧图片（来自图片生成页面）
                5. **进度显示**：顶部显示当前进度（当前/总段数）
                
                > 💡 提示：视频生成需要时间，系统会自动等待API响应完成
                """)
            
            # ==================== Tab 7: 视频拼接 ====================            
            with gr.Tab("🔍视频拼接"):
                gr.Markdown("视频拼接功能")
                gr.Markdown("当前视频序号：")
                
                video_index_str = gr.Markdown(value=str(video_index))
                with gr.Row():
                    with gr.Column():
                        video_preview = gr.Video(label="视频预览", interactive=False,autoplay=True)
                        video_next_btn = gr.Button("下一个", variant="primary")
                        
                    with gr.Column():
                        video_merge_preview = gr.Video(label="拼接预览", interactive=False,autoplay=True)
                        video_merge_btn = gr.Button("确认拼接", variant="primary")

                video_next_btn.click(
                    fn=video_new_view,
                    inputs=[
                        video_list,
                        video_index
                    ],
                    outputs=[
                        video_preview,
                        video_index,
                        video_index_str
                    ]
                )

                video_merge_btn.click(
                    fn=video_splicing,
                    inputs=[video_list],
                    outputs=[
                        video_merge_preview,
                        video_splicing_end
                    ]
                )
            # ==================== Tab 8: 后期处理 ====================
            with gr.Tab("📊后期处理"):
                with gr.Row():
                    with gr.Column():
                        gr.Video(label="提升前预览", interactive=False,autoplay=True)
                        gr.Video(label="提升后预览", interactive=False,autoplay=True)
                        gr.Button("帧数提升", variant="primary")
                        
                    with gr.Column():
                        gr.Video(label="提升前预览", interactive=False,autoplay=True)
                        gr.Video(label="提升后预览", interactive=False,autoplay=True)
                        gr.Button("分辨率提升", variant="primary")


            # ==================== Tab end: 使用帮助 ====================
            with gr.Tab("❓ 使用帮助"):
                gr.Markdown("""
                ## 📖 素材导入管理系统 - 使用指南
                
                ### 系统功能
                本系统提供四类素材的导入管理：
                - **角色素材**：包含角色图片、音频、描述等信息
                - **环境素材**：包含环境图片、描述等信息
                - **剧本素材**：包含剧本扩写、优化和结构化功能
                - **图片素材**：根据剧本内容生成首帧图片
                - **视频生成**：根据剧本内容与首帧内容生成视频
                - **进阶生成**：根据剧本内容与首帧内容,其他图片生成视频
                - **视频拼接**：根据通过的视频拼接成为一个完整的视频
                - **后期处理**：对视频进行帧数提升，分辨率提升
                            
                ### 存储结构
                ```
                ./material_profiles/
                ├── roles/              # 角色素材目录
                │   └── [分类]/
                │       └── [角色名]/
                │           ├── [角色名].png
                │           ├── [角色名].mp3
                │           ├── [角色名]_description.txt
                │           ├── [角色名]_audio_content.txt
                │           └── [角色名]_meta.json
                ├── environments/       # 环境素材目录
                │   └── [环境名]/
                │       ├── [环境名].png
                │       ├── [环境名]_description.txt
                │       ├── [环境名]_config.yaml
                │       └── [环境名]_meta.json
                ├── tmp/            # 临时目录
                └── output/             # 成品输出目录
                ```
                
                ### 常见问题
                1. **为什么提示"已存在"？** 
                   - 同名素材已存在，如需覆盖请勾选"强制覆盖"
                
                2. **图片生成失败？**
                   - 检查API密钥是否正确
                   - 确保网络连接正常
                   - 检查提示词是否符合平台要求
                
                3. **剧本扩写结果不理想？**
                   - 请提供更详细的剧本概括
                   - 可尝试添加改良建议进行优化
                
                4. **如何设置图像生成源？**
                   - 修改配置中的 IMAGE_SOURCE 为 "wan" 或 "seedance"
                   - 填写对应的API密钥和配置
                
                5. **更多**
                   - 功能全部写完再更新这里
                   - 歪比巴卜
                            
                ### 技术支持
                如有问题，请检查控制台日志或联系系统管理员。
                """)
        
        # 页脚
        gr.Markdown("---")
        gr.Markdown("© 2026 素材导入管理系统 | Powered by Gradio")

    return demo


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 确保基础目录存在
    Path(MATERIAL_BASE_PATH).mkdir(parents=True, exist_ok=True)
    Path(MATERIAL_BASE_PATH) / ROLE_SUB_PATH
    Path(MATERIAL_BASE_PATH) / ENV_SUB_PATH
    Path(TMP_PATH).mkdir(parents=True, exist_ok=True)
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=False,#自启动页面
        theme=gr.themes.Soft()
    )