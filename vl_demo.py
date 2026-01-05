import os
import time
import base64
import dashscope

def image_to_base64(image_path):
    """将图片文件转换为base64编码的data URL
    
    Args:
        image_path (str): 图片文件路径
        
    Returns:
        str: base64编码的data URL
    """
    try:
        with open(image_path, 'rb') as img_file:
            # 读取图片文件并转换为base64
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            
            # 获取文件扩展名
            ext = image_path.split('.')[-1].lower()
            
            # 根据文件扩展名确定MIME类型
            mime_types = {
                'jpg': 'jpeg',
                'jpeg': 'jpeg',
                'png': 'png',
                'gif': 'gif',
                'webp': 'webp',
                'bmp': 'bmp'
            }
            
            mime_type = mime_types.get(ext, 'jpeg')  # 默认使用jpeg
            
            # 返回完整的data URL
            return f'data:image/{mime_type};base64,{encoded}'
            
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        return None
    except Exception as e:
        print(f"错误：转换图片 {image_path} 时发生异常 - {str(e)}")
        return None

# 若使用新加坡地域的模型，请取消下列注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# 1. 区域人员闯入
messages1 = [
    {
        "role": "system",
        "content": [
            {"text": "你是一个区域闯入检测助手，只检测图中区域是否有人员闯入。如果有人员闯入图中区域，则回复“【区域闯入】有人员闯入区域”，并描述闯入人员的状态，否则回复“【区域闯入】无人员闯入区域”。"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"image": image_to_base64("vl_demo/区域人员闯入.webp")},
        ]
    }
]

# 2. 危险区域闯入
messages2 = [
    {
        "role": "system",
        "content": [
            {"text": "你是一个危险区域闯入检测助手，只检测图中红色区域是否有人员闯入。如果有人员闯入红色区域，则回复“【危险区域闯入】有人员闯入红色区域”，并描述闯入人员的状态，否则回复“【危险区域闯入】无人员闯入红色区域”。"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"image": image_to_base64("vl_demo/危险区域人员闯入.png")},
        ]
    }
]

# 3. 安全帽检测
messages3 = [
    {
        "role": "system",
        "content": [
            {"text": "你是一个安全帽检测助手，只检测图中的人员是否佩戴安全帽。如果人员佩戴安全帽，则回复“【安全帽】人员已佩戴安全帽”，否则回复“【安全帽】人员未佩戴安全帽”"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"image": image_to_base64("vl_demo/安全帽检测.webp")},
        ]
    }
]

# 4. 安全帽正确佩戴检测
messages4 = [
    {
        "role": "system",
        "content": [
            {"text": "你是一个安全帽正确佩戴检测助手，只检测图中的人员是否正确佩戴安全帽。如果人员正确佩戴安全帽，则回复“【安全帽正确佩戴】人员正确佩戴安全帽”，否则回复“【安全帽正确佩戴】人员未正确佩戴安全帽”，并描述当前人员的状态。"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"image": image_to_base64("vl_demo/安全帽正确佩戴检测.webp")},
        ]
    }
]

print("="*50)
print("🚀 开始调用多模态模型...")
start_time = time.time()

model_name = 'qwen3-vl-plus'
response = dashscope.MultiModalConversation.call(
    # 若没有配置环境变量， 请用百炼API Key将下行替换为： api_key ="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key = os.getenv('DASHSCOPE_API_KEY'),
    # qwen3-vl-flash / qwen3-vl-plus
    model = model_name,  # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    # 1. 区域人员闯入
    # 2. 危险区域闯入
    # 3. 安全帽检测
    # 4. 安全帽正确佩戴检测
    messages = messages1
)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"✅ 模型调用完成")
print(f"🎯 模型名称: {model_name}")
print(f"⏱️  响应耗时: {elapsed_time:.2f} 秒")
print("="*50)
print(response.output.choices[0].message.content[0]["text"])