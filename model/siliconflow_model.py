import aiohttp
from .base_model import BaseModel
import asyncio
import sys
import time
import tiktoken
import json

class RateLimiter:
    def __init__(self, rpm, tpm):
        self.rpm = rpm
        self.tpm = tpm
        self.request_tokens = rpm
        self.token_tokens = tpm
        self.last_request_time = time.time()
        self.last_token_time = time.time()

    async def wait_for_capacity(self, tokens):
        while True:
            current_time = time.time()
            time_passed = current_time - self.last_request_time
            self.request_tokens = min(self.rpm, self.request_tokens + time_passed * (self.rpm / 60))
            self.token_tokens = min(self.tpm, self.token_tokens + time_passed * (self.tpm / 60))
            
            if self.request_tokens >= 1 and self.token_tokens >= tokens:
                self.request_tokens -= 1
                self.token_tokens -= tokens
                self.last_request_time = current_time
                self.last_token_time = current_time
                break
            await asyncio.sleep(0.1)

class SiliconFlowModel(BaseModel):
    def __init__(self, api_key, model=None, base_url=None):
        super().__init__(api_key, model, base_url)
        self.base_url = base_url or "https://api.siliconflow.cn/v1/chat/completions"
        self.model = model or "Qwen/Qwen2.5-72B-Instruct-128K"
        self.rate_limiter = RateLimiter(rpm=1000, tpm=20000)
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    async def get_response(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的文本评估助手。"},
                {"role": "user", "content": prompt}
            ]
        }

        tokens = len(self.encoder.encode(prompt))
        await self.rate_limiter.wait_for_capacity(tokens)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        print(f"SiliconFlow API 调用失败,状态码: {response.status}")
                        return None
        except aiohttp.ClientError as e:
            print(f"SiliconFlow API 连接错误: {str(e)}")
            return None
        except asyncio.TimeoutError:
            print("SiliconFlow API 请求超时")
            return None
        except Exception as e:
            print(f"SiliconFlow API 调用出错: {str(e)}")
            return None

if __name__ == "__main__":
    async def main():
        # 从 config.json 文件中读取配置
        with open('config.json', 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        
        # 获取 SiliconFlow 配置
        siliconflow_config = config['siliconflow']
        
        model = SiliconFlowModel(
            api_key=siliconflow_config['api_key'],
            model=siliconflow_config.get('model'),  # 使用 get 方法,如果 'model' 键不存在则返回 None
            base_url=siliconflow_config.get('base_url')  # 同上
        )
        
        response = await model.get_response("你好")
        print(response)

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
