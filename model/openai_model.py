from openai import OpenAI
import asyncio
from .base_model import BaseModel
import traceback
import aiohttp
import sys
import json

class OpenAIModel(BaseModel):
    def __init__(self, api_key, model=None, base_url=None):
        super().__init__(api_key, model, base_url)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model or "gpt-3.5-turbo"

    async def get_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的文本评估助手。"},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        print(f"API 调用失败，状态码: {response.status}")
                        print(await response.text())
                        return None
        except Exception as e:
            print(f"API 调用出错: {type(e).__name__}: {str(e)}")
            return None

if __name__ == "__main__":
    async def main():
        # 从 config.json 文件中读取配置
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        
        # 获取 OpenAI 配置
        openai_config = config['openai']
        
        model = OpenAIModel(
            api_key=openai_config['api_key'],
            model=openai_config['model'],
            base_url=openai_config['base_url']
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
