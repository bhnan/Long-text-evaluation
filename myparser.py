import re
import json
from log import Logger
import asyncio

class ResponseParser:
    def __init__(self, logger, model):
        self.logger = logger
        self.model = model

    def extract_json_from_markdown(self, markdown_string):
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, markdown_string, re.DOTALL)
        
        if match:
            json_string = match.group(1)
            try:
                json_data = json.loads(json_string)
                return json_data
            except json.JSONDecodeError:
                self.logger.error("无法解析JSON字符串")
                return None
        else:
            self.logger.error("未找到JSON代码段")
            return None

    async def parse_model_response(self, response: str) -> tuple:
        try:
            # 首先尝试从可能的Markdown响应中提取JSON
            extracted_json = self.extract_json_from_markdown(response)
            if extracted_json:
                response_dict = extracted_json
            else:
                # 如果没有提取到JSON，则尝试直接解析整个响应
                response_dict = json.loads(response)
            
            # 检查是否有 'reasoning' 和 'result' 键
            if 'reasoning' in response_dict and 'result' in response_dict:
                result = response_dict['result']
            else:
                # 如果没有预期的键，可能整个响应就是结果
                result = response_dict

            # 从结果中提取分数和解释
            score = result.get('score')
            explanation = result.get('explanation')

            # 验证分数
            if score is not None:
                score = float(score)
                if not (1 <= score <= 5):
                    Logger.log_warning(self.logger, f"Extracted score {score} is not in valid range")
                    score = None

            if score is not None and explanation is not None:
                Logger.log_parsing_result(self.logger, score, explanation)
                return score, explanation
            else:
                raise ValueError("Missing score or explanation in the response")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            Logger.log_parsing_error(self.logger, str(e), response)
            self.logger.info("Attempting AI-assisted parsing...")
            return await self.ai_assisted_parsing(response)

    async def ai_assisted_parsing(self, response: str) -> tuple:
        prompt = f"""
        以下是一个AI模型的响应，但其JSON格式可能不正确。请从中提取评分（1-5的数字）和解释。
        如果无法提取，请返回None作为评分和解释。

        响应内容：
        {response}

        请以以下格式输出结果：
        {{"score": 提取的评分, "explanation": "提取的解释"}}
        """

        parsing_response = await self.model.get_response(prompt)
        
        try:
            # 首先尝试从可能的Markdown响应中提取JSON
            extracted_json = self.extract_json_from_markdown(parsing_response)
            if extracted_json:
                parsing_result = extracted_json
            else:
                # 如果没有提取到JSON，则尝试直接解析整个响应
                parsing_result = json.loads(parsing_response)
            
            score = parsing_result.get("score")
            explanation = parsing_result.get("explanation")
            
            # 验证提取的评分
            if score is not None:
                score = float(score)
                if not (1 <= score <= 5):
                    Logger.log_warning(self.logger, f"Extracted score {score} is not in valid range")
                    score = None
            
            if score is not None and explanation is not None:
                Logger.log_ai_assisted_parsing(self.logger, score, explanation)
                return score, explanation
            else:
                raise ValueError("Missing score or explanation in the AI-assisted parsing response")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            Logger.log_parsing_error(self.logger, str(e), parsing_response)
            return 0, None

if __name__ == "__main__":
    # 示例使用
    markdown_string = '''
    ```json
    {"score": 2, "explanation": "段落内容准确但与主题相关性较低，主要讨论了提示攻击对人工智能模型安全性的威胁，而未涉及人工智能在医疗领域的应用、机遇和挑战。"}```
    '''

    # 创建一个简单的 Logger 和 Model 模拟对象
    class SimpleLogger:
        @staticmethod
        def error(msg): print(f"ERROR: {msg}")
        @staticmethod
        def info(msg): print(f"INFO: {msg}")

    class SimpleModel:
        async def get_response(self, prompt):
            return '{"score": 3, "explanation": "这是一个模拟的AI辅助解析结果"}'

    # 创建 ResponseParser 实例
    logger = SimpleLogger()
    model = SimpleModel()
    parser = ResponseParser(logger, model)

    # 使用 extract_json_from_markdown 方法
    json_data = parser.extract_json_from_markdown(markdown_string)
    print("Extracted JSON:", json_data)

    # 使用 parse_model_response 方法
    async def test_parse_model_response():
        score, explanation = await parser.parse_model_response(markdown_string)
        print(f"Parsed response - Score: {score}, Explanation: {explanation}")

    # 运行异步函数
    asyncio.run(test_parse_model_response())
