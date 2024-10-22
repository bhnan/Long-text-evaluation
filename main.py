import json
import os
import asyncio
import sys
from datetime import datetime
from evaluate import AIEvaluator
from analysis import ResultAnalyzer
from files import TextProcessor

def load_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

async def process_item(item_key, item_data, api_key):
    topic = item_data['topic']
    file_path = item_data['file_path']
    description = item_data['description']
    
    # 使用TextProcessor读取和处理长文本文件
    text_processor = TextProcessor(file_path)
    processed_sections = text_processor.process()
    
    # 创建评估器
    expected_style = "遵循特定人工智能领域的规范，保持准确性、客观性、一致性和清晰性，具备良好的组织结构，并在写作前深入思考核心内容和表达方式。"
    evaluator = AIEvaluator(api_key=api_key, topic=topic, topic_description=description, expected_style=expected_style)
    
    # 进行评估
    evaluated_sections = await evaluator.evaluate_document(processed_sections)
    
    # 保存评估结果
    evaluator.save_results(evaluated_sections, file_path)
    
    # 获取保存的结果文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    result_file_name = f"evaluation_{base_filename}_{timestamp}.json"
    result_file_path = os.path.join('evaluation_results', result_file_name)
    
    print(f"Evaluation completed for {item_key}. Results saved to {result_file_path}")
    
    # 分析评估结果
    analyzer = ResultAnalyzer(result_file_path)
    analyzer.analyze()
    
    print(f"Analysis completed for {item_key}.")
    print("--------------------")

async def main():
    # 加载数据集
    dataset = load_dataset('dataset.json')
    
    # 创建评估结果文件夹
    os.makedirs('evaluation_results', exist_ok=True)

    api_key = "sk-asjddhdkjjbpnknahupsoqnrvjoyhgpefcnqzjvgyrufgczl"  # 请替换为您的实际 API 密钥
    
    # 对每个项目进行评估和分析
    tasks = [process_item(item_key, item_data, api_key) for item_key, item_data in dataset.items()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
