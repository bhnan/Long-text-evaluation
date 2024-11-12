import json
import os
import asyncio
import sys
from datetime import datetime
from evaluate import AIEvaluator
from analysis import ResultAnalyzer
from files import TextProcessor
from model.siliconflow_model import SiliconFlowModel

def load_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_result_file_path(file_path):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    result_file_pattern = f"evaluation_{base_filename}_*.json"
    result_files = [f for f in os.listdir('evaluation_results') if f.startswith(f"evaluation_{base_filename}_") and f.endswith('.json')]
    return os.path.join('evaluation_results', result_files[0]) if result_files else None

async def process_item(item_key, item_data, config):
    topic = item_data['topic']
    file_path = item_data['file_path']
    description = item_data['description']
    
    # 检查评估结果文件是否存在
    result_file_name = f"evaluation_{os.path.splitext(os.path.basename(file_path))[0]}_*.json"
    result_files = [f for f in os.listdir('evaluation_results') if f.startswith(f"evaluation_{os.path.splitext(os.path.basename(file_path))[0]}_") and f.endswith('.json')]
    
    if result_files:
        result_file_path = os.path.join('evaluation_results', result_files[0])
        result_folder = os.path.join('result', topic)
        
        if not os.path.exists(result_folder):
            # 如果结果文件夹不存在，执行分析
            print(f"执行分析：{item_key}")
            analyzer = ResultAnalyzer(result_file_path)
            analyzer.analyze()
        else:
            print(f"跳过 {item_key}：结果文件夹已存在")
    else:
        # 如果评估结果文件不存在，进行评估
        print(f"进行评估：{item_key}")
        # 这里放置评估的代码
        # 使用TextProcessor读取和处理长文本文件
        text_processor = TextProcessor(file_path)
        processed_sections = text_processor.process()
        
        # 创建模型实例
        model = SiliconFlowModel(
            api_key=config['siliconflow']['api_key'],
            model=config['siliconflow'].get('model'),
            base_url=config['siliconflow'].get('base_url')
        )

        # 创建评估器
        expected_style = "遵循特定人工智能领域的规范，保持准确性、客观性、一致性和清晰性，具备良好的组织结构，并在写作前深入思考核心内容和表达方式。"
        evaluator = AIEvaluator(model=model, topic=topic, topic_description=description, expected_style=expected_style)
        
        # 进行评估
        evaluated_sections = await evaluator.evaluate_document(processed_sections)
        evaluator.save_results(evaluated_sections, file_path)

        # 保存评估结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        result_file_name = f"evaluation_{base_filename}_{timestamp}.json"
        result_file_path = os.path.join('evaluation_results', result_file_name)
        
        
        print(f"评估完成：{item_key}。结果保存至 {result_file_path}")

async def main():
    # 加载配置
    config = load_config()
    
    # 加载数据集
    dataset = load_dataset('dataset.json')
    
    # 创建评估结果文件夹
    os.makedirs('evaluation_results', exist_ok=True)
    
    # 对每个项目进行评估和分析
    tasks = [process_item(item_key, item_data, config) for item_key, item_data in dataset.items()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
