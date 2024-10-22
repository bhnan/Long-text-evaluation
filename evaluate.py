from model.siliconflow_model import SiliconFlowModel
from typing import List, Dict, Any
import asyncio
import sys
import json
import os
from datetime import datetime
from tqdm.asyncio import tqdm as tqdm_asyncio
from tqdm import tqdm
from log import Logger
from checkpoint import CheckpointManager
from myparser import ResponseParser

class AIEvaluator:
    def __init__(self, api_key, topic, topic_description, expected_style):
        self.model = SiliconFlowModel(api_key)
        self.criteria = [
            "内容准确性和相关性",
            "逻辑连贯性和结构",
            "语言流畅度和表达",
            "风格和语调一致性",
            "完整性和深度"
        ]
        self.grade_descriptions = {
            "内容准确性和相关性": {
                1: "内容存在明显错误，与主题无关，可能误导读者。",
                2: "有部分错误或不准确的信息，相关性较低。",
                3: "主要内容基本准确，但有细节错误或遗漏，一般相关。",
                4: "内容准确，信息全面，与主题高度相关。",
                5: "内容精准，无任何错误，信息深入且全面，完全契合主题和用需求。"
            },
            "逻辑连贯性和结构": {
                1: "结构混乱，逻辑不通，读者难以理解。",
                2: "结构松散，逻辑关系不明确，过渡生硬。",
                3: "基本结构清晰，但存在逻辑跳跃或过渡不顺畅的地方。",
                4: "结构合理，逻辑清晰，段落和句子衔接顺畅。",
                5: "结构严谨，逻辑缜密，过渡自然，整体连贯性极佳。"
            },
            "语言流畅度和表达": {
                1: "大量语法错误，词不达意，表达生硬。",
                2: "存在明显的语法错误，词汇贫乏，表达不够流畅。",
                3: "语法基本正确，但有偶尔错误，表达尚可。",
                4: "语法正确，词汇丰富，表达流畅自然。",
                5: "语言优美，表达精准，语法和用词无可挑剔。"
            },
            "风格和语调一致性": {
                1: "风格混乱，语调不一致，与预期风格严重不符。",
                2: "风格不统一，偶尔偏离预期语调。",
                3: "基本保持一致的风格和语调，但有少数偏差。",
                4: "风格和语调一致，符合目标受众的预期。",
                5: "风格鲜明，语调高度一致，完美契合目标受众和目的。"
            },
            "完整性和深度": {
                1: "内容片面，重要信息缺失，深度极浅。",
                2: "覆盖了部分主题，但有明显遗漏，深度不足。",
                3: "基本覆盖主要内容，但细节和深度有所欠缺。",
                4: "内容全面，讨论深入，满足读者大部分需求。",
                5: "全面且深入地覆盖主题，超出读者预期，具有高度权威性。"
            }
        }
        self.topic = topic
        self.topic_description = topic_description
        self.expected_style = expected_style
        self.checkpoint_file = "evaluation_checkpoint.json"
        self.logger = Logger.setup_logging()
        self.parser = ResponseParser(self.logger, self.model)

    async def evaluate_section(self, section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        scores = {}
        for criterion in self.criteria:
            if criterion == "内容准确性和相关性":
                scores[criterion] = await self._evaluate_accuracy_relevance(section)
            elif criterion == "逻辑连贯性和结构":
                scores[criterion] = await self._evaluate_coherence(section)
            elif criterion == "语言流畅度和表达":
                scores[criterion] = await self._evaluate_fluency(section)
            elif criterion == "风格和语调一致性":
                scores[criterion] = await self._evaluate_style_consistency(section)
            elif criterion == "完整性和深度":
                scores[criterion] = await self._evaluate_completeness_depth(section)
            else:
                prompt = self._create_evaluation_prompt(section, criterion)
                response = await self._get_model_response(prompt)
                score, explanation = await self._parse_model_response(response)
                scores[criterion] = {"score": score, "explanation": explanation}
        return scores

    async def _evaluate_accuracy_relevance(self, section: Dict[str, Any]) -> Dict[str, Any]:
        paragraph_scores = []
        async for paragraph in tqdm_asyncio(section['paragraphs'], desc="评估准确性和相关性", leave=False):
            prompt = self._create_accuracy_relevance_prompt(paragraph)
            response = await self._get_model_response(prompt)
            score, explanation = await self._parse_model_response(response)  # 在这里添加 await
            paragraph_scores.append({"score": score, "explanation": explanation})
        
        overall_score = round(sum(item['score'] for item in paragraph_scores) / len(paragraph_scores), 1)
        overall_explanation = f"段落评分: {', '.join([str(item['score']) for item in paragraph_scores])}。整体评分为各段落评分的平均值。"
        
        return {
            "overall_score": overall_score,
            "overall_explanation": overall_explanation,
            "paragraph_scores": paragraph_scores
        }

    def _create_accuracy_relevance_prompt(self, paragraph: str) -> str:
        prompt = f"""请评估以下段落的"内容准确性和相关性"，考虑到以下主题和描述：

主题：{self.topic}
主题描述：{self.topic_description}

段落内容：
{paragraph}

评分标准：
1分（很差）：{self.grade_descriptions["内容准确性和相关性"][1]}
2分（较差）：{self.grade_descriptions["内容准确性和相关性"][2]}
3分（一般）：{self.grade_descriptions["内容准确性和相关性"][3]}
4分（良好）：{self.grade_descriptions["内容准确性和相关性"][4]}
5分（优秀）：{self.grade_descriptions["内容准确性和相关性"][5]}

请给出评分（1-5）并简要解释原因。

***输出格式***
输出格式：{{"reasoning":{{"解释一步一步评估的过程"}},"result":{{"score": 评分, "explanation": "简要解释"}}}}

请确保您的回答严格遵循这个格式。
"""
        return prompt

    async def _get_model_response(self, prompt: str) -> str:
        response = await self.model.get_response(prompt)
        Logger.log_model_io(self.logger, prompt, response)
        return response

    async def _evaluate_coherence(self, section: Dict[str, Any]) -> Dict[str, Any]:
        coherence_scores = []
        paragraphs = section['paragraphs']
        titles = section.get('titles', [])
        
        async for i in tqdm_asyncio(range(len(paragraphs) - 1), desc="评估逻辑连贯性", leave=False):
            if i < len(titles) - 1 and titles[i] == titles[i+1]:
                prompt = self._create_coherence_prompt(paragraphs[i], paragraphs[i+1])
                response = await self._get_model_response(prompt)
                score, explanation = await self._parse_model_response(response)
                coherence_scores.append({"score": score, "explanation": explanation})
        
        if not coherence_scores:
            return {
                "overall_score": None,
                "overall_explanation": "该章节只包含一个段落或所有段落属于不同标题，无法评估连贯性。",
                "coherence_scores": []
            }
        
        overall_score = round(sum(item['score'] for item in coherence_scores) / len(coherence_scores), 1)
        overall_explanation = f"段落间连贯性评分: {', '.join([str(item['score']) for item in coherence_scores])}。整体评分为各评分的平均值。"
        
        return {
            "overall_score": overall_score,
            "overall_explanation": overall_explanation,
            "coherence_scores": coherence_scores
        }

    def _create_coherence_prompt(self, paragraph1: str, paragraph2: str) -> str:
        prompt = f"""请评估以下两个连续段落之间的"逻辑连贯性和结构"：

段落1：
{paragraph1}

段落2：
{paragraph2}

评分标准：
1分（很差）：{self.grade_descriptions["逻辑连贯性和结构"][1]}
2分（较差）：{self.grade_descriptions["逻辑连贯性和结构"][2]}
3分（一般）：{self.grade_descriptions["逻辑连贯性和结构"][3]}
4分（良好）：{self.grade_descriptions["逻辑连贯性和结构"][4]}
5分（优秀）：{self.grade_descriptions["逻辑连贯性和结构"][5]}

请特别注意段落之间的衔接是否自然，逻辑是否连贯。给出评分（1-5）并简要解释原因。

***输出格式***
输出格式：{{"reasoning":{{"解释一步一步评估的过程"}},"result":{{"score": 评分, "explanation": "简要解释"}}}}

请确保您的回答严格遵循这个格式。
"""
        return prompt

    async def _evaluate_fluency(self, section: Dict[str, Any]) -> Dict[str, Any]:
        paragraph_scores = []
        async for paragraph in tqdm_asyncio(section['paragraphs'], desc="评估语言流畅度", leave=False):
            prompt = self._create_fluency_prompt(paragraph)
            response = await self._get_model_response(prompt)
            score, explanation = await self._parse_model_response(response)
            paragraph_scores.append({"score": score, "explanation": explanation})
        
        overall_score = round(sum(item['score'] for item in paragraph_scores) / len(paragraph_scores), 1)
        overall_explanation = f"段落流畅度评分: {', '.join([str(item['score']) for item in paragraph_scores])}。整体评分为各段落评分的平均值。"
        
        return {
            "overall_score": overall_score,
            "overall_explanation": overall_explanation,
            "paragraph_scores": paragraph_scores
        }

    def _create_fluency_prompt(self, paragraph: str) -> str:
        prompt = f"""请评估以下段落的"语言流畅度和表达"：

段落内容：
{paragraph}

评分标准：
1分（很差）：{self.grade_descriptions["语言流畅度和表达"][1]}
2分（较差）：{self.grade_descriptions["语言流畅度和表达"][2]}
3分（一般）：{self.grade_descriptions["语言流畅度和表达"][3]}
4分（良好）：{self.grade_descriptions["语言流畅度和表达"][4]}
5分（优秀）：{self.grade_descriptions["语言流畅度和表达"][5]}

请给出评分（1-5）并简要解释原因。

***输出格式***
输出格式：{{"reasoning":{{"解释一步一步评估的过程"}},"result":{{"score": 评分, "explanation": "简要解释"}}}}

请确保您的回答严格遵循这个格式。
"""
        return prompt

    async def _evaluate_style_consistency(self, section: Dict[str, Any]) -> Dict[str, Any]:
        paragraph_scores = []
        async for paragraph in tqdm_asyncio(section['paragraphs'], desc="评估风格一致性", leave=False):
            prompt = self._create_style_consistency_prompt(paragraph)
            response = await self._get_model_response(prompt)
            score, explanation = await self._parse_model_response(response)
            paragraph_scores.append({"score": score, "explanation": explanation})
        
        overall_score = round(sum(item['score'] for item in paragraph_scores) / len(paragraph_scores), 1)
        overall_explanation = f"段落风格一致性评分: {', '.join([str(item['score']) for item in paragraph_scores])}。整体评分为各段落评分的平均值。"
        
        return {
            "overall_score": overall_score,
            "overall_explanation": overall_explanation,
            "paragraph_scores": paragraph_scores
        }
    
    def _create_style_consistency_prompt(self, paragraph: str) -> str:
        prompt = f"""请评估以下段落的"风格和语调一致性"，考虑到预期的风格和语调：

预期风格和语调：{self.expected_style}

段落内容：
{paragraph}

评分标准：
1分（很差）：{self.grade_descriptions["风格和语调一致性"][1]}
2分（较差）：{self.grade_descriptions["风格和语调一致性"][2]}
3分（一般）：{self.grade_descriptions["风格和语调一致性"][3]}
4分（良好）：{self.grade_descriptions["风格和语调一致性"][4]}
5分（优秀）：{self.grade_descriptions["风格和语调一致性"][5]}

请给出评分（1-5）并简要解释原因，特别注意段落的风格和语调是否与预期一致。

***输出格式***
输出格式：{{"reasoning":{{"解释一步一步评估的过程"}},"result":{{"score": 评分, "explanation": "简要解释"}}}}

请确保您的回答严格遵循这个格式。
"""
        return prompt

    async def _evaluate_completeness_depth(self, section: Dict[str, Any]) -> Dict[str, Any]:
        paragraph_scores = []
        async for paragraph in tqdm_asyncio(section['paragraphs'], desc="评估完整性和深度", leave=False):
            prompt = self._create_completeness_depth_prompt(paragraph)
            response = await self._get_model_response(prompt)
            score, explanation = await self._parse_model_response(response)  # 在这里添加 await
            paragraph_scores.append({"score": score, "explanation": explanation})
        
        overall_score = round(sum(item['score'] for item in paragraph_scores) / len(paragraph_scores), 1)
        overall_explanation = f"段落完整性和深度评分: {', '.join([str(item['score']) for item in paragraph_scores])}。整体评分为各段落评分的平均值。"
        
        return {
            "overall_score": overall_score,
            "overall_explanation": overall_explanation,
            "paragraph_scores": paragraph_scores
        }

    def _create_completeness_depth_prompt(self, paragraph: str) -> str:
        prompt = f"""请评估以下段落的"完整性和深度"，考虑到以下主题和描述：

主题：{self.topic}
主题描述：{self.topic_description}

段落内容：
{paragraph}

评分标准：
1分（很差）：{self.grade_descriptions["完整性和深度"][1]}
2分（较差）：{self.grade_descriptions["完整性和深度"][2]}
3分（一般）：{self.grade_descriptions["完整性和深度"][3]}
4分（良好）：{self.grade_descriptions["完整性和深度"][4]}
5分（优秀）：{self.grade_descriptions["完整性和深度"][5]}

请给出评分（1-5）并详细解释原因，特别注意内容的完整性和深度是否符合主题要求。评估应考虑以下方面：
1. 段落是否涵盖了主题的重要方面？
2. 内容是否深入探讨了相关问题？
3. 是否提供了足够的细节和例子来支持观点？
4. 是否考虑了不同视角或潜在的争议？
5. 段落的深度是否满足了预期的专业水平？

***输出格式***
输出格式：{{"reasoning":{{"解释一步一步评估的过程"}},"result":{{"score": 评分, "explanation": "简要解释"}}}}

请确保您的回答严格遵循这个格式。
"""
        return prompt

    async def evaluate_document(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evaluated_sections = CheckpointManager.load_checkpoint(self.checkpoint_file)
        start_index = len(evaluated_sections)
        total_steps = len(sections) * len(self.criteria) + len(sections) - 1  # 章节评估 + 章节间连贯性评估
        
        with tqdm(total=total_steps, desc="评估进度", initial=start_index * len(self.criteria)) as pbar:
            for i in tqdm(range(start_index, len(sections)), desc="评估节"):
                section = sections[i]
                scores = {}
                for criterion in self.criteria:
                    if criterion == "内容准确性和相关性":
                        scores[criterion] = await self._evaluate_accuracy_relevance(section)
                    elif criterion == "逻辑连贯性和结构":
                        scores[criterion] = await self._evaluate_coherence(section)
                    elif criterion == "语言流畅度和表达":
                        scores[criterion] = await self._evaluate_fluency(section)
                    elif criterion == "风格和语调一致性":
                        scores[criterion] = await self._evaluate_style_consistency(section)
                    elif criterion == "完整性和深度":
                        scores[criterion] = await self._evaluate_completeness_depth(section)
                    pbar.update(1)
                section['scores'] = scores
                evaluated_sections.append(section)
                CheckpointManager.save_checkpoint(evaluated_sections, self.checkpoint_file)
            
            # 评估相邻章节之间的连贯性
            for i in range(start_index, len(evaluated_sections) - 1):
                if evaluated_sections[i].get('parent_title') == evaluated_sections[i+1].get('parent_title'):
                    coherence_prompt = self._create_section_coherence_prompt(evaluated_sections[i], evaluated_sections[i+1])
                    response = await self._get_model_response(coherence_prompt)
                    score, explanation = await self._parse_model_response(response)
                    evaluated_sections[i]['section_coherence'] = {"score": score, "explanation": explanation}
                else:
                    evaluated_sections[i]['section_coherence'] = {
                        "score": None,
                        "explanation": "相邻章节属于不同的大标题，不评估连贯性。"
                    }
                pbar.update(1)
                CheckpointManager.save_checkpoint(evaluated_sections, self.checkpoint_file)
        
        return evaluated_sections

    def clear_checkpoint(self):
        CheckpointManager.clear_checkpoint(self.checkpoint_file)

    def _create_section_coherence_prompt(self, section1: Dict[str, Any], section2: Dict[str, Any]) -> str:
        prompt = f"""请评估以下两个连续章节之间的"逻辑连贯性和结构"：

章节1标题：{section1['title']}
章节1最后一段：{section1['paragraphs'][-1]}

章节2标题：{section2['title']}
章节2第一段：{section2['paragraphs'][0]}

评分标准：
1分（很差）：{self.grade_descriptions["逻辑连贯性和结构"][1]}
2分（较差）：{self.grade_descriptions["逻辑连贯性和结构"][2]}
3分（一般）：{self.grade_descriptions["逻辑连贯性和结构"][3]}
4分（良好）：{self.grade_descriptions["逻辑连贯性和结构"][4]}
5分（优秀）：{self.grade_descriptions["逻辑连贯性和结构"][5]}

请特别注意章节之间的衔接是否自然，逻辑是否连贯。给出评分（1-5）并简要解释原因。

***输出格式***
输出格式：{{"reasoning":{{"解释一步一步评估的过程"}},"result":{{"score": 评分, "explanation": "简要解释"}}}}

请确保您的回答严格遵循这个格式。
"""
        return prompt

    def save_results(self, evaluated_sections: List[Dict[str, Any]], original_filename: str):
        """评估结果保存为JSON文件，文件名包含时间戳和原始文件名"""
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 提取原始文件名（不包括路径和扩展名）
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]
        
        # 构造新的文件名
        result_filename = f"evaluation_{base_filename}_{timestamp}.json"
        
        results = {
            "document_title": base_filename,
            "evaluation_time": timestamp,
            "sections": evaluated_sections
        }
        
        # 确保输出目录存在
        os.makedirs("evaluation_results", exist_ok=True)
        
        # 保存结
        with open(os.path.join("evaluation_results", result_filename), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {result_filename}")
        
        # 评估完成后删除检查点文件
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    async def _parse_model_response(self, response: str) -> tuple:
        return await self.parser.parse_model_response(response)

# 使用示例
if __name__ == "__main__":
    api_key = "sk-asjddhdkjjbpnknahupsoqnrvjoyhgpefcnqzjvgyrufgczl"  # 请替换为您的实际 API 密钥
    topic = "人工智能在医疗领域的应用"
    topic_description = "探讨人工智能技术如何在诊断、治疗和医疗管理中的应用，以及其带来的机遇和挑战。"
    expected_style = "专业、客观，使用医学和技术术语，但同时保持通俗易懂。语调应该是信息性的，带有一定的乐观态度，但也要客观指出潜在的问题和挑战。"
    evaluator = AIEvaluator(api_key, topic, topic_description, expected_style)

    # 假设我们已经从 files.py 获得了处理后的 sections
    from files import TextProcessor
    input_file = r"长文本文件\a.txt"
    text_processor = TextProcessor(input_file)
    sections = text_processor.process()

    async def main():
        evaluated_sections = await evaluator.evaluate_document(sections)

        # 保存最终评估结果
        evaluator.save_results(evaluated_sections, input_file)

        # 打印评估结果
        for section in evaluated_sections:
            print(f"标题: {section['title']}")
            print("评分:")
            for criterion, result in section['scores'].items():
                print(f"  {criterion}:")
                if isinstance(result, dict) and 'score' in result and 'explanation' in result:
                    print(f"    分数: {result['score']}")
                    print(f"    解释: {result['explanation']}")
                else:
                    print(f"    结果格式不正确: {result}")
            print()

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
