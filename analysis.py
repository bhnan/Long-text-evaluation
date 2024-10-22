import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict

class ResultAnalyzer:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.file_name = os.path.basename(json_file_path)
        self.evaluation_name = self.file_name.split('_')[1]
        self.result_folder = f"result/{self.evaluation_name}"
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.evaluated_sections = json.load(f)['sections']
        
        self.criteria = [
            "Content Accuracy and Relevance",
            "Logical Coherence and Structure",
            "Language Fluency and Expression",
            "Style and Tone Consistency",
            "Completeness and Depth"
        ]
        self.criteria_mapping = {
            "内容准确性和相关性": "Content Accuracy and Relevance",
            "逻辑连贯性和结构": "Logical Coherence and Structure",
            "语言流畅度和表达": "Language Fluency and Expression",
            "风格和语调一致性": "Style and Tone Consistency",
            "完整性和深度": "Completeness and Depth"
        }
        os.makedirs(self.result_folder, exist_ok=True)

    def calculate_average_scores(self) -> Dict[str, float]:
        total_scores = {criterion: 0 for criterion in self.criteria}
        section_count = len(self.evaluated_sections)

        for section in self.evaluated_sections:
            for cn_criterion, en_criterion in self.criteria_mapping.items():
                score = section['scores'][cn_criterion]['overall_score']
                if score is not None:
                    total_scores[en_criterion] += score

        average_scores = {criterion: total / section_count for criterion, total in total_scores.items() if total != 0}
        return average_scores

    def plot_average_scores(self):
        average_scores = self.calculate_average_scores()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(average_scores.keys(), average_scores.values())
        plt.title("Average Scores for Each Evaluation Criterion")
        plt.xlabel("Evaluation Criteria")
        plt.ylabel("Average Score")
        plt.ylim(0, 5)
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.result_folder}/average_scores.png')
        plt.close()

    def plot_score_distribution(self):
        score_distribution = {criterion: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0} for criterion in self.criteria}
        
        for section in self.evaluated_sections:
            for cn_criterion, en_criterion in self.criteria_mapping.items():
                score = section['scores'][cn_criterion]['overall_score']
                if score is not None:
                    score_distribution[en_criterion][round(score)] += 1

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Score Distribution for Each Evaluation Criterion")
        
        for i, (criterion, distribution) in enumerate(score_distribution.items()):
            ax = axs[i // 3, i % 3]
            ax.bar(distribution.keys(), distribution.values())
            ax.set_title(criterion, fontsize=10)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_ylim(0, len(self.evaluated_sections))

        axs[-1, -1].axis('off')  # Hide the extra subplot
        plt.tight_layout()
        plt.savefig(f'{self.result_folder}/score_distribution.png')
        plt.close()

    def generate_summary_report(self):
        average_scores = self.calculate_average_scores()
        
        report = "Evaluation Results Summary:\n\n"
        report += "Average Scores:\n"
        for criterion, score in average_scores.items():
            report += f"  {criterion}: {score:.2f}\n"
        
        report += "\nScore Distribution for Each Criterion:\n"
        for en_criterion in self.criteria:
            report += f"\n{en_criterion}:\n"
            for section in self.evaluated_sections:
                cn_criterion = next(cn for cn, en in self.criteria_mapping.items() if en == en_criterion)
                score = section['scores'][cn_criterion]['overall_score']
                title = section['title']
                if score is not None:
                    report += f"  {title}: {score}\n"
        
        with open(f'{self.result_folder}/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

    def analyze(self):
        self.plot_average_scores()
        self.plot_score_distribution()
        self.generate_summary_report()
        print(f"Analysis completed. Charts and summary report have been saved in the '{self.result_folder}' folder.")

# Usage example
if __name__ == "__main__":
    analyzer = ResultAnalyzer("evaluation_results/evaluation_a_20241022_170501.json")
    analyzer.analyze()
