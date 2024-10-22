import json
import os

class CheckpointManager:
    @staticmethod
    def save_checkpoint(data, filename="evaluation_checkpoint.json"):
        """
        保存checkpoint到文件
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_checkpoint(filename="evaluation_checkpoint.json"):
        """
        从文件加载checkpoint
        如果文件不存在，返回空列表
        """
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    @staticmethod
    def clear_checkpoint(filename="evaluation_checkpoint.json"):
        """
        清除checkpoint文件
        """
        if os.path.exists(filename):
            os.remove(filename)

