import logging
import os
from datetime import datetime

class Logger:
    @staticmethod
    def setup_logging():
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log', 'a', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)

    @staticmethod
    def log_model_io(logger, prompt, response):
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}")

    @staticmethod
    def log_parsing_result(logger, score, explanation):
        logger.info(f"Parsed response - Score: {score}, Explanation: {explanation}")

    @staticmethod
    def log_parsing_error(logger, error, response):
        logger.error(f"Error parsing response: {error}\nResponse: {response}")

    @staticmethod
    def log_ai_assisted_parsing(logger, score, explanation):
        logger.info(f"AI-assisted parsing result - Score: {score}, Explanation: {explanation}")

    @staticmethod
    def log_warning(logger, message):
        logger.warning(message)
