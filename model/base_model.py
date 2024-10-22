from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, api_key, model=None, base_url=None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    @abstractmethod
    async def get_response(self, prompt):
        pass
