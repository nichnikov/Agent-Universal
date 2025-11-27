"""
Utilities for Langfuse integration and prompt management.
"""

import os
from typing import Optional, Dict, Any
from langfuse import Langfuse


class LangfuseManager:
    """
    Singleton manager for Langfuse operations.
    Handles prompt loading and tracing configuration.
    """
    
    _instance: Optional['LangfuseManager'] = None
    _langfuse_client: Optional[Langfuse] = None
    
    def __new__(cls) -> 'LangfuseManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if self._langfuse_client is None:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Langfuse client with environment variables."""
        try:
            self._langfuse_client = Langfuse(
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Langfuse client: {e}")
            self._langfuse_client = None
    
    def get_prompt(self, prompt_name: str, **kwargs: Any) -> str:
        """
        Retrieve a prompt from Langfuse by name.
        
        Args:
            prompt_name: Name of the prompt in Langfuse
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If prompt not found or Langfuse not configured
        """
        if self._langfuse_client is None:
            # Fallback prompts for development/testing
            return self._get_fallback_prompt(prompt_name, **kwargs)
        
        try:
            prompt = self._langfuse_client.get_prompt(prompt_name)
            if prompt is None:
                raise ValueError(f"Prompt '{prompt_name}' not found in Langfuse")
            
            # Format prompt with provided variables
            return prompt.prompt.format(**kwargs) if kwargs else prompt.prompt
            
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}' from Langfuse: {e}")
            return self._get_fallback_prompt(prompt_name, **kwargs)
    
    def _get_fallback_prompt(self, prompt_name: str, **kwargs: Any) -> str:
        """
        Fallback prompts for development when Langfuse is not available.
        """
        fallback_prompts = {
            "supervisor-system-prompt": """Ты — супервизор мультиагентной системы. У тебя есть помощник:

LegalExpert - специализируется на юридических вопросах, знает российское законодательство (ГК РФ, УК РФ, КоАП РФ).

Твоя задача:
1. Анализировать входящий запрос пользователя
2. Если вопрос требует юридической консультации - передать управление LegalExpert
3. Если ответ уже получен или вопрос не требует специализированной помощи - завершить диалог (FINISH)

Отвечай только в формате JSON с полем "next", где значение либо "LegalExpert", либо "FINISH".""",

            "legal-expert-prompt": """Ты — опытный юрист Российской Федерации с глубокими знаниями законодательства.

Твоя специализация:
- Гражданский кодекс РФ (ГК РФ)
- Уголовный кодекс РФ (УК РФ)
- Кодекс об административных правонарушениях (КоАП РФ)
- Налоговое законодательство
- Корпоративное право

ВАЖНО: 
- Используй ТОЛЬКО предоставленные инструменты для поиска информации
- НЕ выдумывай законы и статьи
- Ссылайся только на найденную через инструменты информацию
- Если информация не найдена, честно об этом сообщи

Отвечай профессионально, структурированно, со ссылками на конкретные статьи законов."""
        }
        
        prompt = fallback_prompts.get(prompt_name, f"Prompt '{prompt_name}' not found")
        return prompt.format(**kwargs) if kwargs else prompt
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        return self._langfuse_client


# Convenience function for getting prompts
def get_prompt(prompt_name: str, **kwargs: Any) -> str:
    """
    Get a prompt from Langfuse.
    
    Args:
        prompt_name: Name of the prompt in Langfuse
        **kwargs: Variables to substitute in the prompt template
        
    Returns:
        Formatted prompt string
    """
    manager = LangfuseManager()
    return manager.get_prompt(prompt_name, **kwargs)
