"""
Utilities for Langfuse integration and prompt management.
"""

import os
from typing import Optional, Dict, Any, Type, TypeVar
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .prompts import get_fallback_prompt, get_fallback_prompt_data

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROXY_URL = "http://llm-audit-proxy-ml.prod.ml.aservices.tech/v1"

def create_llm(config: Dict[str, Any] = None) -> Any:
    """
    Создает простой LLM (ChatOpenAI) без structured output.
    Используется для вспомогательных задач (например, фильтрация).
    
    Args:
        config: Конфигурация модели (из Langfuse или fallback)
    
    Returns:
        Configured ChatOpenAI instance
    """
    # Значения по умолчанию
    model_name = "gpt-4o"
    temperature = 0.0
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_PROXY_URL)
    
    # Применяем конфиг из Langfuse или fallback
    if config:
        model_name = config.get("model", model_name)
        temperature = config.get("temperature", temperature)
        config_base_url = config.get("base_url") or config.get("baseUrl") or config.get("openai_api_base")
        if config_base_url:
            base_url = config_base_url
            
        try:
            temperature = float(temperature)
        except (ValueError, TypeError):
            temperature = 0.0

    llm = None
    
    # Проверяем ключи
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )
    
    if llm is None:
         # Fallback initialization
         llm = ChatOpenAI(
             model=model_name, 
             temperature=temperature,
             base_url=base_url
         )
         
    return llm


def create_structured_llm(response_model: Type[T], config: Dict[str, Any] = None) -> Any:
    """
    Создает LLM с structured output (возвращает объект Pydantic).
    Универсальная фабрика для всех агентов.
    
    Args:
        response_model: Pydantic модель, описывающая структуру ответа
        config: Конфигурация модели (из Langfuse)
    
    Returns:
        Configured LLM with structured output
    """
    llm = create_llm(config)
    return llm.with_structured_output(response_model)


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
        """
        data = self.get_prompt_data(prompt_name, **kwargs)
        return data["content"]

    def get_prompt_data(self, prompt_name: str, force_fallback: bool = False, **kwargs: Any) -> Dict[str, Any]:
        """
        Retrieve prompt text AND configuration from Langfuse.
        
        Args:
            prompt_name: Name of the prompt in Langfuse
            force_fallback: If True, forces the use of fallback prompts from prompts.py
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Dict containing:
            - content: Formatted prompt string
            - config: Model configuration dict (or empty dict if not found)
            - type: Prompt type (chat or text)
        """
        if self._langfuse_client is None or force_fallback:
            # Fallback prompts for development/testing (from prompts.py)
            fallback_data = get_fallback_prompt_data(prompt_name, **kwargs)
            return {
                "content": fallback_data["content"],
                "config": fallback_data["config"],
                "type": "text"
            }
        
        try:
            prompt = self._langfuse_client.get_prompt(prompt_name)
            if prompt is None:
                raise ValueError(f"Prompt '{prompt_name}' not found in Langfuse")
            
            # Format prompt with provided variables
            content = prompt.compile(**kwargs) if kwargs else prompt.compile()
            
            return {
                "content": content,
                "config": prompt.config,
                "type": getattr(prompt, "type", "text")
            }
            
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}' from Langfuse: {e}")
            # Fallback to local prompts on error
            fallback_data = get_fallback_prompt_data(prompt_name, **kwargs)
            return {
                "content": fallback_data["content"],
                "config": fallback_data["config"],
                "type": "text"
            }
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        return self._langfuse_client


# Convenience functions
def get_prompt(prompt_name: str, force_fallback: bool = False, **kwargs: Any) -> str:
    """Get prompt text only."""
    manager = LangfuseManager()
    data = manager.get_prompt_data(prompt_name, force_fallback=force_fallback, **kwargs)
    return data["content"]

def get_prompt_data(prompt_name: str, force_fallback: bool = False, **kwargs: Any) -> Dict[str, Any]:
    """Get prompt text and configuration."""
    manager = LangfuseManager()
    return manager.get_prompt_data(prompt_name, force_fallback=force_fallback, **kwargs)
