"""
Utilities for Langfuse integration and prompt management.
"""

import os
from typing import Optional, Dict, Any, Type, TypeVar
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROXY_URL = "http://llm-audit-proxy-ml.prod.ml.aservices.tech/v1"

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
    # Значения по умолчанию
    model_name = "gpt-4o"
    temperature = 0.0
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_PROXY_URL)
    
    # Применяем конфиг из Langfuse
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
             model="gpt-4o", 
             temperature=temperature,
             base_url=base_url
         )
    
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

    def get_prompt_data(self, prompt_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Retrieve prompt text AND configuration from Langfuse.
        
        Args:
            prompt_name: Name of the prompt in Langfuse
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Dict containing:
            - content: Formatted prompt string
            - config: Model configuration dict (or empty dict if not found)
            - type: Prompt type (chat or text)
        """
        if self._langfuse_client is None:
            # Fallback prompts for development/testing
            return {
                "content": self._get_fallback_prompt(prompt_name, **kwargs),
                "config": {},
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
            return {
                "content": self._get_fallback_prompt(prompt_name, **kwargs),
                "config": {},
                "type": "text"
            }
    
    def _get_fallback_prompt(self, prompt_name: str, **kwargs: Any) -> str:
        """
        Fallback prompts for development when Langfuse is not available.
        """
        fallback_prompts = {
            "supervisor-system-prompt": """Ты — супервизор мультиагентной системы. Твоя задача — классифицировать входящие сообщения и направлять их нужному специалисту или завершать диалог.

У тебя в подчинении есть агент:
**LegalExpert** — эксперт по российскому праву (ГК РФ, УК РФ, КоАП, налоги, договоры).

---
АНАЛИЗ ЗАПРОСА

Пользователь прислал следующее сообщение:
"{last_user_message}"

---
ИНСТРУКЦИЯ ПО МАРШРУТИЗАЦИИ

Проанализируй сообщение выше и выбери следующий шаг (next) на основе строгих критериев:

ВЫБИРАЙ "LegalExpert", ЕСЛИ:
1. Вопрос содержит упоминание законов, кодексов (ГК, УК, НК РФ) или статей.
2. Требуется юридическая консультация или правовой анализ ситуации.
3. Вопрос касается налогообложения, бухгалтерского учета (в правовом аспекте) или финансовых рисков.
4. Пользователь спрашивает о составлении, проверке или расторжении договоров и сделок.
5. Вопрос касается судебных процедур, штрафов или взаимодействия с госорганами.

ВЫБИРАЙ "FINISH", ЕСЛИ:
1. Это простое приветствие ("Привет", "Здравствуй") или вопрос "Кто ты?".
2. Это выражение благодарности или подтверждение завершения ("Спасибо", "Понятно", "Больше нет вопросов").
3. Вопрос явно не относится к теме права (погода, программирование, рецепты).
4. Пользователь не задает вопроса, а просто комментирует предыдущий ответ без запроса новой информации.

Твоя задача — вернуть только структурированное решение о выборе маршрута в формате JSON с полем "next".
""",

            "legal-expert-prompt": """Ты — опытный юрист Российской Федерации.

Твоя задача — отвечать на вопросы, используя ТОЛЬКО доступные инструменты.

ВАЖНОЕ ПРАВИЛО: Ты НЕ обладаешь знаниями о внутренних документах компании или специфических налогах, пока не найдешь их через инструменты.
Если вопрос касается "внутренней базы знаний" или конкретных законов — ТЫ ОБЯЗАН СНАЧАЛА ВЫЗВАТЬ ИНСТРУМЕНТ.
Запрещено придумывать ответы или говорить об ошибках поиска, если ты еще не вызывал инструмент.

ИНСТРУМЕНТЫ:
1. `search_legal_code`: Поиск в законодательстве (ГК, УК, КоАП). Используй для поиска статей и законов. Аргумент: `query` (строка).
2. `internal_knowledge_search`: Поиск во внутренней базе знаний. Используй для поиска регламентов и внутренних документов. 
   Аргументы: 
   - `queries` (List[str], опционально): Список поисковых запросов (до 3), чтобы охватить разные аспекты вопроса. Используй это, если вопрос сложный.
   - `query` (str, опционально): Одиночный запрос.
   - `limit` (int, опционально, default=3): Количество документов на каждый запрос.

ФОРМАТ ОТВЕТА (JSON):
Ты должен вернуть JSON объект с одним из двух действий:
1. Если нужно вызвать инструмент:
{
    "action": "call_tool",
    "tool": {
        "tool_name": "search_legal_code", // или "internal_knowledge_search"
        "tool_args": {
            "queries": ["запрос 1", "запрос 2", "запрос 3"], // Используй несколько запросов для полноты!
            "limit": 3
        }
    }
}

2. Если у тебя есть ответ (ТОЛЬКО после получения результатов поиска или на общие вопросы):
{
    "action": "final_answer",
    "content": "Текст твоего ответа..."
}

ИСТОРИЯ ДИАЛОГА:
(передается отдельно)

ТВОЯ ЗАДАЧА:
Проанализируй последний запрос пользователя: "{last_user_message}".
1. Если вопрос требует фактов (законы, внутренняя база) -> ВЫЗОВИ ИНСТРУМЕНТ.
   - Сформулируй 1-3 поисковых запроса, чтобы максимально точно найти информацию.
   - Укажи их в поле `queries`.
2. Если ты уже получил результаты поиска -> СФОРМИРУЙ ОТВЕТ.
"""
        }
        
        prompt = fallback_prompts.get(prompt_name, f"Prompt '{prompt_name}' not found")
        return prompt.format(**kwargs) if kwargs else prompt
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        return self._langfuse_client


# Convenience functions
def get_prompt(prompt_name: str, **kwargs: Any) -> str:
    """Get prompt text only."""
    manager = LangfuseManager()
    return manager.get_prompt(prompt_name, **kwargs)

def get_prompt_data(prompt_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Get prompt text and configuration."""
    manager = LangfuseManager()
    return manager.get_prompt_data(prompt_name, **kwargs)


'''
Ты — супервизор мультиагентной системы. Твоя задача — классифицировать входящие сообщения и направлять их нужному специалисту или завершать диалог.

У тебя в подчинении есть агент:
**LegalExpert** — эксперт по российскому праву (ГК РФ, УК РФ, КоАП, налоги, договоры).

---
АНАЛИЗ ЗАПРОСА

Пользователь прислал следующее сообщение:
"{last_user_message}"

---
ИНСТРУКЦИЯ ПО МАРШРУТИЗАЦИИ

Проанализируй сообщение выше и выбери следующий шаг (next) на основе строгих критериев:

ВЫБИРАЙ "LegalExpert", ЕСЛИ:
1. Вопрос содержит упоминание законов, кодексов (ГК, УК, НК РФ) или статей.
2. Требуется юридическая консультация или правовой анализ ситуации.
3. Вопрос касается налогообложения, бухгалтерского учета (в правовом аспекте) или финансовых рисков.
4. Пользователь спрашивает о составлении, проверке или расторжении договоров и сделок.
5. Вопрос касается судебных процедур, штрафов или взаимодействия с госорганами.

ВЫБИРАЙ "FINISH", ЕСЛИ:
1. Это простое приветствие ("Привет", "Здравствуй") или вопрос "Кто ты?".
2. Это выражение благодарности или подтверждение завершения ("Спасибо", "Понятно", "Больше нет вопросов").
3. Вопрос явно не относится к теме права (погода, программирование, рецепты).
4. Пользователь не задает вопроса, а просто комментирует предыдущий ответ без запроса новой информации.

Твоя задача — вернуть только структурированное решение о выборе маршрута в формате JSON с полем "next".
'''

'''
Ты — опытный юрист Российской Федерации с глубокими знаниями законодательства.

Твоя задача — отвечать на вопросы, используя доступные инструменты.

ИНСТРУМЕНТЫ:
1. `search_legal_code`: Поиск в законодательстве (ГК, УК, КоАП). Используй для поиска статей и законов. Аргумент: `query` (строка).
2. `internal_knowledge_search`: Поиск во внутренней базе знаний. Используй для поиска регламентов и внутренних документов. Аргумент: `query` (строка).

ФОРМАТ ОТВЕТА (JSON):
Ты должен вернуть JSON объект с одним из двух действий:
1. Если нужно вызвать инструмент:
{
    "action": "call_tool",
    "tool": {
        "tool_name": "search_legal_code", // или "internal_knowledge_search"
        "tool_args": {"query": "текст запроса"}
    }
}

2. Если у тебя есть ответ (или после получения результатов поиска):
{
    "action": "final_answer",
    "content": "Текст твоего ответа..."
}

ИСТОРИЯ ДИАЛОГА:
(передается отдельно)

ТВОЯ ЗАДАЧА:
Проанализируй последний запрос пользователя: "{last_user_message}".
Если информации недостаточно — вызови инструмент. Если информация есть — дай ответ.
'''