"""
Supervisor Node - управляющий узел для маршрутизации запросов.
"""

import os
from typing import Dict, Any, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..state import AgentState
from ..utils import get_prompt_data


class RouteResponse(BaseModel):
    """
    Модель для структурированного ответа супервизора.
    Гарантирует строгий выбор маршрута.
    """
    next: Literal["LegalExpert", "FINISH"]


# Корпоративный прокси для доступа к LLM (OpenRouter)
DEFAULT_PROXY_URL = "http://llm-audit-proxy-ml.prod.ml.aservices.tech/v1"


def create_supervisor_llm(config: Dict[str, Any] = None):
    """
    Создает LLM для Supervisor с structured output.
    
    Args:
        config: Конфигурация модели (из Langfuse)
    
    Returns:
        Configured LLM with structured output
    """
    # Значения по умолчанию
    model_name = "gpt-4o"
    temperature = 0.0
    # По умолчанию используем корпоративный прокси
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_PROXY_URL)
    
    # Применяем конфиг из Langfuse, если есть
    if config:
        model_name = config.get("model", model_name)
        temperature = config.get("temperature", temperature)
        # Если в Langfuse задан специфичный URL, он переопределяет дефолтный
        config_base_url = config.get("base_url") or config.get("baseUrl") or config.get("openai_api_base")
        if config_base_url:
            base_url = config_base_url
            
        try:
            temperature = float(temperature)
        except (ValueError, TypeError):
            temperature = 0.0

    # Выбираем LLM
    # Так как используется OpenRouter/Proxy с OpenAI-compatible API (/v1),
    # мы предпочтительно используем ChatOpenAI даже для Claude моделей, если они идут через этот прокси.
    # Но оставляем возможность использовать ChatAnthropic, если ключи настроены явно иначе.
    
    llm = None
    
    # Проверяем ключи. Для OpenRouter обычно используется OPENAI_API_KEY или специфичный.
    # Предполагаем, что ключ лежит в OPENAI_API_KEY.
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )
    elif os.getenv("ANTHROPIC_API_KEY") and "claude" in model_name.lower() and base_url == DEFAULT_PROXY_URL:
        # Если это Claude и мы НЕ меняли URL на прокси (или прокси поддерживает Anthropic Native),
        # то пробуем нативный клиент. Но URL .../v1 намекает на OpenAI формат.
        # Поэтому это ветка скорее как fallback для прямого доступа.
        llm = ChatAnthropic(
            model=model_name,
            temperature=temperature
        )
    
    if llm is None:
         # Пытаемся инициализироваться хотя бы как-то, библиотека сама проверит наличие ключей
         llm = ChatOpenAI(
             model="gpt-4o", 
             temperature=temperature,
             base_url=base_url
         )
    
    # Настраиваем structured output
    return llm.with_structured_output(RouteResponse)


async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor узел - принимает решения о маршрутизации.
    
    Args:
        state: Текущее состояние агента
        
    Returns:
        Обновление состояния с указанием следующего шага
    """
    try:
        # Получаем системный промт и конфиг из Langfuse
        prompt_data = get_prompt_data("supervisor-system-prompt")
        system_prompt = prompt_data["content"]
        model_config = prompt_data.get("config", {})
        
        # Создаем LLM с structured output, используя конфиг
        llm = create_supervisor_llm(model_config)
        
        # Анализируем историю сообщений для принятия решения
        messages = state["messages"]
        
        # Формируем контекст для анализа
        if messages:
            # Берем последнее сообщение пользователя для анализа
            last_user_message = None
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    last_user_message = msg.content
                    break
            
            if last_user_message:
                analysis_prompt = f"""
{system_prompt}

Проанализируй последний запрос пользователя: "{last_user_message}"

Определи, нужна ли помощь LegalExpert или можно завершить диалог.

Критерии для LegalExpert:
- Вопросы о законах, статьях, кодексах РФ
- Юридические консультации
- Правовые процедуры
- Налогообложение
- Договоры и сделки

Критерии для FINISH:
- Общие вопросы
- Благодарности
- Уже получен полный ответ от LegalExpert
- Вопросы не по теме права
"""
            else:
                analysis_prompt = f"{system_prompt}\n\nНет сообщений для анализа. Завершить диалог."
        else:
            analysis_prompt = f"{system_prompt}\n\nНет сообщений для анализа. Завершить диалог."
        
        # Получаем решение от LLM
        response = await llm.ainvoke([HumanMessage(content=analysis_prompt)])
        
        # Возвращаем обновление состояния
        return {
            "next": response.next
        }
        
    except Exception as e:
        # В случае ошибки завершаем диалог
        print(f"Error in supervisor_node: {e}")
        return {
            "next": "FINISH"
        }