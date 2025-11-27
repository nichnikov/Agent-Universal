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
from ..utils import get_prompt


class RouteResponse(BaseModel):
    """
    Модель для структурированного ответа супервизора.
    Гарантирует строгий выбор маршрута.
    """
    next: Literal["LegalExpert", "FINISH"]


def create_supervisor_llm():
    """
    Создает LLM для Supervisor с structured output.
    
    Returns:
        Configured LLM with structured output
    """
    # Выбираем LLM на основе доступных API ключей
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,  # Нулевая температура для детерминированной маршрутизации
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.0,
        )
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
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
        # Получаем системный промт из Langfuse
        system_prompt = get_prompt("supervisor-system-prompt")
        
        # Создаем LLM с structured output
        llm = create_supervisor_llm()
        
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
