"""
Supervisor Node - управляющий узел для маршрутизации запросов.
"""

# Удаляем старую функцию create_supervisor_llm и импорты, которые переехали в utils
# Но чтобы search_replace сработал точно, заменим импорты и удалим функцию

import os
from typing import Dict, Any, Literal
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI # Удаляем неиспользуемый импорт

from ..state import AgentState
from ..utils import get_prompt_data, create_structured_llm


class RouteResponse(BaseModel):
    """
    Модель для структурированного ответа супервизора.
    Гарантирует строгий выбор маршрута.
    """
    next: Literal["LegalExpert", "AccountingExpert", "FINISH"]



# DEFAULT_PROXY_URL и create_supervisor_llm удалены, так как функционал перенесен в utils.py



async def supervisor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Supervisor узел - принимает решения о маршрутизации.
    
    Args:
        state: Текущее состояние агента
        config: Конфигурация запуска (включая callbacks)
        
    Returns:
        Обновление состояния с указанием следующего шага
    """
    try:
        # Получаем системный промт и конфиг из Langfuse
        prompt_data = get_prompt_data("supervisor-system-prompt")
        model_config = prompt_data.get("config", {})
        
        # Создаем LLM с structured output, используя универсальную фабрику
        # Передаем базовый конфиг при инициализации
        llm_factory_config = model_config 
        
        # Анализируем историю сообщений для принятия решения
        messages = state["messages"]
        
        analysis_prompt = ""
        
        # Формируем контекст для анализа
        if messages:
            # Находим последнее сообщение
            last_message = messages[-1]
            
            # Если последнее сообщение от AI (LegalExpert), то цикл завершен
            # В текущей архитектуре LegalExpert сам управляет своими tool calls и возвращает финальный ответ.
            # Supervisor не должен отправлять обратно на LegalExpert без добавления критики (чего он сейчас не делает),
            # иначе возникает бесконечный цикл.
            if isinstance(last_message, AIMessage):
                return {
                    "next": "FINISH"
                }

            # Находим последнее сообщение пользователя для анализа
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) and msg.content:
                    last_user_message = msg.content
                    break
            
            # Если есть сообщение пользователя (и это не ответ эксперта, который мы обработали выше)
            if last_user_message:
                # Вся логика и критерии теперь внутри промта в Langfuse.
                # Мы просто передаем переменную last_user_message.
                prompt_data = get_prompt_data("supervisor-system-prompt", force_fallback=False, last_user_message=last_user_message)
                analysis_prompt = prompt_data["content"]
                # Обновляем конфиг, если он пришел с промтом
                if prompt_data.get("config"):
                    llm_factory_config = prompt_data.get("config")
            else:
                analysis_prompt = "Нет сообщений для анализа. Завершить диалог."
        else:
            analysis_prompt = "Нет сообщений для анализа. Завершить диалог."
            
        # Создаем LLM с актуальным конфигом
        llm = create_structured_llm(RouteResponse, llm_factory_config)
        
        # Получаем решение от LLM
        # Передаем config, чтобы работали callbacks (в т.ч. Langfuse)
        response = await llm.ainvoke([HumanMessage(content=analysis_prompt)], config=config)
        
        print(f"DEBUG Supervisor: Decision={response.next}")
        
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