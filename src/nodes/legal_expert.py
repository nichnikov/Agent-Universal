"""
Legal Expert Node - специализированный агент для юридических консультаций.
"""

import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..state import AgentState
from ..utils import get_prompt_data
from ..tools.legal_tools import search_legal_code
from ..tools.action_search_tool import create_search_tool


# Корпоративный прокси для доступа к LLM (OpenRouter)
DEFAULT_PROXY_URL = "http://llm-audit-proxy-ml.prod.ml.aservices.tech/v1"


def create_legal_llm(tools: List[Any], config: Dict[str, Any] = None):
    """
    Создает LLM для Legal Expert с привязанными инструментами.
    
    Args:
        tools: Список инструментов для привязки
        config: Конфигурация модели (из Langfuse)
        
    Returns:
        Configured LLM with bound tools
    """
    # Значения по умолчанию
    model_name = "gpt-4o"
    temperature = 0.1
    # По умолчанию используем корпоративный прокси
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_PROXY_URL)
    
    # Применяем конфиг из Langfuse
    if config:
        model_name = config.get("model", model_name)
        temperature = config.get("temperature", temperature)
        # Если в Langfuse задан специфичный URL
        config_base_url = config.get("base_url") or config.get("baseUrl") or config.get("openai_api_base")
        if config_base_url:
            base_url = config_base_url
        
        try:
            temperature = float(temperature)
        except (ValueError, TypeError):
            temperature = 0.1

    # Выбираем LLM
    llm = None
    
    # Проверяем ключи. Для OpenRouter через прокси обычно используется API key.
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )
    elif os.getenv("ANTHROPIC_API_KEY") and "claude" in model_name.lower() and base_url == DEFAULT_PROXY_URL:
        # Fallback на прямой Anthropic, если URL не меняли на прокси
        # (предполагаем, что прокси только для OpenAI-like, или у нас нет ключа для прокси)
        llm = ChatAnthropic(
            model=model_name,
            temperature=temperature
        )
    
    if llm is None:
         llm = ChatOpenAI(
             model="gpt-4o", 
             temperature=temperature,
             base_url=base_url
         )
    
    # Привязываем инструменты к LLM
    return llm.bind_tools(tools)


async def legal_expert_node(state: AgentState) -> Dict[str, Any]:
    """
    Legal Expert узел - обрабатывает юридические запросы.
    
    Args:
        state: Текущее состояние агента
        
    Returns:
        Обновление состояния с новыми сообщениями
    """
    try:
        # Получаем системный промт и конфиг из Langfuse
        prompt_data = get_prompt_data("legal-expert-prompt")
        system_prompt = prompt_data["content"]
        model_config = prompt_data.get("config", {})
        
        # Инициализируем инструменты
        action_search = create_search_tool()
        
        tools = [search_legal_code, action_search]
        tools_map = {
            "search_legal_code": search_legal_code,
            "internal_knowledge_search": action_search
        }
        
        # Создаем LLM с инструментами и конфигом
        llm = create_legal_llm(tools, config=model_config)
        
        # Формируем сообщения для LLM
        messages = [
            HumanMessage(content=system_prompt),
            *state["messages"]
        ]
        
        # Вызываем LLM асинхронно
        response = await llm.ainvoke(messages)
        
        # Обрабатываем ответ
        new_messages = []
        
        # Если LLM хочет вызвать инструмент
        if response.tool_calls:
            # Добавляем сообщение с вызовом инструмента
            new_messages.append(response)
            
            # Выполняем каждый вызов инструмента
            for tool_call in response.tool_calls:
                try:
                    tool_name = tool_call["name"]
                    tool = tools_map.get(tool_name)
                    
                    if tool:
                        # Разная логика вызова для sync/async инструментов
                        if tool_name == "internal_knowledge_search":
                            # BaseTool.ainvoke вызывает _arun
                            tool_result = await tool.ainvoke(tool_call["args"])
                        else:
                            # @tool создает StructuredTool, у которого есть invoke
                            tool_result = tool.invoke(tool_call["args"])
                    else:
                        tool_result = f"Error: Unknown tool '{tool_name}'"
                    
                    # Добавляем результат инструмента
                    new_messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                    
                except Exception as e:
                    # Обрабатываем ошибки инструментов
                    error_msg = f"Error executing tool '{tool_call['name']}': {str(e)}"
                    new_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call["id"]
                        )
                    )
            
            # Получаем финальный ответ после выполнения инструментов
            final_messages = messages + new_messages
            # Второй вызов тоже асинхронный
            final_response = await llm.ainvoke(final_messages)
            new_messages.append(final_response)
            
        else:
            # Если инструменты не нужны, просто добавляем ответ
            new_messages.append(response)
        
        return {
            "messages": new_messages
        }
        
    except Exception as e:
        # Обработка критических ошибок
        import traceback
        traceback.print_exc()
        error_response = AIMessage(
            content=f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
        )
        return {
            "messages": [error_response]
        }