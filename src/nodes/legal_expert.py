"""
Legal Expert Node - специализированный агент для юридических консультаций.
"""

import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..state import AgentState
from ..utils import get_prompt
from ..tools.legal_tools import search_legal_code
from ..tools.action_search_tool import create_search_tool


def create_legal_llm(tools: List[Any]):
    """
    Создает LLM для Legal Expert с привязанными инструментами.
    
    Args:
        tools: Список инструментов для привязки
        
    Returns:
        Configured LLM with bound tools
    """
    # Выбираем LLM на основе доступных API ключей
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,  # Низкая температура для точности в юридических вопросах
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
        )
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
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
        # Получаем системный промт из Langfuse
        system_prompt = get_prompt("legal-expert-prompt")
        
        # Инициализируем инструменты
        # search_legal_code - это уже готовый инструмент (@tool)
        action_search = create_search_tool()
        
        tools = [search_legal_code, action_search]
        tools_map = {
            "search_legal_code": search_legal_code,
            "internal_knowledge_search": action_search
        }
        
        # Создаем LLM с инструментами
        llm = create_legal_llm(tools)
        
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