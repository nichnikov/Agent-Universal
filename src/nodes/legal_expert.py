"""
Legal Expert Node - специализированный агент для юридических консультаций.
"""

import os
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from ..state import AgentState
from ..utils import get_prompt_data, create_structured_llm
from ..tools.legal_tools import search_legal_code
from ..tools.action_search_tool import create_search_tool


# Модели данных для structured output
class ToolRequest(BaseModel):
    tool_name: Literal["search_legal_code", "internal_knowledge_search"]
    tool_args: Dict[str, Any]

class AgentAction(BaseModel):
    action: Literal["call_tool", "final_answer"]
    tool: Optional[ToolRequest] = None
    content: Optional[str] = None


async def legal_expert_node(state: AgentState) -> Dict[str, Any]:
    """
    Legal Expert узел - обрабатывает юридические запросы с использованием structured output.
    """
    try:
        # Получаем решение от LLM
        # Анализируем историю сообщений
        messages = state["messages"]
        
        # Находим последнее сообщение пользователя для формирования контекста промта
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_user_message = msg.content
                break
        
        # Получаем системный промт и конфиг из Langfuse
        # Передаем контекст (историю) в промт, если это предусмотрено в Langfuse
        prompt_context = {"last_user_message": last_user_message} if last_user_message else {}
        prompt_data = get_prompt_data("legal-expert-prompt", **prompt_context)
        system_prompt = prompt_data["content"]
        model_config = prompt_data.get("config", {})
        
        # Инициализируем инструменты
        action_search = create_search_tool()
        
        tools_map = {
            "search_legal_code": search_legal_code,
            "internal_knowledge_search": action_search
        }
        
        # Создаем LLM с structured output, используя универсальную фабрику
        llm = create_structured_llm(AgentAction, config=model_config)
        
        # Формируем сообщения для LLM
        # Добавляем system prompt в начало истории
        # Примечание: В реальном диалоге мы не должны дублировать system prompt каждый раз,
        # но для stateless графа это допустимо.
        current_messages = [HumanMessage(content=system_prompt)] + messages
        
        # Вызываем LLM асинхронно
        response = await llm.ainvoke(current_messages)
        
        print(f"DEBUG LegalExpert: Action={response.action}")
        if response.action == "call_tool":
            print(f"DEBUG LegalExpert: Tool={response.tool}")
        
        # Обрабатываем ответ
        new_messages = []
        
        if response.action == "call_tool" and response.tool:
            # Логика вызова инструмента
            tool_name = response.tool.tool_name
            tool_args = response.tool.tool_args
            tool = tools_map.get(tool_name)
            
            tool_result = f"Error: Tool '{tool_name}' not found."
            
            if tool:
                try:
                    if tool_name == "internal_knowledge_search":
                        tool_result = await tool.ainvoke(tool_args)
                    else:
                        tool_result = tool.invoke(tool_args)
                except Exception as e:
                    tool_result = f"Error executing tool '{tool_name}': {str(e)}"
            
            # Добавляем результат работы инструмента в историю
            # Важно: для модели мы должны сформулировать это как контекст
            # Так как мы не используем нативный tool calling, мы добавляем ToolMessage или AIMessage с результатом
            
            # Формируем сообщение с результатом для следующего шага
            tool_result_message = HumanMessage(
                content=f"Результат выполнения инструмента {tool_name}:\n{str(tool_result)}\n\nТеперь дай финальный ответ на основе этой информации."
            )
            
            # Рекурсивный вызов LLM с результатом (или добавление в историю и ожидание следующего шага)
            # В данном дизайне мы делаем один шаг "думать -> действовать -> отвечать"
            
            # Формируем новый контекст: история + результат инструмента
            next_step_messages = current_messages + [tool_result_message]
            
            # Второй вызов для получения финального ответа
            final_response = await llm.ainvoke(next_step_messages)
            
            # Возвращаем финальный ответ пользователю
            if final_response.content:
                new_messages.append(AIMessage(content=final_response.content))
            else:
                 new_messages.append(AIMessage(content="Извините, не удалось сформировать ответ после поиска."))
                 
        elif response.action == "final_answer" and response.content:
            new_messages.append(AIMessage(content=response.content))
            
        else:
            # Fallback если модель вернула что-то странное
            new_messages.append(AIMessage(content="Извините, я не смог определить дальнейшие действия."))
        
        return {
            "messages": new_messages
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_response = AIMessage(
            content=f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
        )
        return {
            "messages": [error_response]
        }