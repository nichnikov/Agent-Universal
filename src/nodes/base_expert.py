"""
Base Expert Node logic.
Contains common execution logic for expert agents.
"""

from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState
from ..utils import get_prompt_data, create_structured_llm, LangfuseManager
from ..logging_utils import parse_and_log_search_results


class ToolArgs(BaseModel):
    queries: Optional[List[str]] = None
    limit: Optional[int] = 3
    query: Optional[str] = None
    
    class Config:
        extra = "forbid"  # Запрещаем лишние поля, что важно для OpenAI Structured Output


async def execute_expert_node(
    state: AgentState,
    config: RunnableConfig,
    prompt_name: str,
    tools_map: Dict[str, Any],
    response_model: Type[BaseModel],
) -> Dict[str, Any]:
    """
    Универсальная логика выполнения узла эксперта.
    """
    try:
        messages = state["messages"]
        
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_user_message = msg.content
                break
        
        prompt_context = {"last_user_message": last_user_message} if last_user_message else {}
        prompt_data = get_prompt_data(prompt_name, **prompt_context)
        system_prompt = prompt_data["content"]
        model_config = prompt_data.get("config", {})
        
        llm = create_structured_llm(response_model, config=model_config)
        
        current_messages = [HumanMessage(content=system_prompt)] + messages
        
        # Первый вызов LLM
        response = await llm.ainvoke(current_messages, config=config)
        
        print(f"DEBUG Agent: Action={response.action}")
        if response.action == "call_tool":
            print(f"DEBUG Agent: Tool={response.tool}")
            if response.tool and response.tool.tool_args.queries:
                print(f"DEBUG Agent Generated Queries: {response.tool.tool_args.queries}")
        
        new_messages = []
        
        # ВАЛИДАЦИЯ: Проверяем, что эксперт не пытается дать ответ без предварительного поиска
        # Ищем в истории диалога результаты поиска
        has_search_results = False
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                if "Результат выполнения инструмента" in msg.content and "internal_knowledge_search" in msg.content:
                    has_search_results = True
                    break
        
        # Если эксперт пытается дать final_answer без предварительного поиска - принудительно вызываем инструмент
        if response.action == "final_answer" and not has_search_results:
            print("WARNING: Agent attempted to give final_answer without search. Forcing tool call.")
            # Принудительно вызываем инструмент напрямую, минуя структурированную модель
            if last_user_message:
                # Формируем поисковый запрос из вопроса пользователя
                search_query = last_user_message[:200]  # Берем первые 200 символов как запрос
                tool_name = "internal_knowledge_search"
                tool = tools_map.get(tool_name)
                
                if tool:
                    print(f"DEBUG Agent: Forced tool call with query: {search_query}")
                    # Вызываем инструмент напрямую
                    tool_args = {"queries": [search_query], "limit": 3}
                    try:
                        if hasattr(tool, "ainvoke"):
                            tool_result = await tool.ainvoke(tool_args, config=config)
                        else:
                            tool_result = tool.invoke(tool_args, config=config)
                        
                        # Логируем результаты поиска
                        if "search" in tool_name:
                            parse_and_log_search_results(tool_result)
                            
                            # Логируем структурированные результаты в Langfuse
                            try:
                                if hasattr(tool, 'get_last_search_results'):
                                    structured_results = tool.get_last_search_results()
                                    if structured_results:
                                        trace_id = None
                                        if config and "callbacks" in config:
                                            callbacks = config["callbacks"]
                                            for callback in callbacks:
                                                if hasattr(callback, 'get_trace_id'):
                                                    try:
                                                        trace_id = callback.get_trace_id()
                                                        break
                                                    except:
                                                        pass
                                                if hasattr(callback, 'run_manager') and hasattr(callback.run_manager, 'run_id'):
                                                    try:
                                                        if hasattr(callback.run_manager, 'parent_run_id'):
                                                            trace_id = callback.run_manager.parent_run_id
                                                        break
                                                    except:
                                                        pass
                                        
                                        langfuse_manager = LangfuseManager()
                                        if langfuse_manager.client:
                                            try:
                                                event_params = {
                                                    "name": "search_results_structured",
                                                    "metadata": {"search_results": structured_results}
                                                }
                                                if trace_id:
                                                    event_params["trace_id"] = trace_id
                                                
                                                langfuse_manager.client.event(**event_params)
                                            except Exception as e:
                                                print(f"Warning: Could not log structured results to Langfuse: {e}")
                            except Exception as e:
                                print(f"Warning: Error logging structured results: {e}")
                        
                        # Получаем структурированные результаты поиска для включения в промпт
                        search_materials_text = ""
                        if hasattr(tool, 'get_last_search_results'):
                            structured_results = tool.get_last_search_results()
                            if structured_results:
                                # Формируем текстовое представление материалов для промпта
                                materials_list = []
                                for query, docs in structured_results.items():
                                    materials_list.append(f"\nПоисковый запрос: {query}")
                                    for idx, doc in enumerate(docs, 1):
                                        materials_list.append(
                                            f"\nМатериал {idx}:"
                                            f"\n  Наименование: {doc.get('title', 'Без названия')}"
                                            f"\n  URL: {doc.get('url', 'Нет URL')}"
                                            f"\n  Содержание:\n{doc.get('content', '')[:2000]}..."  # Ограничиваем длину
                                        )
                                search_materials_text = "\n".join(materials_list)
                        
                        # Формируем обновленный системный промпт с материалами
                        updated_system_prompt = system_prompt
                        if search_materials_text:
                            updated_system_prompt = f"""{system_prompt}

НАЙДЕННЫЕ МАТЕРИАЛЫ ИЗ ВНУТРЕННЕЙ БАЗЫ ЗНАНИЙ:
{search_materials_text}

КРИТИЧЕСКИ ВАЖНО: Твой ответ должен строиться СТРОГО на основе этих найденных материалов. 
Используй ТОЛЬКО информацию из материалов выше. Запрещено добавлять информацию, которой нет в найденных материалах.
В поле 'references' укажи ТОЛЬКО те материалы, которые реально использованы в ответе (укажи наименования из раздела "Наименование" выше).
"""
                        
                        # Теперь формируем сообщение с результатами и просим LLM дать ответ
                        tool_result_message = HumanMessage(
                            content=f"Результат выполнения инструмента {tool_name}:\n{str(tool_result)}\n\nКРИТИЧЕСКИ ВАЖНО: Теперь дай финальный ответ, используя СТРОГО ТОЛЬКО информацию из результатов поиска выше. Запрещено добавлять информацию, которой нет в найденных материалах. В поле 'references' укажи ТОЛЬКО те материалы, которые реально использованы в ответе."
                        )
                        
                        # Обновляем системный промпт в сообщениях
                        updated_messages = [HumanMessage(content=updated_system_prompt)] + messages + [tool_result_message]
                        final_response = await llm.ainvoke(updated_messages, config=config)
                        
                        if final_response.content:
                            final_content = final_response.content
                            if hasattr(final_response, 'references') and final_response.references:
                                refs = "\n".join([f"- {r}" for r in final_response.references])
                                final_content += f"\n\nИспользованные материалы:\n{refs}"
                            new_messages.append(AIMessage(content=final_content))
                        else:
                            new_messages.append(AIMessage(content="Извините, не удалось сформировать ответ после поиска."))
                        
                        return {
                            "messages": new_messages
                        }
                    except Exception as e:
                        print(f"Error executing forced tool call: {e}")
                        new_messages.append(AIMessage(content=f"Ошибка при выполнении поиска: {str(e)}"))
                        return {
                            "messages": new_messages
                        }
                else:
                    print(f"ERROR: Tool '{tool_name}' not found in tools_map")
        
        if response.action == "call_tool" and response.tool:
            tool_name = response.tool.tool_name
            # Преобразуем модель Pydantic обратно в dict для вызова инструмента
            tool_args = response.tool.tool_args.model_dump(exclude_none=True)
            print(f"DEBUG Agent: Tool Args={tool_args}")
            tool = tools_map.get(tool_name)
            
            tool_result = f"Error: Tool '{tool_name}' not found."
            
            if tool:
                try:
                    # Проверяем, является ли инструмент асинхронным или синхронным
                    if hasattr(tool, "ainvoke"):
                         # Передаем config для трейсинга
                        tool_result = await tool.ainvoke(tool_args, config=config)
                    else:
                        tool_result = tool.invoke(tool_args, config=config)
                        
                    # Логируем результаты поиска, если это search tool
                    # Можно определить по имени инструмента или по наличию результата
                    if "search" in tool_name:
                         parse_and_log_search_results(tool_result)
                         
                         # Логируем структурированные результаты в Langfuse
                         try:
                             # Пытаемся получить структурированные результаты из инструмента
                             if hasattr(tool, 'get_last_search_results'):
                                 structured_results = tool.get_last_search_results()
                                 if structured_results:
                                     # Пытаемся получить trace_id из callback
                                     trace_id = None
                                     if config and "callbacks" in config:
                                         callbacks = config["callbacks"]
                                         for callback in callbacks:
                                             # Пытаемся получить trace_id из Langfuse callback
                                             if hasattr(callback, 'get_trace_id'):
                                                 try:
                                                     trace_id = callback.get_trace_id()
                                                     break
                                                 except:
                                                     pass
                                             # Альтернативный способ - через run_manager
                                             if hasattr(callback, 'run_manager') and hasattr(callback.run_manager, 'run_id'):
                                                 try:
                                                     # В Langfuse trace_id может быть в run_manager
                                                     if hasattr(callback.run_manager, 'parent_run_id'):
                                                         trace_id = callback.run_manager.parent_run_id
                                                     break
                                                 except:
                                                     pass
                                     
                                     # Логируем через Langfuse client
                                     langfuse_manager = LangfuseManager()
                                     if langfuse_manager.client:
                                         try:
                                             # Создаем event для логирования структурированных результатов
                                             # Формат: поисковый запрос: [{"title": ..., "url": ..., "content": ...}, ...]
                                             event_params = {
                                                 "name": "search_results_structured",
                                                 "metadata": {"search_results": structured_results}
                                             }
                                             if trace_id:
                                                 event_params["trace_id"] = trace_id
                                             
                                             langfuse_manager.client.event(**event_params)
                                         except Exception as e:
                                             print(f"Warning: Could not log structured results to Langfuse: {e}")
                         except Exception as e:
                             print(f"Warning: Error logging structured results: {e}")
                         
                except Exception as e:
                    tool_result = f"Error executing tool '{tool_name}': {str(e)}"
            
            print(f"DEBUG Agent: Raw Tool Result Preview (first 200 chars): {str(tool_result)[:200]}...")
            
            # Получаем структурированные результаты поиска для включения в промпт
            search_materials_text = ""
            if tool and "search" in tool_name and hasattr(tool, 'get_last_search_results'):
                structured_results = tool.get_last_search_results()
                if structured_results:
                    # Формируем текстовое представление материалов для промпта
                    materials_list = []
                    for query, docs in structured_results.items():
                        materials_list.append(f"\nПоисковый запрос: {query}")
                        for idx, doc in enumerate(docs, 1):
                            materials_list.append(
                                f"\nМатериал {idx}:"
                                f"\n  Наименование: {doc.get('title', 'Без названия')}"
                                f"\n  URL: {doc.get('url', 'Нет URL')}"
                                f"\n  Содержание:\n{doc.get('content', '')[:2000]}..."  # Ограничиваем длину
                            )
                    search_materials_text = "\n".join(materials_list)
            
            # Формируем обновленный системный промпт с материалами
            updated_system_prompt = system_prompt
            if search_materials_text:
                updated_system_prompt = f"""{system_prompt}

НАЙДЕННЫЕ МАТЕРИАЛЫ ИЗ ВНУТРЕННЕЙ БАЗЫ ЗНАНИЙ:
{search_materials_text}

КРИТИЧЕСКИ ВАЖНО: Твой ответ должен строиться СТРОГО на основе этих найденных материалов. 
Используй ТОЛЬКО информацию из материалов выше. Запрещено добавлять информацию, которой нет в найденных материалах.
В поле 'references' укажи ТОЛЬКО те материалы, которые реально использованы в ответе (укажи наименования из раздела "Наименование" выше).
"""
            
            tool_result_message = HumanMessage(
                content=f"Результат выполнения инструмента {tool_name}:\n{str(tool_result)}\n\nКРИТИЧЕСКИ ВАЖНО: Теперь дай финальный ответ, используя СТРОГО ТОЛЬКО информацию из результатов поиска выше. Запрещено добавлять информацию, которой нет в найденных материалах. В поле 'references' укажи ТОЛЬКО те материалы, которые реально использованы в ответе."
            )
            
            # Обновляем системный промпт в сообщениях
            updated_messages = [HumanMessage(content=updated_system_prompt)] + messages + [tool_result_message]
            final_response = await llm.ainvoke(updated_messages, config=config)
            
            if final_response.content:
                final_content = final_response.content
                if hasattr(final_response, 'references') and final_response.references:
                    refs = "\n".join([f"- {r}" for r in final_response.references])
                    final_content += f"\n\nИспользованные материалы:\n{refs}"
                new_messages.append(AIMessage(content=final_content))
                # Мы не добавляем промежуточные tool messages в историю state, чтобы не путать модель
            else:
                 new_messages.append(AIMessage(content="Извините, не удалось сформировать ответ после поиска."))
                 
        elif response.action == "final_answer" and response.content:
            final_content = response.content
            if hasattr(response, 'references') and response.references:
                refs = "\n".join([f"- {r}" for r in response.references])
                final_content += f"\n\nИспользованные материалы:\n{refs}"
            new_messages.append(AIMessage(content=final_content))
        else:
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

