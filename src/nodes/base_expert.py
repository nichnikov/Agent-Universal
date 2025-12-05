"""
Base Expert Node logic.
Contains common execution logic for expert agents.
"""

import asyncio
from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState
from ..utils import get_prompt_data, create_structured_llm, create_llm, LangfuseManager
class ToolArgs(BaseModel):
    queries: Optional[List[str]] = None
    limit: Optional[int] = None
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
        # Получаем сохраненные результаты поиска из состояния, если они есть
        final_search_results = state.get("search_results")
        final_relevant_materials = state.get("relevant_materials")
        
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                last_user_message = msg.content
                break
        
        prompt_context = {"last_user_message": last_user_message} if last_user_message else {}
        prompt_data = get_prompt_data(prompt_name, force_fallback=False, **prompt_context)
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
        
        # Логика выполнения инструмента
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
                         # Логируем структурированные результаты в Langfuse
                         try:
                             if hasattr(tool, 'get_last_search_results'):
                                 structured_results = tool.get_last_search_results()
                                 if structured_results:
                                     # Обновляем результаты поиска в переменной состояния
                                     final_search_results = structured_results
                                     
                                     # Логируем в консоль для отладки (ГАРАНТИРОВАННО)
                                     import json
                                     # Формируем список объектов для вывода, как просил пользователь
                                     log_output = []
                                     for q, d in structured_results.items():
                                         log_output.append({
                                             "query": q,
                                             "search_results": d
                                         })
                                     
                                     print(f"\nDEBUG: Structured Search Results for Langfuse:\n{json.dumps(log_output, ensure_ascii=False, indent=2)}\n")

                                     # Пытаемся получить trace_id из callback
                                     trace_id = None
                                     if config and "callbacks" in config:
                                         callbacks_obj = config["callbacks"]
                                         handlers = []
                                         # Проверяем тип callbacks - это может быть список или менеджер
                                         if isinstance(callbacks_obj, list):
                                             handlers = callbacks_obj
                                         elif hasattr(callbacks_obj, 'handlers'):
                                              handlers = callbacks_obj.handlers
                                         
                                         for callback in handlers:
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
                                             # Формируем список объектов для метаданных Langfuse
                                             log_output = []
                                             for q, d in structured_results.items():
                                                 log_output.append({
                                                     "query": q,
                                                     "search_results": d
                                                 })
                                             
                                             event_params = {
                                                 "name": "search_results_structured",
                                                 "metadata": {"search_results": log_output}
                                             }
                                             if trace_id:
                                                 event_params["trace_id"] = trace_id
                                             
                                             langfuse_manager.client.event(**event_params)
                                         except Exception as e:
                                             print(f"Warning: Could not log structured results to Langfuse: {e}")
                                     
                                     # === ЛОГИКА ФИЛЬТРАЦИИ ===
                                     try:
                                         # Получаем промпт для фильтрации и конфигурацию
                                         filter_prompt_data = get_prompt_data("filter_results_prompt", force_fallback=True)
                                         filter_prompt_template = filter_prompt_data["content"]
                                         filter_config = filter_prompt_data.get("config", {})
                                         
                                         # Создаем LLM для фильтрации (обычный текстовый, не structured)
                                         filter_llm = create_llm(filter_config)
                                         
                                         async def filter_single_doc(query: str, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                                             """Фильтрует один документ для конкретного запроса."""
                                             try:
                                                 text_content = doc.get('content', '')
                                                 if not text_content:
                                                     return None
                                                     
                                                 # Формируем сообщение
                                                 prompt = filter_prompt_template.format(query=query, text=text_content[:15000]) # Ограничиваем вход
                                                 messages = [HumanMessage(content=prompt)]
                                                 
                                                 # Вызываем LLM
                                                 response = await filter_llm.ainvoke(messages)
                                                 filtered_text = response.content.strip()
                                                 
                                                 # Если вернулся пустой текст или указание на отсутствие информации
                                                 if not filtered_text or "DOES NOT CONTAIN ANSWER" in filtered_text:
                                                     return None
                                                     
                                                 # Возвращаем обновленный документ
                                                 return {
                                                     "title": doc.get('title'),
                                                     "url": doc.get('url'),
                                                     "content": filtered_text
                                                 }
                                             except Exception as e:
                                                 print(f"Error filtering doc: {e}")
                                                 return None

                                         # Запускаем фильтрацию параллельно
                                         relevant_materials = {}
                                         tasks = []
                                         
                                         # Структура для маппинга результатов задач обратно к запросам
                                         task_mapping = [] # [(query), ...]
                                         
                                         for query, docs in structured_results.items():
                                             relevant_materials[query] = []
                                             for doc in docs:
                                                 tasks.append(filter_single_doc(query, doc))
                                                 task_mapping.append(query)
                                         
                                         if tasks:
                                             print(f"DEBUG: Starting filtering for {len(tasks)} documents...")
                                             filtered_results_list = await asyncio.gather(*tasks)
                                             
                                             # Разбираем результаты
                                             for i, res in enumerate(filtered_results_list):
                                                 query = task_mapping[i]
                                                 if res:
                                                     relevant_materials[query].append(res)
                                             
                                             # Удаляем запросы без релевантных материалов
                                             relevant_materials = {k: v for k, v in relevant_materials.items() if v}
                                             
                                             print(f"DEBUG: Filtering complete. Found relevant materials for {len(relevant_materials)} queries.")
                                             
                                             # Сохраняем в состояние
                                             final_relevant_materials = relevant_materials
                                     except Exception as e:
                                         print(f"Error during results filtering: {e}")
                                         # Fallback - не обновляем relevant_materials, используем полные
                                         # final_relevant_materials = None
                         except Exception as e:
                             print(f"Warning: Error logging structured results: {e}")
                         
                except Exception as e:
                    tool_result = f"Error executing tool '{tool_name}': {str(e)}"
            
            print(f"DEBUG Agent: Raw Tool Result Preview (first 200 chars): {str(tool_result)[:200]}...")
            
            # Получаем структурированные результаты поиска (из переменной или состояния) для включения в промпт
            # Используем relevant_materials, если они есть, иначе search_results
            search_materials_text = ""
            results_to_use = final_relevant_materials if final_relevant_materials else final_search_results
            
            if results_to_use:
                # Формируем текстовое представление материалов для промпта
                materials_list = []
                for query, docs in results_to_use.items():
                    materials_list.append(f"\nПоисковый запрос: {query}")
                    for idx, doc in enumerate(docs, 1):
                        materials_list.append(
                            f"\nМатериал {idx}:"
                            f"\n  Наименование: {doc.get('title', 'Без названия')}"
                            f"\n  URL: {doc.get('url', 'Нет URL')}"
                            f"\n  Содержание:\n{doc.get('content', '')}..."
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
В поле 'references' укажи ТОЛЬКО те материалы, которые реально использованы в ответе (укажи наименования и URL в скобках из раздела "Наименование" и "URL" выше).
"""
            
            tool_result_message = HumanMessage(
                content=f"Результат выполнения инструмента {tool_name}:\n{str(tool_result)}\n\nКРИТИЧЕСКИ ВАЖНО: Теперь дай финальный ответ, используя СТРОГО ТОЛЬКО информацию из результатов поиска выше. Запрещено добавлять информацию, которой нет в найденных материалах. В поле 'references' укажи ТОЛЬКО те материалы, которые реально использованы в ответе (наименование + URL)."
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
        
        # Возвращаем обновленное состояние, включая результаты поиска
        return {
            "messages": new_messages,
            "search_results": final_search_results,
            "relevant_materials": final_relevant_materials
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
