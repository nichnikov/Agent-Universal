import uuid
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler
from src.graph import app as agent_app

# Создаем приложение FastAPI
app = FastAPI(
    title="Agent Universal API",
    description="API server for interacting with the Universal Agent",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agent Universal API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Эндпоинт для общения с агентом.
    Принимает сообщение пользователя и возвращает ответ агента.
    """
    try:
        # Генерируем thread_id, если не передан
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Инициализируем Langfuse callback handler
        # Он автоматически подхватит переменные окружения (LANGFUSE_PUBLIC_KEY, etc.)
        # Проверяем наличие ключей, чтобы избежать ошибок инициализации, если они не заданы
        callbacks = []
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                langfuse_handler = CallbackHandler(
                    session_id=thread_id,
                    user_id="user_api" # Можно параметризировать при необходимости
                )
                callbacks.append(langfuse_handler)
            except Exception as e:
                print(f"Warning: Failed to initialize Langfuse callback: {e}")
        
        # Конфигурация для LangGraph
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks
        }
        
        # Формируем входные данные
        inputs = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        # Запускаем граф
        # Используем invoke, так как graph.py экспортирует app (CompiledGraph)
        result = await agent_app.ainvoke(inputs, config=config)
        
        # Извлекаем последний ответ от агента (обычно это последнее сообщение)
        if result and "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            response_content = last_message.content
        else:
            response_content = "Извините, агент не вернул корректный ответ."
            
        return ChatResponse(
            response=response_content,
            thread_id=thread_id
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Запускаем сервер
    # При запуске через docker-compose CMD уже вызывает uvicorn
    # Этот блок нужен для локальной отладки
    uvicorn.run(app, host="0.0.0.0", port=8080)
