"""
Main entry point for testing the Universal Autonomous Agent MVP.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler

from src.graph import app
from src.state import AgentState


async def test_agent(query: str) -> None:
    """
    Тестирует агента с заданным запросом.
    
    Args:
        query: Запрос пользователя для тестирования
    """
    print(f"\n{'='*60}")
    print(f"ТЕСТ: {query}")
    print(f"{'='*60}")
    
    # Создаем начальное состояние
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "next": ""
    }
    
    # Инициализируем Langfuse callback
    langfuse_handler = CallbackHandler()
    
    try:
        # Запускаем граф асинхронно с трейсингом
        result = await app.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})
        
        # Выводим результаты
        print("\nХОД ВЫПОЛНЕНИЯ:")
        for i, message in enumerate(result["messages"], 1):
            print(f"\n{i}. {type(message).__name__}:")
            if hasattr(message, 'content'):
                print(f"   Содержание: {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"   Вызовы инструментов: {message.tool_calls}")
        
        print(f"\nФИНАЛЬНОЕ СОСТОЯНИЕ: {result.get('next', 'ЗАВЕРШЕНО')}")
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Основная функция для запуска тестов."""
    # Загружаем переменные окружения
    load_dotenv()
    
    # Проверяем наличие API ключей
    if not os.getenv("OPENAI_API_KEY"):
        print("ВНИМАНИЕ: Не найден API ключ для LLM.")
        print("Установите OPENAI_API_KEY в .env файле.")
        return
    
    print("🤖 Universal Autonomous Agent MVP")
    print("Supervisor + Legal Expert Node")
    print("\nЗапуск тестовых сценариев...")
    
    # Тестовый сценарий 1: Юридический вопрос (должен вызвать мок-инструмент search_legal_code)
    # await test_agent("Как правильно оформить продажу офисной мебели юрлицом?")
    
    # Тестовый сценарий 2: Поиск во внутренней базе (должен вызвать internal_knowledge_search)
    # Для этого нужно, чтобы LLM выбрала этот инструмент. Промт говорит "используй инструменты".
    # Нам нужно, чтобы промт Legal Expert знал о новом инструменте и его назначении.
    # Новый инструмент называется "internal_knowledge_search".
    # await test_agent("Найди во внутренней базе знаний информацию о налоге на прибыль. Кратко расскажи, что это за налог")
    
    # Тестовый сценарий 3: Общий вопрос
    # await test_agent("Привет, кто ты?")
    
    # Тестовый сценарий 4: Бухгалтерский вопрос (должен вызвать accounting_knowledge_search)
    # await test_agent("какими нормами права, применяемыми в судах регулируется дела о банкротстве")
    await test_agent("какие льготы возможны для АНО образование Передай вопрос бухгалтерскому эксперту.")

    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
