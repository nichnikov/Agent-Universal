"""
Search result logging utilities.
"""

import re
import logging

logger = logging.getLogger(__name__)

def parse_and_log_search_results(tool_result: str):
    """
    Парсит строковый результат поиска и выводит детализированный лог
    по запросам и найденным документам.
    
    Args:
        tool_result: Строковый результат выполнения инструмента поиска
    """
    print(f"\n{'='*20} ДЕТАЛИЗАЦИЯ ПОИСКА {'='*20}")
    
    if not tool_result or "Error" in tool_result or "No documents found" in tool_result:
        print(f"Результат поиска: {tool_result}")
        print(f"{'='*60}\n")
        return

    # Разбиваем на документы
    docs = tool_result.split("\n---\n")
    
    # Группируем по запросам
    results_by_query = {}
    
    for doc in docs:
        # Пропускаем пустые строки
        if not doc.strip():
            continue
            
        # Извлекаем Query Used
        query_match = re.search(r"Query Used: (.*?)\n", doc)
        query = query_match.group(1).strip() if query_match else "Unknown Query"
        
        # Извлекаем Title
        title_match = re.search(r"## Document: (.*?)\n", doc)
        title = title_match.group(1).strip() if title_match else "Untitled"
        
        # Извлекаем URL
        url_match = re.search(r"URL: (.*?)\n", doc)
        url = url_match.group(1).strip() if url_match else "No URL"
        
        # Извлекаем Content (все что после "Content:\n")
        content_match = re.search(r"Content:\n(.*)", doc, re.DOTALL)
        content = content_match.group(1).strip() if content_match else "No Content"

        if query not in results_by_query:
            results_by_query[query] = []
        
        results_by_query[query].append({
            "title": title, 
            "url": url,
            "content": content
        })
    
    # Выводим в лог
    for query, items in results_by_query.items():
        print(f"\n🔍 ПОИСКОВЫЙ ЗАПРОС: '{query}'")
        print(f"   Найдено документов: {len(items)}")
        for idx, item in enumerate(items, 1):
            print(f"   {idx}. {item['title']}")
            print(f"      URL: {item['url']}")
            # Выводим первые 300 символов контента
            preview = item['content'][:300].replace('\n', ' ')
            print(f"      Content: {preview}...")
            
    print(f"{'='*60}\n")

