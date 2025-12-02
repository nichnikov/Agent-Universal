"""
Accounting tools for the Accounting Expert agent.
Mock implementations.
"""

from langchain_core.tools import tool

@tool
def search_accounting_code(query: str | None = None, queries: list[str] | None = None, limit: int | None = 3) -> str:
    """
    Ищет информацию в бухгалтерских нормативных документах (ПБУ, ФСБУ, НК РФ, План счетов).
    
    Используй этот инструмент для поиска:
    - Положений по бухгалтерскому учету (ПБУ, ФСБУ)
    - Проводок и плана счетов
    - Правил отражения хозяйственных операций
    - Налогового учета (в связке с бухгалтерским)
    
    Args:
        query: (Опционально) Одиночный поисковый запрос.
        queries: (Опционально) Список дополнительных запросов.
        limit: (Опционально) Ограничение количества результатов.
        
    Returns:
        str: Найденная информация из нормативной базы
    """
    search_queries = []
    if queries and isinstance(queries, list):
        search_queries.extend(queries)
    if query:
        search_queries.append(query)
    
    search_queries = list(set([q for q in search_queries if q]))
    
    if not search_queries:
        return "Ошибка: Не указан поисковый запрос."
        
    all_results = []
    metadata = f"SEARCH METADATA:\nQueries: {search_queries}\nLimit: {limit}\n---\n"
    all_results.append(metadata)

    for q in search_queries:
        query_lower = q.lower()
        result_text = ""
        
        if "основн" in query_lower or "средств" in query_lower or "фсбу 6" in query_lower:
             result_text = """ФСБУ 6/2020 "Основные средства":
1. Актив признается основным средством, если он имеет материально-вещественную форму, предназначен для использования в ходе обычной деятельности, и срок использования превышает 12 месяцев.
2. Первоначальная стоимость включает все затраты на приобретение и приведение в состояние готовности.
3. Амортизация начисляется с момента признания в учете (или с 1-го числа следующего месяца)."""
    
        elif "проводк" in query_lower or "счет" in query_lower:
            result_text = """Типовые проводки:
Дт 08 Кт 60 - Поступление основных средств
Дт 01 Кт 08 - Ввод в эксплуатацию
Дт 20 (26, 44) Кт 02 - Начисление амортизации
Дт 51 Кт 62 - Поступление оплаты от покупателя
Дт 62 Кт 90.1 - Реализация товаров/услуг"""
    
        elif "налог" in query_lower:
            result_text = """НК РФ Статья 248. Порядок определения доходов. К доходам в целях настоящей главы относятся:
1) доходы от реализации товаров (работ, услуг) и имущественных прав.
2) внереализационные доходы."""
    
        else:
            result_text = "Информация по данному запросу не найдена в базе бухгалтерских стандартов."
            
        all_results.append(f"Query: {q}\nResult:\n{result_text}\n")
    
    return "\n---\n".join(all_results)

