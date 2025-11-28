"""
Tool for searching internal knowledge base (Action).
Implementation based on provided scripts and requirements.
"""

import logging
import os
import asyncio
from typing import Type, Optional, List
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

from .action_internal.client import SearchClient, SearchParams, SearchResult, DOC_API_URL
from .action_internal.json_parser import JsonDocumentParser
from .action_internal.xml_parser import XmlDocumentParser

logger = logging.getLogger(__name__)


class KnowledgeSearchInput(BaseModel):
    """Input schema for the Knowledge Search Tool."""
    query: Optional[str] = Field(
        None, 
        description="Single search query. Use this OR 'queries'."
    )
    queries: Optional[List[str]] = Field(
        None,
        description="List of search queries (up to 3) to run in parallel. Use this for comprehensive coverage."
    )
    limit: int = Field(
        3, 
        description="Maximum number of documents to retrieve. Defaults to 3. Increase only if comprehensive search is needed."
    )
    pub_alias: Optional[str] = Field(
        None,
        description="Optional publication alias filter."
    )


class KnowledgeSearchTool(BaseTool):
    """
    Tool for searching specific technical documentation, articles, and guidelines 
    within the company's internal knowledge base.
    Returns the content of relevant documents.
    """
    name: str = "internal_knowledge_search"
    description: str = (
        "Useful for searching specific technical documentation, articles, and guidelines "
        "within the company's internal knowledge base. "
        "Returns the content of relevant documents."
    )
    args_schema: Type[BaseModel] = KnowledgeSearchInput
    
    # Private attributes for internal services
    _client: SearchClient = PrivateAttr()
    _json_parser: JsonDocumentParser = PrivateAttr()
    _xml_parser: XmlDocumentParser = PrivateAttr()

    def __init__(self, client: Optional[SearchClient] = None):
        super().__init__()
        self._client = client or SearchClient()
        self._json_parser = JsonDocumentParser()
        self._xml_parser = XmlDocumentParser()

    async def _execute_single_search(self, query: str, limit: int, pub_alias: Optional[str]) -> List[str]:
        """
        Executes a single search query and returns formatted document contents as a list.
        """
        logger.info(f"Executing search query: '{query}'")
        
        # Подготовка параметров поиска
        params = SearchParams(
            fstring=query,
            pubAlias=pub_alias or "bss", # Используем дефолт из рабочего примера
            pubdivid=13,                 # Используем дефолт из рабочего примера
            page=1,
            sortby="Relevance" 
        )

        # Вызов клиента
        results: List[SearchResult] = await self._client.fetch_search_pages_and_docs(
            search_params=params,
            pages=1,
            base_search_url="https://1gl.ru/system/content/search-new/" # Обновленный URL поиска
        )

        if not results:
            return []

        # Обработка результатов
        formatted_docs = []
        
        # Берем только топ-N результатов
        for res in results[:limit]:
            if res.error:
                logger.warning(f"Error fetching doc {res.item.id}: {res.error}")
                continue
            
            if not res.document:
                continue

            # Маршрутизация парсеров
            is_xml_gateway = res.item.pubdivid in [3, 13]
            
            parsed_text = ""
            title = res.item.docName or "Untitled"

            try:
                if is_xml_gateway:
                    # XML Parser logic
                    xml_title = self._xml_parser.get_title(res.document)
                    if xml_title:
                        title = xml_title
                    parsed_text = self._xml_parser.parse(res.document)
                else:
                    # JSON Parser logic
                    parsed_text = self._json_parser.parse(res.document)
            except Exception as e:
                logger.error(f"Error parsing document {res.item.id}: {e}")
                continue

            # Очистка и форматирование для LLM
            MAX_CHARS = 4000
            if len(parsed_text) > MAX_CHARS:
                parsed_text = parsed_text[:MAX_CHARS] + "\n...[Content Truncated]..."

            doc_entry = (
                f"## Document: {title}\n"
                f"Query Used: {query}\n"
                f"Source ID: {res.item.id} (Module: {res.item.moduleId})\n"
                f"URL: {res.item.url}\n"
                f"Content:\n{parsed_text}\n"
            )
            formatted_docs.append(doc_entry)
            
        return formatted_docs

    async def _arun(self, query: Optional[str] = None, queries: Optional[List[str]] = None, limit: int = 3, pub_alias: Optional[str] = None) -> str:
        """
        Executes the search and returns formatted document contents.
        Supports single 'query' or multiple 'queries'.
        """
        logger.info(f"Tool '{self.name}' called with query='{query}', queries='{queries}'")

        try:
            # Определяем список запросов
            search_queries = []
            if queries:
                search_queries = queries
            elif query:
                search_queries = [query]
            
            if not search_queries:
                return "Error: No search queries provided. Please provide 'query' or 'queries'."

            # Запускаем поиск параллельно
            tasks = [self._execute_single_search(q, limit, pub_alias) for q in search_queries]
            results_lists = await asyncio.gather(*tasks)
            
            # Объединяем результаты
            all_formatted_outputs = []
            
            # Добавляем метаданные поиска в начало для наглядности в Langfuse
            metadata = f"SEARCH METADATA:\nQueries: {search_queries}\nLimit per query: {limit}\n---\n"
            all_formatted_outputs.append(metadata)

            for res_list in results_lists:
                all_formatted_outputs.extend(res_list)

            if len(all_formatted_outputs) <= 1: # Только метаданные
                return "No documents found matching your queries."

            return "\n---\n".join(all_formatted_outputs)

        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return f"Error executing search: {str(e)}"

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution. Please use ainvoke() or async agent.")


def create_search_tool() -> KnowledgeSearchTool:
    """Factory function to create the search tool."""
    client = SearchClient()
    return KnowledgeSearchTool(client=client)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Загружаем переменные окружения
        load_dotenv()
        print("🔍 Запуск теста KnowledgeSearchTool...")
        
        try:
            # Создаем инструмент
            tool = create_search_tool()
            
            # Тестовый запрос (несколько запросов)
            queries = ["налог на прибыль", "НДС ставки"]
            limit = 2
            
            print(f"\nЗапросы: '{queries}' (limit={limit})")
            
            # Вызываем инструмент
            result = await tool.ainvoke({"queries": queries, "limit": limit})
            
            print(f"\nРезультат:\n{'-'*40}\n{result}\n{'-'*40}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    # Запускаем асинхронный цикл
    asyncio.run(main())
