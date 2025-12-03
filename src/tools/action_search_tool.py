"""
Tool for searching internal knowledge base (Action).
Implementation based on provided scripts and requirements.
"""

import logging
import os
import asyncio
from typing import Type, Optional, List, Dict, Any
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
        description="List of search queries to run in parallel. Use this for comprehensive coverage. The number of queries should be specified in the prompt instructions."
    )
    limit: Optional[int] = Field(
        None, 
        description="Maximum number of documents to retrieve per query. This value should be specified in the prompt instructions and will be passed from there."
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
    _default_pubdivid: int = PrivateAttr()
    _last_search_results: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    def __init__(self, client: Optional[SearchClient] = None, default_pubdivid: int = 13):
        super().__init__()
        self._client = client or SearchClient()
        self._json_parser = JsonDocumentParser()
        self._xml_parser = XmlDocumentParser()
        self._default_pubdivid = default_pubdivid
        self._last_search_results = None

    async def _execute_single_search(self, query: str, limit: int, pub_alias: Optional[str]) -> Dict[str, Any]:
        """
        Executes a single search query and returns structured results.
        """
        logger.info(f"Executing search query: '{query}' with pubdivid={self._default_pubdivid}")
        
        # Подготовка параметров поиска
        params = SearchParams(
            fstring=query,
            pubAlias=pub_alias or "bss", # Используем дефолт из рабочего примера
            pubdivid=self._default_pubdivid, # Используем настроенный pubdivid
            page=1,
            sortby="Relevance" 
        )

        # Вызов клиента
        results: List[SearchResult] = await self._client.fetch_search_pages_and_docs(
            search_params=params,
            pages=1,
            base_search_url="https://1gl.ru/system/content/search-new/" # Обновленный URL поиска
        )

        structured_docs = []

        if results:
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

                structured_docs.append({
                    "title": title,
                    "url": res.item.url,
                    "content": parsed_text,
                    "source_id": res.item.id,
                    "module_id": res.item.moduleId
                })
            
        return {
            "query": query,
            "documents": structured_docs
        }

    async def _arun(self, query: Optional[str] = None, queries: Optional[List[str]] = None, limit: Optional[int] = None, pub_alias: Optional[str] = None) -> str:
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

            # Используем значение limit из промпта, если оно не указано - используем значение по умолчанию
            # Значение по умолчанию должно быть указано в промпте, но на случай если не передано - используем 5
            effective_limit = limit if limit is not None else 5
            
            # Запускаем поиск параллельно
            tasks = [self._execute_single_search(q, effective_limit, pub_alias) for q in search_queries]
            results_list = await asyncio.gather(*tasks)
            
            # Объединяем результаты
            all_formatted_outputs = []
            
            # Добавляем метаданные поиска
            metadata_header = f"SEARCH METADATA:\nQueries: {search_queries}\nLimit per query: {effective_limit}\nPubDivID: {self._default_pubdivid}\n---\n"
            all_formatted_outputs.append(metadata_header)

            for res in results_list:
                q = res['query']
                docs = res['documents']
                
                if not docs:
                    all_formatted_outputs.append(f"Search Query: {q}\nNo documents found.")
                    continue

                doc_strings = []
                for d in docs:
                    doc_entry = (
                        f"Title: {d['title']}\n"
                        f"URL: {d['url']}\n"
                        f"Content:\n{d['content']}\n"
                    )
                    doc_strings.append(doc_entry)
                
                # Формируем блок для запроса
                docs_block = "\n---\n".join(doc_strings)
                all_formatted_outputs.append(f"Search Query: {q}\nResults:\n{docs_block}")

            # Сохраняем структурированные результаты для последующего логирования в Langfuse
            # Формируем структуру: поисковый запрос: [Ответ 1, Ответ 2, ...]
            structured_results = {}
            for res in results_list:
                query = res['query']
                docs = res['documents']
                structured_results[query] = [
                    {
                        "title": doc['title'],
                        "url": doc['url'],
                        "content": doc['content']
                    }
                    for doc in docs
                ]
            
            self._last_search_results = structured_results

            if len(all_formatted_outputs) <= 1: # Только метаданные
                return "No documents found matching your queries."

            return "\n\n====================\n\n".join(all_formatted_outputs)

        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return f"Error executing search: {str(e)}"

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution. Please use ainvoke() or async agent.")
    
    def get_last_search_results(self) -> Optional[Dict[str, Any]]:
        """
        Возвращает структурированные результаты последнего поиска.
        Формат: {поисковый_запрос: [{"title": ..., "url": ..., "content": ...}, ...]}
        """
        return self._last_search_results


def create_search_tool(default_pubdivid: int = 13) -> KnowledgeSearchTool:
    """Factory function to create the search tool."""
    client = SearchClient()
    return KnowledgeSearchTool(client=client, default_pubdivid=default_pubdivid)


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        # Загружаем переменные окружения
        # Запуск: python -m src.tools.action_search_tool
        load_dotenv()
        print("🔍 Запуск теста KnowledgeSearchTool...")
        
        try:
            # Создаем инструмент
            tool = create_search_tool(1)
            
            # Тестовый запрос (несколько запросов)
            queries = ["когда упрощенец платит НДС", "НДС сроки уплаты"]
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
