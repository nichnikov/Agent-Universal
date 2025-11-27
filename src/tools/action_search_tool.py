"""
Tool for searching internal knowledge base (Action).
Implementation based on provided scripts and requirements.
"""

import logging
import os
from typing import Type, Optional, List
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

from .action_internal.client import SearchClient, SearchParams, SearchResult, DOC_API_URL
from .action_internal.json_parser import JsonDocumentParser
from .action_internal.xml_parser import XmlDocumentParser

logger = logging.getLogger(__name__)


class KnowledgeSearchInput(BaseModel):
    """Input schema for the Knowledge Search Tool."""
    query: str = Field(
        ..., 
        description="Search query or question to find information in the internal knowledge base."
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

    async def _arun(self, query: str, limit: int = 3, pub_alias: Optional[str] = None) -> str:
        """
        Executes the search and returns formatted document contents.
        """
        logger.info(f"Tool '{self.name}' called with query: '{query}'")

        try:
            # Подготовка параметров поиска
            params = SearchParams(
                fstring=query,
                pubAlias=pub_alias,
                page=1,
                sortby="Relevance" 
            )

            # Вызов клиента (используем существующую логику пагинации, но ограничиваем 1 страницей для скорости)
            # URL берется из переменной модуля client.py, которая инициализируется из ENV или дефолта
            results: List[SearchResult] = await self._client.fetch_search_pages_and_docs(
                search_params=params,
                pages=1,
                base_search_url="https://site-backend-ss.prod.ss.aservices.tech/api/v1/desktop/search" # TODO: Вынести в конфиг
            )

            if not results:
                return "No documents found matching your query."

            # Обработка результатов
            formatted_outputs = []
            
            # Берем только топ-N результатов
            for res in results[:limit]:
                if res.error:
                    logger.warning(f"Error fetching doc {res.item.id}: {res.error}")
                    continue
                
                if not res.document:
                    continue

                # Маршрутизация парсеров
                # Логика определения типа документа на основе pubdivid
                # Согласно client.py pubdivid 3 и 13 идут через шлюз (XML)
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
                # Обрезаем слишком длинные тексты, чтобы не забить контекст
                MAX_CHARS = 4000
                if len(parsed_text) > MAX_CHARS:
                    parsed_text = parsed_text[:MAX_CHARS] + "\n...[Content Truncated]..."

                doc_entry = (
                    f"## Document: {title}\n"
                    f"Source ID: {res.item.id} (Module: {res.item.moduleId})\n"
                    f"URL: {res.item.url}\n"
                    f"Content:\n{parsed_text}\n"
                )
                formatted_outputs.append(doc_entry)

            if not formatted_outputs:
                return "Documents found but failed to parse content."

            return "\n---\n".join(formatted_outputs)

        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return f"Error executing search: {str(e)}"

    def _run(self, *args, **kwargs):
        raise NotImplementedError("This tool only supports async execution. Please use ainvoke() or async agent.")


def create_search_tool() -> KnowledgeSearchTool:
    """Factory function to create the search tool."""
    client = SearchClient()
    return KnowledgeSearchTool(client=client)

