import asyncio
import logging

from search.action.client import SearchClient
from search.action.json_parser import JsonDocumentParser
from search.action.schemas import SearchParams
from search.action.xml_parser import XmlDocumentParser

logger = logging.getLogger(__name__)


class ActionSearchClient:
    """
    Generic client для поиска в Action без domain-specific логики.

    Выполняет только низкоуровневые операции:
    - Поиск через Action API
    - Парсинг документов
    - Возврат сырых результатов
    """

    def __init__(self) -> None:
        self.parser = JsonDocumentParser()
        self.xml_parser = XmlDocumentParser()

    def search(self, *, search_params: SearchParams, pages: int = 1) -> dict[str, object]:
        """Синхронная обёртка для async поиска."""
        return asyncio.run(self.search_async(search_params=search_params, pages=pages))

    async def search_async(
        self, *, search_params: SearchParams, pages: int = 1, base_search_url: str | None = None
    ) -> dict[str, object]:
        """
        Выполняет поиск и парсинг документов через Action API.

        Args:
            search_params: Параметры поиска
            pages: Количество страниц для загрузки
            base_search_url: URL для поиска (если None, используется дефолтный из client.py)
        """
        client = SearchClient()

        kwargs = {"search_params": search_params, "pages": pages}
        if base_search_url:
            kwargs["base_search_url"] = base_search_url

        doc_results = await client.fetch_search_pages_and_docs(**kwargs)
        parsed = []
        errors = []

        for search_result in doc_results:
            if search_result.error is not None:
                item_dict = search_result.item.model_dump()
                errors.append({"item": item_dict, "error": search_result.error})
                logger.error(
                    "doc fetch error | moduleId=%s id=%s | %s",
                    search_result.item.moduleId,
                    search_result.item.id,
                    search_result.error,
                )
                continue
            try:
                search_item = search_result.item
                doc_response = search_result.document

                # Choose parser based on pubdivid
                if search_item.pubdivid in [3, 13]:
                    # Use XML parser for internal gateway API responses
                    plain = self.xml_parser.parse(doc_response)
                    title = self.xml_parser.get_title(doc_response)
                else:
                    # Use standard parser for regular API responses
                    plain = self.parser.parse(doc_response)
                    title = self.parser._clean_text(doc_response["document"]["content"]["title"])

                out = {
                    "id": search_item.id,
                    "moduleId": search_item.moduleId,
                    "api_url": search_item.url,
                    "title": title,
                    "plain_text": plain,
                }
                parsed.append(out)

                # Лог итогового документа
                title_str = (out["title"] or "") if isinstance(out["title"], str) else ""
                snippet = plain[:800] + ("…" if len(plain) > 800 else "")
                logger.info("DOC #%s/%s | %s\n%s", out["moduleId"], out["id"], title_str, snippet)

            except Exception as e:
                item_dict = search_result.item.model_dump()
                errors.append({"item": item_dict, "error": f"parse error: {e}"})
                logger.exception("parse error | moduleId=%s id=%s", search_result.item.moduleId, search_result.item.id)

        return {
            "items": [result.item.model_dump() for result in doc_results],
            "parsed": parsed,
            "errors": errors,
        }


# ============================== Пример ==============================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)

    try:
        action_client = ActionSearchClient()

        # Пример использования с явными параметрами
        params = SearchParams(
            pubAlias="bss.plus",
            fixedregioncode="77",
            isUseHints="false",
            fstring="Как составить и сдать единую (упрощенную) налоговую декларацию",
            sortby="Relevance",
            status="actual",
            dataformat="json",
            pubdivid=1,
            pubId=220,
        )

        result = action_client.search(search_params=params, pages=1)

        logging.info(f"\nВсего результатов: {len(result['items'])}")
        logging.info(f"Успешно распарсено: {len(result['parsed'])}")
        logging.info(f"Ошибок: {len(result['errors'])}")

        for i, doc in enumerate(result["parsed"][:3], 1):
            logging.info(f"\n--- Документ #{i} ---")
            logging.info(f"ID: {doc['moduleId']}/{doc['id']}")
            logging.info(f"Заголовок: {doc['title']}")
            logging.info(f"Текст: {doc['plain_text'][:200]}{'...' if len(doc['plain_text']) > 200 else ''}")

    except Exception as e:
        logging.error(f"\nПроизошла ошибка во время теста: {e}")
        import traceback

        traceback.print_exc()
