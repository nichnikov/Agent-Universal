import asyncio
import logging
import os
from collections.abc import Iterable

import httpx

from .schemas import SearchItem, SearchParams, SearchResult

DOC_API_URL = os.getenv("DOC_API_URL") #, "https://site-backend-ss.prod.ss.aservices.tech/api/v1/desktop/document_get-by-id")

INTERNAL_GATEWAY_API_URL = os.getenv("INTERNAL_GATEWAY_API_URL") #, "https://internal-gateway-backend-ss.prod.ss.aservices.tech/api/v1/content/part-doc_get")
INTERNAL_GATEWAY_TOKEN = os.getenv("INTERNAL_GATEWAY_TOKEN") #, "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwOi8vc2NoZW1hcy5taWNyb3NvZnQuY29tL3dzLzIwMDgvMDYvaWRlbnRpdHkvY2xhaW1zL3JvbGUiOiJwbGF0IiwiaXNzIjoi0JzQtdC00LjQsNCz0YDRg9C_0L_QsCDQkNC60YLQuNC-0L0t0JzQptCk0K3QoCIsImF1ZCI6InBsYXQifQ.LZ4Ps4Zrq9JAL8abpKmLQbzsTP2g3NdXmb4tbIGD6MQ")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": USER_AGENT,
    "Referer": "https://1gl.ru/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

logger = logging.getLogger(__name__)


class SearchClient:
    def __init__(self) -> None:
        self.timeout = 15.0
        self.max_connections = 50

    def _extract_items(
        self, search_page_json: dict[str, object], pubdivid: int | None = None
    ) -> list[dict[str, object]]:
        """Extract items from search page response."""
        items = search_page_json["data"]["searchResponse"]["items"]
        if not isinstance(items, list):
            return []

        # Validate and convert items to dicts, adding pubdivid
        validated_items = []
        for item in items:
            item_with_pubdivid = {**item, "pubdivid": pubdivid}
            validated_items.append(SearchItem.model_validate(item_with_pubdivid).model_dump())
        return validated_items

    @staticmethod
    def _build_doc_url(base_doc_url: str, module_id: int | str, document_id: int | str, locale: str = "ru") -> str:
        return f"{base_doc_url}?moduleId={module_id}&documentId={document_id}&locale={locale}"

    @staticmethod
    def _build_internal_gateway_url(module_id: int | str, document_id: int | str) -> str:
        """Build URL for internal gateway API."""
        return f"{INTERNAL_GATEWAY_API_URL}?PubId=9&ModuleId={module_id}&Id={document_id}"

    async def _search_pages(
        self,
        *,
        client: httpx.AsyncClient,
        base_search_url: str,
        search_params: SearchParams,
        pages: int,
    ) -> list[dict[str, object]]:
        if pages <= 0:
            return []

        async def fetch_page(p: int) -> dict[str, object]:
            resp = await client.get(base_search_url, params={**search_params.model_dump(exclude_none=True), "page": p})
            logger.info(f"Search url: {str(resp.request.url)}")
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "application/json" not in ct:
                snippet = resp.text[:300].replace("\n", " ")
                raise httpx.HTTPError(f"Unexpected content-type: {ct}. Snippet: {snippet!r}")

            return resp.json()

        return await asyncio.gather(*[asyncio.create_task(fetch_page(p)) for p in range(1, pages + 1)])

    async def _fetch_docs(
        self,
        *,
        client: httpx.AsyncClient,
        items: Iterable[dict[str, object]],
        base_doc_url: str,
    ) -> list[SearchResult]:
        async def fetch_one(item: dict[str, object]) -> SearchResult:
            module_id = item.get("moduleId")
            doc_id = item.get("id")
            pubdivid = item.get("pubdivid")

            search_item = SearchItem.model_validate(item)

            # Choose API based on pubdivid
            if pubdivid in [3, 13]:
                # Use internal gateway API for pubdivid 3 and 13
                url = self._build_internal_gateway_url(module_id, doc_id)
                search_item.url = url

                try:
                    # Request with authorization token
                    headers = {"Authorization": f"Bearer {INTERNAL_GATEWAY_TOKEN}"}
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    ct = resp.headers.get("content-type", "")
                    if "application/json" not in ct:
                        snippet = resp.text[:300].replace("\n", " ")
                        raise httpx.HTTPError(f"Unexpected content-type: {ct}. Snippet: {snippet!r}")
                    json_data = resp.json()

                    return SearchResult(item=search_item, document=json_data, error=None)
                except Exception as e:
                    return SearchResult(item=search_item, document=None, error=str(e))
            else:
                # TODO: Предусмотреть передачу locale
                url = self._build_doc_url(base_doc_url, module_id, doc_id)
                search_item.url = url

                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    ct = resp.headers.get("content-type", "")
                    if "application/json" not in ct:
                        snippet = resp.text[:300].replace("\n", " ")
                        raise httpx.HTTPError(f"Unexpected content-type: {ct}. Snippet: {snippet!r}")
                    json_data = resp.json()

                    return SearchResult(item=search_item, document=json_data, error=None)
                except Exception as e:
                    return SearchResult(item=search_item, document=None, error=str(e))

        items_list = list(items)
        results = await asyncio.gather(*[fetch_one(it) for it in items_list])
        return list(results)

    async def fetch_search_pages_and_docs(
        self,
        *,
        search_params: SearchParams,
        pages: int,
        base_search_url: str,
        base_doc_url: str = DOC_API_URL,
    ) -> list[SearchResult]:
        limits = httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=self.max_connections)
        timeout_cfg = httpx.Timeout(self.timeout)

        async with httpx.AsyncClient(
            headers=HEADERS, limits=limits, timeout=timeout_cfg, follow_redirects=True, http2=True
        ) as client:
            pages_json = await self._search_pages(
                client=client,
                base_search_url=base_search_url,
                search_params=search_params,
                pages=pages,
            )

            all_items = []
            for pj in pages_json:
                all_items.extend(self._extract_items(pj, pubdivid=search_params.pubdivid))

            return await self._fetch_docs(client=client, items=all_items, base_doc_url=base_doc_url)
