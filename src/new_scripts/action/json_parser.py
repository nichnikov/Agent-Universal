import json
import logging
import re

logger = logging.getLogger(__name__)


class JsonDocumentParser:
    def __init__(self):
        self.texts = []

    def parse(self, document: dict) -> str:
        try:
            if "document" in document:
                self._process_document(document["document"])

            return self._concatenate_and_clean_texts()

        except Exception as e:
            logger.warning(f"Error processing document: {e}")
            return ""

    def _process_document(self, document: dict):
        if "content" not in document:
            return

        content = document["content"]

        if "snippetsInfo" in content:
            self._process_snippets_info(content["snippetsInfo"])

        if "body" in content:
            self._process_content_body(content)

        try:
            self._process_documents_element(document)
        except Exception as e:
            logger.warning(f"Warning: Could not process documents element: {e}")

        try:
            self._process_snippet_element(document)
        except Exception as e:
            logger.warning(f"Warning: Could not process snippets: {e}")

    def _process_snippets_info(self, snippets_info: list):
        for snippet_info in snippets_info:
            if "content" in snippet_info:
                snippet_content = snippet_info["content"]
                view_type = snippet_content.get("options", {}).get("viewType", "unknown")
                self._extract_texts_from_children(snippet_content, view_type)

    def _process_content_body(self, content_element):
        if isinstance(content_element, str):
            content_body = json.loads(content_element)
        elif isinstance(content_element, dict) and "body" in content_element:
            if isinstance(content_element["body"], str):
                content_body = json.loads(content_element["body"])
            else:
                content_body = content_element["body"]
        else:
            content_body = content_element

        if "children" not in content_body:
            return

        options = content_body.get("options", {})
        view_type = options.get("viewType")

        if view_type in ["situation", "searchArt", "snippet"]:
            count_attr = f"{view_type}_count"
            current_count = getattr(self, count_attr, 0) + 1
            setattr(self, count_attr, current_count)
            view_type = f"{view_type}_{current_count}"

        valid_children = [ch for ch in content_body["children"] if ch["type"] not in ["image", "div"]]
        self._extract_texts_from_children(valid_children, view_type)

    def _extract_texts_from_children(self, children_data, view_type):
        if isinstance(children_data, dict):
            current_type = children_data.get("type")
            if current_type in [
                "p",
                "list",
                "headerblock",
                "warning",
                "opinion",
                "advice",
                "example",
                "moreAbout",
                "reason",
                "operInfo",
                "importantContent",
                "fullAnswerHL",
                "documentRoot",
                "phrase",
            ]:
                self._extract_text_from_element(children_data, view_type, current_type)

        elif isinstance(children_data, list):
            for child in children_data:
                self._extract_texts_from_children(child, view_type)

    def _extract_text_from_element(self, element: dict, view_type: str, tag: str):
        if element["type"] in ["phrase", "list"]:
            if "options" in element and "number" in element["options"]:
                self.texts.append(f"number_{element['options']['number']}_view_type_{view_type}_tag_{tag}")

        if "children" in element:
            for child in element["children"]:
                self._extract_text_from_element(child, view_type, tag)

        if element["type"] == "text" and "options" in element and "value" in element["options"]:
            text_value = element["options"]["value"]
            if text_value:
                self.texts.append(text_value)

    def _process_documents_element(self, documents_element: dict):
        if not isinstance(documents_element, dict):
            return

        if "content" in documents_element:
            self._process_content_body(documents_element["content"])

        if "documents" in documents_element and documents_element["documents"]:
            for inner_document in documents_element["documents"]:
                self._process_documents_element(inner_document)

    def _process_snippet_element(self, documents_element: dict):
        if not isinstance(documents_element, dict):
            return

        if "content" in documents_element:
            content = documents_element["content"]
            if "snippets" in content:
                for snippet in content["snippets"]:
                    self._process_content_body(snippet["content"])

    def _concatenate_and_clean_texts(self) -> str:
        text_content = []
        for text in self.texts:
            if not text.startswith("number_") or "_view_type_" not in text:
                text_content.append(text)

        combined_text = " ".join(text_content)
        return self._clean_text(combined_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        s = text

        for pat, repl in (
            (r"&#160;", " "),  # HTML-nbsp
            (r";\.\.\.", r"; ..."),  # ;... → ; ...
            (r"\s+([,.;:)\]])", r"\1"),  # пробелы перед пунктуацией
            (r"([(\[])\s+", r"\1"),  # пробелы после открывающих скобок
            (r"([;:])(?!\s|$)", r"\1 "),  # пробел после ; :
        ):
            s = re.sub(pat, repl, s)

        # NBSP
        s = s.replace("\xa0 ", "\xa0").replace(" \xa0", "\xa0")
        s = re.sub(r" {2,}", " ", s)

        # Многоточие, если заканчивается на ; или :
        if re.search(r"(?:;|:)\s*$", s):
            s = re.sub(r"(?:;|:)\s*$", " ...", s)
            s = re.sub(r" {2,}", " ", s)

        return s

    def get_extracted_texts(self) -> list[str]:
        return self.texts.copy()


if __name__ == "__main__":

    def main():
        parser = JsonDocumentParser()
        example_path = "example_document.json"

        try:
            with open(example_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            cleaned_text = parser.parse(document)
            extracted_texts = parser.get_extracted_texts()

            logger.info(f"Extracted {len(extracted_texts)} text elements:")
            for i, text in enumerate(extracted_texts[:10]):
                logger.info(f"{i + 1}: {text}")

            if len(extracted_texts) > 10:
                logger.info(f"... and {len(extracted_texts) - 10} more elements")

            logger.info(cleaned_text)

        except FileNotFoundError:
            logger.info(f"Error: File {example_path} not found")
        except json.JSONDecodeError as e:
            logger.info(f"Error parsing JSON: {e}")
        except Exception as e:
            logger.info(f"Error: {e}")

    main()
