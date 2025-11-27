import html
import logging
import re
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class XmlDocumentParser:
    """
    Parser for XML documents from internal gateway API.

    Extracts and parses XML content from the 'topTextXml' field.
    """

    def parse(self, json_response: dict) -> str:
        """
        Parse JSON response containing XML content.

        Args:
            json_response: JSON response from internal gateway API

        Returns:
            Plain text extracted from XML
        """
        try:
            # Extract topTextXml from response
            # Support both structures: with "data" wrapper and without
            if "data" in json_response:
                data = json_response["data"]
            else:
                data = json_response

            if "topTextXml" not in data:
                logger.warning("No 'topTextXml' field in response")
                return ""

            xml_content = data["topTextXml"]
            if not xml_content:
                logger.warning("Empty topTextXml")
                return ""

            # Decode HTML entities (e.g., &lt; -> <, &gt; -> >)
            decoded_xml = html.unescape(xml_content)

            # Normalize HTML self-closing tags to XML format
            normalized_xml = self._normalize_html_tags(decoded_xml)

            try:
                root = ET.fromstring(normalized_xml)
                # Extract all text content
                text_parts = self._extract_text_from_element(root)
                combined_text = " ".join(text_parts)
                return self._clean_text(combined_text)

            except ET.ParseError as e:
                logger.warning(f"Failed to parse XML: {e}")
                # Fallback: try to extract text using regex
                return self._extract_text_fallback(decoded_xml)

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return ""

    @staticmethod
    def _normalize_html_tags(xml_content: str) -> str:
        """
        Normalize HTML self-closing tags to XML format.

        Converts tags like <br>, <hr>, <img> to <br/>, <hr/>, <img/> etc.
        """
        # List of common HTML self-closing tags
        self_closing_tags = [
            "br",
            "hr",
            "img",
            "input",
            "meta",
            "link",
            "area",
            "base",
            "col",
            "embed",
            "source",
            "track",
            "wbr",
        ]

        # Pattern to match self-closing tags that aren't already closed
        # Matches <tag> or <tag attributes> but not <tag/> or <tag />
        for tag in self_closing_tags:
            # Match opening tag without closing slash
            # Capture attributes in group 1
            pattern = rf"<{tag}(\s[^>]*)?(?<!/)>"
            replacement = rf"<{tag}\1/>"
            xml_content = re.sub(pattern, replacement, xml_content, flags=re.IGNORECASE)

        return xml_content

    def _extract_text_from_element(self, element: ET.Element) -> list[str]:
        """Recursively extract text from XML element."""
        text_parts = []

        # Get direct text content
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())

        # Process children
        for child in element:
            child_texts = self._extract_text_from_element(child)
            text_parts.extend(child_texts)

            # Get tail text (text after the element)
            if child.tail and child.tail.strip():
                text_parts.append(child.tail.strip())

        return text_parts

    def _extract_text_fallback(self, xml_content: str) -> str:
        """Fallback method to extract text using regex if XML parsing fails."""
        # Remove XML tags
        text = re.sub(r"<[^>]+>", " ", xml_content)
        # Decode HTML entities again in case they weren't decoded
        text = html.unescape(text)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        cleaned_text = self._clean_text(text.strip())
        logger.info(f"Cleaned text w fallback: {cleaned_text}")
        return cleaned_text

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text using same approach as JsonDocumentParser.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        s = text

        # Apply cleaning patterns from json_parser
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

        return s.strip()

    def get_title(self, json_response: dict) -> str:
        """
        Extract document title from response.

        Args:
            json_response: JSON response from internal gateway API

        Returns:
            Document title or empty string
        """
        try:
            # Support both structures: with "data" wrapper and without
            if "data" in json_response:
                data = json_response["data"]
            else:
                data = json_response

            # Check title field
            if "title" in data and data["title"]:
                return str(data["title"])
            # Fallback to other common fields
            for field in ["name", "docName", "documentName"]:
                if field in data and data[field]:
                    return str(data[field])
            return ""
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return ""
