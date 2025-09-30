import requests
import time
from typing import List, Dict, Any, Optional
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenLibraryAPI:
    def __init__(self):
        self.base_url = "https://openlibrary.org"
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "BookChatAI/1.0 (Educational Purpose)"}
        )

        self.last_request_time = 0
        self.min_request_interval = 0.1

    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def search_books(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        logger.info(
            f"OpenLibrary: Starting book search for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )
        logger.info(f"OpenLibrary: Requesting {limit} results")

        try:
            logger.info("OpenLibrary: Applying rate limiting...")
            self._rate_limit()

            original_query = query
            query = query.strip().replace(" ", "+")
            logger.info(f"OpenLibrary: Cleaned query: '{original_query}' -> '{query}'")

            url = f"{self.base_url}/search.json"
            params = {
                "q": query,
                "limit": limit,
                "fields": "key,title,author_name,first_publish_year,isbn,subject,publisher,language,number_of_pages_median,cover_i",
            }

            logger.info(f"OpenLibrary: Making API request to {url}")
            logger.info(f"OpenLibrary: Request params: {params}")

            request_start = time.time()
            response = self.session.get(url, params=params, timeout=10)
            request_duration = time.time() - request_start

            logger.info(
                f"OpenLibrary: API request completed in {request_duration:.2f}s"
            )
            logger.info(f"OpenLibrary: Response status: {response.status_code}")

            response.raise_for_status()

            data = response.json()
            books = data.get("docs", [])

            logger.info(f"OpenLibrary: Received {len(books)} raw book results")

            logger.info("OpenLibrary: Processing book data...")
            processed_books = []
            for i, book in enumerate(books):
                logger.info(
                    f"OpenLibrary: Processing book {i + 1}/{len(books)}: '{book.get('title', 'Unknown')}'"
                )
                processed_book = self._process_book_data(book)
                if processed_book:
                    processed_books.append(processed_book)

            logger.info(f"Found {len(processed_books)} books for query: {query}")
            return processed_books

        except requests.RequestException as e:
            logger.error(f"Error searching books: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in book search: {str(e)}")
            return []

    def get_book_details(self, book_key: str) -> Optional[Dict[str, Any]]:
        try:
            self._rate_limit()

            url = f"{self.base_url}{book_key}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            book_data = response.json()

            if book_data.get("works"):
                work_key = book_data["works"][0]["key"]
                work_data = self._get_work_details(work_key)
                if work_data:
                    book_data.update(work_data)

            return self._process_detailed_book_data(book_data)

        except requests.RequestException as e:
            logger.error(f"Error getting book details: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting book details: {str(e)}")
            return None

    def get_author_info(self, author_key: str) -> Optional[Dict[str, Any]]:
        try:
            self._rate_limit()

            url = f"{self.base_url}{author_key}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            author_data = response.json()

            return {
                "name": author_data.get("name", "Unknown"),
                "birth_date": author_data.get("birth_date", ""),
                "death_date": author_data.get("death_date", ""),
                "bio": author_data.get("bio", {}).get("value", "")
                if isinstance(author_data.get("bio"), dict)
                else str(author_data.get("bio", "")),
                "wikipedia": author_data.get("wikipedia", ""),
                "key": author_data.get("key", ""),
            }

        except requests.RequestException as e:
            logger.error(f"Error getting author info: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting author info: {str(e)}")
            return None

    def search_by_author(
        self, author_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        query = f"author:{author_name}"
        return self.search_books(query, limit)

    def search_by_subject(self, subject: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = f"subject:{subject}"
        return self.search_books(query, limit)

    def _get_work_details(self, work_key: str) -> Optional[Dict[str, Any]]:
        try:
            self._rate_limit()

            url = f"{self.base_url}{work_key}.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.warning(f"Error getting work details: {str(e)}")
            return None

    def _process_book_data(self, book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if not book.get("title"):
                return None

            processed = {
                "key": book.get("key", ""),
                "title": book.get("title", "").strip(),
                "author_name": book.get("author_name", []),
                "first_publish_year": book.get("first_publish_year"),
                "isbn": book.get("isbn", []),
                "subject": book.get("subject", []),
                "publisher": book.get("publisher", []),
                "language": book.get("language", []),
                "number_of_pages": book.get("number_of_pages_median"),
                "cover_id": book.get("cover_i"),
            }

            if processed["cover_id"]:
                processed["cover_url"] = (
                    f"https://covers.openlibrary.org/b/id/{processed['cover_id']}-M.jpg"
                )

            return processed

        except Exception as e:
            logger.warning(f"Error processing book data: {str(e)}")
            return None

    def _process_detailed_book_data(self, book: Dict[str, Any]) -> Dict[str, Any]:
        try:
            processed = {
                "title": book.get("title", ""),
                "description": self._extract_description(book.get("description")),
                "subjects": book.get("subjects", []),
                "first_sentence": self._extract_description(book.get("first_sentence")),
                "key": book.get("key", ""),
                "revision": book.get("revision", 0),
                "latest_revision": book.get("latest_revision", 0),
                "created": book.get("created", {}).get("value", ""),
                "last_modified": book.get("last_modified", {}).get("value", ""),
            }

            return processed

        except Exception as e:
            logger.warning(f"Error processing detailed book data: {str(e)}")
            return {}

    def _extract_description(self, description_field) -> str:
        if not description_field:
            return ""

        if isinstance(description_field, str):
            return description_field
        elif isinstance(description_field, dict):
            return description_field.get("value", "")
        elif isinstance(description_field, list) and description_field:
            first_item = description_field[0]
            if isinstance(first_item, str):
                return first_item
            elif isinstance(first_item, dict):
                return first_item.get("value", "")

        return str(description_field)

    def get_trending_books(
        self, subject: str = "fiction", limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            query = f"subject:{subject}"
            books = self.search_books(query, limit * 2)

            return books[:limit]

        except Exception as e:
            logger.error(f"Error getting trending books: {str(e)}")
            return []
