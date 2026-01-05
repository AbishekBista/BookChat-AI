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
