import os
from typing import List, Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langchain.agents import AgentType, initialize_agent
from utils.openlibrary_api import OpenLibraryAPI
from utils.book_rag import BookRAG
from utils.prompt_templates import PromptTemplates
from utils.book_tools import get_book_tools
from utils.lcel_chains import BookChatChains
import boto3
import logging
import time
from langchain.callbacks.tracers import LangChainTracer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


langchain_tracer = None
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    try:
        langchain_tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "bookchat-ai")
        )
        logger.info("LangSmith tracing initialized successfully")
        logger.info(
            f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT', 'bookchat-ai')}"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith tracer: {str(e)}")
        langchain_tracer = None
else:
    logger.info(
        "LangSmith tracing not enabled. Set LANGCHAIN_TRACING_V2=true to enable."
    )


class BookChatSystem:
    def __init__(self, api_key: Optional[str] = None):
        logger.info("Initializing BookChatSystem...")
        start_time = time.time()

        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"AWS Region: {self.aws_region}")
        logger.info("Initializing AWS Bedrock client...")
        try:
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            )
            logger.info("AWS Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            logger.exception("Bedrock client initialization error details:")
            raise ValueError(f"Failed to initialize AWS Bedrock client: {str(e)}")

        model_id = os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "5000"))
        logger.info(
            f"LLM Configuration - Model: {model_id}, Temperature: {temperature}, Max Tokens: {max_tokens}"
        )

        try:
            self.llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=model_id,
                model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
            )
            logger.info("ChatBedrock LLM initialized")
        except Exception as e:
            logger.warning(
                f"ChatBedrock initialization failed ({e}), falling back to FakeListLLM for local testing"
            )
            try:
                from langchain_community.llms import FakeListLLM

                self.llm = FakeListLLM(
                    responses=["[local-fallback] I'm running in test mode."]
                )
                logger.info("FakeListLLM initialized as fallback LLM")
            except Exception:
                logger.exception("Failed to initialize fallback FakeListLLM")
                raise

        logger.info("Initializing OpenLibrary API...")
        self.openlibrary_api = OpenLibraryAPI()
        logger.info("OpenLibrary API initialized")

        logger.info("Initializing BookRAG system...")
        self.book_rag = BookRAG()
        logger.info("BookRAG system initialized")

        logger.info("Initializing PromptTemplates...")
        self.prompt_templates = PromptTemplates()
        logger.info("PromptTemplates initialized")

        logger.info("Initializing conversation memory...")
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
            memory_key="chat_history_summary",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        logger.info("Initializing LangChain Tools and Agent...")

        self.tools = get_book_tools()
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate",
                return_intermediate_steps=False,
            )
            logger.info("LangChain Agent initialized successfully")
        except Exception as e:
            logger.warning(
                f"Agent initialization failed, falling back to basic chat: {str(e)}"
            )
            logger.exception("Agent initialization error details:")
            self.agent = None
        logger.info("Initializing LCEL chains...")
        try:
            self.lcel_chains = BookChatChains(
                self.llm, self.book_rag, self.openlibrary_api
            )
            logger.info("LCEL chains initialized successfully")
        except Exception as e:
            logger.warning(f"LCEL chains initialization failed: {str(e)}")
            self.lcel_chains = None
        self.conversation_history = []
        logger.info("Enhanced conversation memory initialized")
        init_duration = time.time() - start_time
        logger.info(f"BookChatSystem initialized successfully in {init_duration:.2f}s")

    def update_api_key(self, api_key: str):
        logger.info(
            "AWS Bedrock uses environment credentials - no API key update needed"
        )

    def clear_conversation_memory(self):
        logger.info("Clearing conversation memory due to book context change")
        try:
            if hasattr(self.memory, "clear"):
                self.memory.clear()
            elif hasattr(self.memory, "chat_memory"):
                self.memory.chat_memory.clear()
            if hasattr(self.summary_memory, "clear"):
                self.summary_memory.clear()
            elif hasattr(self.summary_memory, "chat_memory"):
                self.summary_memory.chat_memory.clear()
            self.conversation_history = []
            logger.info("Conversation memory cleared successfully")
        except Exception as e:
            logger.warning(f"Error clearing conversation memory: {str(e)}")
            try:
                from langchain.memory import (
                    ConversationBufferWindowMemory,
                    ConversationSummaryBufferMemory,
                )

                self.memory = ConversationBufferWindowMemory(
                    k=5, memory_key="chat_history", return_messages=True
                )
                self.summary_memory = ConversationSummaryBufferMemory(
                    llm=self.llm,
                    max_token_limit=2000,
                    memory_key="chat_history_summary",
                    return_messages=True,
                )
                self.conversation_history = []
                logger.info("Conversation memory reinitialized as fallback")
            except Exception as fallback_error:
                logger.error(f"Failed to reinitialize memory: {str(fallback_error)}")

    def _extract_key_terms(self, query: str) -> str:
        import re

        query_lower = query.lower()

        noise_words = [
            "tell me about",
            "what is",
            "who wrote",
            "who is",
            "what are",
            "how do",
            "can you",
            "please",
            "i want to know",
            "information about",
            "details about",
            "explain",
            "describe",
            "about",
            "?",
        ]

        cleaned_query = query_lower
        for noise in noise_words:
            cleaned_query = cleaned_query.replace(noise, " ")

        stop_words = {
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "as",
            "it",
            "he",
            "she",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
        }

        words = re.findall(r"\b\w{2,}\b", cleaned_query)
        meaningful_words = [w for w in words if w not in stop_words]

        if meaningful_words:
            return " ".join(meaningful_words[:3])

        return query.strip()

    def _validate_book_relevance(self, user_query: str) -> Dict[str, Any]:
        logger.info(
            f"Validating book relevance for query: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'"
        )

        query_lower = user_query.lower().strip()

        explicit_book_terms = [
            "book",
            "books",
            "novel",
            "novels",
            "literature",
            "literary",
            "author",
            "authors",
            "writer",
            "writers",
            "poet",
            "poets",
            "read",
            "reading",
            "write",
            "writing",
            "publish",
            "published",
            "story",
            "stories",
            "plot",
            "character",
            "characters",
            "chapter",
            "chapters",
            "fiction",
            "non-fiction",
            "nonfiction",
            "poetry",
            "poem",
            "poems",
            "prose",
            "library",
            "libraries",
            "bookstore",
            "isbn",
            "edition",
            "manuscript",
            "genre",
            "genres",
            "classic",
            "classics",
            "theme",
            "themes",
            "literary analysis",
            "book review",
            "character analysis",
            "plot analysis",
            "recommend book",
            "suggest book",
            "book recommendation",
        ]

        explicit_matches = [term for term in explicit_book_terms if term in query_lower]
        if explicit_matches:
            logger.info(f"Explicit book terms found: {explicit_matches[:2]}")
            return {
                "is_book_related": True,
                "confidence": "high",
                "reason": "explicit_book_terms",
                "terms_found": explicit_matches[:2],
            }

        obvious_non_book_patterns = [
            "weather today",
            "weather forecast",
            "temperature today",
            "rain today",
            "what is the weather",
            "how hot is it",
            "will it rain",
            "how do i cook",
            "how to cook",
            "recipe for",
            "cooking instructions",
            "cook pasta",
            "make pasta",
            "prepare food",
            "what is 2+2",
            "what is 1+1",
            "2 plus 2",
            "1 plus 1",
            "calculate",
            "what is 3+3",
            "what is 4+4",
            "simple math",
            "math problem",
            "tell me about google",
            "about google",
            "google search",
            "google maps",
            "facebook profile",
            "youtube video",
            "how to use google",
            "news today",
            "breaking news",
            "current events",
            "latest news",
            "politics today",
            "stock market today",
            "election results",
            "medical advice",
            "doctor appointment",
            "legal advice",
            "tax help",
            "financial planning",
            "investment advice",
        ]

        for pattern in obvious_non_book_patterns:
            if pattern in query_lower:
                logger.info(f"Obvious non-book topic detected: {pattern}")
                return {
                    "is_book_related": False,
                    "confidence": "high",
                    "reason": "obvious_non_book_topic",
                    "evidence": pattern,
                }

        logger.info("Checking OpenLibrary API for book relevance...")
        try:
            books = self.openlibrary_api.search_books(user_query, limit=5)
            if books:
                book_relevance_score = self._analyze_book_relevance(user_query, books)

                if book_relevance_score["is_relevant"]:
                    logger.info(
                        f"OpenLibrary API confirmed book relevance: {book_relevance_score['reason']}"
                    )
                    return {
                        "is_book_related": True,
                        "confidence": book_relevance_score["confidence"],
                        "reason": "api_confirmed_books",
                        "analysis": book_relevance_score["reason"],
                        "books_found": len(books),
                    }
                else:
                    logger.info(
                        f"OpenLibrary results not genuinely book-related: {book_relevance_score['reason']}"
                    )

            if not books or len(books) == 0:
                key_terms = self._extract_key_terms(user_query)
                if key_terms != user_query and len(key_terms.strip()) > 0:
                    logger.info(
                        f"Trying search with extracted key terms: '{key_terms}'"
                    )
                    books = self.openlibrary_api.search_books(key_terms, limit=5)
                    if books:
                        book_relevance_score = self._analyze_book_relevance(
                            key_terms, books
                        )

                        if book_relevance_score["is_relevant"]:
                            logger.info(
                                f"OpenLibrary API confirmed book relevance with key terms: {book_relevance_score['reason']}"
                            )
                            return {
                                "is_book_related": True,
                                "confidence": book_relevance_score["confidence"],
                                "reason": "api_confirmed_books_key_terms",
                                "analysis": book_relevance_score["reason"],
                                "books_found": len(books),
                                "key_terms_used": key_terms,
                            }

        except Exception as e:
            logger.warning(f"OpenLibrary search failed: {str(e)}")

        if any(word in query_lower for word in ["recommend", "suggest", "suggestion"]):
            if not any(book_word in query_lower for book_word in explicit_book_terms):
                logger.info("Recommendation request without explicit book references")
                return {
                    "is_book_related": False,
                    "confidence": "high",
                    "reason": "recommendation_without_book_context",
                }

        famous_literary_works = [
            "ramayana",
            "mahabharata",
            "iliad",
            "odyssey",
            "aeneid",
            "beowulf",
            "hamlet",
            "macbeth",
            "othello",
            "romeo and juliet",
            "king lear",
            "pride and prejudice",
            "jane eyre",
            "wuthering heights",
            "emma",
            "great gatsby",
            "to kill a mockingbird",
            "moby dick",
            "war and peace",
            "anna karenina",
            "brothers karamazov",
            "don quixote",
            "divine comedy",
            "paradise lost",
            "canterbury tales",
            "sherlock holmes",
            "harry potter",
            "lord of the rings",
            "hobbit",
            "game of thrones",
            "dune",
        ]

        literary_matches = [
            work for work in famous_literary_works if work in query_lower
        ]
        if literary_matches:
            logger.info(f"Famous literary work detected: {literary_matches[0]}")

            key_terms = self._extract_key_terms(user_query)
            search_query = key_terms if key_terms != user_query else literary_matches[0]

            try:
                books = self.openlibrary_api.search_books(search_query, limit=3)
                if books and len(books) > 0:
                    logger.info(
                        f"Famous literary work confirmed with API: {literary_matches[0]}"
                    )
                    return {
                        "is_book_related": True,
                        "confidence": "high",
                        "reason": "famous_literary_work",
                        "evidence": literary_matches[0],
                        "books_found": len(books),
                        "search_terms": search_query,
                    }
            except Exception as e:
                logger.warning(f"Literary work verification failed: {str(e)}")

        has_context_hint = any(
            hint in query_lower
            for hint in ["about", "history of", "biography", "life of"]
        )

        if has_context_hint:
            logger.info("Query has contextual hints, checking OpenLibrary API...")
            try:
                books = self.openlibrary_api.search_books(user_query, limit=3)
                if books:
                    relevant_books = []
                    for book in books:
                        title = book.get("title", "").lower()

                        if any(
                            word in title
                            for word in user_query.lower().split()
                            if len(word) > 3
                        ):
                            relevant_books.append(book)

                    if relevant_books:
                        logger.info(
                            f"Found {len(relevant_books)} contextually relevant books"
                        )
                        return {
                            "is_book_related": True,
                            "confidence": "medium",
                            "reason": "contextual_api_match",
                            "relevant_books": len(relevant_books),
                        }
                        return {
                            "is_book_related": True,
                            "confidence": "medium",
                            "reason": "contextual_api_match",
                            "relevant_books": len(relevant_books),
                        }
            except Exception as e:
                logger.warning(f"OpenLibrary search failed: {str(e)}")

        obvious_non_book_topics = [
            "google",
            "apple",
            "microsoft",
            "facebook",
            "netflix",
            "spotify",
            "weather",
            "temperature",
            "climate",
            "current events",
            "news",
            "calculate",
            "equation",
            "formula",
            "what is 2 +",
            "math problem",
            "recipe",
            "cook pasta",
            "cooking ingredients",
            "meal prep",
            "football score",
            "basketball game",
            "soccer match",
            "sports news",
            "technical support",
            "troubleshoot",
            "install software",
        ]

        for topic in obvious_non_book_topics:
            if topic in query_lower:
                logger.info(f"Obvious non-book topic detected: {topic}")
                return {
                    "is_book_related": False,
                    "confidence": "high",
                    "reason": "obvious_non_book_topic",
                    "detected_topic": topic,
                }

        logger.info("No strong book indicators found")
        return {
            "is_book_related": False,
            "confidence": "high",
            "reason": "insufficient_book_indicators",
        }

    def _analyze_book_relevance(self, query: str, books: List[Dict]) -> Dict[str, Any]:
        logger.info(f"Analyzing book relevance for {len(books)} results")

        query_words = set(query.lower().split())

        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "tell",
            "me",
            "about",
            "what",
            "is",
            "who",
            "when",
            "where",
            "how",
            "why",
        }
        meaningful_query_words = query_words - stop_words

        relevant_books = 0
        total_relevance_score = 0
        evidence = []

        for book in books:
            title = book.get("title", "").lower()
            author_names = book.get("author_name", [])
            subjects = book.get("subject", [])

            book_relevance_score = 0
            book_evidence = []

            title_words = set(title.split())
            query_in_title = meaningful_query_words.intersection(title_words)
            if query_in_title:
                book_relevance_score += len(query_in_title) * 3
                book_evidence.append(f"Query words in title: {list(query_in_title)}")

            if author_names:
                author_text = " ".join(author_names).lower()
                author_words = set(author_text.split())
                query_in_author = meaningful_query_words.intersection(author_words)
                if query_in_author:
                    book_relevance_score += len(query_in_author) * 2
                    book_evidence.append(
                        f"Query words in author: {list(query_in_author)}"
                    )

            if subjects:
                book_subjects = [s.lower() for s in subjects]
                literary_subjects = [
                    s
                    for s in book_subjects
                    if any(
                        lit_term in s
                        for lit_term in [
                            "literature",
                            "poetry",
                            "epic",
                            "mythology",
                            "classic",
                            "fiction",
                            "drama",
                            "novel",
                        ]
                    )
                ]
                if literary_subjects:
                    book_relevance_score += len(literary_subjects)
                    book_evidence.append(f"Literary subjects: {literary_subjects[:2]}")

            biographical_indicators = ["biography", "life of", "story of", "history of"]
            if any(indicator in title for indicator in biographical_indicators):
                book_relevance_score += 2
                book_evidence.append("Biographical/historical work")

            if book_relevance_score >= 2:
                relevant_books += 1
                total_relevance_score += book_relevance_score
                evidence.extend(book_evidence[:2])

        avg_relevance = total_relevance_score / len(books) if books else 0
        relevance_ratio = relevant_books / len(books) if books else 0

        logger.info(
            f"Relevance analysis: {relevant_books}/{len(books)} books relevant, avg score: {avg_relevance:.1f}"
        )

        if relevance_ratio >= 0.6 and avg_relevance >= 2:
            return {
                "is_relevant": True,
                "confidence": "high",
                "reason": f"Strong book relevance: {relevant_books}/{len(books)} books match query context",
                "evidence": evidence[:3],
            }
        elif relevance_ratio >= 0.4 and avg_relevance >= 1.5:
            return {
                "is_relevant": True,
                "confidence": "medium",
                "reason": f"Moderate book relevance: {relevant_books}/{len(books)} books have contextual matches",
                "evidence": evidence[:2],
            }
        elif len(query.split()) == 1 and relevant_books >= 1:
            return {
                "is_relevant": True,
                "confidence": "medium",
                "reason": f"Single-word query with literary context: {relevant_books} relevant books found",
                "evidence": evidence[:2],
            }
        else:
            return {
                "is_relevant": False,
                "confidence": "high",
                "reason": f"Low book relevance: only {relevant_books}/{len(books)} books contextually relevant",
                "evidence": evidence[:1],
            }

    def _get_non_book_response(
        self, user_query: str, validation_result: Dict[str, Any]
    ) -> str:
        logger.info("Generating non-book response")

        query_lower = user_query.lower()

        if any(
            word in query_lower
            for word in [
                "google",
                "apple",
                "microsoft",
                "tech",
                "technology",
                "computer",
                "software",
            ]
        ):
            return """I'm BookChat AI, a specialized assistant focused exclusively on books, literature, and reading-related topics. 

While technology is fascinating, I can only help with book-related questions such as:
- Book recommendations and reviews
- Author information and literary analysis
- Reading suggestions based on your preferences
- Finding books by genre, theme, or topic
- Literary discussions and interpretations

Is there anything about books or reading I can help you with today?"""

        elif any(
            word in query_lower
            for word in ["what is", "who is", "when did", "where is", "how to"]
        ) and not any(
            book_word in query_lower
            for book_word in ["book", "author", "read", "write"]
        ):
            return """I'm BookChat AI, and I specialize exclusively in books and literature! 

While I'd love to help with general questions, I can only assist with book-related topics like:
- Book recommendations tailored to your interests
- Information about authors and their works
- Literary analysis and book discussions
- Help finding books on specific subjects
- Reading suggestions and book reviews

Do you have any questions about books, authors, or reading that I can help with?"""

        elif any(
            word in query_lower
            for word in [
                "movie",
                "film",
                "music",
                "song",
                "game",
                "tv",
                "show",
                "netflix",
            ]
        ):
            return """I'm BookChat AI, focused exclusively on the wonderful world of books and literature!

While movies, music, and shows are great entertainment, I can only help with book-related topics such as:
- Books that were adapted into movies or shows
- Reading recommendations in any genre
- Author biographies and literary works
- Finding books similar to stories you enjoyed
- Book club suggestions and literary discussions

Would you like book recommendations or have any literature questions I can help with?"""

        elif any(
            word in query_lower
            for word in [
                "how are you",
                "tell me about yourself",
                "what can you do",
                "help me with",
            ]
        ):
            return """Hello! I'm BookChat AI, your dedicated assistant for everything related to books and literature!

I specialize in helping with:
- Book Recommendations - Find your next great read
- Author Information - Learn about writers and their works
- Book Discovery - Search by genre, theme, or topic
- Literary Analysis - Discuss themes, characters, and plots
- Reading Guidance - Get suggestions based on your preferences
- Literary History - Explore classics and literary movements

What kind of books are you interested in, or is there something specific about literature you'd like to explore?"""

        else:
            return """I'm BookChat AI, a specialized assistant dedicated to books, literature, and reading!

I can only help with book-related questions such as:
- Book recommendations and reviews
- Author information and literary analysis
- Finding books by genre, topic, or theme
- Literary discussions and interpretations
- Reading suggestions based on your interests

If you have any questions about books, authors, or reading, I'd be delighted to help! What would you like to know about literature today?"""

    def get_response(
        self,
        user_query: str,
        book_context: List[Dict[str, Any]] = None,
        chat_mode: str = "General Book Discussion",
        use_lcel: bool = False,
    ) -> str:
        logger.info(
            f"Starting response generation for query: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'"
        )
        logger.info(f"Chat mode: {chat_mode}")
        logger.info(f"Book context: {len(book_context) if book_context else 0} books")

        validation_result = self._validate_book_relevance(user_query)

        if not validation_result["is_book_related"]:
            logger.info(f"Non-book query detected: {validation_result['reason']}")
            return self._get_non_book_response(user_query, validation_result)

        logger.info(
            f"Book-related query confirmed: {validation_result['reason']} (confidence: {validation_result['confidence']})"
        )

        start_time = time.time()

        if use_lcel and self.lcel_chains:
            logger.info("Using LCEL chains for response generation")
            try:
                chat_history = [msg for msg in self.memory.chat_memory.messages[-6:]]
                response_content = self.lcel_chains.run_chain(
                    query=user_query, chat_mode=chat_mode, chat_history=chat_history
                )

                self.memory.save_context(
                    {"input": user_query}, {"output": response_content}
                )
                self.summary_memory.save_context(
                    {"input": user_query}, {"output": response_content}
                )

                self.conversation_history.append(
                    {
                        "user": user_query,
                        "assistant": response_content,
                        "context": {
                            "book_context": book_context,
                            "chat_mode": chat_mode,
                            "method": "lcel_chains",
                        },
                        "metadata": {
                            "timestamp": time.time(),
                            "total_duration": time.time() - start_time,
                        },
                    }
                )

                total_duration = time.time() - start_time
                logger.info(
                    f"LCEL response generation completed successfully in {total_duration:.2f}s"
                )
                return response_content

            except Exception as e:
                logger.warning(
                    f"LCEL chains failed, falling back to traditional approach: {str(e)}"
                )

        try:
            logger.info("Step 1: Getting external book context from OpenLibrary API...")
            context_start = time.time()
            external_context = self._get_external_book_context(user_query)
            context_duration = time.time() - context_start
            logger.info(f"External context retrieved in {context_duration:.2f}s")
            logger.info(
                f"External context: {len(external_context.get('relevant_books', []))} books found"
            )

            logger.info("Step 2: Getting RAG context from knowledge base...")
            rag_start = time.time()
            rag_context = self.book_rag.get_relevant_context(user_query)
            rag_duration = time.time() - rag_start
            logger.info(f"RAG context retrieved in {rag_duration:.2f}s")
            logger.info(f"RAG context length: {len(rag_context)} characters")

            logger.info("Step 3: Building system prompt...")
            prompt_start = time.time()
            system_prompt = self._build_system_prompt(
                chat_mode=chat_mode,
                book_context=book_context,
                external_context=external_context,
                rag_context=rag_context,
            )

            system_prompt += """

IMPORTANT GUARDRAILS:
- If you don't have specific information about a book, author, or literary fact, clearly state your uncertainty
- Use phrases like "Based on my knowledge..." or "I'm not certain about specific details, but..."
- Don't invent publication dates, quotes, or biographical details
- Acknowledge the limits of your training data
- When providing book recommendations, base them on well-known works and established reception
"""
            prompt_duration = time.time() - prompt_start
            logger.info(
                f"System prompt built in {prompt_duration:.2f}s ({len(system_prompt)} chars)"
            )

            logger.info("Step 4: Building user prompt...")
            user_prompt = self._build_user_prompt(user_query, chat_mode)
            logger.info(f"User prompt built ({len(user_prompt)} chars)")

            logger.info("Step 5: Preparing messages for LLM...")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            chat_history = self.memory.chat_memory.messages
            if chat_history:
                logger.info(
                    f"Adding LangChain conversation history ({len(chat_history)} messages)"
                )

                for i, msg in enumerate(chat_history[-6:]):
                    messages.insert(1 + i, msg)

            if self.agent and self._should_use_agent(user_query):
                logger.info("Using Agent for tool-assisted response...")
                llm_start = time.time()

                try:
                    agent_input_str = user_query

                    if langchain_tracer:
                        agent_response = self.agent.run(
                            agent_input_str, callbacks=[langchain_tracer]
                        )
                    else:
                        agent_response = self.agent.run(agent_input_str)

                    agent_response = self._sanitize_agent_output(agent_response)

                    class AgentResponse:
                        def __init__(self, content):
                            self.content = content

                    response = AgentResponse(agent_response)
                    llm_duration = time.time() - llm_start
                    logger.info(
                        f"Agent call completed successfully in {llm_duration:.2f}s"
                    )

                except Exception as e:
                    logger.warning(
                        f"Agent failed, falling back to direct LLM: {str(e)}"
                    )
                    logger.exception("Agent execution error details:")

                    if langchain_tracer:
                        response = self._call_llm(
                            messages, callbacks=[langchain_tracer]
                        )
                    else:
                        response = self._call_llm(messages)
                    llm_duration = time.time() - llm_start
            else:
                logger.info("Calling LLM (Bedrock) directly...")
                llm_start = time.time()

                if langchain_tracer:
                    logger.info("📊 Using LangSmith tracing for LLM call")
                    response = self._call_llm(messages, callbacks=[langchain_tracer])
                else:
                    response = self._call_llm(messages)

                llm_duration = time.time() - llm_start
                logger.info(f"LLM call completed successfully in {llm_duration:.2f}s")
            logger.info(f"Response length: {len(response.content)} characters")
            logger.info(
                f"Response preview: {response.content[:200]}{'...' if len(response.content) > 200 else ''}"
            )

            logger.info("Storing conversation in LangChain memory...")
            self.memory.save_context(
                {"input": user_query}, {"output": response.content}
            )

            self.summary_memory.save_context(
                {"input": user_query}, {"output": response.content}
            )

            logger.info("Storing conversation metadata...")
            self.conversation_history.append(
                {
                    "user": user_query,
                    "assistant": response.content,
                    "context": {"book_context": book_context, "chat_mode": chat_mode},
                    "metadata": {
                        "timestamp": time.time(),
                        "external_context_books": len(
                            external_context.get("relevant_books", [])
                        ),
                        "rag_context_length": len(rag_context),
                        "llm_duration": llm_duration,
                        "total_duration": time.time() - start_time,
                    },
                }
            )

            if len(self.conversation_history) > 10:
                logger.info("Trimming conversation history to last 10 exchanges")
                self.conversation_history = self.conversation_history[-10:]

            logger.info(
                f"Memory status: {len(self.memory.chat_memory.messages)} messages in buffer"
            )

            total_duration = time.time() - start_time
            logger.info(
                f"Response generation completed successfully in {total_duration:.2f}s"
            )

            return response.content

        except Exception as e:
            total_duration = time.time() - start_time
            error_msg = f"Error generating response: {str(e)}"
            logger.error(f"{error_msg} (after {total_duration:.2f}s)")
            logger.exception("Response generation error details:")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."

    def _sanitize_agent_output(self, raw_output: str) -> str:
        try:
            if not raw_output:
                return raw_output

            cleaned = raw_output

            for marker in [
                "Thought:",
                "Thought",
                "Action:",
                "Action",
                "Observation:",
                "Observation",
            ]:
                cleaned = cleaned.replace(marker, "")

            footers = [
                "Knowledge Verification:",
                "Knowledge Verification",
                "This response is generated based on my training data and may contain inaccuracies.",
            ]
            for f in footers:
                cleaned = cleaned.replace(f, "")

            import re

            cleaned = re.sub(r"\n{2,}", "\n\n", cleaned).strip()

            return cleaned
        except Exception:
            return raw_output

    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert a list of SystemMessage/HumanMessage into a single text prompt for LLMs that
        expect a string prompt (used for local/testing fallbacks).
        """
        pieces = []
        for m in messages:
            try:
                role = getattr(m, "type", None) or m.__class__.__name__
                content = getattr(m, "content", str(m))
                pieces.append(f"[{role}] {content}")
            except Exception:
                pieces.append(str(m))
        return "\n\n".join(pieces)

    def _call_llm(self, messages: List[Any], callbacks: Optional[List[Any]] = None):
        """Call the configured LLM in a way that works both for chat-style LLMs (invoke/messages)
        and text-only LLMs (FakeListLLM). Returns an object with a 'content' attribute when possible,
        otherwise returns the raw string.
        """

        class DummyResp:
            def __init__(self, text):
                self.content = text

        try:
            if hasattr(self.llm, "invoke"):
                result = (
                    self.llm.invoke(messages, callbacks=callbacks)
                    if callbacks
                    else self.llm.invoke(messages)
                )
                if isinstance(result, str):
                    return DummyResp(result)
                if hasattr(result, "content"):
                    return result

                return DummyResp(str(result))
        except Exception:
            pass

        prompt = self._messages_to_prompt(messages)
        try:
            result = (
                self.llm(prompt, callbacks=callbacks) if callbacks else self.llm(prompt)
            )
            if isinstance(result, str):
                return DummyResp(result)
            if hasattr(result, "content"):
                return result
            return DummyResp(str(result))
        except Exception as e:
            logger.exception("Fallback LLM call failed")
            return DummyResp(f"[llm-error] {str(e)}")

    def _get_external_book_context(self, query: str) -> Dict[str, Any]:
        logger.info(
            f"Fetching external context for query: '{query[:50]}{'...' if len(query) > 50 else ''}'"
        )

        try:
            logger.info("Calling OpenLibrary API...")
            api_start = time.time()
            books = self.openlibrary_api.search_books(query, limit=5)
            api_duration = time.time() - api_start

            logger.info(
                f"OpenLibrary API returned {len(books) if books else 0} books in {api_duration:.2f}s"
            )

            if books:
                context = {"relevant_books": [], "authors": set(), "subjects": set()}

                for i, book in enumerate(books):
                    book_info = {
                        "title": book.get("title", ""),
                        "author": book.get("author_name", [""])[0]
                        if book.get("author_name")
                        else "",
                        "publish_year": book.get("first_publish_year", ""),
                        "subjects": book.get("subject", [])[:3]
                        if book.get("subject")
                        else [],
                    }
                    context["relevant_books"].append(book_info)
                    logger.info(
                        f"Book {i + 1}: '{book_info['title']}' by {book_info['author']} ({book_info['publish_year']})"
                    )

                    if book.get("author_name"):
                        context["authors"].update(book["author_name"][:2])
                    if book.get("subject"):
                        context["subjects"].update(book["subject"][:5])

                context["authors"] = list(context["authors"])
                context["subjects"] = list(context["subjects"])

                logger.info(
                    f"External context processed: {len(context['relevant_books'])} books, {len(context['authors'])} authors, {len(context['subjects'])} subjects"
                )
                return context
            else:
                logger.info("No books found in OpenLibrary API")

        except Exception as e:
            logger.warning(f"Error fetching external context: {str(e)}")
            logger.exception("External context fetch error details:")

        logger.info("Returning empty external context")
        return {"relevant_books": [], "authors": [], "subjects": []}

    def _build_system_prompt(
        self,
        chat_mode: str,
        book_context: List[Dict[str, Any]] = None,
        external_context: Dict[str, Any] = None,
        rag_context: str = "",
    ) -> str:
        template = self.prompt_templates.get_system_template(chat_mode)

        context_info = self.prompt_templates.get_context_template(
            book_context=book_context,
            rag_context=rag_context,
            external_context=external_context,
        )

        full_prompt = template
        if context_info and context_info.strip():
            full_prompt += f"\n\nCONTEXT INFORMATION:\n{context_info}"

        return full_prompt

    def _build_user_prompt(self, user_query: str, chat_mode: str) -> str:
        return self.prompt_templates.get_user_template(chat_mode, user_query)

    def _format_conversation_history(self) -> str:
        if not self.conversation_history:
            return ""

        formatted_history = ""

        recent_history = self.conversation_history[-3:]

        for i, exchange in enumerate(recent_history):
            formatted_history += f"User: {exchange['user']}\\n"
            formatted_history += f"Assistant: {exchange['assistant'][:200]}...\\n\\n"

        return formatted_history

    def _should_use_agent(self, query: str) -> bool:
        query_lower = query.lower()

        agent_triggers = [
            "search for",
            "find books",
            "look up",
            "recommend",
            "suggest",
            "what books",
            "book about",
            "author",
            "genre",
            "similar to",
            "like",
            "recommendations",
        ]

        for trigger in agent_triggers:
            if trigger in query_lower:
                logger.info(f"Agent trigger detected: '{trigger}' in query")
                return True

        question_patterns = ["what", "who", "where", "when", "how", "why", "which"]
        if any(pattern in query_lower for pattern in question_patterns):
            logger.info("Question detected - using agent for enhanced responses")
            return True

        return False

    def clear_history(self):
        self.conversation_history = []
        self.memory.clear()
        self.summary_memory.clear()
        logger.info("Conversation history cleared")

    def get_conversation_stats(self) -> Dict[str, Any]:
        return {
            "total_exchanges": len(self.conversation_history),
            "chat_modes_used": list(
                set(
                    [
                        exchange["context"]["chat_mode"]
                        for exchange in self.conversation_history
                    ]
                )
            ),
            "books_discussed": len(
                set(
                    [
                        book["title"]
                        for exchange in self.conversation_history
                        if exchange["context"].get("book_context")
                        for book in exchange["context"]["book_context"]
                    ]
                )
            ),
        }
