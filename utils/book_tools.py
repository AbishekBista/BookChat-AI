import os
import logging
from typing import List, Optional, Type, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from utils.openlibrary_api import OpenLibraryAPI
from utils.book_rag import BookRAG

logger = logging.getLogger(__name__)


class BookInfo(BaseModel):
    title: str = Field(description="The title of the book")
    authors: List[str] = Field(default=[], description="List of book authors")
    publish_year: Optional[int] = Field(
        default=None, description="Year the book was published"
    )
    isbn: Optional[List[str]] = Field(default=None, description="List of ISBN numbers")
    subjects: Optional[List[str]] = Field(
        default=None, description="List of book subjects/topics"
    )
    pages: Optional[int] = Field(default=None, description="Number of pages")
    description: Optional[str] = Field(
        default=None, description="Book description or summary"
    )


class BookSearchResult(BaseModel):
    books: List[BookInfo] = Field(description="List of books found")
    total_found: int = Field(description="Total number of books found")
    query: str = Field(description="Original search query")


class KnowledgeDocument(BaseModel):
    content: str = Field(description="The document content")
    relevance_score: Optional[float] = Field(
        default=None, description="Relevance score if available"
    )
    source: Optional[str] = Field(default=None, description="Source of the document")


class BookKnowledgeResult(BaseModel):
    documents: List[KnowledgeDocument] = Field(
        default=[], description="Retrieved knowledge documents"
    )
    total_documents: int = Field(description="Total number of documents retrieved")
    query: str = Field(description="Original knowledge query")
    summary: Optional[str] = Field(
        default=None, description="AI-generated summary of the knowledge"
    )


class BookRecommendation(BaseModel):
    title: str = Field(description="Recommended book title")
    authors: List[str] = Field(default=[], description="Book authors")
    publish_year: Optional[int] = Field(default=None, description="Publication year")
    reason: str = Field(description="Reason for recommendation")
    subjects: Optional[List[str]] = Field(
        default=None, description="Book subjects/genres"
    )
    relevance_score: Optional[float] = Field(
        default=None, description="How relevant to user's query"
    )


class BookRecommendationResult(BaseModel):
    recommendations: List[BookRecommendation] = Field(
        default=[], description="List of book recommendations"
    )
    total_recommendations: int = Field(description="Total number of recommendations")
    query: str = Field(description="Original recommendation query")
    context: Optional[str] = Field(
        default=None, description="Additional context or explanation"
    )


class ToolError(BaseModel):
    error_type: str = Field(description="Type of error that occurred")
    error_message: str = Field(description="Human-readable error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    query: Optional[str] = Field(
        default=None, description="Original query that caused the error"
    )


class BookSearchInput(BaseModel):
    query: str = Field(
        description="Search query for books. Can include title, author, keywords, or search options like 'limit:10' to specify maximum results"
    )


class BookRAGInput(BaseModel):
    query: str = Field(
        description="Query to search in book knowledge base. Can include search options like 'k:5' to specify number of documents"
    )


class BookRecommendationInput(BaseModel):
    preferences: str = Field(
        description="User preferences, genres, or liked books/authors. Can include options like 'limit:5' to specify maximum recommendations"
    )


class BookSearchTool(BaseTool):
    name: str = "book_search"
    description: str = """
    Search for books using the OpenLibrary API. 
    Use this tool when you need to find books by title, author, or topic.
    Returns structured information about books including title, authors, publication year, and subjects.
    """
    args_schema: Type[BaseModel] = BookSearchInput

    def __init__(self):
        super().__init__()

        logger.info("BookSearchTool initialized")

    def _get_api(self) -> OpenLibraryAPI:
        return OpenLibraryAPI()

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        limit = 5
        actual_query = query

        if "limit:" in query.lower():
            parts = query.split("limit:")
            if len(parts) == 2:
                try:
                    limit_part = parts[1].strip().split()[0]
                    limit = int(limit_part)
                    actual_query = parts[0].strip()
                except (ValueError, IndexError):
                    pass

        logger.info(f"BookSearchTool: Searching for '{actual_query}' (limit: {limit})")

        try:
            openlibrary_api = self._get_api()
            books_data = openlibrary_api.search_books(actual_query, limit=limit)

            if not books_data:
                logger.info("BookSearchTool: No books found")
                error_result = ToolError(
                    error_type="NO_RESULTS",
                    error_message=f"No books found for query: '{actual_query}'",
                    query=actual_query,
                )
                return error_result.model_dump_json(indent=2)

            books = []
            for book_data in books_data:
                # Validate and clean author information
                author_names = book_data.get("author_name", [])
                if not author_names or (isinstance(author_names, list) and len(author_names) == 0):
                    author_names = ["Author information not available"]
                
                book = BookInfo(
                    title=book_data.get("title", "Unknown"),
                    authors=author_names,
                    publish_year=book_data.get("first_publish_year"),
                    isbn=book_data.get("isbn", []),
                    subjects=book_data.get("subject", [])[:5]
                    if book_data.get("subject")
                    else [],
                    pages=book_data.get("number_of_pages_median"),
                    description=None,
                )
                books.append(book)
                logger.info(f"BookSearchTool: Added '{book.title}' by {', '.join(book.authors[:2])}")

            result = BookSearchResult(
                books=books, total_found=len(books), query=actual_query
            )

            logger.info(f"BookSearchTool: Found {len(books)} books")
            return result.model_dump_json(indent=2)

        except Exception as e:
            error_msg = f"Error searching for books: {str(e)}"
            logger.error(f"BookSearchTool: {error_msg}")
            error_result = ToolError(
                error_type="API_ERROR",
                error_message=error_msg,
                details=str(e),
                query=actual_query,
            )
            return error_result.model_dump_json(indent=2)


class BookKnowledgeTool(BaseTool):
    name: str = "book_knowledge"
    description: str = """
    Retrieve relevant literary knowledge and context from the book knowledge base.
    Use this tool when you need information about literature, literary analysis, 
    reading recommendations, or general book-related topics.
    """
    args_schema: Type[BaseModel] = BookRAGInput

    def __init__(self):
        super().__init__()

        logger.info("BookKnowledgeTool initialized")

    def _get_rag(self) -> BookRAG:
        return BookRAG()

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        k = 5
        actual_query = query

        if "k:" in query.lower():
            parts = query.split("k:")
            if len(parts) == 2:
                try:
                    k_part = parts[1].strip().split()[0]
                    k = int(k_part)
                    actual_query = parts[0].strip()
                except (ValueError, IndexError):
                    pass

        logger.info(
            f"BookKnowledgeTool: Searching knowledge base for '{actual_query}' (k={k})"
        )

        try:
            book_rag = self._get_rag()
            context = book_rag.get_relevant_context(actual_query, k=k)

            if not context or len(context.strip()) < 10:
                logger.info("BookKnowledgeTool: No relevant knowledge found")
                error_result = ToolError(
                    error_type="NO_KNOWLEDGE",
                    error_message=f"No relevant literary knowledge found for query: '{actual_query}'",
                    query=actual_query,
                )
                return error_result.model_dump_json(indent=2)

            documents = []
            if context:
                sections = [
                    section.strip()
                    for section in context.split("\n\n")
                    if section.strip()
                ]
                for i, section in enumerate(sections[:k]):
                    doc = KnowledgeDocument(
                        content=section,
                        relevance_score=1.0 - (i * 0.1),
                        source="BookRAG_Knowledge_Base",
                    )
                    documents.append(doc)

            summary = (
                f"Retrieved {len(documents)} relevant knowledge sections about {actual_query}. "
                f"Synthesize key themes, representative examples, and suggested search queries or related titles that would help find similar works."
            )
            if documents:
                summary += f" Key topics covered include: {actual_query}"

            result = BookKnowledgeResult(
                documents=documents,
                total_documents=len(documents),
                query=actual_query,
                summary=summary,
            )

            logger.info(
                f"BookKnowledgeTool: Retrieved {len(documents)} knowledge sections"
            )
            return result.model_dump_json(indent=2)

        except Exception as e:
            error_msg = f"Error retrieving book knowledge: {str(e)}"
            logger.error(f"BookKnowledgeTool: {error_msg}")
            error_result = ToolError(
                error_type="RAG_ERROR",
                error_message=error_msg,
                details=str(e),
                query=actual_query,
            )
            return error_result.model_dump_json(indent=2)


class BookRecommendationTool(BaseTool):
    name: str = "book_recommendations"
    description: str = """
    Generate personalized book recommendations based on user preferences, genres, 
    authors, or previously enjoyed books. This tool is especially good at handling 
    plot-based descriptions and thematic queries. Use this tool when users ask for 
    book suggestions or recommendations.
    """
    args_schema: Type[BaseModel] = BookRecommendationInput

    def __init__(self):
        super().__init__()

        logger.info("BookRecommendationTool initialized")

    def _get_api(self) -> OpenLibraryAPI:
        return OpenLibraryAPI()

    def _get_rag(self) -> BookRAG:
        return BookRAG()

    def _get_llm(self):
        from langchain_aws import ChatBedrock
        import boto3

        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        )

        return ChatBedrock(
            client=bedrock_client,
            model_id=os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0"),
            model_kwargs={"temperature": 0.3, "max_tokens": 1000},
        )

    def _extract_themes_and_keywords(self, plot_description: str) -> Dict[str, Any]:
        logger.info(
            f"Extracting themes from plot description: '{plot_description[:100]}...'"
        )

        try:
            llm = self._get_llm()

            analysis_prompt = f"""
You are a literary expert. Analyze this plot description and extract key elements for book recommendations:

Plot Description: "{plot_description}"

Please provide a structured analysis in the following format:

THEMES:
- List 3-5 main themes (e.g., coming of age, social class, inheritance, identity)

KEYWORDS:
- List 5-10 specific keywords for book searches (e.g., poor to rich, inheritance, discovery, transformation)

SIMILAR_BOOKS:
- List 3-5 well-known books with similar plots or themes
- Include author names

SEARCH_TERMS:
- Provide 2-3 optimized search terms for finding similar books

Provide a detailed, actionable analysis suitable for guiding searches and recommendations. Aim for about 150-350 words and include short examples or suggested search queries.
"""

            from langchain.schema import HumanMessage

            response = llm([HumanMessage(content=analysis_prompt)])

            analysis_text = response.content
            logger.info(f"LLM analysis completed: {len(analysis_text)} characters")

            themes = []
            keywords = []
            similar_books = []
            search_terms = []

            current_section = None
            for line in analysis_text.split("\n"):
                line = line.strip()
                if line.upper().startswith("THEMES:"):
                    current_section = "themes"
                elif line.upper().startswith("KEYWORDS:"):
                    current_section = "keywords"
                elif line.upper().startswith("SIMILAR_BOOKS:"):
                    current_section = "similar_books"
                elif line.upper().startswith("SEARCH_TERMS:"):
                    current_section = "search_terms"
                elif line.startswith("-") and current_section:
                    item = line[1:].strip()
                    if current_section == "themes":
                        themes.append(item)
                    elif current_section == "keywords":
                        keywords.append(item)
                    elif current_section == "similar_books":
                        similar_books.append(item)
                    elif current_section == "search_terms":
                        search_terms.append(item)

            return {
                "themes": themes[:5],
                "keywords": keywords[:10],
                "similar_books": similar_books[:5],
                "search_terms": search_terms[:3],
                "raw_analysis": analysis_text,
            }

        except Exception as e:
            logger.warning(f"Theme extraction failed: {str(e)}")

            return self._extract_themes_manually(plot_description)

    def _search_with_multiple_strategies(
        self, theme_analysis: Dict[str, Any], limit: int
    ) -> List[Dict]:
        logger.info("Searching books with multiple strategies")

        all_books = []
        search_strategies = []

        if theme_analysis.get("similar_books"):
            for book in theme_analysis["similar_books"][:3]:
                search_strategies.append(("similar_book", book))

        if theme_analysis.get("search_terms"):
            for term in theme_analysis["search_terms"][:2]:
                search_strategies.append(("search_term", term))

        if theme_analysis.get("themes"):
            for theme in theme_analysis["themes"][:2]:
                search_strategies.append(("theme", theme))

        if theme_analysis.get("keywords"):
            keyword_query = " ".join(theme_analysis["keywords"][:5])
            search_strategies.append(("keywords", keyword_query))

        openlibrary_api = self._get_api()

        for strategy_type, query in search_strategies:
            try:
                logger.info(
                    f"Strategy '{strategy_type}': searching for '{query[:50]}...'"
                )
                books = openlibrary_api.search_books(query, limit=3)
                if books:
                    for book in books:
                        book["search_strategy"] = strategy_type
                        book["search_query"] = query
                    all_books.extend(books)
                    logger.info(
                        f"Found {len(books)} books with strategy '{strategy_type}'"
                    )
            except Exception as e:
                logger.warning(f"Search strategy '{strategy_type}' failed: {str(e)}")
                continue

        unique_books = []
        seen_titles = set()

        for book in all_books:
            title = book.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_books.append(book)
                if len(unique_books) >= limit * 2:
                    break

        logger.info(f"Found {len(unique_books)} unique books after deduplication")
        return unique_books[: limit * 2]

    def _run(
        self, preferences: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        limit = 5
        include_context = True
        actual_preferences = preferences

        if "limit:" in preferences.lower():
            parts = preferences.split("limit:")
            if len(parts) == 2:
                try:
                    limit_part = parts[1].strip().split()[0]
                    limit = int(limit_part)
                    actual_preferences = parts[0].strip()
                except (ValueError, IndexError):
                    pass

        if "context:false" in preferences.lower():
            include_context = False
            actual_preferences = actual_preferences.replace("context:false", "").strip()

        logger.info(
            f"BookRecommendationTool: Generating intelligent recommendations for '{actual_preferences}' (limit: {limit})"
        )

        try:
            is_plot_description = self._is_plot_description(actual_preferences)
            logger.info(f"Plot description detected: {is_plot_description}")

            if is_plot_description:
                logger.info("Using enhanced plot-based recommendation strategy")

                theme_analysis = self._extract_themes_and_keywords(actual_preferences)
                logger.info(f"Extracted themes: {theme_analysis.get('themes', [])[:3]}")
                logger.info(
                    f"Generated search terms: {theme_analysis.get('search_terms', [])}"
                )

                books_data = self._search_with_multiple_strategies(
                    theme_analysis, limit
                )

                book_rag = self._get_rag()
                knowledge_context = ""
                if include_context and theme_analysis.get("themes"):
                    theme_query = " ".join(theme_analysis["themes"][:3])
                    knowledge_context = book_rag.get_relevant_context(
                        f"books about {theme_query}", k=3
                    )

            else:
                logger.info(
                    "Using traditional preference-based recommendation strategy"
                )

                book_rag = self._get_rag()
                knowledge_context = ""
                if include_context:
                    knowledge_context = book_rag.get_relevant_context(
                        f"recommendations {actual_preferences}", k=3
                    )

                openlibrary_api = self._get_api()
                books_data = openlibrary_api.search_books(
                    actual_preferences, limit=limit * 2
                )
                theme_analysis = None

            if not books_data:
                if knowledge_context:
                    error_result = ToolError(
                        error_type="LIMITED_RESULTS",
                        error_message=f"No specific books found, but found relevant literary knowledge about '{actual_preferences}'",
                        details=knowledge_context[:500],
                        query=actual_preferences,
                    )
                    return error_result.model_dump_json(indent=2)
                else:
                    error_result = ToolError(
                        error_type="NO_RECOMMENDATIONS",
                        error_message=f"No book recommendations found for '{actual_preferences}'",
                        details="Try describing the plot, themes, or mood you're looking for in more detail",
                        query=actual_preferences,
                    )
                    return error_result.model_dump_json(indent=2)

            recommendations = []
            for i, book_data in enumerate(books_data[:limit]):
                reason = self._generate_recommendation_reason(
                    book_data, actual_preferences, theme_analysis, is_plot_description
                )

                subjects = book_data.get("subject", [])[:3]
                
                # Ensure we always have author information, even if empty
                author_names = book_data.get("author_name", [])
                if not author_names:
                    author_names = ["Author information not available"]
                    logger.warning(f"No author information found for book: {book_data.get('title', 'Unknown')}")
                
                # Validate and clean author names
                cleaned_authors = []
                for author in author_names:
                    if author and isinstance(author, str) and len(author.strip()) > 0:
                        cleaned_authors.append(author.strip())
                
                if not cleaned_authors:
                    cleaned_authors = ["Author information not available"]
                    logger.warning(f"Author names invalid after cleaning for: {book_data.get('title', 'Unknown')}")

                recommendation = BookRecommendation(
                    title=book_data.get("title", "Unknown Title"),
                    authors=cleaned_authors,
                    publish_year=book_data.get("first_publish_year"),
                    reason=reason,
                    subjects=subjects,
                    relevance_score=1.0 - (i * 0.15),
                )
                recommendations.append(recommendation)
                logger.info(f"Recommendation {i+1}: '{recommendation.title}' by {', '.join(cleaned_authors[:2])}")

            context = None
            if (
                include_context
                and knowledge_context
                and len(knowledge_context.strip()) > 50
            ):
                context_prefix = (
                    "Based on your plot description: "
                    if is_plot_description
                    else "Based on your preferences: "
                )
                context = context_prefix + (
                    knowledge_context[:400] + "..."
                    if len(knowledge_context) > 400
                    else knowledge_context
                )

            # Add explicit instruction in the result to use exact author names
            if context:
                context = context + "\n\nIMPORTANT: Use the EXACT author names provided above. Do not invent or modify author information."
            else:
                context = "IMPORTANT: Use the EXACT author names provided in the recommendations above. If author information is missing, explicitly state 'Author information not available' - do not guess or invent author names."
            
            result = BookRecommendationResult(
                recommendations=recommendations,
                total_recommendations=len(recommendations),
                query=actual_preferences,
                context=context,
            )

            logger.info(
                f"BookRecommendationTool: Generated {len(recommendations)} intelligent recommendations"
            )
            return result.model_dump_json(indent=2)

        except Exception as e:
            error_msg = f"Error generating recommendations: {str(e)}"
            logger.error(f"BookRecommendationTool: {error_msg}")
            logger.exception("Recommendation error details:")
            error_result = ToolError(
                error_type="RECOMMENDATION_ERROR",
                error_message=error_msg,
                details=str(e),
                query=actual_preferences,
            )
            return error_result.model_dump_json(indent=2)

    def _is_plot_description(self, preferences: str) -> bool:
        plot_indicators = [
            "boy who",
            "girl who",
            "person who",
            "character who",
            "story about",
            "discovers",
            "finds out",
            "learns that",
            "realizes",
            "grows up",
            "poor",
            "rich",
            "wealthy",
            "inheritance",
            "letter",
            "secret",
            "journey",
            "adventure",
            "quest",
            "transformation",
            "changes",
            "family",
            "parents",
            "orphan",
            "adopted",
            "hidden",
            "mystery",
            "love story",
            "romance",
            "friendship",
            "betrayal",
            "revenge",
            "coming of age",
            "young adult",
            "teenager",
            "child",
        ]

        preferences_lower = preferences.lower()
        plot_score = sum(
            1 for indicator in plot_indicators if indicator in preferences_lower
        )

        has_story_structure = (
            (
                "who" in preferences_lower
                and ("discovers" in preferences_lower or "finds" in preferences_lower)
            )
            or ("about" in preferences_lower and len(preferences.split()) > 5)
            or ("story" in preferences_lower)
            or ("plot" in preferences_lower)
        )

        return plot_score >= 2 or has_story_structure

    def _extract_themes_manually(self, plot_description: str) -> Dict[str, Any]:
        logger.info("Using manual theme extraction as fallback")

        plot_lower = plot_description.lower()

        theme_mappings = {
            "coming of age": [
                "boy who",
                "girl who",
                "young person",
                "teenager",
                "child",
                "grows up",
            ],
            "social class": ["poor", "rich", "wealthy", "poverty", "money", "fortune"],
            "inheritance": [
                "inherits",
                "inheritance",
                "letter",
                "will",
                "estate",
                "legacy",
            ],
            "mystery": [
                "mysterious",
                "secret",
                "hidden",
                "discovers",
                "uncovers",
                "finds out",
            ],
            "family": [
                "family",
                "parents",
                "father",
                "mother",
                "siblings",
                "relatives",
            ],
            "transformation": [
                "discovers",
                "becomes",
                "transforms",
                "changes",
                "realizes",
            ],
            "adventure": ["journey", "quest", "adventure", "travels", "explore"],
            "magic": ["magical", "powers", "magic", "supernatural", "fantasy"],
            "identity": ["who they are", "identity", "true self", "real parents"],
        }

        themes = []
        for theme, indicators in theme_mappings.items():
            if any(indicator in plot_lower for indicator in indicators):
                themes.append(theme)

        search_terms = []

        if "poor" in plot_lower and ("rich" in plot_lower or "wealthy" in plot_lower):
            search_terms.extend(["rags to riches", "poor boy rich", "social mobility"])
            themes.append("rags to riches")

        if "magical" in plot_lower and "powers" in plot_lower:
            search_terms.extend(
                ["young adult fantasy", "magical powers", "coming of age fantasy"]
            )

        if "inherits" in plot_lower and "house" in plot_lower:
            search_terms.extend(
                ["inherited house mystery", "family secrets", "gothic mystery"]
            )

        if "time travel" in plot_lower:
            search_terms.extend(
                ["time travel fiction", "alternate history", "temporal adventure"]
            )

        words = plot_description.split()
        stop_words = {
            "a",
            "an",
            "the",
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
        }
        keywords = [
            word.strip(".,!?")
            for word in words
            if len(word) > 2 and word.lower() not in stop_words
        ]

        similar_books = []
        if "rags to riches" in themes or (
            "poor" in plot_lower and "rich" in plot_lower
        ):
            similar_books.extend(
                [
                    "Great Expectations by Charles Dickens",
                    "The Prince and the Pauper by Mark Twain",
                ]
            )

        if "magical powers" in search_terms or "magic" in themes:
            similar_books.extend(
                ["Harry Potter by J.K. Rowling", "Percy Jackson by Rick Riordan"]
            )

        if "inherited house mystery" in search_terms or "mystery" in themes:
            similar_books.extend(
                [
                    "Rebecca by Daphne du Maurier",
                    "The House of Seven Gables by Nathaniel Hawthorne",
                ]
            )

        if "time travel" in search_terms:
            similar_books.extend(
                [
                    "The Time Machine by H.G. Wells",
                    "A Wrinkle in Time by Madeleine L'Engle",
                ]
            )

        if not search_terms:
            if len(keywords) >= 3:
                search_terms.append(" ".join(keywords[:3]))
            search_terms.append(
                " ".join(keywords[:5]) if len(keywords) >= 5 else plot_description[:50]
            )

        logger.info(f"Manual extraction found themes: {themes[:3]}")
        logger.info(f"Generated search terms: {search_terms[:3]}")

        return {
            "themes": themes[:5],
            "keywords": keywords[:10],
            "similar_books": similar_books[:5],
            "search_terms": search_terms[:3]
            if search_terms
            else [plot_description[:50]],
            "raw_analysis": "Manual analysis of plot description",
        }

    def _generate_recommendation_reason(
        self,
        book_data: Dict,
        original_query: str,
        theme_analysis: Optional[Dict],
        is_plot_description: bool,
    ) -> str:
        subjects = book_data.get("subject", [])[:3]
        search_strategy = book_data.get("search_strategy", "general_search")

        if is_plot_description and theme_analysis:
            if search_strategy == "similar_book":
                reason = (
                    "This book has a similar plot and themes to what you described. "
                )
            elif search_strategy == "theme":
                reason = (
                    "This book explores themes that align with your plot description: "
                )
            else:
                reason = "Based on your plot description, this book shares similar narrative elements: "

            if theme_analysis.get("themes"):
                key_themes = ", ".join(theme_analysis["themes"][:2])
                reason += f"{key_themes}. "

            if subjects:
                reason += f"Literary subjects include: {', '.join(subjects)}."
        else:
            reason = f"Matches your interest in {original_query}. "
            if subjects:
                reason += f"Topics include: {', '.join(subjects)}."

        return reason


def get_book_tools() -> List[BaseTool]:
    tools = [BookSearchTool(), BookKnowledgeTool(), BookRecommendationTool()]
    logger.info(f"Initialized {len(tools)} book tools")
    return tools
