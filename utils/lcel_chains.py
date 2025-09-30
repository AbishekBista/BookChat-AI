import logging
from typing import Dict, List, Any
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws import ChatBedrock
from utils.book_rag import BookRAG
from utils.openlibrary_api import OpenLibraryAPI
from utils.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class BookChatChains:
    def __init__(
        self, llm: ChatBedrock, book_rag: BookRAG, openlibrary_api: OpenLibraryAPI
    ):
        self.llm = llm
        self.book_rag = book_rag
        self.openlibrary_api = openlibrary_api
        self.prompt_templates = PromptTemplates()

        self._setup_chains()
        logger.info("LCEL chains initialized")

    def _setup_chains(self):
        self.basic_chat_chain = self._create_basic_chat_chain()

        self.rag_chain = self._create_rag_chain()

        self.book_search_chain = self._create_book_search_chain()

        self.recommendation_chain = self._create_recommendation_chain()

        logger.info("All LCEL chains created")

    def _create_basic_chat_chain(self) -> Runnable:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are BookChat AI, a knowledgeable assistant specializing in books, literature, and reading. 
            Provide helpful, accurate, and engaging responses about books, authors, literary analysis, and reading recommendations.
            
            Context from conversation:
            {context}""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(lambda x: ""),
                "chat_history": RunnableLambda(lambda x: []),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_rag_chain(self) -> Runnable:
        def get_rag_context(query: str) -> str:
            try:
                context = self.book_rag.get_relevant_context(query, k=5)
                return context if context else "No relevant context found."
            except Exception as e:
                logger.error(f"RAG context retrieval failed: {str(e)}")
                return "Context retrieval failed."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are BookChat AI with access to a comprehensive book knowledge base. 
            Use the provided context to give accurate, detailed responses about books and literature.
            
            Relevant Context:
            {context}
            
            Instructions:
            - Base your response on the provided context when relevant
            - If context doesn't fully answer the question, use your general knowledge but indicate uncertainty
            - Always provide helpful and engaging responses about books and literature""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(get_rag_context),
                "chat_history": RunnableLambda(lambda x: []),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_book_search_chain(self) -> Runnable:
        def search_books(query: str) -> str:
            try:
                books = self.openlibrary_api.search_books(query, limit=5)
                if not books:
                    return f"No books found for '{query}'"

                result = f"Found {len(books)} books for '{query}':\n\n"
                for i, book in enumerate(books, 1):
                    title = book.get("title", "Unknown")
                    authors = book.get("author_name", [])
                    year = book.get("first_publish_year")

                    result += f"{i}. **{title}**\n"
                    if authors:
                        result += f"   Authors: {', '.join(authors[:2])}\n"
                    if year:
                        result += f"   Published: {year}\n"
                    result += "\n"

                return result
            except Exception as e:
                logger.error(f"Book search failed: {str(e)}")
                return f"Book search failed: {str(e)}"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are BookChat AI with book search capabilities. 
            You have access to book search results from OpenLibrary. 
            Use this information to provide helpful responses about books.
            
            Search Results:
            {search_results}
            
            Provide detailed information about the books found and help the user with their book-related query.""",
                ),
                ("human", "{input}"),
            ]
        )

        chain = (
            {
                "input": RunnablePassthrough(),
                "search_results": RunnableLambda(search_books),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _create_recommendation_chain(self) -> Runnable:
        def get_recommendations(query: str) -> Dict[str, str]:
            try:
                rag_context = self.book_rag.get_relevant_context(
                    f"recommendations {query}", k=3
                )

                books = self.openlibrary_api.search_books(query, limit=5)

                search_results = ""
                if books:
                    search_results = (
                        f"Found {len(books)} books related to '{query}':\n\n"
                    )
                    for i, book in enumerate(books, 1):
                        title = book.get("title", "Unknown")
                        authors = book.get("author_name", [])
                        year = book.get("first_publish_year")
                        subjects = book.get("subject", [])

                        search_results += f"{i}. **{title}**\n"
                        if authors:
                            search_results += f"   by {', '.join(authors[:2])}\n"
                        if year:
                            search_results += f"   Published: {year}\n"
                        if subjects:
                            search_results += f"   Topics: {', '.join(subjects[:3])}\n"
                        search_results += "\n"

                return {
                    "rag_context": rag_context
                    if rag_context
                    else "No specific context available.",
                    "search_results": search_results
                    if books
                    else f"No books found for '{query}'",
                }
            except Exception as e:
                logger.error(f"Recommendation gathering failed: {str(e)}")
                return {
                    "rag_context": "Context retrieval failed.",
                    "search_results": f"Book search failed: {str(e)}",
                }

        def format_recommendations(data: Dict[str, Any]) -> str:
            query = data.get("input", "")
            rec_data = get_recommendations(query)

            formatted = (
                f"Literary Context for '{query}':\n{rec_data['rag_context']}\n\n"
            )
            formatted += f"Book Search Results:\n{rec_data['search_results']}"

            return formatted

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are BookChat AI, a book recommendation specialist. 
            You have access to literary knowledge and book search results.
            Provide personalized book recommendations based on the user's preferences.
            
            Available Information:
            {recommendation_data}
            
            Instructions:
            - Use both the literary context and search results to make informed recommendations
            - Explain why you're recommending specific books
            - Consider the user's preferences and reading level
            - Provide a diverse selection when possible
            - Include brief descriptions of recommended books""",
                ),
                ("human", "Please recommend books for: {input}"),
            ]
        )

        chain = (
            {
                "input": RunnablePassthrough(),
                "recommendation_data": RunnableLambda(format_recommendations),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def get_chain_for_query(
        self, query: str, chat_mode: str = "General Book Discussion"
    ) -> Runnable:
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["recommend", "suggest", "should i read", "what to read"]
        ):
            logger.info("Using recommendation chain")
            return self.recommendation_chain
        elif any(
            word in query_lower
            for word in ["search", "find books", "look for", "book about"]
        ):
            logger.info("Using book search chain")
            return self.book_search_chain
        elif any(
            word in query_lower
            for word in ["analyze", "analysis", "theme", "character", "plot", "genre"]
        ):
            logger.info("Using RAG chain for literary analysis")
            return self.rag_chain
        else:
            logger.info("Using basic chat chain")
            return self.basic_chat_chain

    def run_chain(
        self,
        query: str,
        chat_mode: str = "General Book Discussion",
        chat_history: List = None,
    ) -> str:
        try:
            chain = self.get_chain_for_query(query, chat_mode)

            chain_input = query
            if hasattr(chain, "input_schema"):
                chain_input = {"input": query, "chat_history": chat_history or []}

            logger.info(
                f"Running LCEL chain for query: '{query[:50]}{'...' if len(query) > 50 else ''}'"
            )

            result = chain.invoke(chain_input)
            logger.info("LCEL chain execution completed")

            return result

        except Exception as e:
            logger.error(f"LCEL chain execution failed: {str(e)}")

            return "I apologize, but I encountered an error processing your request about books. Please try rephrasing your question."
