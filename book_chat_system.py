import os
from dotenv import load_dotenv
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
import boto3
import logging
import time
from langchain.callbacks.tracers import LangChainTracer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


langchain_tracer = None
if os.getenv("LANGSMITH_TRACING_V2") == "true":
    try:
        langchain_tracer = LangChainTracer(
            project_name=os.getenv("LANGSMITH_PROJECT", "bookchat-ai")
        )
        logger.info("LangSmith tracing initialized successfully")
        logger.info(
            f"LangSmith project: {os.getenv('LANGSMITH_PROJECT', 'bookchat-ai')}"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith tracer: {str(e)}")
        langchain_tracer = None
else:
    logger.info(
        "LangSmith tracing not enabled. Set LANGSMITH_TRACING_V2=true to enable."
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
            # Custom agent prefix to prevent exposing internal tool names
            agent_prefix = """You are BookChat AI, a knowledgeable literary assistant helping users with book-related inquiries.

CRITICAL INSTRUCTION: NEVER reveal the names or technical details of your internal tools, systems, or implementation to users. Do not mention tool names like 'book_search', 'book_knowledge', 'book_recommendations' or any other system internals.

When helping users, respond naturally as a helpful assistant would, without explaining HOW you're obtaining the information. Simply provide the information they need in a conversational, helpful manner.

Instead of saying "Use this tool..." or "I'll use the book_search tool...", simply help the user directly with their request.

You have access to various capabilities to help with:
- Finding and searching for books
- Providing literary knowledge and analysis
- Offering personalized book recommendations

Respond to user queries naturally and helpfully without exposing your internal workings."""

            agent_suffix = """Begin! Remember to NEVER mention tool names or internal system details to the user.

Question: {input}
{agent_scratchpad}"""

            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate",
                return_intermediate_steps=False,
                agent_kwargs={
                    "prefix": agent_prefix,
                    "suffix": agent_suffix,
                },
            )
            logger.info("LangChain Agent initialized successfully")
        except Exception as e:
            logger.warning(
                f"Agent initialization failed, falling back to basic chat: {str(e)}"
            )
            logger.exception("Agent initialization error details:")
            self.agent = None

        self.conversation_history = []
        logger.info("Enhanced conversation memory initialized")
        init_duration = time.time() - start_time
        logger.info(f"BookChatSystem initialized successfully in {init_duration:.2f}s")

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


    def _validate_book_relevance(self, user_query: str, conversation_history: list = None) -> Dict[str, Any]:
        """
        Use LLM to validate if a query is related to books, literature, or reading.
        Returns a dict with is_book_related, confidence, and reason.
        """
        logger.info(
            f"Validating book relevance with LLM for query: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'"
        )

        try:
            # Build conversation context if available
            context_str = ""
            if conversation_history and len(conversation_history) > 0:
                logger.info(f"Including conversation context: {len(conversation_history)} messages")
                context_str = "\n\nRecent conversation context:\n"
                # Get last 3 message pairs (10 messages) for context
                recent_messages = conversation_history[-10:]
                for msg in recent_messages:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get("content", "")[:200]  # Limit length
                    context_str += f"{role}: {content}\n"
                context_str += "\nIMPORTANT: If the current query refers to something mentioned in the conversation context (e.g., 'he', 'she', 'it', 'that book', 'the author'), consider it book-related if the context was about books/authors.\n"
            else:
                logger.info("No conversation context available for validation")

            validation_prompt = f"""You are a classification assistant for a book-focused chatbot called BookChat AI.

Your task is to determine if the user's query is related to books, literature, reading, or writing.{context_str}

Book-related topics include:
- Book recommendations and suggestions
- Questions about specific books, novels, stories, or literary works
- Author information and biographies
- Literary analysis (themes, characters, plot, writing style)
- Reading recommendations and lists
- Book reviews and critiques
- Poetry and poems
- Publishing, editions, and book formats
- Libraries, bookstores, and reading culture
- Writing techniques and creative writing
- Literary genres and movements
- Famous literary works and classics

Non-book-related topics include:
- Weather, cooking, math, technology companies
- Current events, news, politics (unless about books/authors)
- Medical, legal, or financial advice
- Sports, entertainment (unless about books/movies based on books)
- General knowledge questions unrelated to literature

Current user query: "{user_query}"

Respond with ONLY a JSON object in this exact format:
{{
  "is_book_related": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "brief explanation in 5-10 words"
}}

Do not include any other text or explanation outside the JSON object."""

            # Use the LLM to validate
            messages = [
                SystemMessage(content="You are a precise classification assistant that responds only with valid JSON."),
                HumanMessage(content=validation_prompt)
            ]
            
            start_time = time.time()
            response = self.llm.invoke(messages)
            duration = time.time() - start_time
            
            logger.info(f"LLM validation completed in {duration:.2f}s")
            
            # Parse the response
            response_text = response.content.strip()
            logger.info(f"LLM response: {response_text[:200]}")
            
            # Try to extract JSON from response
            import json
            import re
            
            # Look for JSON object in the response
            json_match = re.search(r'\{[^{}]*"is_book_related"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                # Validate the response has required fields
                if "is_book_related" in result and "confidence" in result and "reason" in result:
                    logger.info(
                        f"LLM classification: is_book_related={result['is_book_related']}, "
                        f"confidence={result['confidence']}, reason={result['reason']}"
                    )
                    return result
                else:
                    logger.warning("LLM response missing required fields")
            
            # If parsing failed, fall back to simple keyword check
            logger.warning("Could not parse LLM response, falling back to keyword check")
            return self._fallback_keyword_validation(user_query)
            
        except Exception as e:
            logger.error(f"LLM validation failed: {str(e)}")
            logger.exception("LLM validation error details:")
            # Fall back to simple keyword check
            return self._fallback_keyword_validation(user_query)
    
    def _fallback_keyword_validation(self, user_query: str) -> Dict[str, Any]:
        """Simple keyword-based fallback validation."""
        logger.info("Using fallback keyword validation")
        
        query_lower = user_query.lower().strip()
        
        # Quick book-related keyword check
        book_keywords = [
            "book", "novel", "author", "read", "literature", "story", "write",
            "poem", "poetry", "fiction", "character", "plot", "genre", "classic",
            "library", "publish", "chapter", "theme"
        ]
        
        # Quick non-book keyword check
        non_book_keywords = [
            "weather", "recipe", "cook", "calculate", "news", "stock", "medical",
            "google", "facebook", "sports", "temperature", "election"
        ]
        
        has_book_keyword = any(keyword in query_lower for keyword in book_keywords)
        has_non_book_keyword = any(keyword in query_lower for keyword in non_book_keywords)
        
        if has_book_keyword and not has_non_book_keyword:
            return {
                "is_book_related": True,
                "confidence": "medium",
                "reason": "contains book-related keywords"
            }
        elif has_non_book_keyword:
            return {
                "is_book_related": False,
                "confidence": "medium",
                "reason": "contains non-book topic keywords"
            }
        else:
            # Default to not book-related if uncertain
            return {
                "is_book_related": False,
                "confidence": "low",
                "reason": "no clear book indicators found"
            }

    def get_response(
        self,
        user_query: str,
        book_context: List[Dict[str, Any]] = None,
        chat_mode: str = "General Book Discussion",
    ) -> str:
        logger.info(
            f"Starting response generation for query: '{user_query[:100]}{'...' if len(user_query) > 100 else ''}'"
        )
        logger.info(f"Chat mode: {chat_mode}")
        logger.info(f"Book context: {len(book_context) if book_context else 0} books")

        # Note: Query validation is already done in app.py before calling this method
        # No need to re-validate here as non-book queries are filtered out earlier

        start_time = time.time()

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

CRITICAL: NEVER REVEAL INTERNAL SYSTEM DETAILS
- Do NOT mention tool names like 'book_search', 'book_knowledge', 'book_recommendations' or any internal system components
- Do NOT explain your internal process or how you retrieve information
- Do NOT say things like "Use this tool..." or "Here's how I can help you with my tools..."
- Simply provide helpful, natural responses as a knowledgeable literary assistant would
- Act as if you inherently know how to help, without explaining the mechanics
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

                for i, msg in enumerate(chat_history[-10:]):
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
                    logger.info("Using LangSmith tracing for LLM call")
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
        """Remove any internal agent artifacts and tool names from output."""
        try:
            if not raw_output:
                return raw_output

            cleaned = raw_output

            # Remove ReAct framework artifacts
            for marker in [
                "Thought:",
                "Thought",
                "Action:",
                "Action",
                "Observation:",
                "Observation",
                "Action Input:",
                "Action Input",
                "Final Answer:",
            ]:
                cleaned = cleaned.replace(marker, "")

            # Remove tool name references (case insensitive)
            import re
            tool_patterns = [
                r'book_search',
                r'book_knowledge',
                r'book_recommendations',
                r'Use this tool',
                r'use the .* tool',
                r'I\'ll use',
                r'tool if you need',
                r'Use .* tool when',
            ]
            for pattern in tool_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

            # Remove knowledge verification footers
            footers = [
                "Knowledge Verification:",
                "Knowledge Verification",
                "This response is generated based on my training data and may contain inaccuracies.",
            ]
            for f in footers:
                cleaned = cleaned.replace(f, "")

            # Clean up excessive newlines
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

            return cleaned
        except Exception as e:
            logger.warning(f"Error sanitizing agent output: {str(e)}")
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
                    self.llm.invoke(messages, config={"callbacks": callbacks})
                    if callbacks
                    else self.llm.invoke(messages)
                )
                if isinstance(result, str):
                    return DummyResp(result)
                if hasattr(result, "content"):
                    return result

                return DummyResp(str(result))
        except Exception as e:
            logger.warning(f"invoke() failed: {str(e)}, trying fallback")

        # Fallback for simpler LLMs
        prompt = self._messages_to_prompt(messages)
        try:
            # Don't use callbacks with direct call, only with invoke
            result = self.llm(prompt)
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
