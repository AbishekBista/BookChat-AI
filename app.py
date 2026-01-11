import streamlit as st
import os
from dotenv import load_dotenv
from book_chat_system import BookChatSystem
from utils.openlibrary_api import OpenLibraryAPI
import logging
import time
import random
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


load_dotenv()

if os.getenv("LANGSMITH_TRACING_V2"):
    logger.info("LangSmith tracing enabled")
    logger.info(f"LangSmith project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
else:
    logger.info(
        "LangSmith tracing not enabled. Set LANGSMITH_TRACING_V2=true to enable."
    )


def is_book_related_query(query, book_chat_system=None, conversation_history=None):

    logger.info(
        f"Starting book-related query analysis for: '{query[:100]}{'...' if len(query) > 100 else ''}'"
    )
    start_time = time.time()

    if not book_chat_system:
        logger.warning("No BookChatSystem available, defaulting to not book-related")
        return False, "BookChatSystem not available"

    try:
        validation_result = book_chat_system._validate_book_relevance(query, conversation_history)

        is_book_related = validation_result.get("is_book_related", False)
        reason = validation_result.get("reason", "unknown")
        confidence = validation_result.get("confidence", "unknown")

        total_duration = time.time() - start_time
        logger.info(f"Total query analysis completed in {total_duration:.2f}s")

        if is_book_related:
            logger.info(
                f"Query validated as book-related: {reason} (confidence: {confidence})"
            )
            return True, f"{reason} (confidence: {confidence})"
        else:
            logger.info(
                f"Query validated as non-book-related: {reason} (confidence: {confidence})"
            )
            return False, f"{reason} (confidence: {confidence})"

    except Exception as e:
        logger.warning(f"Intelligent validation failed: {str(e)}")
        logger.exception("Validation error details:")
        return False, f"Validation error: {str(e)}"


def get_non_book_response():
    responses = [
        """I'm BookChat AI, your specialized literary companion!

I focus exclusively on books, literature, authors, and reading-related topics. I can help you with:

• **Book recommendations** based on your preferences

• **Literary analysis** of plots, themes, and characters 

• **Author information** and biographies

• **Reading suggestions** for different genres

• **Book discussions** and interpretations

• **Writing styles** and literary techniques

Feel free to ask me anything about novels, poetry, short stories, classics, or contemporary literature. What literary adventure would you like to explore today?""",
        """Hello! I'm BookChat AI, designed to be your dedicated literary assistant.

I specialize in all things related to books and literature, including:

• **Book discovery** - Find your next great read

• **Literary discussions** - Analyze themes, characters, and plots

• **Author insights** - Learn about writers and their works

• **Genre exploration** - Discover new literary territories

• **Reading guidance** - Get personalized recommendations

• **Book comparisons** - Compare similar works and authors

I'd love to help you dive into the wonderful world of books and literature! What bookish topic interests you today?""",
        """I'm BookChat AI, your personal guide to the world of literature!

My expertise covers everything book-related:

• **Personalized recommendations** tailored to your taste

• **In-depth literary analysis** of classics and contemporary works

• **Author profiles** and literary biographies

• **Genre guides** for fiction, non-fiction, poetry, and more

• **Reading strategies** and book discussion tips

• **Literary history** and cultural context

Whether you're looking for your next favorite book, want to discuss a story you've read, or explore literary themes, I'm here to help! What literary question can I assist you with?""",
    ]

    return random.choice(responses)


def format_response(response, references=None, book_context=None, is_book_query=True):
    if not response:
        return response

    if not is_book_query:
        return response

    response = re.sub(r"\.(?=[A-Z])", ". ", response)

    response = response.replace("\n\n", "\n\n")

    response = re.sub(r'"([^"]*)"', r'**"\1"**', response)

    response = response

    return response.strip()


st.set_page_config(
    page_title="BookChat AI - Your Literary Companion",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    logger.info("Initializing session state...")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized empty messages list")

    if "book_chat_system" not in st.session_state:
        logger.info("Initializing BookChatSystem...")
        try:
            st.session_state.book_chat_system = BookChatSystem()
            logger.info("BookChatSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chat system: {str(e)}")
            logger.exception("BookChatSystem initialization error details:")
            st.error(f"Error initializing chat system: {str(e)}")
            st.session_state.book_chat_system = None

    if "selected_books" not in st.session_state:
        st.session_state.selected_books = []
        logger.info("Initialized empty selected_books list")

    if "previous_book_context" not in st.session_state:
        st.session_state.previous_book_context = []
        logger.info("Initialized previous_book_context tracker")

    if "previous_chat_mode" not in st.session_state:
        st.session_state.previous_chat_mode = None
        logger.info("Initialized previous_chat_mode tracker")

    logger.info("Session state initialization completed")


def display_chat_message(message, is_user=True):
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)


def main():
    initialize_session_state()

    with st.sidebar:
        st.subheader("Chat Mode")

        chat_mode = st.selectbox(
            "Select mode:",
            [
                "General Discussion",
                "Recommendations",
                "Literary Analysis",
                "Author Info",
            ],
            help="Choose how you want to interact with BookChat AI",
        )

        mode_mapping = {
            "General Discussion": "General Book Discussion",
            "Recommendations": "Book Recommendation",
            "Literary Analysis": "Literary Analysis",
            "Author Info": "Author Biography",
        }
        current_chat_mode = mode_mapping[chat_mode]
        
        # Check if chat mode changed and clear memory/messages if so
        if st.session_state.previous_chat_mode is not None and current_chat_mode != st.session_state.previous_chat_mode:
            logger.info(f"Chat mode changed from '{st.session_state.previous_chat_mode}' to '{current_chat_mode}'")
            if st.session_state.book_chat_system:
                st.session_state.book_chat_system.clear_conversation_memory()
            st.session_state.messages = []
            logger.info("Cleared memory and messages due to chat mode change")
        
        st.session_state.chat_mode = current_chat_mode
        st.session_state.previous_chat_mode = current_chat_mode

        st.divider()

        st.subheader("Book Context")

        with st.form("book_search_form", clear_on_submit=False, border=False):
            book_query = st.text_input(
                "Search books:", placeholder="Harry Potter...", key="book_search_input"
            )
            search_submitted = st.form_submit_button(
                "Search", icon=":material/search:", type="primary", width="stretch"
            )

        if search_submitted and book_query.strip():
            with st.spinner("Searching..."):
                try:
                    openlibrary = OpenLibraryAPI()
                    books = openlibrary.search_books(book_query.strip(), limit=3)
                    st.session_state.search_results = books
                    st.session_state.last_search_query = book_query.strip()
                    if books:
                        st.success(f"Found {len(books)} books!")
                    else:
                        st.warning("No books found. Try a different search term.")
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
        elif search_submitted and not book_query.strip():
            if "search_results" in st.session_state:
                del st.session_state.search_results
            if "last_search_query" in st.session_state:
                del st.session_state.last_search_query

        if (
            hasattr(st.session_state, "search_results")
            and st.session_state.search_results
        ):
            st.write("**Search Results:**")
            for i, book in enumerate(st.session_state.search_results):
                title = book.get("title", "Unknown")
                author = (
                    book.get("author_name", ["Unknown"])[0]
                    if book.get("author_name")
                    else "Unknown"
                )

                display_title = title if len(title) <= 35 else title[:32] + "..."
                display_author = author if len(author) <= 25 else author[:22] + "..."

                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{display_title}**")
                        st.caption(f"by {display_author}")
                    with col2:
                        if st.button(
                            "",
                            key=f"add_book_{i}",
                            help="Add to context",
                            icon=":material/add_circle:",
                        ):
                            book_already_selected = any(
                                selected_book.get("key") == book.get("key")
                                for selected_book in st.session_state.selected_books
                            )
                            if not book_already_selected:
                                st.session_state.selected_books.append(book)
                                st.rerun()

        if st.session_state.selected_books:
            st.write("**Books in Context:**")

            for i, book in enumerate(st.session_state.selected_books):
                title = book.get("title", "Unknown")
                author = (
                    book.get("author_name", ["Unknown"])[0]
                    if book.get("author_name")
                    else "Unknown"
                )

                display_title = title if len(title) <= 20 else title[:17] + "..."

                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f0f2f6;
                            border: 1px solid #d4d4d8;
                            border-radius: 15px;
                            padding: 8px 12px;
                            margin: 2px 0;
                        ">
                            <small><strong>{display_title}</strong><br>
                            <em>by {author[:15] + "..." if len(author) > 15 else author}</em></small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button(
                        "",
                        key=f"remove_book_{i}",
                        help=f"Remove '{title}' from context",
                        icon=":material/remove_circle:",
                    ):
                        st.session_state.selected_books.pop(i)
                        st.rerun()

            st.caption(f"{len(st.session_state.selected_books)} book(s) in context")

    # Check for book context changes and clear memory/messages immediately
    current_book_keys = [
        f"{book.get('title', '')}__{book.get('author_name', [''])[0] if book.get('author_name') else ''}"
        for book in st.session_state.selected_books
    ]
    
    if current_book_keys != st.session_state.previous_book_context:
        logger.info(
            f"Book context changed from {len(st.session_state.previous_book_context)} to {len(current_book_keys)} books"
        )
        logger.info(f"Previous context: {st.session_state.previous_book_context}")
        logger.info(f"Current context: {current_book_keys}")
        
        if st.session_state.book_chat_system:
            st.session_state.book_chat_system.clear_conversation_memory()
        st.session_state.messages = []
        logger.info("Cleared memory and messages due to book context change")
        
        st.session_state.previous_book_context = current_book_keys.copy()

    if not st.session_state.messages:
        st.markdown("### Welcome to BookChat AI!")
        st.markdown(
            "*Your intelligent companion for exploring the world of books and literature*"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**I can help you with:**")
            st.markdown("- Book recommendations")
            st.markdown("- Literary analysis")
            st.markdown("- Author biographies")

        with col2:
            st.markdown("**And much more:**")
            st.markdown("- Plot summaries")
            st.markdown("- Book comparisons")
            st.markdown("- Reading insights")

        st.info(
            "**Tip:** Select a chat mode in the sidebar and start asking questions below!"
        )

    else:
        st.markdown("### Conversation")

        messages_container = st.container()
        with messages_container:
            for message in st.session_state.messages:
                display_chat_message(message["content"], message["role"] == "user")

    st.divider()

    if prompt := st.chat_input("Ask me anything about books...", key="main_chat_input"):
        aws_configured = all(
            [
                os.getenv("AWS_ACCESS_KEY_ID"),
                os.getenv("AWS_SECRET_ACCESS_KEY"),
                os.getenv("AWS_REGION"),
            ]
        )

        if not aws_configured:
            st.error(
                "Please configure your AWS credentials in the .env file to start chatting!"
            )
            return

        if not st.session_state.book_chat_system:
            st.error(
                "Chat system not initialized. Please check your AWS configuration."
            )
            return

        st.session_state.messages.append({"role": "user", "content": prompt})

        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_message = st.session_state.messages[-1]["content"]

        with st.spinner("Analyzing your query..."):
            try:
                logger.info(
                    f"Processing user message: '{latest_message[:100]}{'...' if len(latest_message) > 100 else ''}'"
                )

                logger.info("Initializing tools for query analysis...")

                logger.info("Starting book-related query analysis...")

                is_book_query, analysis_reason = is_book_related_query(
                    latest_message, 
                    book_chat_system=st.session_state.book_chat_system,
                    conversation_history=st.session_state.messages
                )

                logger.info(
                    f"Query analysis result - Is book-related: {is_book_query}, Reason: {analysis_reason}"
                )

                if not is_book_query:
                    logger.info(
                        "Query classified as non-book-related, returning redirect response"
                    )
                    non_book_response = get_non_book_response()
                    st.session_state.messages.append(
                        {"role": "assistant", "content": non_book_response}
                    )
                    logger.info("Non-book response added to chat history")

                else:
                    logger.info(
                        "Query classified as book-related, processing with BookChatSystem..."
                    )
                    logger.info(
                        f"Preparing context from {len(st.session_state.selected_books)} selected books..."
                    )
                    book_context = []

                    for i, book in enumerate(st.session_state.selected_books):
                        book_info = {
                            "title": book.get("title", ""),
                            "author": book.get("author_name", [""])[0]
                            if book.get("author_name")
                            else "",
                            "publish_year": book.get("first_publish_year", ""),
                            "subjects": book.get("subject", [])[:5]
                            if book.get("subject")
                            else [],
                        }
                        book_context.append(book_info)
                        logger.info(
                            f"Book {i + 1}: '{book_info['title']}' by {book_info['author']}"
                        )

                    logger.info("Calling BookChatSystem.get_response()...")
                    chat_mode = st.session_state.get(
                        "chat_mode", "General Book Discussion"
                    )
                    logger.info(f"Chat mode: {chat_mode}")

                    response = st.session_state.book_chat_system.get_response(
                        latest_message,
                        book_context=book_context,
                        chat_mode=chat_mode,
                    )

                    logger.info(
                        f"BookChatSystem response received ({len(response)} chars)"
                    )
                    logger.info(
                        f"Response preview: {response[:150]}{'...' if len(response) > 150 else ''}"
                    )

                    logger.info("Formatting response...")
                    formatted_response = format_response(
                        response, None, book_context, is_book_query=True
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": formatted_response}
                    )
                    logger.info("Assistant response added to chat history")

                logger.info("Rerunning Streamlit to display response")
                st.rerun()

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(f"{error_msg}")
                logger.exception("Chat processing error details:")
                st.error(error_msg)


if __name__ == "__main__":
    main()
