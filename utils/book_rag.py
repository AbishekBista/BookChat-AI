import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
import boto3
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BookRAG:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\\n\\n", "\\n", ".", "!", "?", ",", " ", ""],
        )

        self._initialize_embeddings()
        self._initialize_vectorstore()
        self._populate_default_knowledge()
        self._initialize_retrievers()

    def _initialize_embeddings(self):
        try:
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            )

            self.embeddings = BedrockEmbeddings(
                client=bedrock_client,
                model_id=os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
            )
            logger.info("Bedrock embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Bedrock embeddings: {str(e)}")
            raise

    def _initialize_vectorstore(self):
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                logger.info(
                    f"Loaded existing vectorstore from {self.persist_directory}"
                )
            else:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
                logger.info(f"Created new vectorstore at {self.persist_directory}")

        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise

    def _populate_default_knowledge(self):
        try:
            if self.vectorstore._collection.count() > 0:
                logger.info(
                    "Vectorstore already contains documents, skipping default population"
                )
                return

            default_knowledge = [
                {
                    "content": "Genres overview: Literary genres include fiction, non-fiction, poetry, drama, and hybrid forms. Within fiction there are many subgenres such as mystery, romance, science fiction, fantasy, historical fiction, literary fiction, horror, thriller, and speculative fiction. Nonfiction subgenres include biography, memoir, history, science writing, travel writing, essays, and self-help. Understanding genre conventions helps set reader expectations and frames analysis.",
                    "metadata": {"category": "genres", "type": "overview"},
                },
                {
                    "content": "Genre subtypes and hallmarks: For example, mystery often focuses on a central puzzle and an investigation; romance centers emotional relationships and an arc toward intimacy; science fiction explores speculative technologies and social implications; historical fiction reconstructs past eras while blending fact and imagined detail. Each subtype has common tropes and narrative patterns.",
                    "metadata": {"category": "genres", "type": "subtypes"},
                },
                {
                    "content": "Classic literature and why it matters: Classics are works that remain influential across generations, often studied for their craft, cultural impact, and thematic depth. Representative authors include Shakespeare, Jane Austen, Charles Dickens, Tolstoy, Homer, and more. Classics are useful teaching texts because they invite repeated interpretation and contextual study.",
                    "metadata": {"category": "classics", "type": "definition"},
                },
                {
                    "content": "Literary movements timeline: Key movements include Romanticism (late 18th–19th c.), Realism and Naturalism (19th c.), Modernism (late 19th–early 20th c.), Postmodernism (mid–late 20th c.), and Contemporary/Global Literature. Each movement reflects historical, philosophical, and aesthetic priorities of its time.",
                    "metadata": {"category": "movements", "type": "timeline"},
                },
                {
                    "content": "Narrative structures and plot architectures: Common plot models include linear plots, episodic plots, framed narratives, circular narratives, and the three-act structure. Plot devices include exposition, rising action, climax, falling action, and resolution. Recognizing structure improves summary and analysis.",
                    "metadata": {"category": "plot", "type": "structures"},
                },
                {
                    "content": "Character and characterization: Characters can be dynamic (change over time) or static, round (complex) or flat (one-dimensional). Study of motives, backstory, arc, dialogue, and actions reveals theme and moral perspective. Archetypes (hero, mentor, trickster, shadow) are common cross-cultural patterns.",
                    "metadata": {"category": "characters", "type": "archetypes"},
                },
                {
                    "content": "Narrative voice and point of view: First-person, second-person, and third-person (limited, omniscient) narration shape reliability and intimacy. Unreliable narrators, free indirect discourse, and focalization are important concepts when assessing perspective.",
                    "metadata": {"category": "techniques", "type": "narration"},
                },
                {
                    "content": "Literary devices and figurative language: Common devices include metaphor, simile, symbolism, imagery, irony, foreshadowing, allegory, motif, and personification. Devices operate at the sentence and paragraph level to produce tone and layered meaning.",
                    "metadata": {"category": "devices", "type": "reference"},
                },
                {
                    "content": "Themes and thematic analysis: Typical themes across literature include identity, power, freedom, alienation, love, mortality, social justice, and memory. Thematic analysis connects plot and character detail to larger philosophical or cultural questions.",
                    "metadata": {"category": "themes", "type": "analysis"},
                },
                {
                    "content": "Critical approaches and schools: Formalism (close reading, text-focused), New Criticism, Structuralism, Marxist criticism, Feminist criticism, Postcolonial criticism, Reader-Response theory, Psychoanalytic criticism, and Ecocriticism. Each approach asks different questions and highlights different evidence.",
                    "metadata": {"category": "criticism", "type": "approaches"},
                },
                {
                    "content": "Reading and study strategies: Active reading techniques include annotating, summarizing paragraphs, tracking character lists, making margin notes of questions, mapping timelines, and creating chapter summaries. Techniques for teaching include close-reading exercises, Socratic questioning, and comparative reading.",
                    "metadata": {"category": "reading_skills", "type": "techniques"},
                },
                {
                    "content": "Annotation and note-taking best practices: Use consistent symbols, record page and line numbers for quotes, paraphrase complex passages, note patterns and motifs, and distinguish between observation and interpretation. Maintain a reading log for themes and recurring language.",
                    "metadata": {"category": "reading_skills", "type": "annotation"},
                },
                {
                    "content": "Book club facilitation and discussion prompts: Prepare open-ended questions about character motives, turning points, symbolism, and the author's choices. Use prompts like 'What would you change?' or 'How does the setting shape character decisions?' to encourage dialogue.",
                    "metadata": {"category": "book_clubs", "type": "discussion"},
                },
                {
                    "content": "Recommendation methodology: Good recommendation practices combine explicit user preferences (genres, authors, length), mood and context, reading history, and risk tolerance. Use representative sampling (similar authors), Explainable reasons (why a book fits), and tiered suggestions (easy, deep, experimental).",
                    "metadata": {"category": "recommendations", "type": "strategies"},
                },
                {
                    "content": "Famous authors and representative works (annotated): Examples include Jane Austen (Pride and Prejudice - social comedy of manners), George Orwell (1984 - dystopian political allegory), Toni Morrison (Beloved - memory and trauma, African American history), Gabriel García Márquez (One Hundred Years of Solitude - magical realism), and Chimamanda Ngozi Adichie (Half of a Yellow Sun - postcolonial history).",
                    "metadata": {"category": "authors", "type": "representative_works"},
                },
                {
                    "content": "Publishing basics and book metadata: Key bibliographic fields include title, subtitle, author(s), editor(s), publisher, publication date, ISBN, edition, language, page count, format (hardcover, paperback, ebook, audiobook), and identifiers. Accurate metadata supports discovery and citation.",
                    "metadata": {"category": "publishing", "type": "metadata"},
                },
                {
                    "content": "ISBNs, editions, and rights: ISBNs uniquely identify a specific edition/format. Different editions (revisions, new forewords, translations) may change pagination and content. Public domain status depends on country-specific copyright term (author's death + years).",
                    "metadata": {"category": "publishing", "type": "isbn_editions"},
                },
                {
                    "content": "Accessibility and formats: Books are available in physical print, EPUB/MOBI ebooks, PDF, and audiobooks. Consider accessibility features (read-aloud, large print, alt text for images) and DRM/licensing when distributing digital content.",
                    "metadata": {"category": "formats", "type": "accessibility"},
                },
                {
                    "content": "Citation and referencing quick guide: Common citation styles include MLA, APA, and Chicago. For a book: MLA typically uses Author. Title. Publisher, Year. Include edition and translator if relevant. For academic work, provide page numbers for direct quotes.",
                    "metadata": {"category": "reference", "type": "citation"},
                },
                {
                    "content": "Practical teaching resources: Lesson plan ideas include thematic units, author studies, comparative genre projects, and scaffolded close-reading activities. Use formative assessments like reading journals and short analytic essays to measure comprehension.",
                    "metadata": {"category": "education", "type": "teaching_resources"},
                },
                {
                    "content": "Research and bibliography best practices: When researching literature, consult primary texts, peer-reviewed criticism, authoritative editions, and bibliographies. Use academic databases and keep full citation metadata for every source.",
                    "metadata": {"category": "research", "type": "best_practices"},
                },
                {
                    "content": "Reading benefits and cognitive effects: Regular reading improves vocabulary, empathy, concentration, and critical thinking. Fiction in particular has been linked to theory of mind improvements and emotional intelligence increases.",
                    "metadata": {"category": "benefits", "type": "cognitive"},
                },
                {
                    "content": "How to use this knowledge base: Combine these seed documents with book-specific ingestion (PDFs, EPUBs, summaries) to enable both broad literary context and precise textual retrieval. Prefer RAG retrieval for book-specific queries and use the seed knowledge for genre-level, pedagogical, and bibliographic answers.",
                    "metadata": {"category": "meta", "type": "usage"},
                },
            ]

            documents = []
            for item in default_knowledge:
                doc = Document(page_content=item["content"], metadata=item["metadata"])
                documents.append(doc)

            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()

            logger.info(
                f"Added {len(documents)} default knowledge documents to vectorstore"
            )

        except Exception as e:
            logger.error(f"Error populating default knowledge: {str(e)}")

    def _initialize_retrievers(self):
        try:
            if not self.vectorstore:
                logger.warning(
                    "⚠️ Cannot initialize retrievers - vectorstore not available"
                )
                return

            self.similarity_retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            self.mmr_retriever = self.vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
            )

            self.threshold_retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 5},
            )

            logger.info("Multiple retrieval strategies initialized")

        except Exception as e:
            logger.error(f"Error initializing retrievers: {str(e)}")
            self.similarity_retriever = None
            self.mmr_retriever = None
            self.threshold_retriever = None

    def get_retriever(self, strategy: str = "similarity") -> Optional[BaseRetriever]:
        retrievers = {
            "similarity": getattr(self, "similarity_retriever", None),
            "mmr": getattr(self, "mmr_retriever", None),
            "threshold": getattr(self, "threshold_retriever", None),
        }

        retriever = retrievers.get(strategy)
        if retriever is None:
            logger.warning(
                f"⚠️ Retriever strategy '{strategy}' not available, falling back to similarity"
            )
            return retrievers.get("similarity")

        return retriever

    def add_book_knowledge(self, book_info: Dict[str, Any], book_content: str = None):
        try:
            documents = []

            book_summary = f"Title: {book_info.get('title', 'Unknown')}\\n"
            book_summary += f"Author: {book_info.get('author', 'Unknown')}\\n"

            if book_info.get("publish_year"):
                book_summary += f"Published: {book_info['publish_year']}\\n"

            if book_info.get("subjects"):
                book_summary += f"Subjects: {', '.join(book_info['subjects'][:5])}\\n"

            if book_info.get("description"):
                book_summary += f"Description: {book_info['description']}\\n"

            doc = Document(
                page_content=book_summary,
                metadata={
                    "title": book_info.get("title", "Unknown"),
                    "author": book_info.get("author", "Unknown"),
                    "category": "book_info",
                    "type": "metadata",
                },
            )
            documents.append(doc)

            if book_content:
                content_chunks = self.text_splitter.split_text(book_content)
                for i, chunk in enumerate(content_chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "title": book_info.get("title", "Unknown"),
                            "author": book_info.get("author", "Unknown"),
                            "category": "book_content",
                            "type": "text",
                            "chunk_id": i,
                        },
                    )
                    documents.append(doc)

            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()

            logger.info(
                f"Added {len(documents)} documents for book: {book_info.get('title', 'Unknown')}"
            )

        except Exception as e:
            logger.error(f"Error adding book knowledge: {str(e)}")

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        logger.info(
            f"RAG: Searching for relevant context for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )
        logger.info(f"RAG: Requesting {k} most relevant documents")

        start_time = time.time()

        try:
            if not self.vectorstore:
                logger.warning("⚠️ RAG: Vectorstore not initialized")
                return ""

            logger.info("RAG: Performing similarity search...")
            search_start = time.time()
            relevant_docs = self.vectorstore.similarity_search(query, k=k)
            search_duration = time.time() - search_start

            logger.info(f"RAG: Similarity search completed in {search_duration:.2f}s")
            logger.info(f"RAG: Found {len(relevant_docs)} relevant documents")

            if not relevant_docs:
                logger.info("RAG: No relevant documents found")
                return ""

            logger.info("RAG: Formatting context from documents...")
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                content = doc.page_content.strip()
                metadata = doc.metadata

                logger.info(
                    f"RAG: Doc {i + 1}: {len(content)} chars, category: {metadata.get('category', 'N/A')}"
                )

                context_part = f"[Context {i + 1}]\\n"
                if metadata.get("title") and metadata.get("author"):
                    context_part += (
                        f"Source: {metadata['title']} by {metadata['author']}\\n"
                    )
                    logger.info(
                        f"RAG: Doc {i + 1} source: '{metadata['title']}' by {metadata['author']}"
                    )
                elif metadata.get("category"):
                    context_part += f"Topic: {metadata['category']}\\n"
                    logger.info(f"RAG: Doc {i + 1} topic: {metadata['category']}")

                context_part += f"Content: {content}\\n"
                context_parts.append(context_part)

            formatted_context = "\\n".join(context_parts)
            total_duration = time.time() - start_time

            logger.info(f"RAG: Context formatting completed in {total_duration:.2f}s")
            logger.info(
                f"RAG: Total context length: {len(formatted_context)} characters"
            )

            return formatted_context

        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(
                f"RAG: Error retrieving relevant context (after {total_duration:.2f}s): {str(e)}"
            )
            logger.exception("RAG context retrieval error details:")
            return ""

    def search_by_metadata(
        self, category: str = None, author: str = None, title: str = None
    ) -> List[Document]:
        try:
            if not self.vectorstore:
                return []

            filter_dict = {}
            if category:
                filter_dict["category"] = category
            if author:
                filter_dict["author"] = author
            if title:
                filter_dict["title"] = title

            if not filter_dict:
                return []

            results = self.vectorstore.similarity_search("", k=50, filter=filter_dict)
            return results

        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        try:
            if not self.vectorstore:
                return {"total_documents": 0}

            total_docs = self.vectorstore._collection.count()

            sample_docs = self.vectorstore.similarity_search("", k=min(100, total_docs))
            categories = set()
            authors = set()

            for doc in sample_docs:
                if doc.metadata.get("category"):
                    categories.add(doc.metadata["category"])
                if doc.metadata.get("author"):
                    authors.add(doc.metadata["author"])

            return {
                "total_documents": total_docs,
                "categories": list(categories),
                "unique_authors": len(authors),
                "sample_categories": list(categories)[:10],
            }

        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"total_documents": 0, "error": str(e)}

    def clear_database(self):
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()

                self._initialize_vectorstore()
                self._populate_default_knowledge()

                logger.info("Knowledge base cleared and reinitialized")

        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")

    def add_bulk_knowledge(self, knowledge_items: List[Dict[str, Any]]):
        try:
            documents = []

            for item in knowledge_items:
                doc = Document(
                    page_content=item["content"], metadata=item.get("metadata", {})
                )
                documents.append(doc)

            if documents:
                self.vectorstore.add_documents(documents)
                self.vectorstore.persist()

                logger.info(f"Added {len(documents)} knowledge items in bulk")

        except Exception as e:
            logger.error(f"Error adding bulk knowledge: {str(e)}")
