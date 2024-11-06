from typing import List, Optional, Iterator, Dict, Any

from pydantic import ConfigDict

from phi.document import Document
from phi.document.reader.base import Reader
from phi.knowledge.base import AssistantKnowledge
from phi.knowledge.chunks import CharacterChunks, ChunkingStrategy
from phi.vectordb import VectorDb
from phi.utils.log import logger


class AgentKnowledge(AssistantKnowledge):
    """Base class for Agent knowledge

    This class inherits from AssistantKnowledge only to maintain backward compatibility for downstream Knowledge bases.
    In phidata 3.0.0, AssistantKnowledge will be deprecated and this class will inherit directly from BaseModel.
    """

    # Reader for reading documents from files, pdfs, urls, etc.
    reader: Optional[Reader] = None
    # Vector db for storing knowledge
    vector_db: Optional[VectorDb] = None
    # Number of relevant documents to return on search
    num_documents: int = 5
    # Number of documents to optimize the vector db on
    optimize_on: Optional[int] = 1000
    # ChunkingStrategy to chunk documents into smaller documents before storing in vector db
    chunking_strategy: ChunkingStrategy = CharacterChunks()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        """Iterator that yields lists of documents in the knowledge base
        Each object yielded by the iterator is a list of documents.
        """
        raise NotImplementedError

    def search(
        self, query: str, num_documents: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Returns relevant documents matching a query"""
        try:
            if self.vector_db is None:
                logger.warning("No vector db provided")
                return []

            _num_documents = num_documents or self.num_documents
            logger.debug(f"Getting {_num_documents} relevant documents for query: {query}")
            return self.vector_db.search(query=query, limit=_num_documents, filters=filters)
        except Exception as e:
            logger.error(f"Error searching for documents: {e}")
            return []

    def load(
        self,
        recreate: bool = False,
        upsert: bool = False,
        skip_existing: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load the knowledge base to the vector db

        Args:
            recreate (bool): If True, recreates the collection in the vector db. Defaults to False.
            upsert (bool): If True, upserts documents to the vector db. Defaults to False.
            skip_existing (bool): If True, skips documents which already exist in the vector db when inserting. Defaults to True.
            filters (Optional[Dict[str, Any]]): Filters to add to each row that can be used to limit results during querying. Defaults to None.
        """

        if self.vector_db is None:
            logger.warning("No vector db provided")
            return

        if recreate:
            logger.info("Dropping collection")
            self.vector_db.drop()

        logger.info("Creating collection")
        self.vector_db.create()

        logger.info("Loading knowledge base")
        num_documents = 0
        for document_list in self.document_lists:
            documents_to_load = document_list
            # Upsert documents if upsert is True and vector db supports upsert
            if upsert and self.vector_db.upsert_available():
                self.vector_db.upsert(documents=documents_to_load, filters=filters)
            # Insert documents
            else:
                # Filter out documents which already exist in the vector db
                if skip_existing:
                    documents_to_load = [
                        document for document in document_list if not self.vector_db.doc_exists(document)
                    ]
                self.vector_db.insert(documents=documents_to_load, filters=filters)
            num_documents += len(documents_to_load)
            logger.info(f"Added {len(documents_to_load)} documents to knowledge base")

    def load_documents(
        self,
        documents: List[Document],
        upsert: bool = False,
        skip_existing: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load documents to the knowledge base

        Args:
            documents (List[Document]): List of documents to load
            upsert (bool): If True, upserts documents to the vector db. Defaults to False.
            skip_existing (bool): If True, skips documents which already exist in the vector db when inserting. Defaults to True.
            filters (Optional[Dict[str, Any]]): Filters to add to each row that can be used to limit results during querying. Defaults to None.
        """

        logger.info("Loading knowledge base")
        if self.vector_db is None:
            logger.warning("No vector db provided")
            return

        logger.debug("Creating collection")
        self.vector_db.create()

        # Upsert documents if upsert is True
        if upsert and self.vector_db.upsert_available():
            self.vector_db.upsert(documents=documents, filters=filters)
            logger.info(f"Loaded {len(documents)} documents to knowledge base")
            return

        # Filter out documents which already exist in the vector db
        documents_to_load = (
            [document for document in documents if not self.vector_db.doc_exists(document)]
            if skip_existing
            else documents
        )

        # Insert documents
        if len(documents_to_load) > 0:
            self.vector_db.insert(documents=documents_to_load, filters=filters)
            logger.info(f"Loaded {len(documents_to_load)} documents to knowledge base")
        else:
            logger.info("No new documents to load")

    def load_document(
        self,
        document: Document,
        upsert: bool = False,
        skip_existing: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load a document to the knowledge base

        Args:
            document (Document): Document to load
            upsert (bool): If True, upserts documents to the vector db. Defaults to False.
            skip_existing (bool): If True, skips documents which already exist in the vector db. Defaults to True.
            filters (Optional[Dict[str, Any]]): Filters to add to each row that can be used to limit results during querying. Defaults to None.
        """
        self.load_documents(documents=[document], upsert=upsert, skip_existing=skip_existing, filters=filters)

    def load_dict(
        self,
        document: Dict[str, Any],
        upsert: bool = False,
        skip_existing: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load a dictionary representation of a document to the knowledge base

        Args:
            document (Dict[str, Any]): Dictionary representation of a document
            upsert (bool): If True, upserts documents to the vector db. Defaults to False.
            skip_existing (bool): If True, skips documents which already exist in the vector db. Defaults to True.
            filters (Optional[Dict[str, Any]]): Filters to add to each row that can be used to limit results during querying. Defaults to None.
        """
        self.load_documents(
            documents=[Document.from_dict(document)], upsert=upsert, skip_existing=skip_existing, filters=filters
        )

    def load_json(
        self, document: str, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load a json representation of a document to the knowledge base

        Args:
            document (str): Json representation of a document
            upsert (bool): If True, upserts documents to the vector db. Defaults to False.
            skip_existing (bool): If True, skips documents which already exist in the vector db. Defaults to True.
            filters (Optional[Dict[str, Any]]): Filters to add to each row that can be used to limit results during querying. Defaults to None.
        """
        self.load_documents(
            documents=[Document.from_json(document)], upsert=upsert, skip_existing=skip_existing, filters=filters
        )

    def load_text(
        self, text: str, upsert: bool = False, skip_existing: bool = True, filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load a text to the knowledge base

        Args:
            text (str): Text to load to the knowledge base
            upsert (bool): If True, upserts documents to the vector db. Defaults to False.
            skip_existing (bool): If True, skips documents which already exist in the vector db. Defaults to True.
            filters (Optional[Dict[str, Any]]): Filters to add to each row that can be used to limit results during querying. Defaults to None.
        """
        self.load_documents(
            documents=[Document(content=text)], upsert=upsert, skip_existing=skip_existing, filters=filters
        )

    def exists(self) -> bool:
        """Returns True if the knowledge base exists"""
        if self.vector_db is None:
            logger.warning("No vector db provided")
            return False
        return self.vector_db.exists()

    def delete(self) -> bool:
        """Clear the knowledge base"""
        if self.vector_db is None:
            logger.warning("No vector db available")
            return True

        return self.vector_db.delete()
