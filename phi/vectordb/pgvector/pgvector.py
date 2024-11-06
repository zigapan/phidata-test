from typing import Optional, List, Union
from hashlib import md5

try:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.engine import create_engine, Engine
    from sqlalchemy.inspection import inspect
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.schema import MetaData, Table, Column
    from sqlalchemy.sql.expression import text, func, select
    from sqlalchemy.types import DateTime, String
except ImportError:
    raise ImportError("`sqlalchemy` not installed")

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    raise ImportError("`pgvector` not installed")

from phi.document import Document
from phi.embedder import Embedder
from phi.vectordb.base import VectorDb
from phi.vectordb.distance import Distance
from phi.vectordb.pgvector.index import Ivfflat, HNSW
from phi.utils.log import logger


class PgVector(VectorDb):
    def __init__(
        self,
        table_name: str,
        schema: str = "ai",
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        embedder: Optional[Embedder] = None,
        search_type: SearchType = SearchType.vector,
        vector_index: Union[Ivfflat, HNSW] = HNSW(),
        distance: Distance = Distance.cosine,
        prefix_match: bool = False,
        vector_score_weight: float = 0.5,
        content_language: str = "english",
        schema_version: int = 1,
        auto_upgrade_schema: bool = False,
    ):
        """
        Initialize the PgVector instance.

        Args:
            table_name (str): Name of the table to store vector data.
            schema (str): Database schema name.
            db_url (Optional[str]): Database connection URL.
            db_engine (Optional[Engine]): SQLAlchemy database engine.
            embedder (Optional[Embedder]): Embedder instance for creating embeddings.
            search_type (SearchType): Type of search to perform.
            vector_index (Union[Ivfflat, HNSW]): Vector index configuration.
            distance (Distance): Distance metric for vector comparisons.
            prefix_match (bool): Enable prefix matching for full-text search.
            vector_score_weight (float): Weight for vector similarity in hybrid search.
            content_language (str): Language for full-text search.
            schema_version (int): Version of the database schema.
            auto_upgrade_schema (bool): Automatically upgrade schema if True.
        """

        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)

        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")

        # Collection attributes
        self.collection: str = collection
        self.schema: Optional[str] = schema

        # Database attributes
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)

        # Embedder for embedding the document contents
        _embedder = embedder
        if _embedder is None:
            from phi.embedder.openai import OpenAIEmbedder

            _embedder = OpenAIEmbedder()
        self.embedder: Embedder = _embedder
        self.dimensions: Optional[int] = self.embedder.dimensions

        # Distance metric
        self.distance: Distance = distance

        # Index for the table
        self.vector_index: Union[Ivfflat, HNSW] = vector_index
        # Enable prefix matching for full-text search
        self.prefix_match: bool = prefix_match
        # Weight for the vector similarity score in hybrid search
        self.vector_score_weight: float = vector_score_weight

        # Index for the collection
        self.index: Optional[Union[Ivfflat, HNSW]] = index

        # Database session
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.db_engine)

        # Database table for the collection
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.collection,
            self.metadata,
            Column("name", String),
            Column("meta_data", postgresql.JSONB, server_default=text("'{}'::jsonb")),
            Column("content", postgresql.TEXT),
            Column("embedding", Vector(self.dimensions)),
            Column("usage", postgresql.JSONB),
            Column("created_at", DateTime(timezone=True), server_default=text("now()")),
            Column("updated_at", DateTime(timezone=True), onupdate=text("now()")),
            Column("content_hash", String),
            extend_existing=True,
        )

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name, schema=self.schema)
        except Exception as e:
            logger.error(e)
            return False

    def create(self) -> None:
        if not self.table_exists():
            with self.Session() as sess:
                with sess.begin():
                    logger.debug("Creating extension: vector")
                    sess.execute(text("create extension if not exists vector;"))
                    if self.schema is not None:
                        logger.debug(f"Creating schema: {self.schema}")
                        sess.execute(text(f"create schema if not exists {self.schema};"))
            logger.debug(f"Creating table: {self.collection}")
            self.table.create(self.db_engine)

    def doc_exists(self, document: Document) -> bool:
        """
        Validating if the document exists or not

        Args:
            document (Document): Document to validate
        """
        columns = [self.table.c.name, self.table.c.content_hash]
        with self.Session() as sess:
            with sess.begin():
                cleaned_content = document.content.replace("\x00", "\ufffd")
                stmt = select(*columns).where(self.table.c.content_hash == md5(cleaned_content.encode()).hexdigest())
                result = sess.execute(stmt).first()
                return result is not None

    def name_exists(self, name: str) -> bool:
        """
        Validate if a row with this name exists or not

        Args:
            name (str): Name to validate
        """
        with self.Session() as sess:
            with sess.begin():
                stmt = select(self.table.c.name).where(self.table.c.name == name)
                result = sess.execute(stmt).first()
                return result is not None

    def insert(self, documents: List[Document], batch_size: int = 10) -> None:
        with self.Session() as sess:
            counter = 0
            for document in documents:
                document.embed(embedder=self.embedder)
                cleaned_content = document.content.replace("\x00", "\ufffd")
                stmt = postgresql.insert(self.table).values(
                    name=document.name,
                    meta_data=document.meta_data,
                    content=cleaned_content,
                    embedding=document.embedding,
                    usage=document.usage,
                    content_hash=md5(cleaned_content.encode()).hexdigest(),
                )
                sess.execute(stmt)
                counter += 1
                logger.debug(f"Inserted document: {document.name} ({document.meta_data})")

                # Commit every `batch_size` documents
                if counter >= batch_size:
                    sess.commit()
                    logger.debug(f"Committed {counter} documents")
                    counter = 0

            # Commit any remaining documents
            if counter > 0:
                sess.commit()
                logger.debug(f"Committed {counter} documents")

    def upsert(self, documents: List[Document]) -> None:
        """
        Upsert documents into the database.

        Args:
            documents (List[Document]): List of documents to upsert
        """
        with self.Session() as sess:
            with sess.begin():
                for document in documents:
                    document.embed(embedder=self.embedder)
                    cleaned_content = document.content.replace("\x00", "\ufffd")
                    stmt = postgresql.insert(self.table).values(
                        name=document.name,
                        meta_data=document.meta_data,
                        content=cleaned_content,
                        embedding=document.embedding,
                        usage=document.usage,
                        content_hash=md5(cleaned_content.encode()).hexdigest(),
                    )
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["name", "content_hash"],
                        set_=dict(
                            meta_data=document.meta_data,
                            content=stmt.excluded.content,
                            embedding=stmt.excluded.embedding,
                            usage=stmt.excluded.usage,
                        ),
                    )
                    sess.execute(stmt)
                    logger.debug(f"Upserted document: {document.name} ({document.meta_data})")

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform a search based on the configured search type.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        if self.search_type == SearchType.vector:
            return self.vector_search(query=query, limit=limit, filters=filters)
        elif self.search_type == SearchType.keyword:
            return self.keyword_search(query=query, limit=limit, filters=filters)
        elif self.search_type == SearchType.hybrid:
            return self.hybrid_search(query=query, limit=limit, filters=filters)
        else:
            logger.error(f"Invalid search type '{self.search_type}'.")
            return []

    def vector_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform a vector similarity search.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        try:
            # Get the embedding for the query string
            query_embedding = self.embedder.get_embedding(query)
            if query_embedding is None:
                logger.error(f"Error getting embedding for Query: {query}")
                return []

            # Define the columns to select
            columns = [
                self.table.c.id,
                self.table.c.name,
                self.table.c.meta_data,
                self.table.c.content,
                self.table.c.embedding,
                self.table.c.usage,
            ]

            # Build the base statement
            stmt = select(*columns)

            # Apply filters if provided
            if filters is not None:
                stmt = stmt.where(self.table.c.filters.contains(filters))

            # Order the results based on the distance metric
            if self.distance == Distance.l2:
                stmt = stmt.order_by(self.table.c.embedding.l2_distance(query_embedding))
            elif self.distance == Distance.cosine:
                stmt = stmt.order_by(self.table.c.embedding.cosine_distance(query_embedding))
            elif self.distance == Distance.max_inner_product:
                stmt = stmt.order_by(self.table.c.embedding.max_inner_product(query_embedding))
            else:
                logger.error(f"Unknown distance metric: {self.distance}")
                return []

            # Limit the number of results
            stmt = stmt.limit(limit)

            # Log the query for debugging
            logger.debug(f"Vector search query: {stmt}")

            # Execute the query
            try:
                with self.Session() as sess, sess.begin():
                    if self.vector_index is not None:
                        if isinstance(self.vector_index, Ivfflat):
                            sess.execute(text(f"SET LOCAL ivfflat.probes = {self.vector_index.probes}"))
                        elif isinstance(self.vector_index, HNSW):
                            sess.execute(text(f"SET LOCAL hnsw.ef_search = {self.vector_index.ef_search}"))
                    results = sess.execute(stmt).fetchall()
            except Exception as e:
                logger.error(f"Error performing semantic search: {e}")
                logger.error("Table might not exist, creating for future use")
                self.create()
                return []

            # Process the results and convert to Document objects
            search_results: List[Document] = []
            for result in results:
                search_results.append(
                    Document(
                        id=result.id,
                        name=result.name,
                        meta_data=result.meta_data,
                        content=result.content,
                        embedder=self.embedder,
                        embedding=result.embedding,
                        usage=result.usage,
                    )
                )

            return search_results
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def enable_prefix_matching(self, query: str) -> str:
        """
        Preprocess the query for prefix matching.

        Args:
            query (str): The original query.

        Returns:
            str: The processed query with prefix matching enabled.
        """
        # Append '*' to each word for prefix matching
        words = query.strip().split()
        processed_words = [word + "*" for word in words]
        return " ".join(processed_words)

    def keyword_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform a keyword search on the 'content' column.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        try:
            # Define the columns to select
            columns = [
                self.table.c.id,
                self.table.c.name,
                self.table.c.meta_data,
                self.table.c.content,
                self.table.c.embedding,
                self.table.c.usage,
            ]

            # Build the base statement
            stmt = select(*columns)

            # Build the text search vector
            ts_vector = func.to_tsvector(self.content_language, self.table.c.content)
            # Create the ts_query using websearch_to_tsquery with parameter binding
            processed_query = self.enable_prefix_matching(query) if self.prefix_match else query
            ts_query = func.websearch_to_tsquery(self.content_language, bindparam("query", value=processed_query))
            # Compute the text rank
            text_rank = func.ts_rank_cd(ts_vector, ts_query)

            # Apply filters if provided
            if filters is not None:
                # Use the contains() method for JSONB columns to check if the filters column contains the specified filters
                stmt = stmt.where(self.table.c.filters.contains(filters))

            # Order by the relevance rank
            stmt = stmt.order_by(text_rank.desc())

            # Limit the number of results
            stmt = stmt.limit(limit)

            # Log the query for debugging
            logger.debug(f"Keyword search query: {stmt}")

            # Execute the query
            try:
                with self.Session() as sess, sess.begin():
                    results = sess.execute(stmt).fetchall()
            except Exception as e:
                logger.error(f"Error performing keyword search: {e}")
                logger.error("Table might not exist, creating for future use")
                self.create()
                return []

            # Process the results and convert to Document objects
            search_results: List[Document] = []
            for result in results:
                search_results.append(
                    Document(
                        id=result.id,
                        name=result.name,
                        meta_data=result.meta_data,
                        content=result.content,
                        embedder=self.embedder,
                        embedding=result.embedding,
                        usage=result.usage,
                    )
                )

            return search_results
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            return []

    def hybrid_search(
            self,
            query: str,
            limit: int = 5,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform a hybrid search combining vector similarity and full-text search.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.
            filters (Optional[Dict[str, Any]]): Filters to apply to the search.

        Returns:
            List[Document]: List of matching documents.
        """
        try:
            # Get the embedding for the query string
            query_embedding = self.embedder.get_embedding(query)
            if query_embedding is None:
                logger.error(f"Error getting embedding for Query: {query}")
                return []

            # Define the columns to select
            columns = [
                self.table.c.id,
                self.table.c.name,
                self.table.c.meta_data,
                self.table.c.content,
                self.table.c.embedding,
                self.table.c.usage,
            ]

            # Build the text search vector
            ts_vector = func.to_tsvector(self.content_language, self.table.c.content)
            # Create the ts_query using websearch_to_tsquery with parameter binding
            processed_query = self.enable_prefix_matching(query) if self.prefix_match else query
            ts_query = func.websearch_to_tsquery(self.content_language, bindparam("query", value=processed_query))
            # Compute the text rank
            text_rank = func.ts_rank_cd(ts_vector, ts_query)

            # Compute the vector similarity score
            if self.distance == Distance.l2:
                # For L2 distance, smaller distances are better
                vector_distance = self.table.c.embedding.l2_distance(query_embedding)
                # Invert and normalize the distance to get a similarity score between 0 and 1
                vector_score = 1 / (1 + vector_distance)
            elif self.distance == Distance.cosine:
                # For cosine distance, smaller distances are better
                vector_distance = self.table.c.embedding.cosine_distance(query_embedding)
                vector_score = 1 / (1 + vector_distance)
            elif self.distance == Distance.max_inner_product:
                # For inner product, higher values are better
                # Assume embeddings are normalized, so inner product ranges from -1 to 1
                raw_vector_score = self.table.c.embedding.max_inner_product(query_embedding)
                # Normalize to range [0, 1]
                vector_score = (raw_vector_score + 1) / 2
            else:
                logger.error(f"Unknown distance metric: {self.distance}")
                return []

            # Apply weights to control the influence of each score
            # Validate the vector_weight parameter
            if not 0 <= self.vector_score_weight <= 1:
                raise ValueError("vector_score_weight must be between 0 and 1")
            text_rank_weight = 1 - self.vector_score_weight  # weight for text rank

            # Combine the scores into a hybrid score
            hybrid_score = (self.vector_score_weight * vector_score) + (text_rank_weight * text_rank)

            # Build the base statement, including the hybrid score
            stmt = select(*columns, hybrid_score.label("hybrid_score"))

            # Add the full-text search condition
            # stmt = stmt.where(ts_vector.op("@@")(ts_query))

            # Apply filters if provided
            if filters is not None:
                stmt = stmt.where(self.table.c.filters.contains(filters))

            # Order the results by the hybrid score in descending order
            stmt = stmt.order_by(desc("hybrid_score"))

            # Limit the number of results
            stmt = stmt.limit(limit)

            # Log the query for debugging
            logger.debug(f"Hybrid search query: {stmt}")

            # Execute the query
            try:
                with self.Session() as sess, sess.begin():
                    if self.vector_index is not None:
                        if isinstance(self.vector_index, Ivfflat):
                            sess.execute(text(f"SET LOCAL ivfflat.probes = {self.vector_index.probes}"))
                        elif isinstance(self.vector_index, HNSW):
                            sess.execute(text(f"SET LOCAL hnsw.ef_search = {self.vector_index.ef_search}"))
                    results = sess.execute(stmt).fetchall()
            except Exception as e:
                logger.error(f"Error performing hybrid search: {e}")
                return []

            # Process the results and convert to Document objects
            search_results: List[Document] = []
            for result in results:
                search_results.append(
                    Document(
                        id=result.id,
                        name=result.name,
                        meta_data=result.meta_data,
                        content=result.content,
                        embedder=self.embedder,
                        embedding=result.embedding,
                        usage=result.usage,
                    )
                )

            return search_results
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return []

    def delete(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.collection}")
            self.table.drop(self.db_engine)

    def exists(self) -> bool:
        return self.table_exists()

    def get_count(self) -> int:
        with self.Session() as sess:
            with sess.begin():
                stmt = select(func.count(self.table.c.name)).select_from(self.table)
                result = sess.execute(stmt).scalar()
                if result is not None:
                    return int(result)
                return 0

    def optimize(self) -> None:
        from math import sqrt

        logger.debug("==== Optimizing Vector DB ====")
        if self.index is None:
            return

        if self.index.name is None:
            _type = "ivfflat" if isinstance(self.index, Ivfflat) else "hnsw"
            self.index.name = f"{self.collection}_{_type}_index"

        index_distance = "vector_cosine_ops"
        if self.distance == Distance.l2:
            index_distance = "vector_l2_ops"
        if self.distance == Distance.max_inner_product:
            index_distance = "vector_ip_ops"

        if isinstance(self.index, Ivfflat):
            num_lists = self.index.lists
            if self.index.dynamic_lists:
                total_records = self.get_count()
                logger.debug(f"Number of records: {total_records}")
                if total_records < 1000000:
                    num_lists = int(total_records / 1000)
                elif total_records > 1000000:
                    num_lists = int(sqrt(total_records))

            with self.Session() as sess:
                with sess.begin():
                    logger.debug(f"Setting configuration: {self.index.configuration}")
                    for key, value in self.index.configuration.items():
                        sess.execute(text(f"SET {key} = '{value}';"))
                    logger.debug(
                        f"Creating Ivfflat index with lists: {num_lists}, probes: {self.index.probes} "
                        f"and distance metric: {index_distance}"
                    )
                    sess.execute(text(f"SET ivfflat.probes = {self.index.probes};"))
                    sess.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS {self.index.name} ON {self.table} "
                            f"USING ivfflat (embedding {index_distance}) "
                            f"WITH (lists = {num_lists});"
                        )
                    )
        elif isinstance(self.index, HNSW):
            with self.Session() as sess:
                with sess.begin():
                    logger.debug(f"Setting configuration: {self.index.configuration}")
                    for key, value in self.index.configuration.items():
                        sess.execute(text(f"SET {key} = '{value}';"))
                    logger.debug(
                        f"Creating HNSW index with m: {self.index.m}, ef_construction: {self.index.ef_construction} "
                        f"and distance metric: {index_distance}"
                    )
                    sess.execute(
                        text(
                            f"CREATE INDEX IF NOT EXISTS {self.index.name} ON {self.table} "
                            f"USING hnsw (embedding {index_distance}) "
                            f"WITH (m = {self.index.m}, ef_construction = {self.index.ef_construction});"
                        )
                    )
        logger.debug("==== Optimized Vector DB ====")

    def clear(self) -> bool:
        from sqlalchemy import delete

        with self.Session() as sess:
            with sess.begin():
                stmt = delete(self.table)
                sess.execute(stmt)
                return True
