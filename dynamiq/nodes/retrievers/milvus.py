from typing import Any, Literal

from dynamiq.components.retrievers.milvus import MilvusDocumentRetriever as MilvusDocumentRetrieverComponent
from dynamiq.connections import Milvus
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import NodeGroup, VectorStoreNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import MilvusVectorStore


class MilvusDocumentRetriever(VectorStoreNode):
    """Document Retriever using Milvus.

    This class implements a document retriever that uses Milvus as the vector store backend.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group of the node.
        name (str): The name of the node.
        vector_store (PineconeVectorStore | None): The Pinecone vector store.
        filters (dict[str, Any] | None): Filters to apply for retrieving specific documents.
        top_k (int): The maximum number of documents to return.
        document_retriever (PineconeDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Arbitrary keyword arguments.
    """

    @property
    def vector_store_cls(self):
        pass

    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    name: str = "MilvusDocumentRetriever"
    connection: Milvus | None = None
    vector_store: MilvusVectorStore | None = None
    filters: dict[str, Any] | None = None
    top_k: int = 10
    document_retriever: MilvusDocumentRetrieverComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the MilvusDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Milvus connection will be created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Milvus()
        super().__init__(**kwargs)

    @property
    def vector_store_params(self):
        return {
            "connection": self.connection,
            "client": self.client,
        }

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}

    def init_components(self, connection_manager: ConnectionManager = ConnectionManager()):
        """
        Initialize the components of the MilvusDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = MilvusDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        """
        Execute the document retrieval process.

        This method retrieves documents based on the input embedding.

        Args:
            input_data (dict[str, Any]): The input data containing the query embedding.
            config (RunnableConfig, optional): The configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query_embedding = input_data["embedding"]
        filters = input_data.get("filters") or self.filters
        top_k = input_data.get("top_k") or self.top_k

        output = self.document_retriever.run(query_embedding, filters=filters, top_k=top_k)

        return {
            "documents": output["documents"],
        }