from backend.chroma_utils import get_collection
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load Chroma collection
collection = get_collection()

# Wrap Chroma in LangChain vector store
vectorstore = Chroma(
    collection_name="legal_rag_chunks",
    persist_directory=str(collection._client.settings.persist_directory),
    embedding_function=OpenAIEmbeddings()  # or any embedding model
)

# Create retrieval object
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # top 5 results

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


def ask_question(question: str):
    """
    Input: user question (str)
    Output: dict with answer and sources
    """
    result = qa_chain.run(question)
    return result
