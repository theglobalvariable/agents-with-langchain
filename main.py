import os
from operator import itemgetter

from dotenv import load_dotenv

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:1.7b"

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


def format_docs(docs):
    """
    Format retrieved documents into a string format suitable for the prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)


print("Initializing components....")

llm = ChatOllama(model=CHAT_MODEL, temperature=0.9)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = PineconeVectorStore(
    embedding=embeddings, index_name=os.getenv("INDEX_NAME")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:
""")


def main(query):
    print(f"Query: {query}")

    print("=" * 60)
    print("Implementaion 0: Raw LLM Invocation (NO RAG)")
    print("=" * 60)
    result_raw = retrieval_without_rag(query)
    print(f"Result: \n{result_raw.content}\n")

    print("=" * 60)
    print("Implementaion 1: Retrieval without LCEL (LangChain Expression Language)")
    print("=" * 60)
    result_lcel = retrieval_without_lcel(query)
    print(f"Result: \n{result_lcel.content}")

    print("=" * 60)
    print("Implementaion 2: Retrieval using LCEL")
    print("=" * 60)
    result_rag = retrieval_with_lcel(query)
    print(f"Result: \n{result_rag}")


def retrieval_without_rag(query):
    return llm.invoke([HumanMessage(content=query)])


def retrieval_without_lcel(query):
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)
    messages = prompt_template.format_messages(context=context, question=query)

    return llm.invoke(messages)


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL.
    Returns a chain that can be invoked with a query to perform retrieval and answer generation in one step.

    Advantages of using LCEL:
    - Composability: Easily combine multiple components (retriever, prompt template, LLM) into a single chain.
    - Reusability: Components can be reused across different chains or applications.
    - Clarity: The chain structure clearly shows the flow of data from retrieval to answer generation.
    - Flexibility: Easily modify or extend the chain by adding new components or changing existing ones without affecting the overall structure.
    - Less boilerplate: Reduces the amount of code needed to connect components together, making it easier to implement complex workflows.
    - Type Safety: LCEL can provide type safety and validation for the inputs and outputs of each component, reducing the likelihood of errors.
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


def retrieval_with_lcel(query):
    chain = create_retrieval_chain_with_lcel()
    return chain.invoke({"question": query})


if __name__ == "__main__":
    main("What is pinecode in machine learning?")
