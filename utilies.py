# filename: read_docx.py
def read_docx(file_path):
    from docx import Document
    doc = Document(file_path)
    content = []
    for paragraph in doc.paragraphs:
        content.append(paragraph.text)
    return '\n'.join(content)

def langchain_embedding_documents(llm_api_key,
                        postgres_username, 
                        postgres_password, 
                        database_name,
                        collection_name, 
                        document_path, 
                        documents_name_list,
                        chunk_size = 500,
                        chunk_overlap = 80,
                        vector_db = 'FAISS'
                          ):
    from tqdm import tqdm

    from langchain_community.document_loaders import TextLoader
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    #from langchain.embeddings import OpenAIEmbeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    from langchain_core.documents import Document
    from langchain_postgres import PGVector
    from langchain_postgres.vectorstores import PGVector
    from langchain.indexes import SQLRecordManager, index
    import os

    document_names = documents_name_list

    document_list = []
    for each_index, document in tqdm(enumerate(document_names), desc='Loading Documents', unit='doc', total=len(document_names)):
        loader = UnstructuredWordDocumentLoader(os.path.join(document_path, f'{document}.docx'))
        document_loaded_text = loader.load()
        document_list.append(document_loaded_text)

    # Dynamically create variable names based on the number of documents and assigning it to the content of each word documents
    all_content_variable_names = []
    for each_index, content in enumerate(document_list):
        content_variable_name = f"content_{each_index+1}"
        globals()[content_variable_name] = content
        all_content_variable_names.append(content_variable_name)

    # Create a splittings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splitted_text = []
    for each_content_variable_name in tqdm(all_content_variable_names, desc='Splitting Content', unit='content_name', total=len(all_content_variable_names)):
        splitted_text = text_splitter.split_documents(globals()[each_content_variable_name]) #splitted_text are all the chunks in each document
        all_splitted_text.append(splitted_text) #all_splitted_text is the list containing all chuncks from all documents

    print(f'Number of documents: {len(all_splitted_text)}') # shows the number of documents
    print(f'Number of chuck in the first document: {len(all_splitted_text[0])}') # shows the number of chunks in the documents 

    #Flattening the all_splitted_text
    flattened_all_splitted_text = [item for sublist in all_splitted_text for item in sublist]

    # Embedding
    embedding = OpenAIEmbeddings(openai_api_key=llm_api_key)
    content_chunk_vec = embedding.embed_documents([each_split.page_content for each_split in flattened_all_splitted_text]) 


    if vector_db == 'pgvector':

        CONNECTION_STRING = f"postgresql+psycopg://{postgres_username}:{postgres_password}@localhost:5432/{database_name}"  # Uses psycopg3!

        namespace = f"pgvector/{collection_name}"
        record_manager = SQLRecordManager(
            namespace, db_url=CONNECTION_STRING
            )

        record_manager.create_schema()     
       
        db = PGVector(embeddings=embedding, 
                connection=CONNECTION_STRING,
                collection_name=collection_name,
                use_jsonb=True,)
        
        run_indexing = index(flattened_all_splitted_text, record_manager, db, cleanup="incremental", source_id_key="source")
        print(run_indexing)
        
    if vector_db == 'FAISS':
        db = FAISS.from_documents(flattened_all_splitted_text, embedding)

    return db

def langChain_connecting_pgvector(llm_api_key,
                                  postgres_username,
                                  postgres_password,
                                  database_name,
                                  collection_name
                                  ):
    from langchain_openai import OpenAIEmbeddings
    from langchain_openai.chat_models import ChatOpenAI


    from langchain_core.documents import Document
    from langchain_postgres import PGVector
    from langchain_postgres.vectorstores import PGVector

    # Vector Database Name and the Table Name
    database_name = database_name

    # Defining the Database
    embedding = OpenAIEmbeddings(openai_api_key=llm_api_key) 
    CONNECTION_STRING = f"postgresql+psycopg://{postgres_username}:{postgres_password}@localhost:5432/{database_name}"  # Uses psycopg3!

    db = PGVector(embeddings=embedding, 
                connection=CONNECTION_STRING,
                collection_name=collection_name,
                use_jsonb=True,)

    return db