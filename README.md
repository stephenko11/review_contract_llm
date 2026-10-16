# Multi-Agent Contract Review System with AutoGen & LangChain

An exploratory multi-agent legal review pipeline built with **Microsoft AutoGen** and **LangChain**. The system simulates a collaborative legal advisory team (Attorney, Playbook Analyst, Business Consultant, Planner, and Admin) that analyzes a commercial agreement against a negotiation playbook from the perspective of a designated party (e.g., the Manufacturer).

It supports direct contract evaluation as well as Retrieval-Augmented Generation (RAG) powered by **PostgreSQL (pgvector)** and **FAISS**.

> **Status:** Personal, self-taught proof-of-concept. It is not production code and is not legal advice.

---

## Project Findings & Experimental Notes

This project was an exploratory proof-of-concept to evaluate multi-agent orchestration for commercial contract review using **AutoGen**.

**Key observations**

- **RAG vs. direct ingestion:** Naive chunk-based RAG retrieval did not noticeably improve review accuracy compared with passing the full contract text. Contracts require holistic cross-referencing that fixed-size chunks tend to fragment.
- **State and context transfer:** AutoGen's conversational chat structure made it hard to reliably pass structured state, strict sequential constraints, and accumulated feedback between specialized agents.
- **Review quality:** Accuracy needs tuning. Some nuanced clauses were occasionally missed during multi-agent handoffs.
- **Tooling and UX:** Text logs are not enough for contract analysis. An interactive UI with side-by-side diffs and redlines is essential for verifying proposed clause amendments.

**Next steps**

- Migrate the architecture to **LangGraph** for deterministic state transitions, cyclic graph routing, and persistent shared memory.
- Build a small test set of contracts with known issues, so accuracy can be measured instead of judged by eye.

---

## Architecture & Agent Roles

```text
                  +-------------------+
                  |   Admin (User)    |
                  +---------+---------+
                            |
                  +---------v---------+
                  |      Planner      |
                  +---------+---------+
                            |
    +-----------------------+-----------------------+
    |                       |                       |
+---v-----------+       +---v-----------+       +---v-----------+
|   Attorney    | <---> |   Playbook    | <---> |   Business    |
|   Chatbot     |       |   Analyst     |       |  Consultant   |
+---------------+       +---------------+       +---------------+
```

- **Admin (`UserProxyAgent`):** Oversees execution and approves plans suggested by the Planner.
- **Planner (`AssistantAgent`):** Outlines and coordinates the sequence of review actions across the specialist agents.
- **Attorney Chatbot (`AssistantAgent`):** Analyzes legal language, identifies dispute risks, incorporates specialist feedback, and drafts proposed clause amendments.
- **Playbook Analyst (`AssistantAgent`):** Parses negotiation guidelines and checks the agreement for missing terms or required deviations.
- **Business Consultant (`AssistantAgent`):** Assesses business risk, commercial feasibility, and IP/property rights allocation for the represented party.
- **RAG Retrieval Agent (`RetrieveUserProxyAgent`):** Interfaces with vector stores to pull relevant contract chunks or external legal references into the chat context.

---

## Project Structure

```text
.
├── main.py              # Orchestration, AutoGen agent definitions, and chat modes
├── utilities.py         # Document parsing (.docx), text chunking, and vector database utilities
├── documents/
│   └── agreement/
│       └── agreement.docx   # Contract file to analyze
├── .env                 # Environment variables (API keys, DB credentials)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to GPT-4o / GPT-4o-mini
- (Optional, for pgvector mode) A running PostgreSQL instance with the `vector` extension enabled

### Install dependencies

```bash
pip install pyautogen langchain langchain-community langchain-openai langchain-postgres \
  python-docx python-dotenv tqdm psycopg2 faiss-cpu unstructured
```

### Configuration

Create a `.env` file in the root directory:

```text
OPENAI_API_KEY=your_openai_api_key_here
POSTGRES_KEY=your_postgres_password_here
```

Adjust the database, model, and party settings inside `main.py` if needed:

```python
postgres_username = 'postgres'
database_name = 'llm_vector_1'
collection_name = 'autogen_contract'
llm_model = 'gpt-4o-mini-2024-07-18'

task = 'agreement'
acting_party = 'Manufacturer'
agreement_to_review_file_name = 'agreement.docx'
```

### Run

Place the contract to review in `documents/agreement/`, choose a mode in `main.py` (see below), then run:

```bash
python main.py
```

---

## Core Modules

### `utilities.py`

File handling and vector indexing utilities:

- `read_docx(file_path)`: Reads and extracts clean text from `.docx` files.
- `langchain_embedding_documents(...)`: Loads Word documents with `UnstructuredWordDocumentLoader`, splits text with `RecursiveCharacterTextSplitter`, embeds chunks with `OpenAIEmbeddings`, and stores them in pgvector or FAISS.
- `langChain_connecting_pgvector(...)`: Connects to an existing pgvector table and collection.

### `main.py`

Initializes the agents, assigns system prompts, and provides three execution modes:

- **Direct mode, `norag_chat()`:** Reviews the document directly, using the full contract text in the prompt. A custom speaker function routes initial control to the Planner.
- **Native RAG mode, `rag_chat()`:** Uses AutoGen's `RetrieveUserProxyAgent` in a round-robin group chat to retrieve document chunks by semantic similarity.
- **Function-call RAG mode, `call_rag_chat()`:** Registers retrieval as a callable tool (`retrieve_content`) for all agents, letting them pull context on demand.

---

## Customizing the Playbook

Edit the multi-line `playbook` string in `main.py` to match your transaction requirements:

```python
playbook = """Background information and playbook:
<Basic Mode Transaction>
- Ex-factory sales; title passes upon shipment.
<Financial Term>
- 30% advance deposit, 70% before delivery.
<Governing & Dispute Law Resolution>
- Governed by and construed in accordance with the laws of the State of New York
"""
```
