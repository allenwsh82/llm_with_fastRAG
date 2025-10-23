# Building LLM Inference with fastRAG Framework

fastRAG is a research framework for efficient and optimized retrieval augmented generative pipelines, incorporating state-of-the-art LLMs and Information Retrieval. fastRAG is designed to empower researchers and developers with a comprehensive tool-set for advancing retrieval augmented generation.

Key Features:

Optimized RAG: Build RAG pipelines with SOTA efficient components for greater compute efficiency.
Optimized for Intel Hardware: Leverage Intel extensions for PyTorch (IPEX), 🤗 Optimum Intel and 🤗 Optimum-Habana for running as optimal as possible on Intel® Xeon® Processors and Intel® Gaudi® AI accelerators.
Customizable: fastRAG is built using Haystack and HuggingFace. All of fastRAG's components are 100% Haystack compatible.

What is the differences between Retrieval Augmented Generation vs Pre-Training vs Fine Tuning?

![RAG_vs_Fine_Tuning](https://github.com/allenwsh82/llm_with_fastRAG/assets/44453417/dead4bd6-f317-454b-a074-a15e3ac8b267)

![tco](https://github.com/allenwsh82/llm_with_fastRAG/assets/44453417/b67059ee-f45d-4c5f-bad2-99dab9a33328)


**End to End fastRAG Pipeline Block Diagram :**

<img width="900" alt="QnA_fastRAG_3png" src="https://github.com/user-attachments/assets/7957cb01-fb21-4432-a79c-73f78c862cf5">



**🧪 ETL Demonstration Using Pandas and Haystack**

This demo illustrates a simple ETL pipeline using Pandas for data handling and Haystack for document indexing with FAISS.


**📥 Extract**

We begin by reading a CSV file using Pandas

---------------------------------------------------------------------------------------------------------------------------------

**df = pd.read_csv(f"{doc_dir}/small_generator_dataset.csv", sep=",")**

---------------------------------------------------------------------------------------------------------------------------------

This loads the dataset into a DataFrame for further processing.

**🔧 Transform**

We perform minimal data cleaning to ensure robustness during document creation:

---------------------------------------------------------------------------------------------------------------------------------

**df.fillna(value="", inplace=True)**
**print(df.head(n=5))**

---------------------------------------------------------------------------------------------------------------------------------

All NaN values are replaced with empty strings to prevent issues when converting rows into text-based documents.

**📦 Load**

Each row is converted into a Haystack Document object:

---------------------------------------------------------------------------------------------------------------------------------

**from haystack import Document**

**titles = list(df["title"].values)**
**texts = list(df["text"].values)**
**documents = [Document(content=text, meta={"name": title or ""}) for title, text in zip(titles, texts)]**

---------------------------------------------------------------------------------------------------------------------------------

- content: Holds the main text.
- meta: Stores metadata like the title.


**🧠 Initialize FAISS Document Store**

We set up a FAISS-based vector index to store and search documents efficiently

---------------------------------------------------------------------------------------------------------------------------------

**from haystack.document_stores import FAISSDocumentStore**
**document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)**

---------------------------------------------------------------------------------------------------------------------------------

This enables fast similarity search over embedded documents.

**This script demonstrates a complete retrieval-augmented generation (RAG) pipeline using Haystack, Hugging Face Transformers, and FAISS—integrating document indexing, semantic search, reranking, and prompt-based generation with a local LLaMA 2 model. Here's a structured walkthrough of what it's doing:**


**🧠 Model Setup**

---------------------------------------------------------------------------------------------------------------------------------

**model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=hf_token)**
**tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=hf_token)**

---------------------------------------------------------------------------------------------------------------------------------

- Loads the LLaMA 2 7B Chat model locally using Hugging Face Transformers.
- Uses a Hugging Face token for authentication.


**🔍 Retriever + Reranker + PromptNode**

---------------------------------------------------------------------------------------------------------------------------------

**retriever = EmbeddingRetriever(...)**
**reranker = SentenceTransformersRanker(...)**

---------------------------------------------------------------------------------------------------------------------------------

- Retriever: Uses sentence-transformers for dense vector search.
- Reranker: Refines top results using a cross-encoder for relevance scoring


**📝 Prompt Template**

---------------------------------------------------------------------------------------------------------------------------------

**lfqa_prompt = PromptTemplate(...)**

---------------------------------------------------------------------------------------------------------------------------------

- Defines a long-form QA prompt with a 50-word constraint.
- Uses {join(documents)} to inject retrieved context.


**PromptNode Configuration**

---------------------------------------------------------------------------------------------------------------------------------

**local_model = PromptModel(...)**
**prompt = PromptNode(...)**

---------------------------------------------------------------------------------------------------------------------------------

- Wraps the LLaMA model with HFLocalInvocationLayer to support non-standard Hugging Face models.
- Configures generation parameters like max_length, torch_dtype, and streaming

**📚 Document Store Operations**

---------------------------------------------------------------------------------------------------------------------------------

**document_store.delete_documents()**
**document_store.write_documents(documents)**
**document_store.update_embeddings(retriever=retriever)**

---------------------------------------------------------------------------------------------------------------------------------

- Clears existing documents.
- Writes new documents from your earlier ETL step.
- Updates vector embeddings for semantic search.

**🔗 Pipeline Assembly**

---------------------------------------------------------------------------------------------------------------------------------

**p = Pipeline()**
**p.add_node(...)**

---------------------------------------------------------------------------------------------------------------------------------

- Connects retriever → reranker → prompt node.
- Executes the pipeline with a user query.

**🧪 Sample Query Execution**

---------------------------------------------------------------------------------------------------------------------------------

**a = p.run(query="who got the first nobel prize in physics", debug=True)**
**print(a['answers'][0].answer)**

---------------------------------------------------------------------------------------------------------------------------------

- Runs the full RAG pipeline.
- Retrieves, reranks, and generates an answer using your custom prompt.


**How to run the Question-Answer with fastRAG Demo:**

1) Clone the project:
```
   git clone https://github.com/allenwsh82/llm_with_fastRAG.git
```

2) Create a new environment for this project: 
```
   python -m venv rag_env
```

3) Activate the environment: 
```
   source rag_evn/bin/activate
```

4) Setup the environment with all the dependencies: 
```
   pip install -r requirements.txt
```

5) You can create your own .csv file or download a .csv format file from this link:
   
   https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip
   
   You can modify or add into your own dataset into the .csv file
   
   <img width="1000" alt="QnA_fastRAG_4" src="https://github.com/user-attachments/assets/b9b11e8a-5d65-4f47-9212-e05f0fa18441">
   
6) Then execute the program by running the following command:
```
   python Generative_QA_Haystack_PromptNode_CSV_database_llama2_Demo.py   
```

User Interface Question-Answer with fastRAG with Gradio: 

<img width="1100" alt="QnA_fastRAG_1" src="https://github.com/user-attachments/assets/83cbdfbf-b946-4b37-ad37-b8250789f538">

Comparison of RAG vs Non-Rag Generated Answers:

<img width="900" alt="QnA_fastRAG_5" src="https://github.com/user-attachments/assets/2d11397d-8699-4fa9-ab66-4995700ab0ae">

