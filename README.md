# llm_with_fastRAG

fastRAG is a research framework for efficient and optimized retrieval augmented generative pipelines, incorporating state-of-the-art LLMs and Information Retrieval. fastRAG is designed to empower researchers and developers with a comprehensive tool-set for advancing retrieval augmented generation.

Key Features:

Optimized RAG: Build RAG pipelines with SOTA efficient components for greater compute efficiency.
Optimized for Intel Hardware: Leverage Intel extensions for PyTorch (IPEX), 🤗 Optimum Intel and 🤗 Optimum-Habana for running as optimal as possible on Intel® Xeon® Processors and Intel® Gaudi® AI accelerators.
Customizable: fastRAG is built using Haystack and HuggingFace. All of fastRAG's components are 100% Haystack compatible.

What is the differences between Retrieval Augmented Generation vs Pre-Training vs Fine Tuning?

![RAG_vs_Fine_Tuning](https://github.com/allenwsh82/llm_with_fastRAG/assets/44453417/dead4bd6-f317-454b-a074-a15e3ac8b267)

![tco](https://github.com/allenwsh82/llm_with_fastRAG/assets/44453417/b67059ee-f45d-4c5f-bad2-99dab9a33328)



How to run the Question-Answer with fastRAG Demo:
1) Clone the project: git clone https://github.com/allenwsh82/llm_with_fastRAG.git
2) Create a new environment for this project: python -m venv rag_env
3) Activate the environment: source rag_evn/bin/activate
4) Setup the environment with all the dependencies: pip insstall -r requirements.txt
5) You can create your own .csv file or download a .csv format file from this link:
   https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/small_generator_dataset.csv.zip
6) Then execute the program by running the following command:
   python Generative_QA_Haystack_PromptNode_CSV_database_llama2_Demo.py
7) 

<img width="960" alt="QnA_fastRAG_1" src="https://github.com/user-attachments/assets/83cbdfbf-b946-4b37-ad37-b8250789f538">
