---
language:
- ko
license: mit
---


# Allganize RAG Leaderboard
Allganize RAG 리더보드는 5개 도메인(금융, 공공, 의료, 법률, 커머스)에 대해서 한국어 RAG의 성능을 평가합니다.  
일반적인 RAG는 간단한 질문에 대해서는 답변을 잘 하지만, 문서의 테이블과 이미지에 대한 질문은 답변을 잘 못합니다.  

RAG 도입을 원하는 수많은 기업들은 자사에 맞는 도메인, 문서 타입, 질문 형태를 반영한 한국어 RAG 성능표를 원하고 있습니다.  
평가를 위해서는 공개된 문서와 질문, 답변 같은 데이터 셋이 필요하지만, 자체 구축은 시간과 비용이 많이 드는 일입니다.  
이제 올거나이즈는 RAG 평가 데이터를 모두 공개합니다. 

RAG는 Parser, Retrieval, Generation 크게 3가지 파트로 구성되어 있습니다.  
현재, 공개되어 있는 RAG 리더보드 중, 3가지 파트를 전체적으로 평가하는 한국어로 구성된 리더보드는 없습니다.

Allganize RAG 리더보드에서는 문서를 업로드하고, 자체적으로 만든 질문을 사용해 답변을 얻었습니다.  
생성한 답변과 정답 답변을 자동 성능 평가 방법을 적용해 각 RAG 방법별 성능 측정을 했습니다.  


# RAG Benchmark
| RAG | 금융 | 공공 | 의료 | 법률 | 커머스 | Average |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Alli (claude3.5-sonnet) | **0.85 (51/60)** | **0.983 (59/60)** | 0.85 (51/60) | **0.767 (46/60)** | 0.783 (47/60) | **0.847 (254/300)** | 
| Alli (claude3-opus) | 0.817 (49/60) | 0.95 (57/60) | **0.9 (54/60)** | 0.75 (45/60) | 0.767 (46/60) | 0.837 (251/300) | 
| Alli (gpt-4o) | 0.8 (48/60) | 0.9 (54/60) | 0.817 (49/60) | 0.683 (41/60) | 0.717 (43/60) | 0.783 (235/300) | 
| Alli (gpt-4) | 0.833 (50/60) | 0.85 (51/60) | 0.733 (44/60) | 0.733 (44/60) | 0.733 (44/60) | 0.777 (233/300) | 
| Alli (gpt-4-turbo) | 0.783 (47/60) | 0.9 (54/60) | 0.733 (44/60) | 0.717 (43/60) | 0.733 (44/60) | 0.773 (232/300) | 
| Alli (alpha-ko-202411-32B) | 0.8 (48/60) | 0.85 (51/60) | 0.75 (45/60) | 0.717 (43/60) | 0.733 (44/60) | 0.77 (231/300) | 
| Alli (gpt-4o-mini) | 0.75 (45/60) | 0.883 (53/60) | 0.7 (42/60) | 0.733 (44/60) | 0.75 (45/60) | 0.763 (229/300) | 
| Upstage (gpt-4-turbo) | 0.617 (37/60) | 0.85 (51/60) | 0.833 (50/60) | 0.6 (36/60) | **0.817 (49/60)** | 0.743 (223/300) | 
| OpenAI Assistant (gpt-4-turbo) | 0.533 (32/60) | 0.883 (53/60) | 0.733 (44/60) | 0.733 (44/60) | 0.783 (47/60) | 0.733 (220/300) | 
| OpenAI Assistant (gpt-4) | 0.717 (43/60) | 0.783 (47/60) | 0.767 (46/60) | 0.517 (31/60) | 0.75 (45/60) | 0.707 (212/300) | 
| Upstage (gpt-4) | 0.6 (36/60) | 0.783 (47/60) | 0.75 (45/60) | 0.583 (35/60) | 0.783 (47/60) | 0.7 (210/300) | 
| Alli (Llama-3-Alpha-Ko-8B-Instruct-Pro) | 0.683 (41/60) | 0.767 (46/60) | 0.633 (38/60) | 0.583 (35/60) | 0.7 (42/60) | 0.673 (202/300) | 
| Alli ([KONI-Llama3-8B-Instruct-20240729](https://huggingface.co/KISTI-KONI/KONI-Llama3-8B-Instruct-20240729)) | 0.683 (41/60) | 0.7 (42/60) | 0.533 (32/60) | 0.567 (34/60) | 0.75 (45/60) | 0.647 (194/300) | 
| Upstage (solar) | 0.6 (36/60) | 0.683 (41/60) | 0.733 (44/60) | 0.433 (26/60) | 0.717 (43/60) | 0.633 (190/300) | 
| Langchain (gpt-4-turbo) | 0.617 (37/60) | 0.517 (31/60) | 0.667 (40/60) | 0.567 (34/60) | 0.683 (41/60) | 0.61 (183/300) | 
| Cohere (command-r-plus) | 0.483 (29/60) | 0.65 (39/60) | 0.433 (26/60) | 0.517 (31/60) | 0.683 (41/60) | 0.553 (166/300) | 
| Cohere (command-r) | 0.5 (30/60) | 0.633 (38/60) | 0.417 (25/60) | 0.533 (32/60) | 0.667 (40/60) | 0.55 (165/300) | 
| Upstage (gpt-3.5-turbo) | 0.5 (30/60) | 0.517 (31/60) | 0.567 (34/60) | 0.417 (25/60) | 0.617 (37/60) | 0.523 (157/300) | 
| Alli ([Llama-3-Alpha-Ko-8B-Instruct](https://huggingface.co/allganize/Llama-3-Alpha-Ko-8B-Instruct)) | 0.533 (32/60) | 0.55 (33/60) | 0.533 (32/60) | 0.417 (25/60) | 0.55 (33/60) | 0.517 (155/300) | 
| Langchain (gpt-3.5-turbo) | 0.4 (24/60) | 0.333 (20/60) | 0.417 (25/60) | 0.35 (21/60) | 0.467 (28/60) | 0.393 (118/300) | 
| Anything LLM (gpt-4-turbo) | 0.267 (16/60) | 0.067 (4/60) | 0.55 (33/60) | 0.283 (17/60) | 0.283 (17/60) | 0.29 (87/300) | 
| Anything LLM (claude3-opus) | 0.267 (16/60) | 0.067 (4/60) | 0.55 (33/60) | 0.317 (19/60) | 0.45 (27/60) | 0.33 (99/300) | 
| Anything LLM (gpt-3.5-turbo) | 0.133 (8/60) | 0.033 (2/60) | 0.233 (14/60) | 0.15 (9/60) | 0.233 (14/60) | 0.157 (47/300) | 



# Auto Evaluate
총 4개의 LLM Eval을 사용하여 평가한 후, voting 하여 "O" 혹은 "X"를 결정했습니다.
- TonicAI : answer_similarity (threshold=4)
- MLflow : answer_similarity/v1/score (threshold=4)
- MLflow : answer_correctness/v1/score (threshold=4)
- Allganize Eval : answer_correctness/claude3-opus

LLM 기반 평가 방법이기 때문에, 오차율이 존재합니다.  
Finance 도메인을 기반으로 사람이 평가한 것과 오차율을 비교하였을 때, 약 8%의 오차율을 보였습니다.  
Colab에 Auto Evaluate를 사용할 수 있게 정리하였습니다.  
- [Colab](https://colab.research.google.com/drive/1c9hH429iAqw4xkgKoQq1SC9f_4p_nwcc?usp=sharing)


# Dataset

### Domain
다양한 도메인 중, 다섯개를 선택해 성능 평가를 진행했습니다.    
- finance(금융)
- public(공공)
- medical(의료)
- law(법률)
- commerce(커머스)


### Documents
도메인별로 PDF 문서를 수집하여 질문들을 생성했습니다.  
각 도메인별 문서의 페이지 수 총합이 2~300개가 되도록 문서들을 수집했습니다.  
각 문서의 이름, 페이지 수, 링크 또한 [documents.csv](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO/blob/main/documents.csv) 파일을 다운받으면 확인하실 수 있습니다.  
각 도메인별 pdf 문서 갯수는 다음과 같습니다.
- finance: 10개 (301 page)
- public: 12개 (258 page)
- medical: 20개 (276 page)
- law: 12개 (291 page)
- commerce: 9개 (211 page)


### Question and Target answer
문서의 페이지 내용을 보고 사용자가 할만한 질문 및 답변들을 만들었습니다.  
각 도메인별로 60개의 질문들을 가지고 있습니다.  


### Context type
문서의 페이지를 보고 여기에서 나올 수 있는 질문들을 생성했습니다.  
이때 질문에 대한 근거가 문단(paragraph)인지, 테이블(table)인지, 이미지(image)인지를 구분했습니다.  
각 질문별 근거 유형을 context_type이라 하여 컬럼을 추가해두었습니다.  
각 도메인별 context_type의 비율은 문서의 페이지에 등장한 빈도수를 반영해 설정했습니다. (ex. 금융 도메인 문서 210, 테이블 127, 이미지26)  
도메인별 context_type의 비율은 다음과 같습니다.  

| domain   | paragraph | table    | image    |
| :--------: | :---------: | :--------: | :--------: |
| finance  | 30 (50%)  | 10 (17%) | 20 (33%) |
| public   | 40 (67%)  | 15 (25%) | 5 (8%)   |
| medical  | 45 (75%)  | 5 (8%) | 10 (17%)   |
| law      | 40 (67%)  | 15 (25%) | 5 (8%)   |
| commerce | 38 (64%)  | 5 (8%) | 17 (28%)   |



# RAG Solution
### Alli
Alli는 Allganize의 RAG 솔루션입니다.  
Parser는 page 단위로 Allganize Parser를 사용해 구현했습니다.  
Retrieval는 Hybrid Search를 사용해 구현했습니다.  
Generation은 OpenAI, Cluade, Allganize에서 만든 금융모델 등 간단하게 선택해서 사용할 수 있습니다.
- [Allganize](https://www.allganize.ai/ko/home)


### LangChain
LangChain은 LLM으로 구동되는 애플리케이션을 개발하기 위한 프레임워크입니다.  
LangChain RAG Quick Start를 기반으로 성능을 평가했습니다.  
Parser는 pypdf를 사용했습니다.  
chunk size와 overlap은 튜토리얼에 나와있는데로 1000과 200으로 설정했습니다.  
Retrieval은 OpenAI Embedding을 사용했습니다.  
Generation은 Langchain에서 지원하는 모델을 자유롭게 사용할 수 있습니다.  
- [LangChain Tutorial](https://python.langchain.com/v0.1/docs/use_cases/question_answering/quickstart/)
- [Colab](https://colab.research.google.com/drive/1Jlzs8ZqFOqqIBBT2T5XGBhr23XxEsvHb?usp=sharing)


### OpenAI Assistant
OpenAI Assistant는 File Search, Code Interperter 같은 특정 기능을 지원하는 툴입니다.  
문서를 업로드할 수 있으며, 자체 vector stores에 저장됩니다.  
질문을 입력하면 vector stores에서 관련된 chunk를 가져와 모델에 입력해 답변을 출력합니다.  
어떤 chunk를 사용했는지 citation이 달리며 확인할 수 있습니다.
- [OpenAI](https://platform.openai.com/docs/assistants/tools/file-search/quickstart)
- [Colab](https://colab.research.google.com/drive/1Ag3ylvk3oucQsOPorjgc1C8qZ4JFrJgu?usp=sharing)


### Cohere
Cohere에서는 text embedding 모델과 generation 모델을 제공하고 있습니다.  
Parser로 Cohere에는 문서를 업로드하고 파싱하는 기능은 없어서 Langchain의 기본 parser를 사용했습니다.  
chunk_size는 500으로 overlap은 200으로 설정했습니다.  
Cohere의 임베딩 최대 길이가 512 토큰이라 상대적으로 짧기 때문에 짧게 설정했습니다.  
Retrieval는 `embed-multilingual-v3.0`을 사용했습니다.  
Generation은 `command-r`과 `command-r-plus`를 사용해 성능을 평가했습니다.
- [Cohere](https://cohere.com/command)
- [Colab](https://colab.research.google.com/drive/1QwozvB-SCeeHhRe6MmlnCETw3bGu9SJe?usp=sharing)


### Anything LLM
Anything LLM은 사용하고 싶은 LLM과 벡터DB를 선택하여 RAG 파이프라인을 로컬에 구축할 수 있는 프로그램입니다.  
문서들을 "Workspace" 라는 개체로 구분합니다. 각 Workspace에 업로드된 문서들만을 대상으로 대화를 수행합니다.  
프로그램을 다운로드하여 사용할 수도 있고, github 코드를 clone하여 docker compose로 실행할 수도 있습니다.  
Parser와 Retrieval는 Anything LLM 자체 방법으로 구현되어 있습니다.  
Generation model은 OpenAI나 Anthropic 모델을 API key만 등록하면 사용할 수 있습니다.  
- [Github link](https://github.com/Mintplex-Labs/anything-llm)
- [Download link](https://useanything.com/download)


### Upstage
Upstage에서는 text embedding 모델과 generation 모델을 제공하고 있습니다.  
Parser로 Upstage에는 문서를 업로드하고 파싱하는 기능은 없어서 Langchain의 기본 parser를 사용했습니다.  
chunk size와 overlap은 튜토리얼에 나와있는데로 1000과 200으로 설정했습니다.  
Retrieval는 `solar-embedding-1-large`를 사용했습니다.  
Generation은 `solar-1-mini-chat`을 사용해 성능을 평가했습니다.  
`gpt4-turbo`, `gpt4`, `gpt3.5-turbo`는 임베딩만 `solar-embedding-1-large`를 사용해서 성능 평가한 방법입니다.  
- [Upstage](https://developers.upstage.ai/docs/apis/embeddings)
- [Colab](https://colab.research.google.com/drive/1JE2IXCACSkWeGiu9xvG8kmr0jmtzVzB1?usp=sharing)

<br>  

# Contributor
- Junghoon Lee (junghoon.lee@allganize.ai)  
- Sounghan Kim (sounghan.kim@allganize.ai)  
- Yujung Kim (yujung.kim@allganize.ai)


# History Note
### 2024.08.09 
- Auto Evaluate를 5개에서 4개로 변경.
- 모델 추가 : Alli (gpt-4o-mini), Alli (KONI-Llama3-8B-Instruct-20240729), Alli (Llama-3-Ko-8B-Finance-Evol), Alli (Llama-3-Alpha-Ko-8B-Instruct)
