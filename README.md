# LangChain과 RAG란 무엇일까? MEETUP 

## 구성

### LangChain_meetup.ipynb
- RAG와 LangChain이 무엇인지 공부하는 노트북 파일 입니다.
- OpenAI 와 Ollama 로 예제가 구성되어 있습니다.

### app.py
- Streamlit을 활용하여 OpenAI의 모델을 활용한 RAG 웹 애플리케이션
- 파일 업로드 기능 (PDF / docx /pptx ).
- OpenAI API 키 입력을 통해 LLM 사용
- 텍스트 처리, 벡터 스토어에 저장, 챗봇 인터페이스를 통한 질의 기능.

### streamlit app 실행방법 
```bash
streamlit run app.py
```