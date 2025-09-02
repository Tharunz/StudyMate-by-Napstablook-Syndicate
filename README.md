# PDF-Powered Academic Assistant 🎓

This project is an AI-powered academic assistant designed to help students interact with their study materials in a conversational Q&A format. It allows users to upload one or more PDFs, such as textbooks, notes, or research papers, and ask natural language questions. The system then provides direct, contextualized answers sourced from the uploaded content, making studying more efficient and engaging.


<img width="1920" height="1080" alt="{764E6CCF-6949-4C31-B77E-AF7C17A2CC30}" src="https://github.com/user-attachments/assets/79ad7860-497e-41ec-815a-546ffc255650" />

<img width="1920" height="1080" alt="{7DF44C69-DE97-46A5-B78E-28F3AF13F3D0}" src="https://github.com/user-attachments/assets/58970a89-00c8-42a7-9078-ebc92b6098c6" />

<img width="1920" height="1080" alt="{1B4F0642-ED5A-4594-9C61-6D98D21F7647}" src="https://github.com/user-attachments/assets/88375c3a-d57a-44a4-a9df-1cab93112ea1" />


---

## Steps to Run.

1. Download the source code.
2. Install python 3.9+
3. Install CUDA 12.9
4. install requirements.txt [pip install -r requirements.txt]
5. Run using streamlit - [ streamlit run app.py ]

Note: The model uses IBM granite v3 2b param's instruct as the LLM and all-MiniLM-L6-v2 as the embedding model.

## Response takes 1-4min approx depending on the query and the pdf content.

[ Model was tested on RTX3050(notebook) 4gb Vram , 32gb ram and i5 11th gen 6 core @2.70GHz ]



## 🌟 Features & Uniqueness

Instead of manually searching through documents, our assistant simplifies the process by grounding its responses in your specific materials. This tailored application of **Retrieval-Augmented Generation (RAG)** is specifically adapted for academic and educational contexts, ensuring that the answers are not only accurate but also verifiable and directly relevant to the content you're studying. This approach promotes a more responsible use of AI by encouraging critical thinking and preventing over-reliance on unverified information.

---

## 📈 Business & Social Impact

### Business Impact

This solution offers a new revenue stream and a unique selling point in the competitive ed-tech market. It can be licensed as software, offered as an in-app purchase for educational websites, or provided as a premium service to institutions. With a relatively short development time, a commercial launch would primarily require the integration of a robust user management and payment system. The primary budget considerations are server and LLM API costs.

### Social Impact

By reducing the time students spend searching for information, our assistant helps them focus on understanding core concepts, making education more accessible and effective. This tool has the potential to transform how students engage with their study materials, fostering a more focused and productive learning experience.

---

## 💻 Technology Architecture

The system is built using **Python** and a modular pipeline that handles document processing, information retrieval, and answer generation. The core technologies include:

* **PyMuPDF**: Used for efficient text extraction from PDF files.
* **SentenceTransformers + FAISS**: Embeds text into vector representations and performs rapid, semantically relevant chunk retrieval based on user queries.
* **IBM Granite**: Generates the final, contextualized answer.
* **Streamlit or Next.js**: Provides a local, interactive user interface.



*Note: The technologies used may be updated in the future to improve performance and functionality.*

---

## 🚀 Scope of Work

The project aims to develop a fully functional, AI-powered PDF-based Q&A system. This involves the complete implementation of the modular pipeline from end-to-end:
1.  **Document Processing**: Handling the ingestion and parsing of PDF files.
2.  **Information Retrieval**: Building the RAG system to efficiently find the most relevant information.
3.  **Answer Generation**: Integrating the LLM to create accurate and contextualized responses.

This scope ensures a robust and reliable study companion that meets the needs of students and educators.
