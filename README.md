# 🎓 Campus AI — Intelligent College Assistant

An AI-powered college assistant built using **LangGraph, Retrieval-Augmented Generation (RAG), FAISS, Hugging Face embeddings, and Groq LLMs**.

Campus AI helps students find information about academic regulations, attendance, examinations, fee structures, and general college-related queries through a conversational interface.

## ✨ Features

* **Conditional RAG:** Automatically routes queries to the appropriate knowledge source.
* **Academic Assistant:** Retrieves information from the academic handbook.
* **Fee Assistant:** Answers fee-related questions using college fee documents.
* **General Chat:** Handles greetings and general questions without document retrieval.
* **Semantic Search:** Uses Hugging Face embeddings and FAISS for document retrieval.
* **Student Personalization:** Supports BCA, BBA, and B.Com (H) programmes.
* **Interactive UI:** Built with Streamlit, including chat history and quick-question suggestions.
* **Modular Architecture:** Uses LangGraph nodes and conditional edges to organize the workflow.

## 🛠️ Tech Stack

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Core application             |
| LangGraph    | Workflow orchestration       |
| LangChain    | LLM and document integration |
| Groq         | Language model inference     |
| Hugging Face | Text embeddings              |
| FAISS        | Vector similarity search     |
| Streamlit    | Web interface                |
| PyPDF        | PDF document loading         |
| uv           | Python dependency management |

## 🏗️ Architecture

```text
                 Student Query
                       |
                       v
                  Classifier
                       |
             +---------+---------+
             |         |         |
             v         v         v
        Academic      Fee      General
           RAG        RAG       Chat
             |         |         |
             +---------+---------+
                       |
                       v
                 Response Node
                       |
                       v
                  Final Answer
```

## 📁 Project Structure

```text
Langraph/
├── app.py
├── academics_handbook.pdf
├── fee_structure.pdf
├── requirements.txt
├── README.md
├── .gitignore
└── .env                 # Not committed
```

The PDF filenames in the code must match the actual filenames in your project.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/campus-ai.git
cd campus-ai
```

### 2. Install uv

Install uv by following the official installation instructions.

### 3. Create the environment

```bash
uv venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Obtain an API key through the official Groq Console.

Never commit your API key to GitHub.

### 6. Add the required documents

Place your academic handbook and fee structure PDFs in the project root.

Make sure their filenames match the paths configured in `app.py`.

### 7. Run the application

```bash
uv run --active streamlit run app.py
```

Open the local URL printed in the terminal.

## 🔍 How It Works

1. The student submits a question through the Streamlit interface.
2. LangGraph classifies the question as academic, fee-related, or general.
3. Academic and fee queries are routed to their corresponding FAISS retrievers.
4. Relevant document chunks are included in the LLM prompt.
5. General queries bypass document retrieval.
6. The response node generates an answer and returns it to the interface.

## 🔒 Security

* API credentials are stored in environment variables.
* `.env` and `.venv` are excluded through `.gitignore`.
* Avoid committing private student records or confidential institutional documents.

## 🚀 Future Improvements

* Add conversation-aware retrieval.
* Display document citations and page numbers.
* Persist FAISS indexes to reduce startup time.
* Add automated tests for routing and retrieval.
* Deploy the application to a cloud platform.
* Add document upload and source selection.
* Improve classification accuracy and retrieval evaluation.

## 👩‍💻 Author

**Sneha Choudhary**

AI & Machine Learning Engineering Student

GitHub: https://github.com/sneha140305
