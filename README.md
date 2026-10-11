
# 🎬 YouTube Video Q&A

### Turn YouTube videos into interactive AI-powered learning experiences.

Ask questions about any supported YouTube video, explore its transcript, jump to specific timestamps, and get AI-generated answers grounded in the video's content.

<p align="center">
  <b>Python · Flask · LangChain · FAISS · Groq API</b>
</p>

---

## ✨ Features

- 🎥 **YouTube Integration** — Load videos using their URLs.
- 📝 **Timestamped Transcripts** — View transcript segments with timestamps.
- 🔎 **Transcript Navigation** — Search transcript text and navigate to specific moments.
- 🤖 **AI-Powered Q&A** — Ask questions and get context-aware answers.
- ⏱️ **Timestamp-Based Answers** — Find relevant moments in the original video.
- 🧠 **Semantic Search** — Retrieve relevant transcript chunks using vector embeddings.
- ⚡ **FAISS Vector Search** — Efficient similarity search over transcript content.
- 🌐 **Interactive Web Interface** — A frontend built with HTML, CSS, and JavaScript.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web server and API |
| LangChain | LLM orchestration and retrieval |
| HuggingFace Embeddings | Convert transcript text into vectors |
| FAISS | Similarity search |
| Groq API | AI-powered question answering |
| YouTube Transcript API | Retrieve video transcripts |
| HTML, CSS, JavaScript | Frontend and user interface |

---

## 🔄 How It Works

1. **Load a video** — Enter a supported YouTube URL.
2. **Fetch the transcript** — Retrieve transcript segments and timestamps.
3. **Generate embeddings** — Convert transcript chunks into vector representations.
4. **Build the search index** — Store the vectors in FAISS.
5. **Ask a question** — Enter a question about the video.
6. **Retrieve relevant context** — Find transcript chunks related to the question.
7. **Generate an answer** — Use the Groq API to answer using the retrieved transcript context, with timestamps where available.

---

## 📁 Project Structure

```text
youtube-video-qa/
│
├── main.py             # Flask backend and API routes
├── index2.html         # Frontend user interface
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment configuration
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

> **Note:** Your actual project structure may contain additional files.

---

## 🚀 Getting Started

### Prerequisites

Before running the project, make sure you have:

- Python 3.10 or a compatible version for your installed dependencies
- Git
- A Groq API key

### 1. Clone the Repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/youtube-video-qa.git
cd youtube-video-qa
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from the [Groq Console](https://console.groq.com/keys).

**Important:** Replace the placeholder with your own API key. Never commit your actual `.env` file to GitHub.

### 5. Run the Application

```bash
python main.py
```

### 6. Open in Your Browser

Visit:

**http://127.0.0.1:5000**

Load a YouTube video and start asking questions!

---

## 🔐 Security

- Never upload `.env` files or API keys to GitHub.
- Keep `.env` in your `.gitignore`.
- Use your hosting provider's secret or environment-variable settings when deploying.
- If an API key is exposed, revoke it and create a replacement.
- Do not share private credentials in screenshots, logs, or public issues.

---

## ⚠️ Notes and Limitations

- A video must have an accessible transcript in the selected language.
- Transcript retrieval may fail when captions are unavailable or access is restricted.
- AI answers depend on the transcript content retrieved for the question.
- The current implementation keeps transcript data and FAISS indexes in memory, so they are cleared when the backend restarts.
- Groq API usage is subject to account limits and the provider's terms.

---

## 🔮 Future Improvements

- [ ] Support additional transcript languages.
- [ ] Add persistent storage for vector indexes.
- [ ] Improve transcript filtering and search.
- [ ] Add chat history for each video.
- [ ] Deploy the application online.
- [ ] Add automated tests and error handling.

---

## 👨‍💻 Author

**YOUR NAME**

- GitHub: [Akul0003](https://github.com/Akul0003)

---

## 📄 License

This project is available under the MIT License if you choose to distribute it under that license. Add a `LICENSE` file to the repository before claiming that the project is licensed under MIT.

---

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>
