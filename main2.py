
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from threading import Lock
from collections import defaultdict
import uuid

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

_embeddings = None
_llm = None

# (video_id, language) -> FAISS vector store
vectorstores = {}

# (video_id, language) -> original transcript segments
transcript_segments = {}

# Loading progress, indexed by request ID
progress = defaultdict(dict)

data_lock = Lock()
progress_lock = Lock()


# --------------------------------------------------
# Progress helpers
# --------------------------------------------------

def update_progress(request_id, stage, detail, percent):
    if not request_id:
        return

    with progress_lock:
        progress[request_id] = {
            "stage": stage,
            "detail": detail,
            "percent": percent,
        }


# --------------------------------------------------
# YouTube helpers
# --------------------------------------------------

def get_youtube_id(url: str):
    """Extract a YouTube video ID from a URL or bare ID."""

    url = (url or "").strip()

    if not url:
        return None

    # Accept a bare video ID.
    if "://" not in url:
        if "/" not in url and " " not in url:
            return url
        return None

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/").split("/")[0] or None

    if host in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    ):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        if parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            return parts[1] if len(parts) > 1 else None

    return None


def normalize_segments(fetched):
    """
    Normalize transcript entries from different versions
    of youtube-transcript-api into dictionaries.
    """

    normalized = []

    for item in fetched:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start", 0)
            duration = item.get("duration", 0)
        else:
            text = getattr(item, "text", "")
            start = getattr(item, "start", 0)
            duration = getattr(item, "duration", 0)

        text = str(text).strip()

        if not text:
            continue

        normalized.append({
            "text": text,
            "start": float(start or 0),
            "duration": float(duration or 0),
        })

    return normalized


def fetch_transcript(video_id: str, language: str):
    """Fetch transcript segments while preserving their timestamps."""

    # Support the older youtube-transcript-api interface.
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        fetched = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=[language],
        )
    else:
        # Support the newer instance-based interface.
        api = YouTubeTranscriptApi()
        fetched = api.fetch(
            video_id,
            languages=[language],
        )

    return normalize_segments(fetched)


def format_timestamp(seconds):
    seconds = max(0, int(float(seconds or 0)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


# --------------------------------------------------
# Transcript chunking
# --------------------------------------------------

def make_timestamped_chunks(segments, video_id, language):
    """
    Create searchable chunks while retaining the start
    and end timestamps for each chunk.
    """

    chunks = []
    current_segments = []
    current_length = 0

    chunk_size = 1000
    overlap_chars = 200

    for segment in segments:
        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        start = float(segment.get("start", 0))
        duration = float(segment.get("duration", 0))
        end = start + duration

        if current_segments and current_length + len(text) > chunk_size:
            chunks.append(
                create_chunk(
                    current_segments,
                    video_id,
                    language,
                )
            )

            # Retain some previous text for retrieval overlap.
            overlap = []
            overlap_length = 0

            for item in reversed(current_segments):
                overlap.insert(0, item)
                overlap_length += len(item["text"])

                if overlap_length >= overlap_chars:
                    break

            current_segments = overlap
            current_length = sum(
                len(item["text"]) for item in current_segments
            )

        current_segments.append({
            "text": text,
            "start": start,
            "end": end,
        })

        current_length += len(text)

    if current_segments:
        chunks.append(
            create_chunk(
                current_segments,
                video_id,
                language,
            )
        )

    return chunks


def create_chunk(segments, video_id, language):
    start = segments[0]["start"]
    end = segments[-1]["end"]

    content = (
        f"[Timestamp: {format_timestamp(start)}]\n"
        + " ".join(segment["text"] for segment in segments)
    )

    return Document(
        page_content=content,
        metadata={
            "video_id": video_id,
            "language": language,
            "start_time": start,
            "end_time": end,
        },
    )


# --------------------------------------------------
# AI model helpers
# --------------------------------------------------

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings


def get_llm():
    global _llm

    if _llm is None:
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.2,
        )

    return _llm


PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a YouTube video.

Answer using ONLY the transcript context below.

Rules:
1. Cite relevant timestamps using [MM:SS].
2. Use only timestamps present in the supplied context.
3. Place a timestamp after the claim it supports.
4. Cite separate timestamps for claims from different parts.
5. Never invent a timestamp.
6. If the transcript does not contain the answer, say you
   could not find the answer in the video.
7. Answer clearly and directly.

Transcript context:
{context}

Question:
{question}

Answer:
""")


# --------------------------------------------------
# Frontend route
# --------------------------------------------------

@app.route("/")
def index():
    """Serve index2.html as the main page."""

    return send_from_directory(
        str(BASE_DIR),
        "index2.html",
    )


# --------------------------------------------------
# Load progress API
# --------------------------------------------------

@app.get("/api/progress")
def get_progress():
    request_id = request.args.get("request_id", "")

    with progress_lock:
        current = progress.get(request_id)

    if not current:
        return jsonify(
            stage="starting",
            detail="Preparing your request...",
            percent=0,
        )

    return jsonify(current)


# --------------------------------------------------
# Load video API
# --------------------------------------------------

@app.post("/api/load")
def load_video():
    body = request.get_json(silent=True) or {}

    url = (body.get("url") or "").strip()
    language = (
        body.get("language") or "en"
    ).strip().lower()

    # Support HTML pages that do not send a request ID.
    request_id = (
        body.get("request_id") or str(uuid.uuid4())
    ).strip()

    video_id = get_youtube_id(url)

    if not video_id:
        return jsonify(
            error="Please enter a valid YouTube video URL or video ID."
        ), 400

    index_key = (video_id, language)

    try:
        # Reuse the index when the same video and language
        # have already been loaded.
        with data_lock:
            cached = index_key in vectorstores

        if cached:
            update_progress(
                request_id,
                "ready",
                "Using the existing searchable index.",
                100,
            )

            return jsonify(
                video_id=video_id,
                language=language,
                message="Video loaded successfully.",
                cached=True,
            )

        # Step 1: Fetch transcript.
        update_progress(
            request_id,
            "fetching",
            "Fetching transcript and timestamps...",
            10,
        )

        segments = fetch_transcript(video_id, language)

        if not segments:
            update_progress(
                request_id,
                "error",
                "The transcript is empty.",
                0,
            )

            return jsonify(
                error="The transcript in this language is empty."
            ), 400

        # Save original segments for the transcript panel.
        # The UI can now request /api/transcript.
        with data_lock:
            transcript_segments[index_key] = segments

        # Step 2: Create timestamped chunks.
        update_progress(
            request_id,
            "processing",
            "Preparing transcript chunks...",
            30,
        )

        chunks = make_timestamped_chunks(
            segments,
            video_id,
            language,
        )

        if not chunks:
            return jsonify(
                error="No usable transcript text was found."
            ), 400

        # Step 3: Generate embeddings.
        update_progress(
            request_id,
            "embedding",
            f"Generating embeddings for {len(chunks)} chunks...",
            45,
        )

        embeddings = get_embeddings()

        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        vectors = embeddings.embed_documents(texts)

        # Step 4: Build the FAISS index.
        update_progress(
            request_id,
            "indexing",
            "Building the searchable index...",
            80,
        )

        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, vectors)),
            embedding=embeddings,
            metadatas=metadatas,
        )

        with data_lock:
            vectorstores[index_key] = vectorstore

        update_progress(
            request_id,
            "ready",
            f"Video ready. Indexed {len(chunks)} chunks.",
            100,
        )

        return jsonify(
            video_id=video_id,
            language=language,
            message="Video loaded successfully.",
            chunks=len(chunks),
        )

    except TranscriptsDisabled:
        update_progress(
            request_id,
            "error",
            "Transcripts are disabled for this video.",
            0,
        )

        return jsonify(
            error="Transcripts are disabled for this video."
        ), 400

    except NoTranscriptFound:
        update_progress(
            request_id,
            "error",
            "No transcript was found in the selected language.",
            0,
        )

        return jsonify(
            error=(
                f"No transcript was found in '{language}'. "
                "Try another language available for this video."
            )
        ), 400

    except Exception as e:
        update_progress(
            request_id,
            "error",
            "Video loading failed.",
            0,
        )

        app.logger.exception("Error while loading video")

        return jsonify(
            error=f"Could not load video: {str(e)}"
        ), 500


# --------------------------------------------------
# Transcript API - used by the transcript panel
# --------------------------------------------------

@app.get("/api/transcript")
def get_transcript():
    video_id = (request.args.get("video_id") or "").strip()
    language = (
        request.args.get("language") or "en"
    ).strip().lower()

    if not video_id:
        return jsonify(
            error="Missing video_id."
        ), 400

    index_key = (video_id, language)

    with data_lock:
        segments = transcript_segments.get(index_key)

    if segments is None:
        return jsonify(
            error="Load this video in the selected language first.",
            segments=[],
        ), 404

    # Return the exact field names expected by index2.html.
    return jsonify(
        video_id=video_id,
        language=language,
        segments=[
            {
                "text": segment["text"],
                "start": segment["start"],
                "duration": segment["duration"],
                "timestamp": format_timestamp(segment["start"]),
            }
            for segment in segments
        ],
    )


# --------------------------------------------------
# Ask a question API
# --------------------------------------------------

@app.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}

    video_id = (body.get("video_id") or "").strip()
    language = (
        body.get("language") or "en"
    ).strip().lower()
    question = (body.get("question") or "").strip()

    if not video_id:
        return jsonify(
            error="Please load a video first."
        ), 400

    if not question:
        return jsonify(
            error="Please type a question."
        ), 400

    index_key = (video_id, language)

    with data_lock:
        vectorstore = vectorstores.get(index_key)

    if vectorstore is None:
        return jsonify(
            error="Load the video in the selected language first."
        ), 400

    try:
        # Retrieve the most relevant transcript chunks.
        docs = vectorstore.similarity_search(
            question,
            k=4,
        )

        if not docs:
            return jsonify(
                answer="I couldn't find relevant information in this video."
            )

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        result = (PROMPT | get_llm()).invoke({
            "context": context,
            "question": question,
        })

        answer = result.content

        # Handle models that return multiple text blocks.
        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
                for item in answer
            )

        return jsonify(
            answer=str(answer),
        )

    except Exception as e:
        app.logger.exception("Error while answering question")

        return jsonify(
            error=f"Something went wrong: {str(e)}"
        ), 500


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000,
        threaded=True,
        use_reloader=False,
    )
