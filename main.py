import logging
import os
import re
from urllib.parse import urlparse, parse_qs

from flask import Flask, jsonify, request
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from google import genai
from google.genai import types
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Vercel detects this top-level Flask WSGI application.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB JSON request limit
logging.basicConfig(level=logging.INFO)

_embedding_client = None
_llm = None

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a YouTube video.

Answer using ONLY the transcript context below.

Rules:
1. Cite relevant timestamps using [MM:SS].
2. Use only timestamps present in the supplied context.
3. Place a timestamp after the claim it supports.
4. Cite separate timestamps for claims from different parts.
5. Never invent a timestamp.
6. If the transcript does not contain the answer, say you could not find the answer in the video.
7. Answer clearly and directly.

Transcript context:
{context}

Question:
{question}

Answer:
""")


def get_youtube_id(url: str):
    """Extract a YouTube video ID from a URL or bare ID."""
    url = (url or "").strip()
    if not url:
        return None

    if "://" not in url:
        if "/" not in url and " " not in url and len(url) == 11:
            return url
        return None

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/").split("/")[0] or None

    if host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            return parts[1] if len(parts) > 1 else None
    return None


def normalize_segments(fetched):
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
        if text:
            normalized.append({
                "text": text,
                "start": float(start or 0),
                "duration": float(duration or 0),
            })
    return normalized


def fetch_transcript(video_id: str, language: str):
    """Support both old and current youtube-transcript-api interfaces."""
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    else:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=[language])
    return normalize_segments(fetched)


def format_timestamp(seconds):
    seconds = max(0, int(float(seconds or 0)))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def make_timestamped_chunks(segments):
    """Create text chunks and retain the timestamp of the first segment."""
    chunks = []
    current = []
    current_length = 0

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = float(segment.get("start", segment.get("start_time", 0)) or 0)
        duration = float(segment.get("duration", 0) or 0)

        if current and current_length + len(text) > CHUNK_SIZE:
            chunks.append(_make_chunk(current))
            overlap = []
            overlap_length = 0
            for item in reversed(current):
                overlap.insert(0, item)
                overlap_length += len(item["text"])
                if overlap_length >= CHUNK_OVERLAP:
                    break
            current = overlap
            current_length = sum(len(item["text"]) for item in current)

        current.append({"text": text, "start": start, "duration": duration})
        current_length += len(text)

    if current:
        chunks.append(_make_chunk(current))
    return chunks


def _make_chunk(segments):
    start = segments[0]["start"]
    body = " ".join(item["text"] for item in segments)
    return {
        "text": f"[Timestamp: {format_timestamp(start)}]\n{body}",
        "start": start,
    }


def get_embedding_client():
    global _embedding_client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in Vercel Environment Variables.")
    if _embedding_client is None:
        _embedding_client = genai.Client(api_key=api_key)
    return _embedding_client


def embed_text(text, task_type):
    response = get_embedding_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return response.embeddings[0].values


def get_llm():
    global _llm
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured in Vercel Environment Variables.")
    if _llm is None:
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            api_key=api_key,
        )
    return _llm


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_relevant_chunks(question, chunks):
    """Embed the question and supplied chunks for this request; no server memory/index."""
    if not chunks:
        return []

    # Keep request cost bounded if a very long transcript is supplied.
    chunks = chunks[:250]
    query_vector = embed_text(question, "RETRIEVAL_QUERY")

    scored = []
    for chunk in chunks:
        vector = embed_text(chunk["text"], "RETRIEVAL_DOCUMENT")
        score = cosine_similarity(query_vector, vector)
        scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:TOP_K]]


@app.get("/")
def health():
    return jsonify({
        "name": "ClipMind AI API",
        "status": "running",
        "endpoints": ["/api/load", "/api/ask"],
    })


@app.post("/api/load")
def load_video():
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    language = (body.get("language") or "en").strip().lower()

    video_id = get_youtube_id(url)
    if not video_id:
        return jsonify(error="Please enter a valid YouTube video URL or video ID."), 400

    try:
        segments = fetch_transcript(video_id, language)
        if not segments:
            return jsonify(error="The transcript in this language is empty."), 400

        # Return segments to the browser. No process-local transcript storage.
        return jsonify(
            success=True,
            video_id=video_id,
            language=language,
            message="Transcript loaded successfully.",
            segments=segments,
        )

    except TranscriptsDisabled:
        return jsonify(error="Transcripts are disabled for this video."), 400
    except NoTranscriptFound:
        return jsonify(
            error=f"No transcript was found in '{language}'. Try another language."
        ), 400
    except Exception as exc:
        app.logger.exception("Error fetching YouTube transcript")
        message = str(exc)
        if "IpBlocked" in type(exc).__name__ or "cloud provider" in message.lower():
            return jsonify(
                error=(
                    "YouTube blocked transcript access from this cloud server. "
                    "This is a hosting/IP restriction, not a problem with your video. "
                    "Try again later or add a manual-transcript option."
                )
            ), 502
        return jsonify(error=f"Could not load transcript: {message}"), 500


@app.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}
    video_id = (body.get("video_id") or "").strip()
    question = (body.get("question") or "").strip()
    segments = body.get("transcript")

    if not video_id:
        return jsonify(error="Please load a video first."), 400
    if not question:
        return jsonify(error="Please type a question."), 400
    if not isinstance(segments, list) or not segments:
        return jsonify(error="Transcript is missing. Please load the video again."), 400
    if len(segments) > 10000:
        return jsonify(error="Transcript is too large. Please use a shorter video."), 413

    # Validate and normalize browser-provided transcript data.
    clean_segments = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(item.get("start", item.get("start_time", 0)) or 0)
            duration = float(item.get("duration", 0) or 0)
        except (TypeError, ValueError):
            start, duration = 0.0, 0.0
        clean_segments.append({"text": text, "start": start, "duration": duration})

    if not clean_segments:
        return jsonify(error="No usable transcript text was supplied."), 400

    try:
        chunks = make_timestamped_chunks(clean_segments)
        relevant = retrieve_relevant_chunks(question, chunks)
        if not relevant:
            return jsonify(answer="I couldn't find relevant information in this video.")

        context = "\n\n".join(chunk["text"] for chunk in relevant)
        result = (PROMPT | get_llm()).invoke({
            "context": context,
            "question": question,
        })

        answer = result.content
        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in answer
            )
        return jsonify(answer=str(answer))

    except Exception as exc:
        app.logger.exception("Error while answering question")
        return jsonify(error=f"Something went wrong: {str(exc)}"), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="The transcript request is too large."), 413


if __name__ == "__main__":
    # Local development only. Vercel uses the exported `app` object.
    app.run(host="0.0.0.0", port=5000, debug=True)
