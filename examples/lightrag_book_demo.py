"""Index a UTF-8 text file with LightRAG and run an interactive query.

This example is intentionally small and explicit so it can be used as a learning
entry point. It supports the OpenAI-compatible services configured in ``.env``
and a dependency-free local embedding fallback for smoke tests.
"""

import argparse
import asyncio
import hashlib
import os
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

# Import after loading .env because LightRAG reads several defaults at import time.
from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.llm.openai import (  # noqa: E402
    openai_complete_if_cache,
    openai_embed,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402


LOCAL_EMBEDDING_DIM = 2048


def configure_console() -> None:
    """Keep Chinese output readable in Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def read_book(path: Path, start_char: int, max_chars: int | None) -> str:
    """Read the source strictly as UTF-8 so bad input fails visibly."""
    text = path.read_text(encoding="utf-8-sig")[start_char:]
    if max_chars is not None:
        text = text[:max_chars]
    if not text.strip():
        raise ValueError(f"The input file is empty: {path}")
    return text


def _text_features(text: str) -> list[str]:
    """Create lexical features that work for both Chinese and Latin text."""
    units = re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())
    bigrams = [f"{left}{right}" for left, right in zip(units, units[1:])]
    return units + bigrams


async def local_hash_embed(texts: list[str]) -> np.ndarray:
    """Return deterministic character n-gram embeddings without extra models."""
    vectors = np.zeros((len(texts), LOCAL_EMBEDDING_DIM), dtype=np.float32)

    for row, value in enumerate(texts):
        for feature in _text_features(value):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            column = int.from_bytes(digest[:4], "little") % LOCAL_EMBEDDING_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vectors[row, column] += sign

        norm = np.linalg.norm(vectors[row])
        if norm:
            vectors[row] /= norm

    return vectors


def make_embedding_func(mode: str) -> EmbeddingFunc:
    if mode == "local":
        return EmbeddingFunc(
            embedding_dim=LOCAL_EMBEDDING_DIM,
            max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
            model_name="local-char-ngram-hash-v1",
            func=local_hash_embed,
        )

    required = (
        "EMBEDDING_BINDING_API_KEY",
        "EMBEDDING_BINDING_HOST",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing remote embedding settings: {', '.join(missing)}")

    return EmbeddingFunc(
        embedding_dim=int(os.environ["EMBEDDING_DIM"]),
        max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
        model_name=os.environ["EMBEDDING_MODEL"],
        supports_asymmetric=True,
        func=partial(
            openai_embed.func,
            model=os.environ["EMBEDDING_MODEL"],
            api_key=os.environ["EMBEDDING_BINDING_API_KEY"].strip(),
            base_url=os.environ["EMBEDDING_BINDING_HOST"].strip(),
        ),
    )


def make_llm_func(model: str):
    book_api_key = os.getenv("BOOK_LLM_API_KEY")
    book_base_url = os.getenv("BOOK_LLM_BINDING_HOST")
    anthropic_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "")

    if book_api_key and book_base_url:
        api_key = book_api_key.strip()
        base_url = book_base_url.strip()
    elif anthropic_token and "api.deepseek.com" in anthropic_base_url.lower():
        # The same DeepSeek token works with its OpenAI-compatible endpoint.
        api_key = anthropic_token.strip()
        base_url = "https://api.deepseek.com"
    else:
        api_key = (
            os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        ).strip()
        base_url = os.getenv("LLM_BINDING_HOST", "").strip()

    if not api_key or not base_url:
        raise RuntimeError(
            "Set BOOK_LLM_API_KEY and BOOK_LLM_BINDING_HOST, or configure the "
            "corresponding LLM_BINDING_* fallback values"
        )

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        # LightRAG may pass this through llm_model_kwargs. This demo deliberately
        # disables chain-of-thought and must avoid forwarding the argument twice.
        kwargs.pop("enable_cot", None)
        is_deepseek = "api.deepseek.com" in base_url.lower()
        if is_deepseek:
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            # DeepSeek currently rejects the Pydantic response schema used by
            # openai_complete_if_cache. Its keyword prompt already requests JSON,
            # which LightRAG validates with json_repair after this call.
            keyword_extraction=keyword_extraction and not is_deepseek,
            api_key=api_key,
            base_url=base_url,
            timeout=int(os.getenv("BOOK_LLM_TIMEOUT", "90")),
            enable_cot=False,
            **kwargs,
        )

    return llm_model_func


def default_llm_model() -> str:
    if os.getenv("BOOK_LLM_MODEL"):
        return os.environ["BOOK_LLM_MODEL"].strip()
    if os.getenv("ANTHROPIC_AUTH_TOKEN") and "api.deepseek.com" in os.getenv(
        "ANTHROPIC_BASE_URL", ""
    ).lower():
        return os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro").strip()
    return "glm-4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        type=Path,
        default=REPO_ROOT / "book.txt",
        help="UTF-8 text file to index",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=REPO_ROOT / "rag_storage" / "book_demo",
        help="Index and cache directory",
    )
    parser.add_argument(
        "--embedding",
        choices=("local", "remote"),
        default="local",
        help="Local lexical smoke-test vectors or the embedding service from .env",
    )
    parser.add_argument(
        "--llm-model",
        default=default_llm_model(),
        help="OpenAI-compatible chat model used for extraction and answers",
    )
    parser.add_argument(
        "--query",
        default="CCMD-3中精神分裂症的主要诊断标准是什么？",
        help="Question asked after indexing",
    )
    parser.add_argument(
        "--mode",
        choices=("naive", "local", "global", "hybrid", "mix"),
        default="naive",
        help="LightRAG retrieval mode",
    )
    parser.add_argument(
        "--start-char",
        type=int,
        default=0,
        help="Start indexing at this character offset; useful for one chapter",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Index only the first N characters; useful for a low-cost smoke test",
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true",
        help="Reuse an existing index and only run the query",
    )
    parser.add_argument(
        "--only-context",
        action="store_true",
        help="Print retrieved evidence without asking the LLM for a final answer",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    book_path = args.book.expanduser().resolve()
    working_dir = args.working_dir.expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"Book: {book_path}")
    print(f"Index: {working_dir}")
    print(f"LLM: {args.llm_model}; embedding: {args.embedding}; mode: {args.mode}")

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=make_llm_func(args.llm_model),
        llm_model_name=args.llm_model,
        llm_model_max_async=2,
        embedding_func=make_embedding_func(args.embedding),
        embedding_func_max_async=4,
        max_parallel_insert=2,
        entity_extract_max_gleaning=0,
        addon_params={
            "language": "Chinese",
            "entity_types": [
                "疾病",
                "症状",
                "诊断标准",
                "病程",
                "治疗",
                "药物",
                "人群",
                "概念",
            ],
        },
    )

    await rag.initialize_storages()
    try:
        if not args.skip_insert:
            text = read_book(book_path, args.start_char, args.max_chars)
            print(
                f"Indexing {len(text):,} characters "
                f"from offset {args.start_char:,}..."
            )
            track_id = await rag.ainsert(text, file_paths=str(book_path))
            documents = await rag.doc_status.get_docs_by_track_id(track_id)
            failures = [
                document
                for document in documents.values()
                if document.status.value == "failed"
            ]
            if failures:
                details = "; ".join(
                    document.error_msg or "unknown indexing error"
                    for document in failures
                )
                raise RuntimeError(f"Indexing failed for track {track_id}: {details}")
            print(f"Indexing succeeded. Track ID: {track_id}")

        print(f"\nQuestion: {args.query}")
        answer = await rag.aquery(
            args.query,
            param=QueryParam(
                mode=args.mode,
                only_need_context=args.only_context,
                response_type="Chinese concise answer with evidence",
                enable_rerank=False,
            ),
        )
        print("\nResult:\n")
        print(answer)
    finally:
        await rag.finalize_storages()


def main() -> None:
    configure_console()
    args = parse_args()
    if args.start_char < 0:
        raise SystemExit("--start-char cannot be negative")
    if args.max_chars is not None and args.max_chars <= 0:
        raise SystemExit("--max-chars must be greater than zero")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
