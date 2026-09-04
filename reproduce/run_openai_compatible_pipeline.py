import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
QUESTIONS_DIR = DATASETS_DIR / "questions"
WORKING_ROOT = REPO_ROOT / "rag_storage" / "reproduce"


def load_project_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []

    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("enable_cot", None)

    return await openai_complete_if_cache(
        os.getenv("REPRO_LLM_MODEL", "glm-4-flash"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
        timeout=int(os.getenv("REPRO_LLM_TIMEOUT", "60")),
        enable_cot=False,
        **kwargs,
    )


def hash_embed(texts: list[str], dim: int) -> np.ndarray:
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    for row, text in enumerate(texts):
        tokens = token_pattern.findall(text.lower())
        if not tokens:
            tokens = [text]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vectors[row, index] += sign

        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm

    return vectors


class EmbeddingRouter:
    def __init__(self, mode: str, dim: int):
        self.mode = mode
        self.dim = dim
        self.use_remote = mode == "remote"

    async def detect(self) -> None:
        if self.mode == "local":
            print("Embedding mode: local hash embedding")
            return

        try:
            embedding = await self.remote(["embedding smoke test"])
            print(f"Embedding mode: remote API, detected shape {embedding.shape}")
            self.use_remote = True
        except Exception as exc:
            if self.mode == "remote":
                raise
            print(
                "Embedding mode: remote API unavailable, falling back to local hash "
                f"embedding ({type(exc).__name__}: {str(exc)[:160]})"
            )
            self.use_remote = False

    async def remote(self, texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts,
            model=os.getenv("EMBEDDING_MODEL"),
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST"),
            embedding_dim=self.dim,
            max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
            client_configs={"timeout": int(os.getenv("REPRO_EMBEDDING_TIMEOUT", "30"))},
        )

    async def __call__(self, texts: list[str]) -> np.ndarray:
        if self.use_remote:
            return await self.remote(texts)
        return hash_embed(texts, self.dim)


def unique_contexts_path(domain: str) -> Path:
    return DATASETS_DIR / "unique_contexts" / f"{domain}_unique_contexts.json"


def output_stem(domain: str, run_name: str) -> str:
    return f"{domain}_{run_name}" if run_name else domain


def questions_path(domain: str, run_name: str) -> Path:
    return QUESTIONS_DIR / f"{output_stem(domain, run_name)}_questions.txt"


def result_path(domain: str, run_name: str) -> Path:
    return QUESTIONS_DIR / f"{output_stem(domain, run_name)}_result.json"


def error_path(domain: str, run_name: str) -> Path:
    return QUESTIONS_DIR / f"{output_stem(domain, run_name)}_errors.json"


async def initialize_rag(
    domain: str, run_name: str, embedding_router: EmbeddingRouter
) -> LightRAG:
    working_dir = WORKING_ROOT / (run_name or "full") / domain
    working_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_model_func,
        llm_model_name=os.getenv("REPRO_LLM_MODEL", "glm-4-flash"),
        llm_model_max_async=int(os.getenv("REPRO_LLM_MAX_ASYNC", "1")),
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_router.dim,
            max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
            func=embedding_router,
        ),
        embedding_func_max_async=int(os.getenv("REPRO_EMBEDDING_MAX_ASYNC", "2")),
    )
    await rag.initialize_storages()
    return rag


async def insert_contexts(
    domain: str,
    rag: LightRAG,
    max_contexts: int | None,
    max_context_chars: int | None,
) -> None:
    path = unique_contexts_path(domain)
    if not path.exists():
        raise FileNotFoundError(f"Missing Step_0 output: {path}")

    with path.open("r", encoding="utf-8") as file:
        contexts = json.load(file)

    if max_contexts is not None:
        contexts = contexts[:max_contexts]
    if max_context_chars is not None:
        contexts = [context[:max_context_chars] for context in contexts]

    print(f"[{domain}] inserting {len(contexts)} unique contexts")
    await rag.ainsert(contexts)
    print(f"[{domain}] insert complete")


def get_summary(context: str, tokenizer, total_tokens: int = 2000) -> str:
    tokens = tokenizer.tokenize(context)
    half_tokens = total_tokens // 2
    start_tokens = tokens[1000 : 1000 + half_tokens]
    end_tokens = tokens[-(1000 + half_tokens) : 1000]
    return tokenizer.convert_tokens_to_string(start_tokens + end_tokens)


async def generate_questions(
    domain: str,
    run_name: str,
    max_contexts: int | None,
    max_context_chars: int | None,
) -> None:
    from transformers import GPT2Tokenizer

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    path = unique_contexts_path(domain)
    with path.open("r", encoding="utf-8") as file:
        contexts = json.load(file)

    if max_contexts is not None:
        contexts = contexts[:max_contexts]
    if max_context_chars is not None:
        contexts = [context[:max_context_chars] for context in contexts]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    summaries = [get_summary(context, tokenizer) for context in contexts]
    total_description = "\n\n".join(summaries)

    prompt = f"""
Given the following description of a dataset:

{total_description}

Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

Output the results in the following structure:
- User 1: [user description]
    - Task 1: [task description]
        - Question 1:
        - Question 2:
        - Question 3:
        - Question 4:
        - Question 5:
    - Task 2: [task description]
        ...
    - Task 5: [task description]
- User 2: [user description]
    ...
- User 5: [user description]
    ...
"""
    print(f"[{domain}] generating questions")
    result = await llm_model_func(prompt)
    questions_path(domain, run_name).write_text(result, encoding="utf-8")
    print(f"[{domain}] questions written to {questions_path(domain, run_name)}")


def extract_queries(domain: str, run_name: str, max_queries: int | None) -> list[str]:
    data = questions_path(domain, run_name).read_text(encoding="utf-8").replace("**", "")
    queries = re.findall(r"- Question \d+:\s*(.+)", data)
    if max_queries is not None:
        queries = queries[:max_queries]
    return queries


async def run_queries(
    domain: str, run_name: str, rag: LightRAG, max_queries: int | None
) -> None:
    queries = extract_queries(domain, run_name, max_queries)
    print(f"[{domain}] running {len(queries)} queries")
    results = []
    errors = []

    for index, query in enumerate(queries, start=1):
        print(f"[{domain}] query {index}/{len(queries)}")
        try:
            result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
            results.append({"query": query, "result": result})
        except Exception as exc:
            errors.append({"query": query, "error": str(exc)})

    result_path(domain, run_name).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    error_path(domain, run_name).write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{domain}] results written to {result_path(domain, run_name)}")
    if errors:
        print(f"[{domain}] {len(errors)} errors written to {error_path(domain, run_name)}")


async def run_domain(
    domain: str,
    run_name: str,
    embedding_router: EmbeddingRouter,
    skip_insert: bool,
    skip_questions: bool,
    skip_queries: bool,
    max_queries: int | None,
    max_contexts: int | None,
    max_context_chars: int | None,
) -> None:
    rag = await initialize_rag(domain, run_name, embedding_router)
    if not skip_insert:
        await insert_contexts(domain, rag, max_contexts, max_context_chars)
    if not skip_questions and not questions_path(domain, run_name).exists():
        await generate_questions(domain, run_name, max_contexts, max_context_chars)
    elif not skip_questions:
        print(f"[{domain}] using existing questions at {questions_path(domain, run_name)}")
    if not skip_queries:
        await run_queries(domain, run_name, rag, max_queries)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["agriculture"],
        choices=["agriculture", "cs", "legal", "mix"],
    )
    parser.add_argument("--embedding", choices=["auto", "remote", "local"], default="auto")
    parser.add_argument("--skip-insert", action="store_true")
    parser.add_argument("--skip-questions", action="store_true")
    parser.add_argument("--skip-queries", action="store_true")
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    load_project_env()
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_router = EmbeddingRouter(args.embedding, embedding_dim)
    await embedding_router.detect()

    print(f"LLM model: {os.getenv('REPRO_LLM_MODEL', 'glm-4-flash')}")
    for domain in args.domains:
        await run_domain(
            domain,
            args.run_name,
            embedding_router,
            args.skip_insert,
            args.skip_questions,
            args.skip_queries,
            args.max_queries,
            args.max_contexts,
            args.max_context_chars,
        )


if __name__ == "__main__":
    asyncio.run(main())
