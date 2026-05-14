"""
Evaluation suite for fine-tuned LLMs.
Supports standard benchmarks (MMLU, HellaSwag, ARC) and custom metrics.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""
    benchmark: str
    score: float
    num_samples: int
    details: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


class EvalSuite:
    """
    Comprehensive evaluation suite for LLMs.

    Supports:
    - Standard benchmarks via lm-eval-harness
    - Custom text generation quality metrics
    - Perplexity computation
    - Speed/throughput benchmarks
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path

        logger.info("eval_suite_init", model=model_path)

    def run_benchmarks(
        self,
        benchmarks: Optional[list[str]] = None,
    ) -> list[EvalResult]:
        """Run all configured benchmarks."""
        benchmarks = benchmarks or self.settings.eval_benchmarks_list
        results = []

        for benchmark in benchmarks:
            logger.info("running_benchmark", benchmark=benchmark)
            try:
                result = self._run_single_benchmark(benchmark)
                results.append(result)
                logger.info(
                    "benchmark_complete",
                    benchmark=benchmark,
                    score=result.score,
                )
            except Exception as e:
                logger.error(
                    "benchmark_failed",
                    benchmark=benchmark,
                    error=str(e),
                )
                results.append(EvalResult(
                    benchmark=benchmark,
                    score=0.0,
                    num_samples=0,
                    details={"error": str(e)},
                ))

        return results

    def _run_single_benchmark(self, benchmark: str) -> EvalResult:
        """Run a single benchmark using lm-eval-harness."""
        start = time.perf_counter()

        try:
            from lm_eval import evaluator
            from lm_eval.models.huggingface import HFLM

            model = HFLM(
                pretrained=self.model_path,
                tokenizer=self.tokenizer_path,
                batch_size=self.settings.per_device_eval_batch_size,
            )

            results = evaluator.simple_evaluate(
                model=model,
                tasks=[benchmark],
                num_fewshot=self.settings.eval_num_fewshot,
            )

            task_results = results["results"].get(benchmark, {})
            score = task_results.get("acc,none", task_results.get("acc_norm,none", 0.0))
            n_samples = task_results.get("samples", 0)

            duration = time.perf_counter() - start
            return EvalResult(
                benchmark=benchmark,
                score=float(score),
                num_samples=n_samples,
                details=task_results,
                duration_seconds=duration,
            )
        except ImportError:
            logger.warning("lm_eval_not_installed_using_fallback")
            return self._fallback_eval(benchmark)

    def _fallback_eval(self, benchmark: str) -> EvalResult:
        """Fallback evaluation using text generation quality metrics."""
        return EvalResult(
            benchmark=benchmark,
            score=0.0,
            num_samples=0,
            details={"note": "lm-eval-harness not installed; install for full benchmarks"},
        )

    def compute_perplexity(
        self,
        texts: list[str],
        batch_size: int = 4,
    ) -> float:
        """Compute perplexity on a set of texts."""
        logger.info("computing_perplexity", num_texts=len(texts))

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.settings.max_seq_length,
                    padding=True,
                ).to(model.device)

                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        logger.info("perplexity_computed", perplexity=perplexity)
        return perplexity

    def benchmark_throughput(
        self,
        prompt: str = "Explain the theory of relativity in simple terms.",
        max_new_tokens: int = 256,
        num_runs: int = 5,
    ) -> dict:
        """Measure inference throughput (tokens/second)."""
        logger.info("benchmarking_throughput", num_runs=num_runs)

        pipe = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        latencies = []
        tokens_generated = []

        for _ in range(num_runs):
            start = time.perf_counter()
            output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            elapsed = time.perf_counter() - start

            gen_text = output[0]["generated_text"][len(prompt):]
            n_tokens = len(pipe.tokenizer.encode(gen_text))

            latencies.append(elapsed)
            tokens_generated.append(n_tokens)

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        throughput = avg_tokens / avg_latency

        result = {
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_tokens_generated": round(avg_tokens, 1),
            "throughput_tokens_per_sec": round(throughput, 1),
            "num_runs": num_runs,
        }

        logger.info("throughput_benchmark", **result)
        return result

    def save_results(self, results: list[EvalResult], output_path: str) -> None:
        """Save evaluation results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "benchmark": r.benchmark,
                "score": r.score,
                "num_samples": r.num_samples,
                "duration_seconds": r.duration_seconds,
                "details": r.details,
            }
            for r in results
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("results_saved", path=str(path))
