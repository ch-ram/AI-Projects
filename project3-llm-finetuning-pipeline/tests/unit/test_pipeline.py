"""Unit tests for LLM Fine-Tuning Pipeline."""
import pytest
from unittest.mock import MagicMock, patch


class TestDataPipeline:
    def test_format_alpaca(self):
        from src.data.pipeline import format_alpaca
        example = {
            "instruction": "Summarize the text.",
            "input": "The quick brown fox.",
            "output": "A fox jumps.",
        }
        result = format_alpaca(example)
        assert "text" in result
        assert "### Instruction:" in result["text"]
        assert "### Response:" in result["text"]

    def test_format_alpaca_no_input(self):
        from src.data.pipeline import format_alpaca
        example = {
            "instruction": "Say hello.",
            "input": "",
            "output": "Hello!",
        }
        result = format_alpaca(example)
        assert "### Input:" not in result["text"]

    def test_format_chatml(self):
        from src.data.pipeline import format_chatml
        example = {
            "system": "You are helpful.",
            "user": "Hi",
            "assistant": "Hello!",
        }
        result = format_chatml(example)
        assert "<|im_start|>system" in result["text"]
        assert "<|im_start|>user" in result["text"]
        assert "<|im_start|>assistant" in result["text"]

    def test_format_sharegpt(self):
        from src.data.pipeline import format_sharegpt
        example = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        result = format_sharegpt(example)
        assert "<|user|>" in result["text"]
        assert "<|assistant|>" in result["text"]


class TestModelRegistry:
    def test_register_and_get(self, tmp_path):
        from src.serving.server import ModelRegistry
        registry = ModelRegistry(str(tmp_path / "registry.json"))
        registry.register(
            model_id="test-v1",
            model_path="/models/test",
            base_model="llama-3.1",
            metrics={"eval_loss": 0.5},
        )
        model = registry.get("test-v1")
        assert model is not None
        assert model["base_model"] == "llama-3.1"

    def test_list_models(self, tmp_path):
        from src.serving.server import ModelRegistry
        registry = ModelRegistry(str(tmp_path / "registry.json"))
        registry.register("v1", "/m/v1", "llama", {"loss": 0.5})
        registry.register("v2", "/m/v2", "mistral", {"loss": 0.3})
        models = registry.list_models()
        assert len(models) == 2

    def test_get_best_model(self, tmp_path):
        from src.serving.server import ModelRegistry
        registry = ModelRegistry(str(tmp_path / "registry.json"))
        registry.register("v1", "/m/v1", "llama", {"eval_loss": 0.5})
        registry.register("v2", "/m/v2", "llama", {"eval_loss": 0.3})
        best = registry.get_best_model("eval_loss")
        assert best == "v2"


class TestEvalResult:
    def test_eval_result_creation(self):
        from src.evaluation.benchmarks import EvalResult
        result = EvalResult(
            benchmark="mmlu",
            score=0.75,
            num_samples=1000,
        )
        assert result.benchmark == "mmlu"
        assert result.score == 0.75


class TestConfig:
    def test_lora_target_modules_list(self):
        with patch("src.utils.config.Path.exists", return_value=False):
            from src.utils.config import Settings
            s = Settings(lora_target_modules="q_proj,k_proj,v_proj")
            assert s.lora_target_modules_list == ["q_proj", "k_proj", "v_proj"]

    def test_eval_benchmarks_list(self):
        with patch("src.utils.config.Path.exists", return_value=False):
            from src.utils.config import Settings
            s = Settings(eval_benchmarks="mmlu,hellaswag")
            assert s.eval_benchmarks_list == ["mmlu", "hellaswag"]
