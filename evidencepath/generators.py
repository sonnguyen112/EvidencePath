"""Answer-generation backends and the paper's evidence-only QA prompt."""

from __future__ import annotations

from dataclasses import dataclass


def build_qa_prompt(question: str, context: str) -> str:
    """Build the no-few-shot prompt used for answer generation."""

    return (
        "You answer a multi-hop question using only the supplied evidence. "
        "Return only the final short answer. If the evidence is insufficient, return Unknown.\n\n"
        f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


class AnswerGenerator:
    """Answer-generator protocol."""

    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError


class UnknownGenerator(AnswerGenerator):
    """Deterministic generator used when retrieval-only evaluation is requested."""

    def answer(self, question: str, context: str) -> str:
        del question, context
        return "Unknown"


@dataclass(frozen=True)
class GenerationConfig:
    model_name_or_path: str
    quantize_8bit: bool = True
    device_map: str = "auto"
    max_input_tokens: int = 16_384
    max_answer_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


class HuggingFaceAnswerGenerator(AnswerGenerator):
    """Transformers causal-LM generator with deterministic decoding."""

    def __init__(self, config: GenerationConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Answer generation requires `torch` and `transformers`."
            ) from exc
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
        model_kwargs = {"device_map": config.device_map}
        if config.quantize_8bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "8-bit generation requires bitsandbytes and a recent transformers release."
                ) from exc
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render_prompt(self, question: str, context: str) -> str:
        prompt = build_qa_prompt(question, context)
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        messages = [
            {"role": "system", "content": "Answer only from the supplied evidence."},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (TypeError, ValueError):
            return prompt

    def answer(self, question: str, context: str) -> str:
        prompt = self._render_prompt(question, context)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_answer_tokens,
                do_sample=False,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_generator(
    model_name_or_path: str | None,
    *,
    quantize_8bit: bool = True,
    max_input_tokens: int = 16_384,
    max_answer_tokens: int = 64,
) -> AnswerGenerator:
    """Return a model-backed generator or the retrieval-only Unknown generator."""

    if not model_name_or_path:
        return UnknownGenerator()
    return HuggingFaceAnswerGenerator(
        GenerationConfig(
            model_name_or_path=model_name_or_path,
            quantize_8bit=quantize_8bit,
            max_input_tokens=max_input_tokens,
            max_answer_tokens=max_answer_tokens,
        )
    )

