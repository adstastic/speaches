from __future__ import annotations

from collections import OrderedDict
import logging
import threading
from typing import TYPE_CHECKING, Any

from speaches.model_manager import SelfDisposingModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import-time optional dependency
    from numpy.typing import NDArray
    from numpy import float32


class _TranscriptionOptions:
    def __init__(self, *, word_timestamps: bool) -> None:
        self.word_timestamps = word_timestamps


class _TranscriptionInfo:
    def __init__(self, *, language: str, duration: float, word_timestamps: bool) -> None:
        self.language = language
        self.duration = duration
        self.transcription_options = _TranscriptionOptions(word_timestamps=word_timestamps)


class MlxWhisperRuntime:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def transcribe(
        self,
        audio: "NDArray[float32]",
        *,
        task: str,
        language: str | None,
        initial_prompt: str | None,
        word_timestamps: bool,
        temperature: float,
    ) -> tuple[list["TranscriptionSegment"], _TranscriptionInfo]:
        """Run MLX Whisper transcription and adapt to speaches types.

        Notes: Requires `mlx-whisper` to be installed.
        """
        from mlx_whisper.transcribe import transcribe as mlx_transcribe  # type: ignore[import-not-found]

        # MLX expects 16kHz float32 mono PCM in range [-1, 1]
        result = mlx_transcribe(
            self.model,
            self.tokenizer,
            audio,
            task=task,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            temperature=temperature,
        )

        # Result may be a dict with 'segments' or a simple object; handle flexibly
        raw_segments: list[dict[str, Any]]
        if isinstance(result, dict) and "segments" in result:
            raw_segments = list(result.get("segments", []))
            detected_language = result.get("language") or language or "unknown"
        else:
            # Fallback: assume result is an iterable of segments
            raw_segments = list(getattr(result, "segments", result))  # type: ignore[arg-type]
            detected_language = language or "unknown"

        from speaches.api_types import TranscriptionSegment, TranscriptionWord

        segments: list[TranscriptionSegment] = []
        max_end = 0.0
        for i, seg in enumerate(raw_segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = str(seg.get("text", "")).strip()

            words_data = seg.get("words") or []
            words = [
                TranscriptionWord(
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    word=str(w.get("word") or w.get("text") or ""),
                    probability=float(w.get("probability", w.get("prob", 0.0))),
                )
                for w in words_data
            ] if words_data else None

            segments.append(
                TranscriptionSegment(
                    id=i,
                    seek=0,
                    start=start,
                    end=end,
                    text=text,
                    tokens=list(seg.get("tokens", [])) or [],
                    temperature=float(seg.get("temperature", temperature or 0.0)),
                    avg_logprob=float(seg.get("avg_logprob", 0.0)),
                    compression_ratio=float(seg.get("compression_ratio", 0.0)),
                    no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                    words=words,
                )
            )
            max_end = max(max_end, end)

        info = _TranscriptionInfo(
            language=detected_language,
            duration=max_end,
            word_timestamps=word_timestamps,
        )
        return segments, info


class MlxWhisperModelManager:
    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self.loaded_models: OrderedDict[str, SelfDisposingModel[MlxWhisperRuntime]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_fn(self, model_id: str) -> MlxWhisperRuntime:
        try:
            from mlx_whisper.utils import load_model  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover - import error at runtime if dependency missing
            raise RuntimeError(
                "mlx-whisper is not installed. Install with: pip install mlx mlx-whisper"
            ) from e

        model, tokenizer = load_model(model_id)
        return MlxWhisperRuntime(model=model, tokenizer=tokenizer)

    def _handle_model_unloaded(self, model_id: str) -> None:
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            model = self.loaded_models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id} not found")
            self.loaded_models[model_id].unload()

    def load_model(self, model_id: str) -> SelfDisposingModel[MlxWhisperRuntime]:
        with self._lock:
            if model_id in self.loaded_models:
                logger.debug(f"{model_id} model already loaded")
                return self.loaded_models[model_id]
            self.loaded_models[model_id] = SelfDisposingModel[MlxWhisperRuntime](
                model_id,
                load_fn=lambda: self._load_fn(model_id),
                ttl=self.ttl,
                model_unloaded_callback=self._handle_model_unloaded,
            )
            return self.loaded_models[model_id]

