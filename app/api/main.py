"""HTTP and static web surface for retrieval-grounded diagnosis."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.kg.loader import GraphStore, load_seed_graph
from app.llm.protocol import ChatMessage, LLMClient
from app.pipeline import DiagnosisStatus, LLMOutputError, diagnose

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"

GraphStoreFactory = Callable[[], GraphStore]
LLMClientFactory = Callable[[], LLMClient]


class DiagnoseRequest(BaseModel):
    """Stable request body for POST /diagnose."""

    symptoms: Annotated[str, Field(strict=True)]


class DiagnosisStatementResponse(BaseModel):
    text: str
    referenced_entity_ids: list[str]


class ModelSuggestionResponse(DiagnosisStatementResponse):
    authoritative: bool
    label: str


class DiagnosisResponse(BaseModel):
    """Same top-level shape for diagnosed, abstained, and invalid outcomes."""

    status: DiagnosisStatus
    reason: str
    matched_symptoms: list[str]
    diagnosis: DiagnosisStatementResponse | None
    verified_knowledge: list[dict[str, Any]]
    model_suggestions: list[ModelSuggestionResponse]
    evidence_chain: list[dict[str, Any]]
    grounding_rejections: int


class DiagnosisProviderUnavailableError(RuntimeError):
    """Raised when the configured LLM client cannot be constructed."""


class DiagnosisProviderResponseError(RuntimeError):
    """Raised when the configured LLM client call fails."""


class _LazyLLMClient:
    """Create one provider client on the first retrieval hit and own its lifetime."""

    def __init__(self, factory: LLMClientFactory) -> None:
        self._factory = factory
        self._client: LLMClient | None = None

    @property
    def initialized(self) -> bool:
        return self._client is not None

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        if self._client is None:
            try:
                self._client = self._factory()
            except Exception as exc:
                raise DiagnosisProviderUnavailableError from exc

        try:
            return self._client.generate(messages)
        except Exception as exc:
            raise DiagnosisProviderResponseError from exc

    def close(self) -> None:
        client = self._client
        self._client = None
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _create_deepseek_client() -> LLMClient:
    # Keep the vendor SDK off invalid-input and abstention import paths.
    from app.llm.client import DeepSeekClient

    return DeepSeekClient()


def get_graph_connection(request: Request) -> Any:
    return request.app.state.graph_store.connection


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_diagnosis_lock(request: Request) -> Any:
    return request.app.state.diagnosis_lock


def create_app(
    *,
    graph_store_factory: GraphStoreFactory = load_seed_graph,
    llm_client_factory: LLMClientFactory = _create_deepseek_client,
) -> FastAPI:
    """Create an application with injectable graph and generation resources."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        graph_store = graph_store_factory()
        llm_client = _LazyLLMClient(llm_client_factory)
        application.state.graph_store = graph_store
        application.state.llm_client = llm_client
        application.state.diagnosis_lock = threading.Lock()
        logger.info("diagnosis_service_started graph_store=ready llm_client=lazy")
        try:
            yield
        finally:
            llm_was_initialized = llm_client.initialized
            try:
                llm_client.close()
            finally:
                graph_store.close()
            logger.info(
                "diagnosis_service_stopped graph_store=closed llm_client=%s",
                "closed" if llm_was_initialized else "unused",
            )

    application = FastAPI(
        title="Agri-Agents Diagnosis API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.post(
        "/diagnose",
        response_model=DiagnosisResponse,
        status_code=status.HTTP_200_OK,
    )
    def diagnose_route(
        payload: DiagnoseRequest,
        connection: Annotated[Any, Depends(get_graph_connection)],
        llm_client: Annotated[LLMClient, Depends(get_llm_client)],
        diagnosis_lock: Annotated[Any, Depends(get_diagnosis_lock)],
    ) -> dict[str, object]:
        try:
            # Kuzu connection sharing is intentionally serialized for the
            # single-process demo until a concurrent connection policy exists.
            with diagnosis_lock:
                result = diagnose(
                    connection,
                    payload.symptoms,
                    llm_client=llm_client,
                )
        except DiagnosisProviderUnavailableError as exc:
            root_error = exc.__cause__ or exc
            logger.error(
                "diagnosis_request_failed stage=provider_configuration error_type=%s",
                type(root_error).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="诊断生成服务暂不可用",
            ) from exc
        except (DiagnosisProviderResponseError, LLMOutputError) as exc:
            root_error = exc.__cause__ or exc
            logger.error(
                "diagnosis_request_failed stage=provider_response error_type=%s",
                type(root_error).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="诊断生成服务返回无效响应",
            ) from exc
        except Exception as exc:
            logger.error(
                "diagnosis_request_failed stage=pipeline error_type=%s",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="诊断服务内部错误",
            ) from exc

        logger.info(
            "diagnosis_request status=%s grounding_rejections=%d",
            result.status.value,
            result.grounding_rejections,
        )
        return result.to_dict()

    # Register API routes first because the root mount captures unmatched paths.
    application.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return application


app = create_app()
