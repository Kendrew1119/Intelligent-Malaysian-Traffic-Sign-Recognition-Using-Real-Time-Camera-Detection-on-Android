from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.inference import DetectorUnavailableError, SignDetector, decode_image
from webapp.hard_cases import HardCaseStore
from webapp.sign_catalog import build_catalog


WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_app(
    detector: SignDetector | None = None,
    hard_case_store: HardCaseStore | None = None,
) -> FastAPI:
    active_detector = detector or SignDetector()
    active_hard_case_store = hard_case_store or HardCaseStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(active_detector.load)
        except DetectorUnavailableError as exc:
            app.state.startup_error = str(exc)
        yield

    app = FastAPI(
        title="MYSignVoice",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.detector = active_detector
    app.state.hard_case_store = active_hard_case_store
    app.state.startup_error = None
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        model = app.state.detector
        return {
            "ready": model.ready,
            "model": model.manifest.get("model", "YOLO26s"),
            "format": "OpenVINO",
            "classes": len(model.class_names),
            "image_size": model.image_size,
            "speed_limit_mode": {
                "mode": "yolo-plus-ocr",
                "reader_enabled": False,
                "ocr": model.speed_limit_ocr.status()
                if hasattr(model, "speed_limit_ocr")
                else {"enabled": False, "error": "OCR status is unavailable."},
                "yolo_classes": [5, 15, 30, 40, 50, 60, 80],
            },
            "error": app.state.startup_error,
        }

    @app.get("/api/signs")
    async def signs() -> dict:
        try:
            catalogue = build_catalog(app.state.detector.class_names)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "count": len(catalogue),
            "source": "Malaysian JKR traffic-sign categories",
            "signs": catalogue,
        }

    @app.get("/api/hard-cases")
    async def hard_case_status() -> dict:
        return {
            "saved": await asyncio.to_thread(app.state.hard_case_store.count),
            "location": "dataset/hard_cases",
        }

    @app.post("/api/detect")
    async def detect(
        image: UploadFile = File(...),
        confidence: float = Form(0.20),
    ) -> dict:
        suffix = Path(image.filename or "").suffix.lower()
        if image.content_type not in ALLOWED_IMAGE_TYPES and suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Use a JPG, PNG, WebP, or BMP image.",
            )
        if not 0.05 <= confidence <= 0.90:
            raise HTTPException(
                status_code=422,
                detail="Confidence must be between 0.05 and 0.90.",
            )

        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
        if not payload:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")

        request_started = time.perf_counter()
        try:
            decoded = await asyncio.to_thread(decode_image, payload)
            prediction = await asyncio.to_thread(
                app.state.detector.predict,
                decoded,
                confidence,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DetectorUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

        prediction["total_ms"] = round(
            (time.perf_counter() - request_started) * 1000,
            1,
        )
        prediction["confidence_threshold"] = confidence
        return prediction

    @app.post("/api/hard-cases")
    async def save_hard_case(
        image: UploadFile = File(...),
        source: str = Form(...),
        issue_type: str = Form(...),
        expected_class: str = Form(""),
        predicted_classes: str = Form("[]"),
        notes: str = Form(""),
        confidence: float = Form(0.20),
    ) -> dict:
        suffix = Path(image.filename or "").suffix.lower()
        if image.content_type not in ALLOWED_IMAGE_TYPES and suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=415, detail="Use a JPG, PNG, WebP, or BMP image.")
        if not 0.05 <= confidence <= 0.90:
            raise HTTPException(status_code=422, detail="Confidence must be between 0.05 and 0.90.")

        valid_classes = set(app.state.detector.class_names.values())
        expected = expected_class.strip()
        if expected and expected not in valid_classes:
            raise HTTPException(status_code=422, detail="Choose a class from the 63-sign list.")
        if issue_type in {"missed", "wrong-class"} and not expected:
            raise HTTPException(status_code=422, detail="Enter the correct sign class for this issue.")
        try:
            predicted = json.loads(predicted_classes)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Predicted classes are invalid.") from exc
        if not isinstance(predicted, list) or any(
            not isinstance(name, str) or name not in valid_classes for name in predicted
        ):
            raise HTTPException(status_code=422, detail="Predicted classes are invalid.")

        payload = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller.")
        if not payload:
            raise HTTPException(status_code=400, detail="The frame is empty.")

        try:
            record = await asyncio.to_thread(
                app.state.hard_case_store.save,
                payload,
                suffix=suffix,
                source=source,
                issue_type=issue_type,
                expected_class=expected or None,
                predicted_classes=predicted,
                notes=notes,
                confidence_threshold=confidence,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "saved": True,
            "id": record["id"],
            "total_saved": await asyncio.to_thread(app.state.hard_case_store.count),
        }

    return app


app = create_app()
