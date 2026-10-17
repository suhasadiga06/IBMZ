"""AgentGuard web API and demo UI.

Run on the LinuxONE VM:  uvicorn agentguard.app.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentguard.core import scenarios
from agentguard.core.explain import incident_report
from agentguard.core.guard import ART, AgentGuard, Guards
from agentguard.core.llm import get_llm

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="AgentGuard", version="1.0")
guard = AgentGuard(Guards.load())
llm = get_llm()


IBM_Z_MODELS = {"9175": "IBM z17", "9176": "IBM z17", "3931": "IBM z16", "3932": "IBM z16",
                "8561": "IBM z15", "8562": "IBM z15"}


def _cpu_model() -> str:
    """Name the hardware. On IBM Z, /proc/cpuinfo carries the machine type."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor() or "unknown"
    m = re.search(r"machine\s*=\s*(\d{4})", text)
    if m:
        return f"{IBM_Z_MODELS.get(m.group(1), 'IBM Z')} (machine type {m.group(1)})"
    m = re.search(r"model name\s*:\s*(.+)", text)
    return m.group(1).strip() if m else platform.processor() or "unknown"


@app.get("/api/system")
def system() -> dict:
    return {
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "is_ibm_z": platform.machine() == "s390x",
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "generator": llm.name,
    }


@app.get("/api/metrics")
def metrics() -> dict:
    path = ART / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Train first: python3 -m agentguard.scripts.train")
    return json.loads(path.read_text())


@app.get("/api/scenarios")
def list_scenarios() -> list[dict]:
    return [{k: s[k] for k in ("id", "kind", "title", "user", "document", "source")}
            | {"plan": [p["tool"] for p in (s["plan"] or [])]} for s in scenarios.all_scenarios()]


@app.post("/api/run/{sid}")
def run(sid: str) -> dict:
    try:
        s = scenarios.get(sid)
    except KeyError:
        raise HTTPException(404, f"unknown scenario {sid}")
    u = guard.run_unprotected(s["user"], s["document"], s["plan"])
    p = guard.run_protected(s["user"], s["document"], s["plan"])
    return {"scenario": sid, "unprotected": asdict(u), "protected": asdict(p),
            "report": incident_report(s, u, p, llm)}


class ChatIn(BaseModel):
    message: str = Field(..., max_length=4000)
    document: str = Field("", max_length=20000)


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    """Free-form test: a normal (non-hijacked) turn, judged by all three layers."""
    u = guard.run_unprotected(body.message, body.document)
    p = guard.run_protected(body.message, body.document)
    s = {"title": "Live input", "kind": "attack" if p.caught_by else "benign"}
    return {"unprotected": asdict(u), "protected": asdict(p), "report": incident_report(s, u, p, llm)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
