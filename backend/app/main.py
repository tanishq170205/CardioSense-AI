from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.api import predict, compare, fuzzy_graphs
import os

app = FastAPI(title="Heart Disease Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(compare.router, prefix="/api", tags=["compare"])
app.include_router(fuzzy_graphs.router, prefix="/api", tags=["fuzzy"])


@app.get("/")
def read_root():
    return RedirectResponse(url="/web/index.html")


# Serve the HTML frontend at /web/
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
frontend_dir = os.path.abspath(frontend_dir)
if os.path.isdir(frontend_dir):
    app.mount("/web", StaticFiles(directory=frontend_dir), name="frontend")
