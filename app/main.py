import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routes.yolo_wsd import router  as yolo_router
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# from app.routes import run_pipeline  # your import

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/results", StaticFiles(directory="app/results"), name="results")
app.mount("/results", NoCacheStaticFiles(directory="app/results"), name="results")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    yolo_router,
    prefix="/yolo-wsd",
    tags=["app"],
)

@app.get("/")
async def get_index():
    return {"message": "Hello route triggered!"}

# Entry point for running the application
def run():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    run()