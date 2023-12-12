
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from schemas import *
from services import *
from .classify_text_router import router as TopicRouter
from utils.config import *

app = FastAPI(
    docs_url= setting.ROUTER_PREFIX + '/docs',
    redoc_url=setting.ROUTER_PREFIX + '/redoc',
    openapi_url= setting.ROUTER_PREFIX+'/openapi.json'
)

app.include_router(TopicRouter, prefix=setting.ROUTER_PREFIX, tags=["Content Moderation API"])

def sample_response(endpoint,method_api,status_code,description,examples):
    responses = app.openapi()["paths"][endpoint][method_api]["responses"]
    if status_code not in responses:
        responses[status_code] = {"description": description}
    if "content" not in responses[status_code]:
        responses[status_code]["content"] = {}
    if "application/json" not in responses[status_code]["content"]:
        responses[status_code]["content"]["application/json"] = {"examples":{}}
    for i in range(0,len(examples)):
        existed_number=len(app.openapi()["paths"][endpoint][method_api]["responses"][status_code]["content"]["application/json"]["examples"])
        app.openapi()["paths"][endpoint][method_api]["responses"][status_code]["content"]["application/json"]["examples"][f"Example{i+existed_number}"]=examples[i]
   
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Content Moderation API",
        version="0.0.1",
        description="This is very simple API of content moderation AI system",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    
    return app.openapi_schema

app.openapi = custom_openapi
sample_response("/content_moderation/classify_text/","post","400","Null text",[null_text_response])
sample_response("/content_moderation/classify_text/","post","401","Bad text",[bad_text_response])



    


