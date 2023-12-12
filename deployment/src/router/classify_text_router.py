from fastapi.responses import JSONResponse
from fastapi import APIRouter
from schemas import *
from services import *

router = APIRouter()

@router.get('/')
def health_method():
    return 'OK'

@router.post("/classify_text/")
async def classify_text(user_input: ClassifyRequest) -> ClassifyResponse:
    """
    This API is using for classifying topic for a text

    Args:

    - text (string): Text you want to classify

    - return_all_probabilities (bool): Set to true to return sorted probabilities of all topics. Default: true

    - enable_profanity_check (bool):  Set to true to enable profanity checking. If the text contains bad words, the system will not classify the topic. Default: true

    Returns:

    - status code 200:
        {
            "result": {},
            "message": "Success"
        }

    - status code 400:
        {
            "result": {},
            "message": "Failed, text is null !"
        }

    - status code 401:
        {
            "result": {},
            "message": "Failed, text contains bad word !"
        }
    """

    text = user_input.text.strip()
    if not text: # if text is null
        return JSONResponse(
        status_code=400,
        content={"result": {},"message": "Failed, text is null !"},
    )
    if user_input.enable_profanity_check and does_have_bad_words(text): # If profanity checking is enabled and offensive words are found in the text
        return JSONResponse(
        status_code=401,
        content={"result": {},"message": "Failed, text contains bad word !"},
    )
    result = predict(text,return_all_probabilities=user_input.return_all_probabilities)
    return JSONResponse(
        status_code=200,
        content={"result": result,"message": "Success"},
    )

