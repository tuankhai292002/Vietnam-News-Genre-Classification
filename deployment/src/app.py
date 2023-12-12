import uvicorn
from router import app
from utils.config import setting

if __name__ == '__main__':
    uvicorn.run("app:app",host='127.0.0.1',port=setting.SERVER_PORT,reload=True)