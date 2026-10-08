import uvicorn

if __name__ == '__main__':
    uvicorn.run(app='app.app:app',host='0.0.0.0',port=8000,reload=True)
    
# reload = True means reload the app whenever i will make change to my app.py