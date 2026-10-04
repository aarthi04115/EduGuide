import sys
sys.path.insert(0, r"C:\Users\Aarthi G D\Desktop\EduGuide\backend")
from main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)