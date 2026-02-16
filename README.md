# Palmistry API (self-contained backend)

This folder contains everything required to run the palmistry API.  You can
upload the entire `backend/` directory to GitHub and it will run on any
machine, provided Python and the listed packages are installed.

## Contents

```
backend/
├─ app.py           # Flask application (expects model in model/best.pt)
├─ requirements.txt # dependencies
├─ README.md        # this file
├─ .gitignore       # ignores venv, logs, etc.
├─ start_server.bat # convenience batch script
└─ model/           # put your best.pt weights here
```

## Usage

```bash
cd backend
python -m venv venv      # optional
.\venv\Scripts\activate # Windows
pip install -r requirements.txt
# development mode (use Flask’s built-in server):
python app.py

# production mode with gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
Then POST images to `http://localhost:5000/upload` as described in the
main README.
