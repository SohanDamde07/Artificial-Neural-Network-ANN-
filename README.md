# Deploy to a public URL (Streamlit Community Cloud, free)

1. Create a GitHub repo and upload `app.py` and `requirements.txt`.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "Create app", pick the repo, set the main file to `app.py`, click Deploy.
4. After a minute you get a URL like https://your-app-name.streamlit.app

Run locally first (optional):
    pip install -r requirements.txt
    streamlit run app.py
