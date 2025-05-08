# test_news.py
import os, requests

key = os.getenv("NEWSAPI_KEY")
print("Using key:", key)
resp = requests.get(
    "https://newsapi.org/v2/top-headlines",
    params={"apiKey": key, "pageSize":1, "language":"en"}
)
print("HTTP", resp.status_code, resp.json().get("status"))
