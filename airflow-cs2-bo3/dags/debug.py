import requests
import json
slug = "fokus-cs-vs-1win-10-09-2026"
r = requests.get(f'https://api.bo3.gg/api/v1/matches/{slug}/short_players_stats', headers={'Accept': 'application/json'})
print("Status code:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("Type:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
        if "data" in data:
            print("Type of data['data']:", type(data["data"]))
    elif isinstance(data, list):
        print("Length:", len(data))
        if len(data) > 0:
            print("First item keys:", data[0].keys() if isinstance(data[0], dict) else data[0])
