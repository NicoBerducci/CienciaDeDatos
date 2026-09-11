import requests

url = "https://api.bo3.gg/api/v2/matches/finished"
params = {
    "filter[tier][in]": "s,a",
    "filter[discipline_id][eq]": "1",
    "filter[matches.start_date][gt]": "2024-01-01 00:00",
    "date": "2026-09-10",
    "utc_offset": "0"
}
headers = {"Accept": "application/json"}

print("Testing with date and utc_offset and filter...")
r = requests.get(url, params=params, headers=headers)
print(f"Status: {r.status_code}")
try:
    print(r.json())
except Exception as e:
    print(r.text)

print("\nTesting WITHOUT filter[matches.start_date][gt]...")
params2 = {
    "filter[tier][in]": "s,a",
    "filter[discipline_id][eq]": "1",
    "date": "2026-09-10",
    "utc_offset": "0"
}
r2 = requests.get(url, params=params2, headers=headers)
print(f"Status: {r2.status_code}")
try:
    print(r2.json())
except Exception as e:
    print(r2.text)
