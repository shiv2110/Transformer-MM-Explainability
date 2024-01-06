import json

f = open('valid_mega.json')
data = json.load(f)
print(len(data))