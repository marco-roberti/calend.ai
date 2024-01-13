import os
import json
from pprint import pprint
data = json.loads(os.popen("twurl '/1.1/friends/list.json?count=200'").read())
data = data['users']
users= [
    (u['name'], u['screen_name'], u['followers_count']) for u in data
]
users.sort(key=lambda u: u[-1], reverse=True)
for i, u in enumerate(users):
    print(f'{i:3d} | @{u[1]:20s} - {u[0]:50s}\t{u[2]:7d}')

