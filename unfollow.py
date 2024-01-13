import time
import json
import os

cursor = -1
while cursor != 0:
    response = os.popen(
        f'twurl "/1.1/friends/list.json?count=200&cursor={cursor}"'
    ).read()
    response = json.loads(response)
    if 'errors' in response:
        time.sleep(60)
        continue
    cursor = response['next_cursor']
    users = response['users']
    print(f'\nanalyzing {len(users)} users:')
    for i, user in enumerate(users):
        screen_name = user['screen_name']
        followers_cnt = user['followers_count']
        print(f'{i:3d}. @{screen_name:30s} {followers_cnt:7d}')
        if followers_cnt < 5000:
            user_id = user['id']
            response = os.popen(
                f'twurl -d "user_id={user_id}" /1.1/friendships/destroy.json'
            ).read()
            response = json.loads(response)
            if 'errors' in response:
                time.sleep(60)
                continue

