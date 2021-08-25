import os
from datetime import datetime
from time import sleep, time

import requests

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER_TOKEN")


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


# noinspection PyProtectedMember
# noinspection PyUnresolvedReferences
def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    if int(response.headers._store['x-rate-limit-remaining'][1]) <= 1:
        # Avoiding HTTP response 429 (Rate limit exceeded)
        print('sleeping until ' + str(datetime.fromtimestamp(int(response.headers._store['x-rate-limit-reset'][1]))))
        sleep(1 + int(response.headers._store['x-rate-limit-reset'][1]) - time())
    return response.json()
