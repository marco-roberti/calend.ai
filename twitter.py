import json
import logging
import os
from datetime import datetime
from time import sleep, time

import requests
from requests_oauthlib import OAuth1Session

bearer_token = os.environ.get("BEARER_TOKEN")
consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")

access_token_url = "https://api.twitter.com/oauth/access_token"
request_token_url = "https://api.twitter.com/oauth/request_token"
base_authorization_url = "https://api.twitter.com/oauth/authorize"
search_url = 'https://api.twitter.com/2/tweets/search/all'
tweet_url = 'https://api.twitter.com/2/tweets/'
profile_url = 'https://api.twitter.com/2/users/'
rules_url = "https://api.twitter.com/2/tweets/search/stream/rules"
following_url = 'https://api.twitter.com/2/users/1438223629456662543/following'

# Get request token
oauth_keys = OAuth1Session(consumer_key, client_secret=consumer_secret)
try:
    fetch_response = oauth_keys.fetch_request_token(request_token_url)
except ValueError:
    print("There may have been an issue with the consumer_key or consumer_secret you entered.")
    fetch_response = None
if fetch_response:
    resource_owner_key = fetch_response.get("oauth_token")
    resource_owner_secret = fetch_response.get("oauth_token_secret")
    # Get authorization
    authorization_url = oauth_keys.authorization_url(base_authorization_url)
    print("Please go here and authorize: %s" % authorization_url)
    verifier = input("Paste the PIN here: ")


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2Python"
    return r


# noinspection PyProtectedMember
# noinspection PyUnresolvedReferences
def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    assert response.status_code == 200, (response.status_code, response.text)
    if int(response.headers._store['x-rate-limit-remaining'][1]) <= 1:
        # Avoiding HTTP response 429 (Rate limit exceeded)
        print('sleeping until ' + str(datetime.fromtimestamp(int(response.headers._store['x-rate-limit-reset'][1]))))
        sleep(1 + int(response.headers._store['x-rate-limit-reset'][1]) - time())
    return response.json()


def get_profile_data(profile_id):
    logging.info(f'Getting profile of user {profile_id}')
    input_user = connect_to_endpoint(profile_url + profile_id, {})['data']
    return input_user['username'], input_user['name']


def get_tweet_author(tweet_id):
    logging.info(f'Getting author of tweet {tweet_id}')
    response = connect_to_endpoint(tweet_url + tweet_id, {'expansions': 'author_id'})
    response = response['includes']['users'][0]
    return response['username'], response['name'], response['id']


def post_reply(reply, to_tweet):
    logging.info(f'Posting tweet >>> {reply}')
    reply = reply.replace("'", "’")
    response = os.popen(
        f"twurl -d 'status={reply}&in_reply_to_status_id={to_tweet.tweet_id}' /1.1/statuses/update.json"
    ).read()
    response = json.loads(response)
    if reply_id := response.get('id_str', None):
        logging.info(f'https://twitter.com/Calend_AI/status/{reply_id}')
    else:
        logging.error(response)


def get_oauth_access():
    # Get the access token
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )

    oauth_tokens = oauth.fetch_access_token(access_token_url)
    return oauth_tokens["oauth_token"], oauth_tokens["oauth_token_secret"]


def follow_author(tweet):
    logging.info(f'Following {tweet.username}')

    # Authenticating
    access_token, access_token_secret = get_oauth_access()
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret
    )

    # Making the request
    payload = {
        "target_user_id": tweet.author_id
    }
    response = oauth.post(following_url, json=payload)
    assert response.status_code == 200, \
        f'Cannot follow @{tweet.username} (HTTP {response.status_code}): {response.text}'


class Tweet:
    def __init__(self, text, username, name, tweet_id=None, author_id=None):
        self.text = text
        self.username = username
        self.name = name
        self.tweet_id = tweet_id
        self.author_id = author_id

    def __str__(self) -> str:
        return f'{self.name} @{self.username} : {self.text}'

    @classmethod
    def from_http(cls, http_response):
        username, name, author_id = get_tweet_author(http_response['id'])
        return cls(http_response['text'], username, name, tweet_id=http_response['id'], author_id=author_id)


class Stream:

    def __init__(self, rules):
        # Retrieve and delete previously set rules
        old_rules = self._get_rules()
        self._delete_all_rules(old_rules)
        # Set new rules
        self.rules = rules
        self._set_rules(self.rules)

    @staticmethod
    def _get_rules():
        logging.info('Getting stream rules')
        response = requests.get(rules_url, auth=bearer_oauth)
        assert response.status_code == 200, f'Cannot get rules (HTTP {response.status_code}): {response.text}'
        response = response.json()
        logging.info(f'Got rules: {response}')
        return response

    @staticmethod
    def _delete_all_rules(rules):
        logging.info('Deleting stream rules')
        if rules is None or "data" not in rules:
            return None

        ids = list(map(lambda rule: rule["id"], rules["data"]))
        payload = {"delete": {"ids": ids}}
        response = requests.post(
            rules_url,
            auth=bearer_oauth,
            json=payload
        )
        assert response.status_code == 200, f'Cannot delete rules (HTTP {response.status_code}): {response.text}'
        response = response.json()
        assert response['meta']['summary']['not_deleted'] == 0, \
            f"{response['meta']['summary']['not_deleted']} rules have not been deleted!"

    @staticmethod
    def _set_rules(rules):
        logging.info(f'Setting stream rules {rules}')
        payload = {"add": rules}
        response = requests.post(
            rules_url,
            auth=bearer_oauth,
            json=payload,
        )
        assert response.status_code == 201, f'Cannot add rules (HTTP {response.status_code}): {response.text}'
        response = response.json()
        assert response['meta']['summary']['not_created'] == 0, \
            f"{response['meta']['summary']['not_created']} rules have not been created!"

    @staticmethod
    def watch(handler):
        logging.info('Watching stream...')
        stream = requests.get(
            "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
        )
        assert stream.status_code == 200, f'Cannot get stream (HTTP {stream.status_code}): {stream.text}'
        for response in stream.iter_lines():
            if response:
                handler(response)
