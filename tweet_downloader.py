import json
import os
import random
from os import path

from tqdm import tqdm

from twitter import connect_to_endpoint

search_url = 'https://api.twitter.com/2/tweets/search/all'
tweet_url = 'https://api.twitter.com/2/tweets?ids='
profile = 'CarloCalenda'
profile_id = '2416067982'
eval_size = 100

start_time = '2018-06-01T00:00:00+01:00'


def process_tweet(response_tweet):
    # Ensure this tweet answers to an existing one
    if 'referenced_tweets' not in response_tweet or len(response_tweet['referenced_tweets']) != 1:
        return None, response_tweet['created_at']
    # Get the tweet the current one answers to
    input_tweet = connect_to_endpoint(
        tweet_url + response_tweet['referenced_tweets'][0]['id'], {'tweet.fields': 'author_id'})
    # Ensure this tweet answers to an existing one (again)
    if 'errors' in input_tweet:
        return None, response_tweet['created_at']
    input_tweet = input_tweet['data'][0]
    # Exclude self-answers
    if input_tweet['author_id'] == profile_id:
        return None, response_tweet['created_at']
    # Return actual data
    input_text = input_tweet['text']
    response_text = response_tweet['text']
    return {'input': input_text, 'output': response_text}, response_tweet['created_at']


def download_tweets():
    if not path.isdir('data'):
        os.mkdir('data')

    tweets = []
    query_params = {
        'query': f'from:{profile} (is:quote OR is:reply) -is:retweet',  # has:hashtags
        'start_time': start_time,
        'tweet.fields': 'referenced_tweets,created_at',
        'max_results': 100
    }
    pbar = tqdm(desc=profile, unit='Tw')

    def process_response():
        for tw in json_response['data']:
            tw, time = process_tweet(tw)
            if tw is not None:
                tweets.append(tw)
                pbar.set_postfix_str(time)
                pbar.update()

    json_response = connect_to_endpoint(search_url, query_params)
    process_response()

    while 'next_token' in json_response['meta']:
        query_params['next_token'] = json_response['meta']['next_token']
        json_response = connect_to_endpoint(search_url, query_params)
        process_response()

    random.shuffle(tweets)
    with open(path.join('data', f'valid.json'), 'w') as f:
        for _ in range(eval_size):
            f.write(json.dumps(tweets.pop()) + '\n')
    with open(path.join('data', f'train.json'), 'w') as f:
        while tweets:
            f.write(json.dumps(tweets.pop()) + '\n')


if __name__ == "__main__":
    download_tweets()
