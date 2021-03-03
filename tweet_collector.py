"""Tweet collector for chatbot"""
import os
import re
import time
import socket
import argparse
from typing import Tuple

import requests_oauthlib
import emoji
import neologdn

from models.encoder_decoder import build_tokenizer
from logger import logger

MAX_ATTEMPTS = 10
WAIT_SEC = 30
MIN_SLEEP_SEC = 2
MIN_CHARS = 2

BASE_TIMESTAMP = 1288834974657
RATE_LIMIT_URL = 'https://api.twitter.com/1.1/application/rate_limit_status.json'
SEARCH_URL = 'https://api.twitter.com/1.1/search/tweets.json'
LOOKUP_URL = 'https://api.twitter.com/1.1/statuses/lookup.json'

# Regex patterns to format text
screen_name_pattern = re.compile('@[\w_]+ *')
hash_tag_pattern = re.compile('#[^ ]+ *')
url_pattern = re.compile('https?://[\w/:%#\$&\?\(\)~\.=\+\-]+ *')
emoji_pattern = re.compile(
    u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
period_pattern = re.compile('。+')
valid_char_pattern = re.compile(r'[^、。!?ー〜1-9a-zA-Zぁ-んァ-ヶ亜-腕纊-黑一-鿕]')
space_pattern = re.compile('\s+')


class Conversation:
    def __init__(self, status_id: str, original_id: str, text: str):
        self.status_ids = []
        self.original_ids = []
        self.texts = []
        self.latest_id = status_id

        self.add_status(status_id, original_id, text)

    def __len__(self) -> int:
        return len(self.status_ids)

    def add_status(self, status_id: str, original_id: str, text: str):
        self.status_ids.append(status_id)
        self.original_ids.append(original_id)
        self.texts.append(text)


class TweetCollector:
    def __init__(self, api_key: str, api_secret_key: str, access_token: str,
                 access_token_secret: str, tokenizer_name: str,
                 max_token_length: int):
        self.session = requests_oauthlib.OAuth1Session(
            client_key=api_key, client_secret=api_secret_key,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret)
        self.tokenizer = build_tokenizer(tokenizer_name)
        self.max_token_length = max_token_length

    @staticmethod
    def wait(sec: int):
        logger.info(f'[Wait] {sec:.3f} sec.')
        time.sleep(sec)

    def get_content(self, url: str, params: dict = {}) -> dict:
        for _ in range(MAX_ATTEMPTS):
            try:
                res = self.session.get(url, params=params)
            except socket.error as e:
                logger.error(f'Failed to connect socket for {url}: {e.errno}.')
                self.wait(WAIT_SEC)
                continue

            if res.status_code == 200:
                return res.json()
            else:
                logger.warning(f'Status code is {res.status_code} for {url}.')
                self.wait(WAIT_SEC)
        else:
            raise Exception(f'Failed to connect to {url} '
                            f'more than max attempts ({MAX_ATTEMPTS}).')

    def check_and_wait_rate_limit(self):
        def get_remaining_and_reset(resource: dict):
            return resource['remaining'], resource['reset']

        for _ in range(MAX_ATTEMPTS):
            content = self.get_content(RATE_LIMIT_URL)
            resources = content['resources']

            search_remaining, search_reset = get_remaining_and_reset(
                resources['search']['/search/tweets'])
            lookup_remaining, lookup_reset = get_remaining_and_reset(
                resources['statuses']['/statuses/lookup'])
            limit_remaining, limit_reset = get_remaining_and_reset(
                resources['application']['/application/rate_limit_status'])

            logger.info(f'[Remaining counts] search: {search_remaining}, '
                        f'lookup: {lookup_remaining}, limit: {limit_remaining}')

            if min(search_remaining, lookup_remaining, limit_remaining) <= 1:
                reset = max(search_reset, lookup_reset, limit_reset)
                self.wait(reset - int(time.time()) + WAIT_SEC)
            else:
                break
        else:
            raise Exception('Failed to wait rate limit')

    @staticmethod
    def status_id_to_timestamp(status_id: int) -> int:
        """Return ms timestamp from status id"""
        return (status_id >> 22) + BASE_TIMESTAMP

    @staticmethod
    def format_text(text: str):
        if text.startswith('RT '):
            return ''

        text = screen_name_pattern.sub('', text)
        text = hash_tag_pattern.sub('', text)
        text = url_pattern.sub('', text)
        text = space_pattern.sub(' ', text)

        # Replace emoji to period
        text = emoji_pattern.sub('。', text)
        text = ''.join(
            c if c not in emoji.UNICODE_EMOJI else '。' for c in text)
        text = period_pattern.sub('。', text)

        # Normalize text
        text = neologdn.normalize(text, repeat=4)
        text = valid_char_pattern.sub('', text)

        return text

    def search_conversations(
            self, query: str, start_timestamp: int,
            len_conversations: int) -> Tuple[int, dict, int]:
        """Search reply tweets include `query` strings after `start_timestamp`
        """

        # Search reply
        params = {'q': query, 'count': 100, 'tweet_mode': 'extended'}
        content = self.get_content(SEARCH_URL, params)
        conversations = {}
        latest_timestamp = start_timestamp
        num_past_statuses = 0
        for status in content['statuses']:
            original_id = status['in_reply_to_status_id_str']
            if original_id is None:
                # Ignore original tweet
                continue

            status_id = status['id_str']
            tweet_timestamp = self.status_id_to_timestamp(int(status_id))
            if tweet_timestamp <= start_timestamp:
                # Ignore past status
                num_past_statuses += 1
                continue

            if latest_timestamp < tweet_timestamp:
                latest_timestamp = tweet_timestamp

            if status['in_reply_to_user_id'] == status['user']['id']:
                continue

            text = self.format_text(status['full_text'])
            if len(text) < MIN_CHARS:
                continue

            conversations[original_id] = Conversation(
                status_id, original_id, text)

        # Get conversations
        for i in range(1, len_conversations):
            ids = ','.join(conversations)
            params = {'id': ids, 'count': len(conversations),
                      'tweet_mode': 'extended'}
            content = self.get_content(LOOKUP_URL, params)

            for status in content:
                original_id = status['in_reply_to_status_id_str']
                if original_id is None:
                    if i == len_conversations - 1:
                        original_id = '0'
                    else:
                        # Ignore original tweet
                        continue

                if status['in_reply_to_user_id'] == status['user']['id']:
                    continue

                text = self.format_text(status['full_text'])
                if len(text) < MIN_CHARS:
                    continue
                if len(self.tokenizer.encode(text)) > self.max_token_length:
                    continue

                status_id = status['id_str']
                conversations[status_id].add_status(status_id, original_id, text)
                conversations[original_id] = conversations.pop(status_id)

            # Remove missing conversations
            conversations = {oid: c for oid, c in conversations.items()
                             if len(c) == i + 1}

            if len(conversations) == 0:
                break

        return latest_timestamp, conversations, num_past_statuses


def main(args):
    collector = TweetCollector(
        api_key=args.api_key, api_secret_key=args.api_secret_key,
        access_token=args.access_token,
        access_token_secret=args.access_token_secret,
        tokenizer_name=args.tokenizer_name,
        max_token_length=args.max_token_length)

    start_timestamp = BASE_TIMESTAMP
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f'tweet_conversations_{args.query}_{args.len_conversations}.tsv'
    filename = os.path.join(args.output_dir, filename)
    total_conversations = 0
    sleep_sec = MIN_SLEEP_SEC
    while True:
        process_start = time.time()
        collector.check_and_wait_rate_limit()
        start_timestamp, conversations, num_past_tweet = \
            collector.search_conversations(
                query=args.query, start_timestamp=start_timestamp,
                len_conversations=args.len_conversations)

        with open(filename, 'a', encoding='utf-8') as f:
            for c in conversations.values():
                line = '\t'.join(c.texts) + '\n'
                f.write(line)

        total_conversations += len(conversations)
        logger.info(f'[Stats] total: {total_conversations}, '
                    f'current: {len(conversations)}')

        process_sec = time.time() - process_start
        if process_sec < sleep_sec:
            collector.wait(sleep_sec - process_sec)
            if num_past_tweet > 0:
                sleep_sec += 1
            elif num_past_tweet == 0 and sleep_sec > MIN_SLEEP_SEC:
                sleep_sec -= 1
            # logger.info(f'Set sleep sec to {sleep_sec} due to '
            #             f'{num_past_tweet} duplicated tweets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, required=True,
                        help='Twitter API key')
    parser.add_argument('--api_secret_key', type=str, required=True,
                        help='Twitter API secret key')
    parser.add_argument('--access_token', type=str, required=True,
                        help='Twitter access token')
    parser.add_argument('--access_token_secret', type=str, required=True,
                        help='Twitter access token secret')
    parser.add_argument('--query', type=str, required=True,
                        help='Query to search tweets')
    parser.add_argument('--len_conversations', type=int, default=5,
                        help='Length of conversations to collect')
    parser.add_argument('--max_token_length', type=int, default=256)
    parser.add_argument('--tokenizer_name', type=str,
                        default='cl-tohoku/bert-base-japanese-whole-word-masking')
    parser.add_argument('--output_dir', type=str, default='dataset/tweet_data')
    args = parser.parse_args()
    main(args)

