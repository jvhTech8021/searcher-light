
import traceback
import json
from urllib.parse import urlencode
import asyncio
import os
import aiohttp
from dotenv import load_dotenv
load_dotenv()

ENV = os.getenv('ENVIRONMENT')
api_url = ''
api_url_completion = ''

# NEED TO MAKE 'enviormnet' VARIABLE logic based
embedding_api_url = ''
dev_searcher_api_url = ''
prod_searcher_api_url = ''
elastic_API_key = os.getenv('')

# ternary operator to set elastic_APO_key based on ENV
# elastic_API_key = prod_elastic_API_key if ENV == 'production' else dev_elastic_API_key

# ternary operator to set searcher_api_url based on ENV
searcher_api_url = prod_searcher_api_url if ENV == 'production' else dev_searcher_api_url

print(searcher_api_url)

cached_embeddings = {}
articles_cache = {}
max_retries = 1

question_and_insight_output = [
    {
        "name": "get_questions_and_insights",
        "description": "Analyze the text and create 5 to 10 standalone insights. For each insight, formulate a question that this insight answers. Remember, the goal is not to make questions that directly reference the text, but rather to derive valuable insights that can stand independently of the original text.",
        "parameters": {
            "type": "object",
            "properties": {
                "question1": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight1": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question2": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight2": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question3": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight3": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question4": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight4": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question5": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight5": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question6": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight6": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question7": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight7": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question8": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight8": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question9": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight9": {
                    "type": "string",
                    "description": "One insight to answer the question"
                },
                "question10": {
                    "type": "string",
                    "description": "One interesting question from the text"
                },
                "insight10": {
                    "type": "string",
                    "description": "One insight to answer the question"
                }
            },
            "required": [
                "question1", "insight1",
                "question2", "insight2",
                "question3", "insight3",
                "question4", "insight4",
                "question5", "insight5",
                "question6", "insight6",
                "question7", "insight7",
                "question8", "insight8",
                "question9", "insight9",
                "question10", "insight10",
            ]
        }
    }
]

final_insight = [
    {
        "name": "create_question_and_answer",
        "description": "The text below are the results of a specific semantic search, please identify the question that was likely asked and answer it throughly. Please remember that the question is very specific, so review the text multiple times and deeply consider the options of what the question might be. Assume your first instinct is too broad and iterate deeper. To instill accuracy rely much more heavily on information from the text that is mentioned at least twice or more. Be sure to provide the specific information that is being seeked out including numerical reference data. Answer as if you have the knowledge inherently. Do not refer back to the text or indicate that you're getting the answer from it. Just provide a straightforward response of a question and answer. Here is the text",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Standalone interesting question from the text"
                },
                "answer": {
                    "type": "string",
                    "description": "Detailed and comprehensive answer to the question"
                }
            },
            "required": [
                "question1", "insight1",
            ]
        }
    }
]


class URLSearchParams:
    def __init__(self, params):
        self.params = params

    def toString(self):
        return "&".join([f"{key}={value}" for key, value in self.params.items()])


def logError(error, method_name, extra=None):
    error_msg = str(error)
    tb_info = traceback.format_exc()
    print(f"Issue with {method_name} call:", error_msg)
    print("Traceback:", tb_info)
    print("Extra Details:", extra)


async def search_index(api_url, params):
    try:
        # Encode the parameters
        query_string = urlencode(params)

        # Construct the full URL
        full_url = api_url + "?" + query_string
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url, verify_ssl=True, headers={'jwt': elastic_API_key}, timeout=100000) as response:
                # data = await response.json()
                try:
                    data = await response.json()
                except Exception:
                    print("Index Response is not JSON.", full_url)
                    return None
                headers = response.headers
                if data is None or headers is None:
                    print("No data or headers in get_report_articles_async")
                    # raise Exception(
                    #     "No data or headers in get_report_articles_async")
                return data, headers

    except aiohttp.ClientError as error:
        logError(error, "search_index()")
        print(
            f'Error in API Request: {full_url} | Response: {response} | Error: {error}')
        # return None, None  # Return a default value indicating an error occurred

async def get_paginated_data(api_url, filter_params, start=0, step=500, range_=None):
    master_data = []

    while True:
        # If range_ is specified and the start exceeds it, break the loop
        if range_ and start >= range_:
            break

        params = {
            "filter": json.dumps(filter_params),
            "range": json.dumps([start, min(start + step - 1, range_ - 1) if range_ else start + step - 1])
        }

        data, headers = await search_index(api_url, params)

        if not data:
            break

        master_data.extend(data)

        # Check if fetched data size is less than step, which means we've fetched all available data
        if len(data) < step:
            break

        start += step

        # If range_ is specified and we've fetched enough data, break the loop
        if range_ and len(master_data) >= range_:
            break

    return master_data, headers


"""
This method retrieves articles for a given report ID.

Parameters:
type_ (str): The type of report.
report_id (str): The ID of the report.
range_ (int): The number of articles to retrieve. Default is 100.
fetchAll (int): Whether to fetch all articles or not. Default is 0.
enablePhrases (str): Whether to enable phrases or not. Default is "enable".
question (str): The question to search for.
search_type (str): The type of search to perform.
url (str): The URL to search for.
sort (str): The sorting order for the results.

Returns:
A list of articles.
"""

async def get_report_articles_async(type_, report_id, range_=100, fetchAll=0, enablePhrases="enable", question=None, search_type=None, url=None, sort=None):

    max_retries = 3
    retries = 0
    while retries < max_retries:
        user_search_query = ''
        try:
            cache_key = json.dumps({
                "type": type_,
                "reportId": report_id,
                "range": range_,
                "limit": fetchAll,
                "question": question,
                "searchType": search_type
            })

            filter_params = {
                "reportId": report_id
            }

            if question is not None:
                filter_params["q"] = question

            if url is not None:
                filter_params["url"] = url

            if sort is not None:
                filter_params["sort"] = sort

            if search_type is not None:
                filter_params["searchType"] = search_type

            if enablePhrases is not None:
                filter_params["enablePhrases"] = enablePhrases

            api_url = f"{searcher_api_url}{type_}"

            # Get all paginated data
            if (fetchAll == 1):
                data, headers = await get_paginated_data(api_url, filter_params)
            else:
                data, headers = await get_paginated_data(api_url, filter_params, range_=int(range_))

            # Rest of the processing logic as in your code
            if data and len(data) > 0 and isinstance(data[0], dict):
                user_search_query = data[0].get("searchEngineQuery", "")
            else:
                print("Data is either empty or not formatted as expected.",
                      api_url, filter_params)

            # ... [initialize lists and start data processing]
            article_text_values = []
            result_ids = []  # list to store IDs
            article_titles = []  # list to store titles
            article_meta_titles = []
            article_urls = []
            date_published = []
            full_results = []
            serpDescriptions = []
            metaDescriptions = []
            url = []
            embeddings = []
            value = "articleText" if type_ == "articles" else "mcText"

            for item in data:
                # ... [your data processing logic for each item in data]
                if value in item and "url" in item:
                    article_text_values.append(item[value])
                    result_ids.append(item['id'])  # append ID to list
                    # append title to list
                    article_titles.append(item['title'])
                    article_meta_titles.append(item['metaTitle'])
                    article_urls.append(item['url'])
                    if 'searchEngineSnippet' in item:
                        serpDescriptions.append(item['searchEngineSnippet'])
                    else:
                        serpDescriptions.append(None)  # or some default value

                    # Check if metaDescription exists for the item before using it
                    if 'metaDescription' in item:
                        metaDescriptions.append(item['metaDescription'])
                    else:
                        metaDescriptions.append(None)  # or some default value

                    date_published.append(item['datePublished'])
                    url.append(item['url'])
                    embeddings.append(item['embeddings'])

                    full_obj = {
                        "id": item['id'],
                        "Article_Title": item['title'] if 'title' in item else None,
                        "metaTitle": item['metaTitle'] if 'metaTitle' in item else None,
                        'serpDescription': item['searchEngineSnippet'] if 'searchEngineSnippet' in item else None,
                        'metaDescription': item['metaDescription'] if 'metaDescription' in item else None,
                        "datePublished": item['datePublished'],
                        "mcText": item[value],
                        "url": item['url'] if 'url' in item else None,
                        "score": item['score'] if 'score' in item else None,
                        "embeddings": item['embeddings'] if 'embeddings' in item else None
                    }
                    full_results.append(full_obj)

            content_range = headers.get("content-range")

            articles_cache[cache_key] = {
                "fullResults": full_results,
                "articleTitles": article_titles,
                "articleMetaTitles": article_meta_titles,
                "articleURLS": article_urls,
                "serpDescriptions": serpDescriptions,
                "metaDescriptions": metaDescriptions,
                "resultIDs": result_ids,
                "articleTextValues": article_text_values,
                "contentRange": content_range,
                "userSearchQuery": user_search_query
            }
            return articles_cache[cache_key]

        except Exception as error:
            if retries == max_retries - 1:
                logError(
                    error, "get_report_articles_async maxed out retries", cache_key)
                raise
            retries += 1
            logError(error, "get_report_articles_async", cache_key)
            print('Waiting and Retrying...')
            await asyncio.sleep(10)