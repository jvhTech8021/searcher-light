from scipy.cluster.hierarchy import linkage
import traceback
import aiohttp
import json
from urllib.parse import urlencode
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import string
import re
import numpy as np
import os
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


async def search_index_not_encoded(api_url, params):
    try:
        full_url = api_url + "?" + URLSearchParams(params).toString()
        # print(f"Calling API: {full_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url, headers={'': ''}, timeout=100000) as response:
                data = await response.json()
                headers = response.headers
                # print(data, headers)
                return data, headers
    except Exception as error:
        print(
            f'Error in API Request: {full_url} | Response: {response}', error)
        return None, None  # Return a default value indicating an error occurred


async def call_embedding_api_async(input_text):
    # Limiting the number of connections per host
    connector = aiohttp.TCPConnector(limit_per_host=15)
    if input_text in cached_embeddings:
        return cached_embeddings[input_text]

    data = {
        "provider": "sagemaker",
        "input": input_text
    }

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(embedding_api_url, json=data) as response:
                embedding_api_result = await response.json()
                cached_embeddings[input_text] = embedding_api_result
                return embedding_api_result

    except Exception as error:
        logError(error, "call_embedding_api_async")
        # print(f"Error calling embedding API: {error} | Response: {response}")
        return None


async def get_report_articles(type_, report_id, range_, limit, question=None, search_type=None):
    try:
        cache_key = json.dumps({
            "type": type_,
            "reportId": report_id,
            "range": range_,
            "limit": limit,
            "question": question,
            "searchType": search_type
        })
        if cache_key in articles_cache:
            return articles_cache[cache_key]

        filter_params = {
            "reportId": report_id
        }

        if question is not None:
            filter_params["q"] = question

        if search_type is not None:
            filter_params["searchType"] = search_type

        params = {
            "filter": json.dumps(filter_params),
            "range": json.dumps([0, range_]),
            "sort": json.dumps(["score", "ASC"])
        }

        api_url = f"{searcher_api_url}{type_}"
        data, headers = await search_index(api_url, params)

        if data is None or headers is None:
            return {
                "articleTextValues": [],
                "contentRange": None
            }

        user_search_query = data[0].get("searchEngineQuery", "")

        article_text_values = [item.get(
            "articleText", "") for item in data if "articleText" in item and "url" in item]
        article_titles = [item.get("title", "")
                          for item in data if "articleText" in item and "url" in item]
        result_ids = [item.get("id", "")
                      for item in data if "articleText" in item and "url" in item]

        full_results = [{"id": item["id"], "Article_Title": item["title"], "mcText": item.get("articleText", "")}
                        for item in data if "articleText" in item and "url" in item]

        content_range = headers.get("content-range")

        articles_cache[cache_key] = {
            "fullResults": full_results,
            "articleTitles": article_titles,
            "resultIDs": result_ids,
            "articleTextValues": article_text_values,
            "contentRange": content_range,
            "userSearchQuery": user_search_query
        }

        return articles_cache[cache_key]
    except Exception as error:
        print(f"Error searching index: {error}")


async def get_report_articles_async_without_paginate(type_, report_id, range_, limit, enablePhrases=None, question=None, search_type=None, url=None, sort=None):

    max_retries = 3
    retries = 0
    while retries < max_retries:
        user_search_query = ''
        try:
            cache_key = json.dumps({
                "type": type_,
                "reportId": report_id,
                "range": range_,
                "limit": limit,
                "question": question,
                "searchType": search_type
            })
            # if cache_key in articles_cache:
            #     return articles_cache[cache_key]

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

            params = {
                "filter": json.dumps(filter_params),
                "range": json.dumps([0, int(range_)])
            }

            api_url = f"{searcher_api_url}{type_}"
            data, headers = await search_index(api_url, params)

            if data and len(data) > 0 and isinstance(data[0], dict):
                user_search_query = data[0].get("searchEngineQuery", "")
            else:
                print("Data is either empty or not formatted as expected.",
                      api_url, params)

            article_text_values = []
            result_ids = []  # list to store IDs
            article_titles = []  # list to store titles
            article_meta_titles = []
            date_published = []
            full_results = []
            serpDescriptions = []
            metaDescriptions = []
            url = []
            embeddings = []
            value = "articleText" if type_ == "articles" else "mcText"

            for item in data:
                if value in item and "url" in item:
                    article_text_values.append(item[value])
                    result_ids.append(item['id'])  # append ID to list
                    # append title to list
                    article_titles.append(item['title'])
                    article_meta_titles.append(item['metaTitle'])
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
                "serpDescriptions": serpDescriptions,
                "metaDescriptions": metaDescriptions,
                "resultIDs": result_ids,
                "articleTextValues": article_text_values,
                "contentRange": content_range,
                "userSearchQuery": user_search_query
            }
            processing = False
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


async def get_report_articles_async_without_cache(type_, report_id, range_, limit, question=None, search_type=None):
    try:
        filter_params = {
            "reportId": report_id
        }

        if question is not None:
            filter_params["q"] = question

        if search_type is not None:
            filter_params["searchType"] = search_type

        # if enablePhrases is not None:
        #     filter_params["enablePhrases"] = enablePhrases

        params = {
            "filter": json.dumps(filter_params),
            "range": json.dumps([0, range_]),
            "sort": json.dumps(["score", "ASC"])
        }

        api_url = f"{searcher_api_url}{type_}"
        data, headers = await search_index(api_url, params)

        if data is None or headers is None:
            print("No data or headers in get_report_articles_async")
            return {
                "articleTextValues": [],
                "contentRange": None
            }

        user_search_query = data[0].get("searchEngineQuery", "")
        article_text_values = []
        result_ids = []  # list to store IDs
        article_titles = []  # list to store titles
        article_meta_titles = []
        date_published = []
        full_results = []
        url = []
        value = "articleText" if type_ == "articles" else "mcText"

        for item in data:
            if value in item and "url" in item:
                article_text_values.append(item[value])
                result_ids.append(item['id'])  # append ID to list
                article_titles.append(item['title'])  # append title to list
                article_meta_titles.append(item['metaTitle'])
                date_published.append(item['datePublished'])
                url.append(item['url'])

                full_obj = {
                    "id": item['id'],
                    "Article_Title": item['title'],
                    "Meta_title": item['metaTitle'],
                    "datePublished": item['datePublished'],
                    "mcText": item[value],
                    "url": item['url']
                }
                full_results.append(full_obj)

        content_range = headers.get("content-range")

        return {
            "fullResults": full_results,
            "articleTitles": article_titles,
            "artilceMetaTitles": article_meta_titles,
            "resultIDs": result_ids,
            "articleTextValues": article_text_values,
            "contentRange": content_range,
            "userSearchQuery": user_search_query
        }
    except Exception as error:
        print(f"Error searching index: {error}, {data}, {params}")


async def calculate_cosine_similarity(text1, text2):
    max_retries = 2
    retries = 0

    while retries < max_retries:
        try:
            embeddings1 = await call_embedding_api_async(text1)
            embeddings2 = await call_embedding_api_async(text2)

            if embeddings1 and embeddings2:
                return cosine_similarity([embeddings1['vector']], [embeddings2['vector']])[0][0]
            else:
                return None

        except Exception as error:
            retries += 1
            if retries >= max_retries:
                logError(
                    f"Error calculating cosine after {retries} attempts: {error}")
                return None
            continue


async def find_average_sentences(sentences):
    try:
        num_sentences = len(sentences)

        # Create a matrix for storing pairwise similarities
        similarity_matrix = np.zeros((num_sentences, num_sentences))

        # Calculate pairwise cosine similarities
        for i in range(num_sentences):
            for j in range(num_sentences):
                if i != j:  # Don't compute similarity of a sentence with itself
                    similarity = await calculate_cosine_similarity(sentences[i], sentences[j])
                    # print(f'similarity: {similarity} between {sentences[i]} and {sentences[j]}')
                    similarity_matrix[i][j] = similarity if similarity else 0

        # Sum the similarities for each sentence
        summed_similarities = np.sum(similarity_matrix, axis=1)

        # Get indices of the top 3 sentences based on summed similarities
        top_indices = summed_similarities.argsort()[-3:][::-1]

        # Return the top 3 average sentences
        return [sentences[i] for i in top_indices]
    except Exception as error:
        logError(f"Error finding average sentences: {error}")


async def dedupe_ngrams(ngrams, fuzzy_threshold):
    try:
        copied_ngrams = ngrams.copy()
        semaphore = asyncio.Semaphore(1500)  # limit concurrent writes

        async def compare_and_dedupe(i):
            nonlocal copied_ngrams
            is_similar_ngram_found = False
            for j in range(i + 1, len(copied_ngrams)):
                similarity = await calculate_cosine_similarity(copied_ngrams[i]['ngram'], copied_ngrams[j]['ngram'])
                if similarity is not None and similarity > fuzzy_threshold:
                    async with semaphore:
                        # Check the similarity property and pop the one with lower similarity
                        if copied_ngrams[i]['similarity'] < copied_ngrams[j]['similarity']:
                            copied_ngrams.pop(i)
                        else:
                            copied_ngrams.pop(j)
                        is_similar_ngram_found = True
                        break

            return not is_similar_ngram_found

        i = 0
        while i < len(copied_ngrams):
            should_increment = await compare_and_dedupe(i)
            if should_increment:
                i += 1

        return copied_ngrams
    except Exception as error:
        print(f"Error dedupe ngrams: {error}")


async def dedupe_questions(questions, fuzzy_threshold):
    try:
        copied_questions = questions.copy()
        semaphore = asyncio.Semaphore(1500)  # limit concurrent writes

        async def compare_and_dedupe(i):
            nonlocal copied_questions
            is_similar_ngram_found = False
            for j in range(i + 1, len(copied_questions)):
                similarity = await calculate_cosine_similarity(copied_questions[i]['question'], copied_questions[j]['question'])
                if similarity is not None and similarity > fuzzy_threshold:
                    async with semaphore:
                        # Check the similarity property and pop the one with lower similarity
                        if copied_questions[i]['similarity'] < copied_questions[j]['similarity']:
                            copied_questions.pop(i)
                        else:
                            copied_questions.pop(j)
                        is_similar_ngram_found = True
                        break

            return not is_similar_ngram_found

        i = 0
        while i < len(copied_questions):
            should_increment = await compare_and_dedupe(i)
            if should_increment:
                i += 1

        return copied_questions
    except Exception as error:
        print(f"Error dedupe ngrams: {error}")


async def dedupe_summaries(summaries, fuzzy_threshold):
    try:
        copied_summaries = summaries.copy()
        semaphore = asyncio.Semaphore(1500)  # limit concurrent writes

        async def compare_and_dedupe(i):
            nonlocal copied_summaries
            is_similar_ngram_found = False
            for j in range(i + 1, len(copied_summaries)):
                similarity = await calculate_cosine_similarity(str(copied_summaries[i]['build_insight']), str(copied_summaries[j]['build_insight']))
                if similarity is not None and similarity > fuzzy_threshold:
                    async with semaphore:
                        # Check the similarity property and pop the one with lower similarity
                        if copied_summaries[i]['similarity'] < copied_summaries[j]['similarity']:
                            copied_summaries.pop(i)
                        else:
                            copied_summaries.pop(j)
                        is_similar_ngram_found = True
                        break

            return not is_similar_ngram_found

        i = 0
        while i < len(copied_summaries):
            should_increment = await compare_and_dedupe(i)
            if should_increment:
                i += 1

        return copied_summaries
    except Exception as error:
        print(f"Error dedupe summaries: {error}")


async def dedupe_final_insights(insights, fuzzy_threshold):
    try:
        copied_insights = insights.copy()
        semaphore = asyncio.Semaphore(1500)  # limit concurrent writes

        async def compare_and_dedupe(i):
            nonlocal copied_insights
            is_similar_ngram_found = False
            for j in range(i + 1, len(copied_insights)):
                similarity = await calculate_cosine_similarity(str(copied_insights[i]['final_insight_question']), str(copied_insights[j]['final_insight_question']))
                if similarity is not None and similarity > fuzzy_threshold:
                    async with semaphore:
                        # Check the similarity property and pop the one with lower similarity
                        if copied_insights[i]['similarity'] < copied_insights[j]['similarity']:
                            copied_insights.pop(i)
                        else:
                            copied_insights.pop(j)
                        is_similar_ngram_found = True
                        break

            return not is_similar_ngram_found

        i = 0
        while i < len(copied_insights):
            should_increment = await compare_and_dedupe(i)
            if should_increment:
                i += 1

        return copied_insights
    except Exception as error:
        print(f"Error dedupe summaries: {error}")


async def dedupe_question_answer(qANDa, fuzzy_threshold):
    try:
        copied_questions = qANDa.copy()
        semaphore = asyncio.Semaphore(1500)  # limit concurrent writes

        async def compare_and_dedupe(i):
            nonlocal copied_questions
            is_similar_ngram_found = False
            for j in range(i + 1, len(copied_questions)):
                similarity = await calculate_cosine_similarity(copied_questions[i]['question_answer'], copied_questions[j]['question_answer'])
                if similarity is not None and similarity > fuzzy_threshold:
                    async with semaphore:
                        # Check the similarity property and pop the one with lower similarity
                        if copied_questions[i]['similarity'] < copied_questions[j]['similarity']:
                            copied_questions.pop(i)
                        else:
                            copied_questions.pop(j)
                        is_similar_ngram_found = True
                        break

            return not is_similar_ngram_found
        i = 0
        while i < len(copied_questions):
            should_increment = await compare_and_dedupe(i)
            if should_increment:
                i += 1

        return copied_questions

    except Exception as error:
        logError(f"Error dedupe question answer: {error}")
        # print(f"Error dedupe question answer: {error}")


async def process_in_batches(items, batch_size, process_item):
    try:
        coroutines = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            coroutine = process_item(batch)
            coroutines.append(coroutine)

        await asyncio.gather(*coroutines)
    except Exception as error:
        print(f"Error process in batches: {error}")


async def process_in_batches_wait(items, batch_size, process_item):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        # tasks = [process_item(item) for item in batch]
        coroutine = [process_item(batch)]

        # Wait for all tasks in the current batch to complete
        await asyncio.gather(*coroutine)


async def process_in_batches_dedupe(ngrams, batch_size, dedupe_func, fuzzy_threshold):
    try:
        results = []
        total_batches = len(ngrams) // batch_size + \
            (1 if len(ngrams) % batch_size != 0 else 0)
        completed_batches = 0
        # Adjust the value based on your desired concurrency
        semaphore = asyncio.Semaphore(5)

        async def process_batch(batch):
            nonlocal completed_batches
            async with semaphore:
                batch_result = await dedupe_func(batch, fuzzy_threshold)
                results.extend(batch_result)
                completed_batches += 1
                remaining_batches = total_batches - completed_batches
                print(
                    f"Completed batch {completed_batches}/{total_batches}. {remaining_batches} batches remaining.")

        tasks = []
        for i in range(0, len(ngrams), batch_size):
            batch = ngrams[i:i + batch_size]
            print(len(batch))
            tasks.append(process_batch(batch))

        await asyncio.gather(*tasks)

        return results
    except Exception as error:
        logError(f"Error process in batches dedupe: {error}")
        # print(f"Error process in batches dedupe: {error}")


def preprocess_text(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    except Exception as error:
        print(f"Error preprocess text: {error}")


def determine_optimal_distance(tfidf_matrix):
    Z = linkage(tfidf_matrix.toarray(), method='ward')
    distances = Z[:, 2]
    distance_diff = np.diff(distances)
    optimal_idx = np.argmax(distance_diff)
    return distances[optimal_idx]


def cluster_questions(raw_data, distance_threshold):

    try:
        # Load the data
        data_list = raw_data['questions_and_insights']

        # Convert list of dictionaries to DataFrame
        data = pd.DataFrame(data_list)

        # Preprocess the questions
        data['Processed Question'] = data['question'].apply(preprocess_text)

        # Transform the processed text data into a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['Processed Question'])

        # Use the AgglomerativeClustering algorithm on this matrix to cluster the questions
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage='average')
        model.fit(tfidf_matrix.toarray())

        # Add the cluster labels to the data
        # Adjust cluster labels to start from 1
        data['cluster'] = model.labels_ + 1

        # drop duplicates
        data = data.drop_duplicates(subset='question')

        # Create a new DataFrame with 'Question', 'Insight' and 'Cluster'
        output_data = data[['question', 'insight', 'similarity', 'cluster']]

        return output_data.to_dict('records')
    except Exception as error:
        print(f"Error cluster questions: {error}")


def cluster_insights(raw_data, distance_threshold):
    try:
        # Convert list of dictionaries to DataFrame
        data = pd.DataFrame(raw_data)

        # Preprocess the questions
        data['Processed Insight'] = data['insight'].apply(preprocess_text)

        # Transform the processed text data into a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['Processed Insight'])

        # Use the AgglomerativeClustering algorithm on this matrix to cluster the questions
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage='average')
        model.fit(tfidf_matrix.toarray())

        # Add the cluster labels to the data
        # Adjust cluster labels to start from 1
        data['cluster'] = model.labels_ + 1

        # drop duplicates
        data = data.drop_duplicates(subset='insight')

        # Create a new DataFrame with 'Question', 'Insight' and 'Cluster'
        output_data = data[['question', 'insight', 'cluster', 'prev_count']]

        return output_data.to_dict('records')
    except Exception as error:
        print(f"Error cluster insights: {error}")


def cluster_snippets_old(raw_data):
    try:
        # Convert list of dictionaries to DataFrame
        data = pd.DataFrame(raw_data)

        # Check if 'mcText' exists in the data
        if 'mcText' not in data.columns:
            raise ValueError("'mcText' field not found in the provided data")

        # Preprocess the questions
        data['Processed Snippet'] = data['mcText'].apply(preprocess_text)

        # Transform the processed text data into a matrix of TF-IDF features
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data['Processed Snippet'])

        # Get the optimal distance threshold
        distance_threshold = determine_optimal_distance(tfidf_matrix)

        # Use the AgglomerativeClustering algorithm on this matrix to cluster the questions
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage='average')
        model.fit(tfidf_matrix.toarray())

        # Add the cluster labels to the data
        # Adjust cluster labels to start from 1
        data['cluster'] = model.labels_ + 1

        data['distance'] = distance_threshold

        # Drop duplicates
        data = data.drop_duplicates(subset='mcText')

        # Create a new DataFrame with 'Question', 'Insight' and 'Cluster'
        output_data = data[['mcText', 'cluster', 'distance']]

        return output_data.to_dict('records')
    except Exception as error:
        raise ValueError(f"Error cluster insights: {error}")


def cluster_snippets_elbow(input_data, stopwords_list):
    print("Starting the clustering process...")

    try:
        # 1. Convert list of dictionaries to DataFrame
        data = pd.DataFrame(input_data)

        # 2. Preprocessing
        print("Preprocessing text data...")

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = text.split()
            tokens = [token for token in tokens if token not in stopwords_list]
            return ' '.join(tokens)

        data['processed_mcText'] = data['mcText'].apply(preprocess_text)

        # 3. TF-IDF vectorization
        print("Vectorizing text data...")
        tfidf_vectorizer = TfidfVectorizer(max_features=800)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_mcText'])

        # 4. Determining the dynamic threshold using the elbow method
        print("Determining dynamic threshold using the elbow method...")

        def find_elbow(data):
            first_point = [0, data[0]]
            last_point = [len(data) - 1, data[-1]]
            m = (last_point[1] - first_point[1]) / \
                (last_point[0] - first_point[0])
            c = first_point[1] - m * first_point[0]
            distances = []
            for i, y in enumerate(data):
                x = i
                line_y = m*x + c
                distance = np.abs(y - line_y) / np.sqrt(1 + m**2)
                distances.append(distance)
            return np.argmax(distances)

        Z = linkage(tfidf_matrix.toarray(), method='ward')
        merging_distances = Z[:, 2]
        sorted_distances = np.sort(merging_distances)[::-1]
        elbow_index = find_elbow(sorted_distances)
        elbow_distance = sorted_distances[elbow_index]-0.2

        # 5. Applying Agglomerative Clustering with the dynamic threshold
        print(
            f"Applying Agglomerative Clustering with a {elbow_distance} threshold...")
        dynamic_clusterer = AgglomerativeClustering(
            distance_threshold=elbow_distance, n_clusters=None, linkage='ward')
        dynamic_cluster_labels = dynamic_clusterer.fit_predict(
            tfidf_matrix.toarray())
        data['dynamic_cluster'] = dynamic_cluster_labels
        data['cluster_threshold'] = elbow_distance

        # 6. Convert DataFrame to list of dictionaries
        output_data = data.to_dict('records')

        print("\nClustering process completed!")

        return output_data
    except Exception as error:
        logError(error, "Error in cluster_snippets()")


def cluster_snippets_tfidf(input_data, stopwords_list, n_clusters):
    print("Starting the clustering process...")

    try:
        # 1. Convert list of dictionaries to DataFrame
        data = pd.DataFrame(input_data)

        data.to_csv('before_clustering_data_tfidf.csv')
        print("Data saved to before_clustering_data_tfidf.csv")

        # 2. Preprocessing
        print("Preprocessing text data...")

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = text.split()
            tokens = [token for token in tokens if token not in stopwords_list]
            return ' '.join(tokens)

        data['processed_mcText'] = data['mcText'].apply(preprocess_text)

        # 3. TF-IDF vectorization
        print("Vectorizing text data...")
        tfidf_vectorizer = TfidfVectorizer(max_features=800)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_mcText'])

        # 4. Applying Agglomerative Clustering with the predefined number of clusters
        print(
            f"Applying Agglomerative Clustering with {n_clusters} clusters...")
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(tfidf_matrix.toarray())
        data['dynamic_cluster'] = cluster_labels

        # 5. Calculate average cluster size
        cluster_sizes = data.groupby('dynamic_cluster').size()
        avg_cluster_size = cluster_sizes.mean()
        data['avg_cluster_size'] = avg_cluster_size

        data.to_csv('after_clustering_data_tfidf.csv')
        print("Data saved to after_clustering_data_tfidf.csv")

        # 5. Convert DataFrame to list of dictionaries
        output_data = data.to_dict('records')

        print("\nClustering process completed!")

        return output_data
    except Exception as error:
        logError(error, "Error in cluster_snippets_tfidf()")


def cluster_snippets_vector(input_data, stopwords_list, n_clusters):
    print("Starting the clustering process...")

    try:
        # 1. Convert list of dictionaries to DataFrame
        data = pd.DataFrame(input_data)

        # data.to_csv('before_clustering_data.csv')
        # print("Data saved to before_clustering_data.csv")

        # 2. Preprocessing
        print("Preprocessing text data...")

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = text.split()
            tokens = [token for token in tokens if token not in stopwords_list]
            return ' '.join(tokens)

        data['processed_mcText'] = data['mcText'].apply(preprocess_text)

        # 3. Extract embeddings from data
        print("Extracting embeddings from data...")
        embeddings = np.stack(data['embeddings'].to_numpy())

        # 4. Applying Agglomerative Clustering with the predefined number of clusters
        print(
            f"Applying Agglomerative Clustering with {n_clusters} clusters...")
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward')
        cluster_labels = clusterer.fit_predict(embeddings)
        data['dynamic_cluster'] = cluster_labels

        # data.to_csv('final_clustering_data.csv')
        # print("Data saved to after_clustering_data.csv")

        # 5. Convert DataFrame to list of dictionaries
        output_data = data.to_dict('records')

        print("\nClustering process completed!")

        return output_data
    except Exception as error:
        logError(error, "Error in cluster_snippets_vector()")
        

async def ask_gpt_async_function(question, model, function=question_and_insight_output, role=None, retries=max_retries):
    try:
        if (function == 0):
            request_data = {
                "promptOpts": {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            }
        else:
            request_data = {
                "promptOpts": {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    "functions": function,
                    "function_call": "auto"
                }
            }

        if (role != None):
            # If 'role' exists, append a new message to the 'messages' list
            request_data["promptOpts"]["messages"].append({
                "role": "system",
                "content": role
            })
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, json=request_data) as response:
                    gpt_api_result = await response.json()
                    if 'choices' in gpt_api_result and isinstance(gpt_api_result['choices'], list) and len(gpt_api_result['choices']) > 0:
                        if (function == 0):
                            return gpt_api_result['choices'][0]['message']['content']
                        else:
                            return gpt_api_result['choices'][0]['message']
                    else:
                        print(gpt_api_result)
                        raise Exception('ask_gpt call failed')
                    # return api_result['choices'][0]['message']['content']
            except Exception as err:
                logError(err, 'ask_gpt call failed')
                if retries > 0:
                    print(
                        f"Retrying in 15 seconds... ({retries} retries left)")
                    await asyncio.sleep(15)  # Wait for 15 seconds
                    # Retry the same function
                    async with session.post(api_url, json=request_data) as response:
                        gpt_api_result = await response.json()
                        if 'choices' in gpt_api_result and isinstance(gpt_api_result['choices'], list) and len(gpt_api_result['choices']) > 0:
                            if (function == 0):
                                return gpt_api_result['choices'][0]['message']['content']
                            else:
                                return gpt_api_result['choices'][0]['message']
                        else:
                            print(gpt_api_result)
                            raise Exception(
                                f'ask_gpt retry {retries} call failed')
                else:
                    print('No more retries left. Operation failed.')
    except Exception as error:
        logError(error, 'ask_gpt_async_function', request_data)


def restructure_questions_and_insights(data):
    try:
        dict = json.loads(data)
        structured_data = []
        for i in range(1, len(dict)//2 + 1):
            item = {
                'question': dict[f'question{i}'],
                'insight': dict[f'insight{i}'],
            }
            structured_data.append(item)
        return structured_data
    except Exception as error:
        logError(error, 'restructure_questions_and_insights')
        # print(f"Error restructure questions and insights: {error}")


def truncate_text_to_3000_words(text):
    words = text.split()

    if len(words) <= 2500:
        return text

    truncated_words = words[:2500]
    truncated_text = ' '.join(truncated_words)

    return truncated_text
