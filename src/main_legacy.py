from nltk.util import ngrams
from collections import defaultdict
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import time
import asyncio
import aiohttp
import os
import traceback
import pandas as pd
from datetime import datetime
from .helpers import get_report_articles_async, cluster_snippets_tfidf, calculate_cosine_similarity, logError, dedupe_final_insights, ask_gpt_async_function, cluster_snippets_vector, process_in_batches_dedupe, truncate_text_to_3000_words, find_average_sentences
from .operations import write_to_table, read_clusters_from_table
from .sqs_tasks import send_message, delete_sqs_message
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
ENV = os.getenv('ENVIRONMENT')

stopwords = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'able', 'about', 'after', 'again', 'all', 'almost',
                 'already', 'also', 'although', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 'be', 'because', 'been',
                 'before', 'being', 'between', 'both', 'but', 'by', 'came', 'can', 'come', 'comes', 'could', 'did', 'do',
                 'does', 'doing', 'done', 'each', 'else', 'etc', 'even', 'every', 'for', 'from', 'gets', 'going', 'got',
                 'had', 'has', 'have', 'here', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'looks',
                 'many', 'may', 'might', 'more', 'most', 'much', 'must', 'never', 'not', 'now', 'of', 'off', 'often',
                 'ok', 'on', 'only', 'or', 'other', 'our', 'out', 'over', 'own', 'rather', 'really', 'see', 'she', 'should',
                 'since', 'some', 'still', 'stuff', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
                 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'up', 'us', 'very', 'via', 'was', 'we', 'well',
                 'were', 'whether', 'while', 'whilst', 'will', 'with', 'within', 'would', 'yes', 'yet', 'you', 'your', 'what',
                 '&', '#', '.', ',',';',"'",':','amp',"!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "-", "=", "{", "}", "[", "]", "|", "\\", ":", ";", "'", "\"", ",", "<", ".", ">", "/", "?", "~", "`"])

search_p_limit = ThreadPoolExecutor(30)
hybrid_search_batch = 50
cluster_cosine_limit = ThreadPoolExecutor(80)
gpt_limit_num = 20
fuzzy_threshold = 0.4
question_fuzzy_threshold = 0.85
fuzzy_question_threshold = 0.6
ngram_q_measurement_limit = 25
batch_size = 1500
gpt_limit = ThreadPoolExecutor(gpt_limit_num)
semaphore = asyncio.Semaphore(30)
# get_questions_and_insights_batch_size = 35
final_insights_semaphore = asyncio.Semaphore(30)
custom_ngram_amount = None
max_retries = 1
full_report_data = {}
blacklisted_urls = set()

measureToQuery = 1
consolidate = 0

chat_gpt_role = "You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where it might help the explanation. If you are provided information to help respond, respond as if you have the knowledge inherently"

def variable_exists(variable_name):
    return variable_name in locals() or variable_name in globals()

async def main(report_id):
    # try:
    report_id = report_id
    print('Fetching full report...')
    full_start_time = time.time()
    fetching_report_start_time = time.time()
    content_range_response = await get_report_articles_async("articles", report_id, 1, 0)
    # print(json.dumps(content_range_response))
    report_creation_query = content_range_response["userSearchQuery"]
    print('CREATION QUERY', report_creation_query)
    moneyball_prompt = f'''Gather all of the most relevant and important information listed in this list of article titles and descriptions, each title and description is seperated by "-". Your job is to create a perfect search query from this text to be used for an embedded index to get a vast yet detailed amount of data. 
    The output must be no longer than 200 words. Ensure the output is in paragraph form and can be read well. Do not include anything in the response other than the query that would be pasted into google. Here is the text:'''
    moneyball= ""
    if(report_creation_query):
        find_top_10_article = await get_report_articles_async("articles", report_id, 10, 0, report_creation_query)
        processed_article_titles_and_desc = []
        for article in find_top_10_article['fullResults']:
            # Check metaDescription, and use searchEngineSnippet if it's empty
            description = article['metaDescription'] if article['metaDescription'] else (article['serpDescription'] if article.get('serpDescription') else '')

            title = article['metaTitle'] if article['metaTitle'] else (article['title'] if article.get('title') else '')

            
            processed_article_titles_and_desc.append({
                'description': description,
                'title': title
            })
        formatted_titles_and_desc = ""
        for entry in processed_article_titles_and_desc:
            formatted_titles_and_desc += "{} - {}\n\n".format(entry["title"], entry["description"])
        cut_article = truncate_text_to_3000_words(formatted_titles_and_desc)
        moneyball = await ask_gpt_async_function(f"{moneyball_prompt} '{cut_article}'", "gpt-4", 0, chat_gpt_role)
    else:
        article_text_values_response = await get_report_articles_async("articles", report_id, content_range_response["contentRange"], 0)
        full_results = article_text_values_response["fullResults"]
        first_ten_titles = [article['metaTitle'] for article in full_results[:10]]
        top_three_avg = await find_average_sentences(first_ten_titles)
        moneyball_query = ''.join(top_three_avg)
        find_top_article = await get_report_articles_async("microchapters", report_id, 0, 0, 'disable', moneyball_query, 'hybrid')
        top_article_url = find_top_article["fullResults"][0]["url"]
        top_article_data = await get_report_articles_async("articles", report_id, 1, 0, 'disable', url=top_article_url)
        top_article_text = top_article_data['articleTextValues'][0]
        cut_article = truncate_text_to_3000_words(top_article_text)
        moneyball = await ask_gpt_async_function(f"{moneyball_prompt} '{cut_article}'", "gpt-4", 0, chat_gpt_role)

    print(moneyball)

    cluster_semaphore = asyncio.Semaphore(45)  # Limit of concurrent tasks
    measure_to_query_semaphore = asyncio.Semaphore(150)  # Limit of 5 concurrent tasks
    gpt_batch_size = 10
    all_texts_by_cluster = {}

    final_insight_prompt = "The text below are the results of a specific semantic search, please identify the question that was likely asked and answer it throughly. Please remember that the question is very specific, so review the text multiple times and deeply consider the options of what the question might be. Assume your first instinct is too broad and iterate deeper. To instill accuracy rely much more heavily on information from the text that is mentioned at least twice or more. Be sure to provide the specific information that is being seeked out including numerical reference data. Answer as if you have the knowledge inherently. If the answer does not exist, find a new question that can be answered. NEVER refer back to the text or indicate that you're getting the answer from it. Here is the text:"
    final_insight_prompt_json = '''The text below are the results of a specific semantic search, please identify the question that was likely asked and answer it throughly. 
    Please remember that the question is specific, so review the text multiple times and deeply consider the options of what the question might be. 
    Assume your first instinct is too broad and iterate deeper. To instill accuracy rely much more heavily on information from the text that is mentioned at least twice or more. 
    Be sure to provide the specific information that is being seeked out including numerical reference data. Answer as if you have the knowledge inherently.
    If the answer does not exist, find a new question that can be answered. NEVER refer back to the text or indicate that you're getting the answer from it. The output MUST be in json format like this: {
    "question": {'question'},
    "answer": {'answer'}}......Here is the text:'''
    question_and_answer_prompt = f'Create one insightful question that could be answered from the text. Then answer the question in a consice manner. The output should be two (2) sentences, one for the question and one for the answer. Here is the text:'
    final_insight_gpt_func = [
                        {
                            "name": "create_question_and_answer",
                            "description": "Question and answer",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "Standalone interesting question that can be answered from the text"
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "Detailed and comprehensive answer to the question that can be answered from the text"
                                    }
                                },
                                "required": [
                                    "question", "answer",
                                ]
                            }
                        }
                    ]
    
    async def gatherData():
        max_moneyball_retries = 3
        retries = 0

        while retries < max_moneyball_retries:
            try:
                money_ball_search_results_range = await get_report_articles_async("microchapters", report_id, 1, 0, 'enable', str(moneyball), 'vectors')
                money_ball_search_results = await get_report_articles_async("microchapters", report_id, money_ball_search_results_range['contentRange'], 1, 'enable', moneyball, 'vectors')
                # If successful, break out of the loop
                break
            except Exception as e:
                # Print or log the error
                # print(f"Error searching moneyball: {e}. Retry {retries + 1} of {max_moneyball_retries}.")
                logError(e, "Error searching moneyball")
                await asyncio.sleep(120)
                retries += 1

        # Optionally handle if all retries failed
        if retries == max_moneyball_retries:
            print("All moneyball retries failed.")

        half = len(money_ball_search_results['fullResults']) // 2  # Find the halfway point, using floor division to ensure an integer
        top_half_results = money_ball_search_results['fullResults'][:half]

        print(f"Top half results length: {len(top_half_results)}")

        return top_half_results
    
    async def clusterData(top_fifty_results):

        number_of_tfidf_clusters = math.ceil(len(top_fifty_results) / 50)
        print("NUMBER OF TFIDF CLUSTERS", number_of_tfidf_clusters)

        snippet_tfidf_clusters = cluster_snippets_tfidf(top_fifty_results, stopwords, number_of_tfidf_clusters)
        # snippet_clusters = cluster_snippets_vector(top_fifty_results, stopwords, 10)

        # print(json.dumps(snippet_clusters))

        # Convert list of dictionaries to DataFrame
        report_df = pd.DataFrame(snippet_tfidf_clusters)

        final_clusters = []  # List to hold the final clustered data

        # Initialize counters for clusters formed and clusters not split into 10
        total_sub_clusters_formed = 0
        clusters_not_split_into_10 = 0

        # Counter to manage the next cluster label
        next_cluster_label = 0

        # Step 2: For each of the 10 clusters, further cluster them using embeddings
        for cluster_label in range(10):
            sub_data = report_df[report_df['dynamic_cluster'] == cluster_label]
            num_sub_clusters = min(len(sub_data), 10)  # Determine number of sub-clusters
            
            # If there's 10 or more records, perform sub-clustering
            if num_sub_clusters >= 10:
                sub_clustered_data = cluster_snippets_vector(sub_data.to_dict('records'), stopwords, 5)

                # Adjusting the labels of the sub-clusters using the global counter
                for record in sub_clustered_data:
                    record['dynamic_cluster'] = next_cluster_label + record['dynamic_cluster']
                
                final_clusters.extend(sub_clustered_data)
                
                # Add to the total count of sub clusters formed
                total_sub_clusters_formed += num_sub_clusters

                # Adjust the global counter
                next_cluster_label += 10

            else:
                # For clusters with fewer than 10 records, keep them in the same cluster
                current_cluster_label = next_cluster_label  # Assign a new cluster label to this cluster
                for record in sub_data.to_dict('records'):
                    record['dynamic_cluster'] = current_cluster_label
                final_clusters.extend(sub_data.to_dict('records'))
                
                next_cluster_label += 1  # Increment the global cluster label for the next cluster
                clusters_not_split_into_10 += 1  # Increment the count of clusters not split into 10

        cluster_counts = defaultdict(int)
        for record in final_clusters:
            cluster_counts[record['dynamic_cluster']] += 1

        # 2. Filter out the clusters with counts of 4 or less.
        # clusters_to_remove = set([cluster for cluster, count in cluster_counts.items() if count < 3])

        clusters_greater_than_3 = sum(1 for count in cluster_counts.values() if count >= 3)


        if clusters_greater_than_3 > 1:
            clusters_to_remove = set(cluster for cluster, count in cluster_counts.items() if count <= 3)
            # print the number of clusters to remove
            print(f"Number of clusters with less than 4: {len(clusters_to_remove)}")
            final_clusters_before_removal = pd.DataFrame(final_clusters)
            # write the report_df to a csv file
            final_clusters_before_removal.to_csv('FINAL_BEFORE_REMOVAL_clusters_report_df.csv')
            filtered_clusters = [record for record in final_clusters if record['dynamic_cluster'] not in clusters_to_remove]
        else:
            filtered_clusters = final_clusters


        # filtered_clusters = [record for record in final_clusters if record['dynamic_cluster'] not in clusters_to_remove]

        print(f"Total sub-clusters formed: {total_sub_clusters_formed}")
        print(f"Number of clusters not split into 10: {clusters_not_split_into_10}")

        print(report_df)

        final_clusters = pd.DataFrame(filtered_clusters)

        # write the report_df to a csv file
        final_clusters.to_csv('FINAL_clusters_report_df.csv')

        final_clusters['moneyball'] = moneyball

        # Sort by dynamic_cluster and then by score in descending order
        final_clusters.sort_values(by=['dynamic_cluster', 'score'], ascending=[True, False])

        # Group by dynamic_cluster and aggregate
        grouped = final_clusters.groupby('dynamic_cluster').agg({
            'mcText': list,
            'Article_Title': list,
            'url': list,
            'datePublished': list,
            'score': list
        }).reset_index()

        # Convert back to a list of dictionaries, where each mcText is represented by its associated Article_Title, url, and datePublished
        result = []
        for _, row in grouped.iterrows():
            cluster_items = [{
                'mcText': mc,
                'Article_Title': title,
                'url': u,
                'datePublished': date,
                'score': s
            } for mc, title, u, date, s in zip(row['mcText'], row['Article_Title'], row['url'], row['datePublished'], row['score'])]
            result.append({
                'dynamic_cluster': row['dynamic_cluster'],
                'items': cluster_items
            })
        
        return result

    def count_words(s):
        return len(s.split())
    
    processed_clusters = []
    skipped_clusters = []

    async def fetch_unique_snippets(query, count):
        unique_snippets = []
        offset = 0
        while len(unique_snippets) < count:
            snippets = await get_report_articles_async("microchapters", report_id, count, offset, 'enable', query, 'hybrid')
            
            for snippet in snippets['fullResults']:
                url = snippet.get('url')  # Assuming the snippet contains a 'url' key. Modify as needed.

                print(url)
                
                if url not in blacklisted_urls:
                    unique_snippets.append(snippet)
                    blacklisted_urls.add(url)
                
                if len(unique_snippets) == count:
                    break
            offset += count
        return unique_snippets

    async def process_cluster(cluster):
        duplicate_question = ''
        async with cluster_semaphore:  # This will make sure we don't exceed the given limit of concurrent tasks

            try:

                clusterContents = cluster['items']

                for item in clusterContents:
                    calculate_similarity_to_moneyball = await calculate_cosine_similarity(item['mcText'], moneyball)
                    item['similarity_to_moneyball'] = calculate_similarity_to_moneyball

                # Sort the items based on similarity_to_moneyball
                sorted_clusterContents = sorted(clusterContents, key=lambda x: x['similarity_to_moneyball'], reverse=True)
                # print('HERE IT IS', json.dumps(clusterContents))

                # ****************************************************************************************************************************************************
                # Check if the number of items in the cluster is greater than 15
                if len(sorted_clusterContents) > 15:
                    print('greater than 15')
                    sorted_clusterContents = sorted_clusterContents[:15]

                combined_mcText = ' '.join([item['mcText'] for item in sorted_clusterContents])
                cluster_id = cluster['dynamic_cluster']

                get_question_answer = await ask_gpt_async_function(f"{question_and_answer_prompt} '{combined_mcText}'", "gpt-3.5-turbo", 0, chat_gpt_role)

                cluster['question_answer'] = get_question_answer

                # print(json.dumps(cluster['question_answer']))

                # Check if the current cluster's question_answer is similar to already processed clusters
                skip_current = False  # Flag to determine if current cluster should be skipped
                for processed_cluster in processed_clusters:
                    similarity = await calculate_cosine_similarity(cluster['question_answer'], processed_cluster)
                    if similarity is not None and similarity > question_fuzzy_threshold:
                        duplicate_question = processed_cluster

                        # Extract the portion after "Answer" from the duplicate string
                        answer_start_index = duplicate_question.find("Answer")
                        if answer_start_index != -1:
                            # Get the part after "Answer"
                            duplicate_answer = duplicate_question[answer_start_index + len("Answer"):].strip()

                            # Append this part to the original string
                            cluster['question_answer'] += f" Additional Answer: {duplicate_answer}"

                        # We still want to process this cluster, but we skip the original duplicate
                        skip_current = True
                        skipped_clusters.append(processed_cluster)  # Assuming skipped_clusters is initialized earlier in your code
                        break

                # If the current cluster's question_answer is not similar to any processed cluster, add it to the list
                if not skip_current:
                    processed_clusters.append(cluster['question_answer'])

                    # print("QUESTION AND ANSWER", str(get_question_answer))
                    search_question_answer = await get_report_articles_async("microchapters", report_id, 15, 0, 'enable', str(get_question_answer), 'hybrid')

                    # search_question_answer = await fetch_unique_snippets(str(get_question_answer), 15)

                    # print('Waiting 60 seconds in between requests')
                    # await asyncio.sleep(60)

                    # gpt_summary = await ask_gpt_async_function(f"Create a detailed overview of this text that aims to explain the topic of the text. The goal is to create a full intersting insight and layout of the text. Build the summary as if it were a journal article: '{combined_mcText}'", "gpt-3.5-turbo", 0)
                    # gpt_title = await ask_gpt_async_function(f"Create a title for this text: '{str(gpt_summary)}'", "gpt-3.5-turbo", 0)

                    combined_text_to_create_insight = ' '.join([entry['mcText'] for entry in search_question_answer['fullResults']])
                    # build_insight_1 = await ask_gpt_async_function(f"{final_insight_prompt} '{str(combined_text_to_create_insight)}'", "gpt-3.5-turbo-0613", final_insight_gpt_func)
                    build_insight_1 = await ask_gpt_async_function(f"{final_insight_prompt_json} '{str(combined_text_to_create_insight)}'", "gpt-3.5-turbo", final_insight_gpt_func)
                    # print(json.dumps(build_insight_1))
                    if isinstance(build_insight_1, dict):
                        if build_insight_1.get("function_call"):
                            function_args = build_insight_1["function_call"]["arguments"]
                            build_insight = json.loads(str(f'{function_args}'))
                            final_insight_question = build_insight["question"]
                            formatted_insight = await ask_gpt_async_function(f'''Add proper HTML formatting to this text so that it is broken up in a organized matter if it is long enough where it is needed. 
                                                                             Do not add anything like a title or header, just add formatting to the text where needed such as line breaks. The output should be HTML like this <p>example text</p> and inlcude line breaks after 4 sentences or so within the <p>.
                                                                             Here is the text: {build_insight["answer"]}''', "gpt-3.5-turbo", 0)
                            final_insight_answer = formatted_insight
                    else:
                        print('GPT did not respond with a function call')
                        print('BEFORE CALL', build_insight_1)
                        build_insight = build_insight_1["content"]
                        json_build_insight = json.loads(str(f'{build_insight}'))
                        print("FUNCTION CALL FAILED", json_build_insight)
                # print(final_insight_question, final_insight_answer)
                    gpt_title = await ask_gpt_async_function(f"Create a title for this text: '{str(build_insight)}'", "gpt-3.5-turbo", 0, chat_gpt_role)

                    # await summarize_and_answer_cluster()

                    full_text_used_for_gpt =  str(question_and_answer_prompt) + ' ' + str(combined_mcText) + ' ' +  str(chat_gpt_role) + ' ' +  str(chat_gpt_role) + ' ' +  str(chat_gpt_role) + ' ' +  str(combined_text_to_create_insight) + ' ' +  str(build_insight)

                    # print(full_text_used_for_gpt)
                    gpt_request_wc = count_words(full_text_used_for_gpt)
                    # print('GPT REQUEST WORD COUNT', gpt_request_wc)

                    all_texts_by_cluster[cluster_id] = combined_mcText
                    cluster['gpt_question_request_text'] = combined_mcText
                    cluster['search_question_answer'] = search_question_answer
                    cluster['gpt_insight_request_text'] = combined_text_to_create_insight
                    cluster['gpt_title'] = gpt_title
                    cluster['build_insight'] = build_insight
                    cluster['final_insight_question'] = final_insight_question
                    cluster['final_insight_answer'] = final_insight_answer

                    final_insight = await ask_gpt_async_function(f"Given this question and answer: '{str(build_insight)}'. summarize and transform this into a consumer-friendly paragraph suitable for an online article that addresses the question.", "gpt-3.5-turbo", 0, chat_gpt_role)
                    cluster['final_insight'] = final_insight

                    # print('CLUSTER  ', json.dumps(cluster))
                
                else:
                    print('DUPLICATE QUESTION FOUND', cluster['question_answer'])
                    cluster['duplicate'] = f"Duplicate question found: {duplicate_question}"

            except Exception as e:
                # print(f"Error processing cluster: {e}")
                logError(e, "Error processing cluster")
                # error_msg = str(e)
                # tb_info = traceback.format_exc()
                # print("Error:", error_msg)
                # print("Traceback:", tb_info)

    async def process_all_clusters(clustered_data, batch_size):
        clusters = list(clustered_data)  # Making sure result is a list
        print(f'Starting to process all ({len(clusters)}) clusters..')

        total_batches = math.ceil(len(clusters) / batch_size)

        for i in range(0, len(clusters), batch_size):
            start_time = time.time()  # Start the timer for the current batch

            batch_num = i // batch_size + 1  # Calculate the current batch number
            batch = clusters[i:i+batch_size]

            max_retries = 3  # Maximum number of retries
            retry_count = 0  # Current retry count

            while retry_count < max_retries:
                try:
                    tasks = [process_cluster(c) for c in batch]
                    await asyncio.gather(*tasks)
                    break  # If successful, break out of the retry loop
                except Exception as e:
                    retry_count += 1
                    print(f"Error processing batch {batch_num}: {e}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        print("Max retries reached. Moving on to the next batch...")
                        # Handle the error as required, for example, log it or notify admins.
            
            # Calculate how much time has passed since the start of the batch
            elapsed_time = time.time() - start_time
            sleep_time = max(60 - elapsed_time, 0)  # Calculate the remaining time up to 60 seconds

            # Print the number of batches completed and remaining
            print(f"Completed batch {batch_num} of {total_batches}.")
            print(f"Batches remaining: {total_batches - batch_num}.")
            
            if batch_num != total_batches:  # Don't wait after processing the last batch
                print(f'Waiting {sleep_time:.2f} seconds...')
                await asyncio.sleep(sleep_time+15)

        final_df_result = pd.DataFrame(clustered_data)

        # Expand the 'items' column
        rows = []
        for _, row in final_df_result.iterrows():
            for item in row['items']:
                item['dynamic_cluster'] = row['dynamic_cluster']
                rows.append(item)

        # Convert the list of dictionaries to a DataFrame
        expanded_df = pd.DataFrame(rows)

        # Write to CSV
        expanded_df.to_csv('cluster_items_output.csv', index=False)
        return final_df_result


    top_fifty_results = await gatherData()
    clustered_data = await clusterData(top_fifty_results)
    insight_creation_data = await process_all_clusters(clustered_data, gpt_batch_size)
    
    # Write the DataFrame to CSV
    insight_creation_data.to_csv(f'cluster_output_{report_id}.csv', index=False)

    print('Running similarity calculation to moneyball...')

    async def compute_similarity(item, moneyBall):
        async with measure_to_query_semaphore:
            insight = f"{item['final_insight_question']} , {item['final_insight_answer']}"
            # print(json.dumps(insight))
            similarity = await calculate_cosine_similarity(f'{insight}', moneyBall)
            return similarity

    async def compute_similarity_for_all_items(df, moneyBall):
        tasks = [compute_similarity(item, moneyBall) for _, item in df.iterrows()]
        similarities = await asyncio.gather(*tasks)

        df['similarity'] = similarities
        df_sorted = df.sort_values(by='similarity', ascending=False)
        return df_sorted

    # 1. Identify and save rows with 'duplicate' column.
    has_duplicate = 'duplicate' in insight_creation_data.columns
    if has_duplicate:
        rows_with_duplicate = insight_creation_data.dropna(subset=['duplicate'])

    # 2. Drop rows with 'duplicate' column from original dataframe.
    if has_duplicate:
        insight_creation_data = insight_creation_data.drop(rows_with_duplicate.index)

    # 3. Process the rows without 'duplicate' column.
    processed_rows = await compute_similarity_for_all_items(insight_creation_data, str(moneyball))

    df_before_dedupe = pd.DataFrame(processed_rows)

    df_before_dedupe.to_csv(f'before_dedupe_{report_id}_sorted.csv', index=False)

    final_dict = processed_rows.to_dict('records')
    initial_count = len(final_dict)

    print('Running dedupe...')

    # deduped_summaries = await process_in_batches_dedupe(final_dict, 500, dedupe_summaries, 0.75)

    deduped_final_insights = await process_in_batches_dedupe(final_dict, 500, dedupe_final_insights, 0.95)

        # Number of items after deduping
    after_dedupe_count = len(deduped_final_insights)

    # Calculate the number of items removed
    deduped_items_removed = initial_count - after_dedupe_count

    print(f'Number of items removed after deduping: {deduped_items_removed}')

    final_sorted_report = pd.DataFrame(deduped_final_insights)

    # Concatenate the processed rows with rows that have 'duplicate'
    if variable_exists('rows_with_duplicate'):
        sorted_df = pd.concat([final_sorted_report, rows_with_duplicate], axis=0).reset_index(drop=True)
    else:
        sorted_df = final_sorted_report

    sorted_df['moneyball'] = moneyball
    sorted_df['report_id'] = report_id
    sorted_df['number_of_insights_deduped'] = deduped_items_removed

    # Get current date and time
    current_timestamp = datetime.now()
    sorted_df['created_at'] = current_timestamp
    # sorted_df['insight_num'] = range(1, len(sorted_df) + 1)

    final_sorted_dict = sorted_df.to_dict('records')

    preview_system_message = '''Your role is to craft concise previews of extensive textual content, enticing readers to delve deeper. 
    Think of these previews as literary teasers, capturing the content's core while igniting curiosity. 
    Here are two examples that include a note about why they are effective: 1.) input: "The Pythagorean theorem is a foundational principle in trigonometry, elucidating the relationship between the sides of a right triangle.
    It's indispensable in myriad fields, ranging from architecture to computer graphics." 
    output: "Discover why a mathematical theorem resonates from buildings to digital screens.”  
    Note: This preview employs curiosity by asking readers to "discover" and juxtaposes seemingly unrelated fields (architecture and digital screens).
    2.) input: "Photosynthesis is a transformative process where green plants harness sunlight for energy. This natural phenomenon underpins the food chain and ensures a consistent energy flow within ecosystems.” 
    output: "Ever wondered how a sunbeam fuels our planet's life?" Note: This preview utilizes a rhetorical question to pique interest, inviting readers to explore a common natural phenomenon's deeper significance. 
    Your goal: kindle interest, driving readers to the full text.'''

    async def createPreivewText(sorted_data):
        for item in sorted_data:
            prompt = f'''Create a preview text between 150 and 250 characters from a larger piece of text and summarize it generally. 
                        Here is the text: "{item['final_insight_answer']}". It must be between 150 and 250 characters. and the output should not be anything besides the preview text. Do not label it as preview text or anything else, just the string of text'''
            preview_text = await ask_gpt_async_function(prompt, "gpt-3.5-turbo", 0, preview_system_message)
            item['preview_text'] = preview_text
        return sorted_data
    
    all_sorted_data = await createPreivewText(final_sorted_dict)
    
    # Sorting the list of dictionaries by 'similarity' in descending order
    all_sorted_data = sorted(final_sorted_dict, key=lambda x: x['similarity'], reverse=True)


    # Getting the top 3
    top_three = all_sorted_data[:3]
    joined_string = ' '.join([item['final_insight_question'] for item in top_three])

    title_builder_role = '''You are a profesional writer which specializes in creating titles for articles which earn a high click through rate. 
                            Here is information you need to reference which provide examples and context about what a good title/headline looks like, review these thouroughly when being asked to create titles. 
                            Here is the information with examples and context: 
                            1.) 5 Tips To Make YOUR videos Cinematic” (2.51% CTR) Context: This headline is effective because it uses the list format which promises readers a defined number of actionable steps or items to explore. The personal touch with "YOUR" also makes it feel tailored for the reader. Guidance for New Headlines: Start with a clear, defined number indicating a list. Make sure the subject of the list promises value or actionable steps. If possible, make it personal. Example: "7 Ways to Boost YOUR Instagram Engagement" 
                            2.) The Best Camera For Beginners In 2022” (7.33% CTR) Context: The appeal here is the direct call-out to beginners. This suggests the content is specifically curated for those just starting out, making it more enticing for that audience.Guidance for New Headlines: Identify a specific group or audience, preferably one that is often looking for guidance (like beginners), and promise tailored advice or solutions for them. Example: "Top 5 Programming Languages for New Coders in 2023"
                            3.) This Will Change The Way You Take Photos Forever” (4.04% CTR) Context: The headline promises a transformative experience or technique that will have lasting effects. This creates intrigue and an urge to find out what this game-changing method might be. Guidance for New Headlines: Use powerful verbs and promises of transformation or long-lasting effects. The idea is to generate curiosity about what this significant change might be.Example: "This Technique Will Revolutionize How You Cook Pasta Forever"
                            4.) Dumb Things Jeff Bezos Wastes His Billions On” (3.13% CTR) Context: The use of negativity and the reference to a well-known personality creates a click-worthy title. The reader might be intrigued about the choices of such a successful person and the audacity of the title to label them as "dumb". Guidance for New Headlines: While treading carefully, consider merging a popular or trending topic with a touch of controversy or unexpected negativity. Example: "Mistakes Elon Musk Made with SpaceX No One Talks About"
                            5.) Hardest Thing Golden Retriever Puppy Owners Go Through” (7.08% CTR) Context: The headline taps into the extremes by using the term "hardest". This can elicit empathy from current dog owners or curiosity from potential dog owners.Guidance for New Headlines: Use superlatives to describe challenges, bests, or worsts. This can spark curiosity as readers will want to know what the extreme aspect is about. Example: "Most Challenging Phase Every New Parent Faces"'''

    report_title = await ask_gpt_async_function(f"Create a unqiue title for this text and ensure the response is only the title and nothing else. Here is the text: {joined_string}", "gpt-3.5-turbo", 0, title_builder_role)

    for item in all_sorted_data:
        item['report_title'] = report_title

    # print(json.dumps(final_sorted_report.to_dict('records')))
    # Sorting the list of dictionaries by the 'similarity' field from highest to lowest
    complete_sorted_data = sorted(all_sorted_data, key=lambda x: x['similarity'], reverse=True)

    final_csv = pd.DataFrame(complete_sorted_data)

    final_csv.to_csv(f'cluster_output_{report_id}_sorted.csv', index=False)

    # Taking the top 50
    # top_50 = complete_sorted_data[:50]

    # remove items that are not related at all
    complete_sorted_data = [item for item in complete_sorted_data if item['similarity'] >= 0.25]

    write_to_table('insight_data_table', complete_sorted_data)


    full_end_time = time.time()
    full_elapsed_time = full_end_time - full_start_time
    print(f"Full elapsed time: {full_elapsed_time:.2f} seconds.")
    # delete_sqs_message(report_id)


    return 
    # except Exception as e:
    #     print(f"Error in report {report_id} creation: {e}")
    #     error_msg = str(e)
    #     tb_info = traceback.format_exc()
    #     print("Error:", error_msg)
    #     print("Traceback:", tb_info)