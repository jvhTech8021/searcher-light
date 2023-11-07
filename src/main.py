import time
import os
import json
from .helpers import get_report_articles_async
from .prompts.suammarize_prompt import (
    summarize_dense_entity, 
    build_system_prompt, 
    built_gen_sys_prompt, 
    build_article_prompt, 
    build_outline_prompt, 
    style_and_build_html)
from .openai.openai import ask_av_gpt
from .operations import write_to_table

async def build_summary_prompt(num_of_articles, num_of_mcs, report_id, report_creation_query):
    article_creation_input = []
    combined_texts = []

    # Waiting to add report searcher query to this request
    articles = await get_report_articles_async("articles", report_id, num_of_articles, 0)
    for article_url in articles["articleURLS"]:
        # lets use 
        top_7_article_mcs = await get_report_articles_async("microchapters", report_id, num_of_mcs, 0, 'disable', report_creation_query, 'hybrid', article_url)
        for element in top_7_article_mcs["fullResults"]:
            del element["embeddings"]  # remove the "embeddings" field
        article_creation_input.append(top_7_article_mcs)

    for article_text in article_creation_input:
        combined_text = ""

        for text_value in article_text['articleTextValues']:
            combined_text += text_value

        combined_texts.append(combined_text)

    prompt = summarize_dense_entity(combined_texts, 5)
    response = {
        "prompt": prompt,
        "articles": articles
    }
    return response


async def main(report_id):
    # try:
    report_id = report_id
    print('Fetching full report...')
    full_start_time = time.time()
    content_range_response = await get_report_articles_async("articles", report_id, 1, 0)
    report_creation_query = content_range_response["userSearchQuery"]
    print('CREATION QUERY', report_creation_query)

    summary = await build_summary_prompt(5, 5, report_id, report_creation_query)

    gpt_summary_params = {
        "type": "summarize",
        "prompt": summary["prompt"],
        "model": "gpt-4",
        "maxTokens": 1000
    }

    gpt_summary_res = ask_av_gpt(gpt_summary_params)
    summary_content = json.dumps(gpt_summary_res[0]["message"]["content"], separators=(',', ':'))

    summary_content = summary_content.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '')

    system_prompt = build_system_prompt()

    outline_prompt = build_outline_prompt(str(summary_content))

    gpt_outline_params = {
        "type": "outline",
        "prompt": outline_prompt,
        "model": "gpt-4",
        "systemPrompt": system_prompt,
        "maxTokens": 800,
    }

    gpt_outline_res = ask_av_gpt(gpt_outline_params)
    outline_content = json.dumps(gpt_outline_res[0]["message"]["content"], separators=(',', ':'))
    outline_content = outline_content.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '')

    outline_sys_prompt = built_gen_sys_prompt(str(summary_content))
    article_prompt = build_article_prompt(str(outline_content))

    gpt_article_params = {
        "type": "article",
        "prompt": article_prompt,
        "model": "gpt-4",
        "systemPrompt": outline_sys_prompt,
        "maxTokens": 800,
    }

    gpt_article_res = ask_av_gpt(gpt_article_params)
    article_content = json.dumps(gpt_article_res[0]["message"]["content"], separators=(',', ':'))
    article_content = article_content.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '')

    gpt_article_title_params = {
        "type": "title",
        "prompt": f'Write an engaging title for this article. Here is the content: {str(summary_content)}',
        "model": "gpt-3.5-turbo",
        "maxTokens": 50,
    }

    gpt_article_title_res = ask_av_gpt(gpt_article_title_params)
    article_title = json.dumps(gpt_article_title_res[0]["message"]["content"], separators=(',', ':'))
    article_title = article_title.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '')

    gpt_article_summary_params = {
        "type": "summary",
        "prompt": f'Write an engaging summary for this article. Here is the content: {str(article_content)}',
        "model": "gpt-4",
        "maxTokens": 1000,
    }

    gpt_article_summary_res = ask_av_gpt(gpt_article_summary_params)
    article_summary = json.dumps(gpt_article_summary_res[0]["message"]["content"], separators=(',', ':'))
    article_summary = article_summary.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '')

    full_article_elements = {
        "report_id": report_id,
        "title": article_title,
        "summary": article_summary,
        "content": article_content,
        "sources": summary["articles"]
    }

    # print(full_article_elements)

    data_list = [full_article_elements]
    write_to_table("searcher_article_data", data_list)

    fileTitle = report_creation_query.replace(' ', '_')
    with open(f'{fileTitle}.txt', 'w') as f:
        f.write(article_content)
        print(f'{report_creation_query} -- HTML file created')

    full_end_time = time.time()
    full_elapsed_time = full_end_time - full_start_time
    print(f"Full elapsed time: {full_elapsed_time:.2f} seconds.")

    return
