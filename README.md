# Asynchronous Report Summarization Tool

## Overview
This Python-based tool automates the generation and summarization of reports using asynchronous operations and AI-powered models to streamline the content creation process.

## Functionality
- **Article Retrieval:** Asynchronously fetches articles based on report IDs and user queries.
- **Summarization:** Summarizes articles, preparing AI model prompts for succinct summaries.
- **Content Creation:** Generates structured content like outlines and full articles from summaries.
- **Engaging Titles and Summaries:** Produces compelling titles and summaries for articles.
- **Data Management:** Writes content to databases and saves articles as text files.

## Components
- `helpers.py`: Functions for asynchronous article fetching.
- `prompts/`: Modules for constructing AI model prompts.
- `openai/`: Interface for AI model interactions, `ask_av_gpt`.
- `operations.py`: Database interaction functionality, `write_to_table`.
