import requests

av_open_ai_url = ""

def ask_av_gpt(params, retries=3):
    # Body data
    data = {
        "text": params["prompt"],
        "model": params["model"],
        "maxTokens": params["maxTokens"]
    }

    type = params["type"]

    headers = {
        "User-Agent": "PostmanRuntime/7.33.0",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    for i in range(retries):
        response = requests.post(
            av_open_ai_url, json=data, headers=headers, timeout=10000)

        if response.status_code == 200:
            json_result = response.json()
            print('USAGE', json_result["output"]["data"]["usage"])
            print(f"Successfully received response: {type}")
            # print(json_result["output"]["data"]["choices"])
            return json_result["output"]["data"]["choices"]
        else:
            print(
                f"Failed to get response. Status code: {response.status_code}")
            if i < retries - 1:
                print(f"Retrying... ({i+1})")
                # Use time.sleep for synchronous sleep
                import time
                time.sleep(2 ** i)  # Exponential backoff
            else:
                print("Max retries reached. Exiting.")
                return None
