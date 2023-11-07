import concurrent.futures
import json
import time
import asyncio
from .sqs_tasks import fetch_message, delete_sqs_message, send_message
from .main import main
import traceback


MAX_RETRIES = 3
WAIT_TIME_ON_ERROR = 10  # seconds

def process_task(request, receipt_handle):
    report_id = request['reportId']

    print('Starting report:', report_id)
    for _ in range(MAX_RETRIES):
        try:
            # fetch new message with report_id and run main
            asyncio.run(main(report_id))
            delete_sqs_message(receipt_handle, report_id)
            send_message(json.dumps({'reportId': report_id}))
            print('Finished report:', report_id)
            return
        except Exception as e:
            error_msg = str(e)
            tb_info = traceback.format_exc()
            print(f"Error processing report {report_id}: {error_msg}")
            print("Traceback:", tb_info)
            time.sleep(WAIT_TIME_ON_ERROR)  # wait and then try again
    print(f"Failed to process report {report_id} after {MAX_RETRIES} attempts")

def process_messages(num_threads):
    print('Waiting for new reports...')
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        futures = set()
        while True:
            if len(futures) < num_threads:
                try:
                    message = fetch_message()
                    if message is None:
                        time.sleep(30)
                        continue
                    request = json.loads(message['Body'])
                    future = executor.submit(process_task, request, message['ReceiptHandle'])
                    futures.add(future)
                except Exception as e:
                    print(f"Error fetching or processing message: {e}")
                    time.sleep(WAIT_TIME_ON_ERROR)  # wait before trying again
            else:
                done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                # Checking for exceptions
                for f in done:
                    try:
                        f.result()
                    except Exception as e:
                        print(f"Error in future: {e}")
                
                futures = not_done

if __name__ == "__main__":
    # use num_thread threads so that at any given point of time a max of {num_threads} messages are processed
    num_threads = 1
    process_messages(num_threads)