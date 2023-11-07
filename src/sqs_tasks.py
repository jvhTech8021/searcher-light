import boto3
import time
import os
# from dotenv import load_dotenv

# load_dotenv()
ENV = os.getenv('ENVIRONMENT')

# Replace 'your_queue_url' with the actual URL of your SQS queue
DEV_FETCH_QUEUE_URL = ''
DEV_SEND_QUEUE_URL = ''

PROD_FETCH_QUEUE_URL = ''
PROD_TEST_FETCH_QUEUE_URL = ''
PROD_SEND_QUEUE_URL = ''

WAIT_TIME_SECONDS = 180

FETCH_QUEUE_URL = PROD_FETCH_QUEUE_URL if ENV == 'production' else DEV_FETCH_QUEUE_URL
SEND_QUEUE_URL = PROD_SEND_QUEUE_URL if ENV == 'production' else DEV_SEND_QUEUE_URL

sqs = boto3.client('sqs')

def fetch_message():
    try:
      response = sqs.receive_message(
          QueueUrl=FETCH_QUEUE_URL,
          MaxNumberOfMessages=1,
          WaitTimeSeconds=10,
      )
      if len(response.get('Messages', [])) > 0:
        messages = response.get("Messages", [])
        message = messages[0]
        print(message)
        return message
      else:
          return None
    except Exception as e:
      print(f"An error occurred when fetching messages: {e}")
      return None
    
def delete_sqs_message(receipt_handle, report_id):
    # Delete received message from queue
    # *************** !!!!!! uncomment this when live ************
    print('deleting message for report:', report_id)
    sqs.delete_message(
        QueueUrl=FETCH_QUEUE_URL,
        ReceiptHandle=receipt_handle
    )

def send_message(body):
    """
    Send a message to the SQS queue.

    Parameters:
    - body (str): The message body to send to the queue.

    Returns:
    - dict: The response from SQS.
    """
    response = sqs.send_message(
        QueueUrl=SEND_QUEUE_URL,
        MessageBody=body
    )
    return response