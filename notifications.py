import requests
import config

def send_plain_message(message):
  return requests.post("https://api.pushover.net/1/messages.json", data = {
    "token": config.api_token,
    "user": config.user_key,
    "message": message
  })

def send_message_with_attachment(message: str, attachment: bytes):
  return requests.post("https://api.pushover.net/1/messages.json", data = {
    "token": config.api_token,
    "user": config.user_key,
    "message": message
  }, files={
    "attachment": attachment
  })
