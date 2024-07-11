from linebot.models import FlexSendMessage
from linebot.models import (
    MessageEvent, TextSendMessage
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import (
    AsyncLineBotApi, WebhookParser
)
from fastapi import Request, FastAPI, HTTPException
import google.generativeai as genai
import os
import sys
from io import BytesIO

import aiohttp
import PIL.Image


# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
gemini_key = os.getenv('GEMINI_API_KEY')
imgage_prompt = '''
Describe this image with scientific detail, reply in zh-TW:
'''

if channel_secret is None:
    print('Specify ChannelSecret as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify ChannelAccessToken as environment variable.')
    sys.exit(1)
if gemini_key is None:
    print('Specify GEMINI_API_KEY as environment variable.')
    sys.exit(1)

# Initialize the FastAPI app for LINEBot
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Initialize the Gemini Pro API
genai.configure(api_key=gemini_key)


@app.post("/")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if (event.message.type == "text"):
            # Provide a default value for reply_msg
            msg = event.message.text
            ret = generate_gemini_text_complete(f'你是一位來自臺灣大學機械所製造組的碩二學生，名字叫狗哥，指導教授為覺文郁教授，覺文郁教授會帶我們跨領域研究，實驗室中有其他小組長包含謝東興(興哥)、謝東賢(賢哥)、陳柏宇(哈好美)、許家銘、陳嘉宏，其中謝東興與謝東賢關係不好，謝東興與陳柏宇關係好，謝東興與許家銘關係不好。謝東興的組別專業為IOT聯網。謝東賢組別專業為工具機主軸熱溫升。許家銘組別專業為LRT(Laser R-test)應用與研究，組別內部有甩哥、政中，有實驗室學生對它們有怨言。陳嘉宏組別專業為數位生產履歷。陳柏宇組別專業為工具機。狗哥在實驗中喜歡另一名男學生叫做吳晨仕，狗哥的口頭禪有(他媽的)、(你在演我)、(你在演我)、(1..1111)、(靠夭喔)。請用以上資料回答以下問題:{msg}')
            reply_msg = TextSendMessage(text=ret.text)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
        elif (event.message.type == "image"):
            message_content = await line_bot_api.get_message_content(
                event.message.id)
            image_content = b''
            async for s in message_content.iter_content():
                image_content += s
            img = PIL.Image.open(BytesIO(image_content))

            result = generate_result_from_image(img, imgage_prompt)
            reply_msg = TextSendMessage(text=result.text)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
            return 'OK'
        else:
            continue

    return 'OK'


def generate_gemini_text_complete(prompt):
    """
    Generate a text completion using the generative model.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response


def generate_result_from_image(img, prompt):
    """
    Generate a image vision result using the generative model.
    """

    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    return response
