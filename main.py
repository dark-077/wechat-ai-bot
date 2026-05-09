import base64
import hashlib
import time
import xml.etree.ElementTree as ET
import os
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse, Response
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI()

WECHAT_TOKEN = os.getenv("WECHAT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "你是一个有用、友好的AI助手，运行在微信公众号中。"
    "用中文回答，简洁明了。"
    "如果用户发来图片，请详细描述图片内容或回答用户关于图片的问题。"
)

# ========== 多轮对话记忆 ==========
MEMORY_TTL = timedelta(minutes=30)
MAX_HISTORY = 20

conversations: dict[str, dict] = {}


def get_session(user_id: str) -> dict:
    now = datetime.now()
    session = conversations.get(user_id)
    if session and (now - session["last_active"]) < MEMORY_TTL:
        session["last_active"] = now
    else:
        session = {"messages": [], "last_active": now}
        conversations[user_id] = session
    return session


def save_turn(session: dict, role: str, content):
    session["messages"].append({"role": role, "content": content})
    if len(session["messages"]) > MAX_HISTORY:
        session["messages"] = session["messages"][-MAX_HISTORY:]


# ========== 微信签名验证 ==========
def verify_signature(signature: str, timestamp: str, nonce: str) -> bool:
    tmp_list = sorted([WECHAT_TOKEN, timestamp, nonce])
    tmp_str = "".join(tmp_list)
    calculated = hashlib.sha1(tmp_str.encode("utf-8")).hexdigest()
    return calculated == signature


# ========== XML 解析与构建 ==========
def parse_wechat_xml(xml_data: bytes) -> dict:
    root = ET.fromstring(xml_data)
    return {child.tag: child.text or "" for child in root}


def build_reply_xml(to_user: str, from_user: str, content: str) -> str:
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "</xml>"
    )


# ========== 图片处理 ==========
async def download_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=15) as http_client:
        resp = await http_client.get(url)
        resp.raise_for_status()
        return resp.content


def image_to_vision_block(image_data: bytes) -> dict:
    b64 = base64.b64encode(image_data).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": b64,
        },
    }


# ========== Claude API 调用 ==========
async def call_claude(session: dict, user_content) -> str:
    api_messages = list(session["messages"])
    api_messages.append({"role": "user", "content": user_content})

    resp = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=api_messages,
    )
    return resp.content[0].text


# ========== 路由 ==========
@app.get("/wechat")
async def verify_server(
    signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...),
):
    if verify_signature(signature, timestamp, nonce):
        return PlainTextResponse(echostr)
    return PlainTextResponse("fail", status_code=403)


@app.post("/wechat")
async def handle_message(request: Request):
    body = await request.body()
    msg = parse_wechat_xml(body)

    from_user = msg.get("FromUserName", "")
    to_user = msg.get("ToUserName", "")
    msg_type = msg.get("MsgType", "")
    reply = ""

    session = get_session(from_user)

    if msg_type == "event":
        if msg.get("Event") == "subscribe":
            reply = "你好！我是 AI 助手，随时向我提问。支持文字和图片。"

    elif msg_type == "text":
        user_text = msg.get("Content", "").strip()
        if not user_text:
            reply = "请发送文字消息。"
        else:
            try:
                reply = await call_claude(session, user_text)
                save_turn(session, "user", user_text)
                save_turn(session, "assistant", reply)
            except Exception:
                reply = "抱歉，AI 服务暂时不可用，请稍后再试。"

    elif msg_type == "image":
        pic_url = msg.get("PicUrl", "")
        if not pic_url:
            reply = "无法获取图片链接。"
        else:
            try:
                image_data = await download_image(pic_url)
                vision_block = image_to_vision_block(image_data)
                user_content = [
                    vision_block,
                    {"type": "text", "text": "请描述这张图片的内容"},
                ]
                reply = await call_claude(session, user_content)
                save_turn(session, "user", user_content)
                save_turn(session, "assistant", reply)
            except Exception:
                reply = "抱歉，无法处理这张图片，请稍后再试。"

    else:
        reply = "暂不支持此类型消息，请发文字或图片。"

    if not reply:
        return Response(content="success")

    return Response(content=build_reply_xml(from_user, to_user, reply), media_type="application/xml")


@app.get("/health")
async def health():
    return {"status": "ok"}
