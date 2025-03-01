from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters
from lingua import Language, LanguageDetectorBuilder
from ffmpeg import FFmpeg
from openai import OpenAI

import requests
import whisper
import tempfile
import edge_tts
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
TG_TOKEN = "7861410110:AAESUKyflijY3CR65IHd7BGOSveM-6a9H_k"
whisper = whisper.load_model("turbo")
deepseek = OpenAI(api_key=os.environ["TRANSLATOR_API_KEY"], base_url="https://api.deepseek.com")

system_prompt = """Ты — языковой переводчик.
Ты знаешь следующие языки: русский, английский, немецкий, французский, испанский, китайский, японский.
По умолчанию общаешься на русском.
Отвечай начиная с "voice:" если собеседник явно хочет услышать ответ голосом.
Если пользователь просит выполнить перевод, то отвечай как google translate - только перевод без пояснений, комментариев, транскрипций или любой другой дополнительной информации.

Пример вопроса пользователя:
Скажи мне голосом как по японски будет Спасибо?

Пример корректного ответа:
voice: ありがとう

Пример недопустимого ответа:
voice: ありがとう (Arigatō)"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["messages"] = [{"role": "system", "content": system_prompt}]
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Я нейропереводчик! Я умею общаться текстом и с помощью голосовых сообщений!")


def lang_detect(text):
    languages = [Language.RUSSIAN, Language.ENGLISH, Language.SPANISH, Language.GERMAN, Language.CHINESE, Language.FRENCH, Language.JAPANESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    detected_lang = detector.detect_language_of(text)
    if detected_lang:
        return detected_lang.iso_code_639_1.name.lower()
    else:
        return None


async def tts(text, lang, file_name):
    voice = "en-US-EmmaMultilingualNeural"

    match lang:
        case "ru": voice = "ru-RU-DmitryNeural"
        case "en": voice = "en-US-AvaNeural"
        case "de": voice = "de-DE-ConradNeural"
        case "fr": voice = "fr-FR-HenriNeural"
        case "es": voice = "es-ES-XimenaNeural"
        case "zh": voice = "zh-CN-XiaoyiNeural"
        case "ja": voice = "ja-JP-NanamiNeural"

    # преобразуем текст в аудиофайл в формате wav
    await edge_tts.Communicate(text, voice).save(file_name + ".wav")
    # конвертируем файл в формате wav в формат ogg
    FFmpeg().option("y").input(file_name + ".wav").output(file_name, {"codec:a": "libopus"}).execute()


async def process_user_message(update, context, user_message):
    if len(user_message) > 1000:
        return "text: Ваше сообщение слишком большое. Я не могу на него ответить."

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    messages = context.user_data["messages"]

    if len(messages) > 20:
        del messages[0]

    messages.append({"role": "user", "content": user_message})

    response = deepseek.chat.completions.create(model="deepseek-chat", messages=messages)

    response_text = response.choices[0].message.content

    messages.append({"role": "assistant", "content": response_text})

    return response_text


async def process_bot_response(response_text, context, chat_id, message_id):
    if response_text.startswith("voice:"):
        response_text = response_text.replace("voice:", "")
        lang = lang_detect(response_text)
        voice_file_name = str(chat_id) + ".ogg"
        await tts(response_text, lang, voice_file_name)
        await context.bot.send_voice(chat_id=chat_id,
                                     voice=voice_file_name,
                                     reply_to_message_id=message_id)
    else:
        await context.bot.send_message(chat_id=chat_id,
                                       text=response_text,
                                       reply_to_message_id=message_id)


async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(update.effective_chat.id, user_message)
    response_text = await process_user_message(update, context, user_message)
    print(update.effective_chat.id, response_text)
    await process_bot_response(response_text, context, update.effective_chat.id, update.message.message_id)


async def process_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await context.bot.get_file(update.message.voice.file_id)  # получаем ссылку на файл в облаке телеграма
    resp = requests.get(file.file_path)  # скачиваем файл содержащий голосовое сообщение

    with tempfile.NamedTemporaryFile(mode="wb", delete=True) as f:
        f.write(resp.content)
        f.flush()
        result = whisper.transcribe(f.name)
        user_message = result["text"]

    print(update.effective_chat.id, user_message)
    response_text = await process_user_message(update, context, user_message)
    print(update.effective_chat.id, response_text)
    await process_bot_response(response_text, context, update.effective_chat.id, update.message.message_id)


application = Application.builder().token(TG_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
application.add_handler(MessageHandler(filters.VOICE, process_voice))
application.run_polling()
