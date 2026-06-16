import os
import logging
import httpx
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Environment Variables ────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
INSTRUCTOR_ID   = int(os.environ["INSTRUCTOR_CHAT_ID"])

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
)

# ─── AI Feedback Prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert IELTS Speaking examiner and English language coach.
Students will send you transcripts of their spoken answers. These can be:
1. IELTS Speaking answers (Part 1, Part 2, or Part 3)
2. Retelling tasks — summarizing an article or podcast they listened to

For every response, provide feedback in this EXACT format. Do not skip any section.

━━━━━━━━━━━━━━━━━━━━━━━━
📝 TRANSCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━
[Copy the student's transcript exactly as given — change nothing]

━━━━━━━━━━━━━━━━━━━━━━━━
❌ MISTAKES & CORRECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━
List every grammar and vocabulary mistake:

1. ❌ They said: "[exact words]"
   ✅ Correct: "[corrected version]"
   📌 Why: [brief, clear explanation]

If there are no mistakes, write: "No significant mistakes — great job!"

━━━━━━━━━━━━━━━━━━━━━━━━
💬 FEEDBACK & COMMENTS
━━━━━━━━━━━━━━━━━━━━━━━━
• Content: Did they answer the question fully? What was missing?
• Vocabulary: Were words too simple? What topic-specific words could improve it?
• Structure: Was the answer well-organised with a clear flow?
• Key Advice: The single most important thing this student should work on.

━━━━━━━━━━━━━━━━━━━━━━━━
⭐ MODEL ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━
Write a Band 8+ model answer for the same question or task.
- Use advanced, topic-specific vocabulary naturally
- Use varied grammar structures
- Sound fluent and natural, not robotic
- Put the most impressive words or phrases in *asterisks* like *this*

Be honest, direct, and encouraging. Students are preparing for a real exam."""


# ─── Core Functions ───────────────────────────────────────────────────────────

async def transcribe_audio(file_bytes: bytes) -> str:
    """Transcribe voice message using Groq's free Whisper API."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("voice.ogg", file_bytes, "audio/ogg")},
            data={"model": "whisper-large-v3", "language": "en"},
        )
        response.raise_for_status()
        return response.json()["text"]


async def get_feedback(transcript: str) -> str:
    """Send transcript to Gemini and get structured IELTS feedback."""
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\nHere is the student's spoken answer:\n\n" + transcript}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048,
        }
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def build_instructor_alert(update: Update, transcript: str) -> str:
    """Build the notification message sent to the instructor."""
    user = update.effective_user
    name = user.full_name or "Unknown"
    username = f"@{user.username}" if user.username else f"No username — ID: {user.id}"
    return (
        f"🔔 New Speaking Submission!\n\n"
        f"👤 Name: {name}\n"
        f"📱 Telegram: {username}\n"
        f"🆔 User ID: {user.id}\n\n"
        f"📝 What they said:\n{transcript}"
    )


# ─── Bot Handlers ─────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *Welcome to your IELTS Speaking Practice Bot!*\n\n"
        "🎤 Send me a *voice message* with your speaking answer and you'll get:\n\n"
        "📝 A full transcript of what you said\n"
        "❌ Every mistake corrected with clear explanations\n"
        "💬 Detailed feedback and comments\n"
        "⭐ A Band 8+ model answer with advanced vocabulary\n\n"
        "Just press and hold 🎙️ and start speaking!",
        parse_mode="Markdown"
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main handler: download → transcribe → notify instructor → give feedback."""
    msg = update.message
    status = await msg.reply_text("⏳ Received! Analysing your answer… (takes about 20 seconds)")

    try:
        # 1. Download voice file from Telegram
        voice_file = await context.bot.get_file(msg.voice.file_id)
        file_bytes = bytes(await voice_file.download_as_bytearray())

        # 2. Transcribe with Groq (free Whisper)
        await status.edit_text("🔊 Transcribing your voice…")
        transcript = await transcribe_audio(file_bytes)

        if not transcript.strip():
            await status.edit_text(
                "❌ I couldn't hear anything clearly.\n\n"
                "Please re-record in a quieter place and speak close to your microphone."
            )
            return

        # 3. Notify instructor with student info + transcript
        try:
            alert = build_instructor_alert(update, transcript)
            await context.bot.send_message(chat_id=INSTRUCTOR_ID, text=alert)
        except Exception as e:
            logger.warning(f"Could not notify instructor: {e}")

        # 4. Get AI feedback from Gemini (free)
        await status.edit_text("🤖 Checking your answer with AI…")
        feedback = await get_feedback(transcript)

        # 5. Send feedback to student
        await status.delete()

        if len(feedback) <= 4096:
            await msg.reply_text(feedback)
        else:
            chunks = [feedback[i:i+4000] for i in range(0, len(feedback), 4000)]
            for chunk in chunks:
                await msg.reply_text(chunk)

    except httpx.HTTPStatusError as e:
        logger.error(f"API error: {e.response.status_code} — {e.response.text}")
        await status.edit_text("❌ There was an error processing your audio. Please try again.")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        await status.edit_text("❌ Something went wrong. Please try again or contact your instructor.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎙️ Please send a *voice message*, not text!\n\n"
        "In Telegram, press and hold the 🎙️ microphone button to record your answer.",
        parse_mode="Markdown"
    )


# ─── Launch ───────────────────────────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("✅ IELTS Bot is running…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
