from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")

CHATTING = range(1)

async def process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_text = update.message.text
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot that replies to messages in just a few sentences",
        },
        {"role": "user", "content": message_text},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    await update.message.reply_text(outputs[0]["generated_text"][len(prompt):])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts the conversation"""

    await update.message.reply_text(
        "Hi! My name is JuBot. I will hold a conversation with you. "
        "Send /cancel to stop talking to me.\n\n",
    )

    return CHATTING


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


def main() -> None:
    API_TOKEN = ""
    
    app = Application.builder().token(API_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHATTING: [MessageHandler(filters.TEXT & ~filters.COMMAND, process)]
        },
        fallbacks=[CommandHandler("cancel", cancel)])
    app.add_handler(conv_handler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
