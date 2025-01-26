from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import asyncio
import nest_asyncio
nest_asyncio.apply()

TELEGRAM_BOT_TOKEN = "6776228304:AAG1GZIDeWaS7jMO9g4uxTc3Bo2dFrqQG00"

MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing the model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(DEVICE)
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)
print("Model and tokenizer initialized successfully!")

async def welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a greeting message when the bot is started."""
    await update.message.reply_text(
        "Hello there! I'm your AI assistant. What do you have in mind?"
    )
async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes user messages and generates AI-based responses."""
    user_input = update.message.text
    chat_context = [
        {"role": "system", "content": "You are a helpful assistant providing detailed responses."},
        {"role": "user", "content": user_input},
    ]
    try:
        generated_prompt = tokenizer.apply_chat_template(
            chat_context,
            tokenize=False,
            add_generation_prompt=True
        )
        generated_text = pipe(
            generated_prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        assistant_response = generated_text[0]["generated_text"].split("<|assistant|>")[-1].strip()
        await update.message.reply_text(assistant_response)
    except Exception as error:
        await update.message.reply_text("Oops, something went wrong. Please try again later.")
        print(f"Error encountered during response generation: {error}")

def start_bot():
    """Configures and starts the Telegram bot."""
    bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot_app.add_handler(CommandHandler("start", welcome_message))
    bot_app.add_handler(MessageHandler(filters.ALL, handle_user_message))
    print("Bot is now live and ready for messages...")
    bot_app.run_polling()

if __name__ == "__main__":
    start_bot()
