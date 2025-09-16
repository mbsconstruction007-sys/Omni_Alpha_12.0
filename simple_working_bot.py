# simple_working_bot.py
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

# Your actual token (hardcoded for now)
TOKEN = '8271891791:AAGmxaL1XIXjjib1WAsjwIndu-c4iz4SrFk'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(f'Hello {user.first_name}! Bot is working!')

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Commands:\n/start - Start bot\n/help - Show help')

print('Starting bot...')
app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler('start', start))
app.add_handler(CommandHandler('help', help))

print('Bot is running! Go to Telegram and send /start')
app.run_polling()
