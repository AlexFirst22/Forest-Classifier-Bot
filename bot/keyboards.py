from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

def main_menu():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🍷 Predict Wine"), KeyboardButton(text="📊 Model Metrics")],
            [KeyboardButton(text="📈 Graphs"), KeyboardButton(text="ℹ️ How does RF work?")]
        ],
        resize_keyboard=True
    )
    return keyboard

def graphs_menu():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🌟 Feature Importance", callback_data="plot_importance")],
        [InlineKeyboardButton(text="🎯 Confusion Matrix", callback_data="plot_confusion")],
        [InlineKeyboardButton(text="📉 ROC Curve", callback_data="plot_roc")],
        [InlineKeyboardButton(text="🌲 Single Tree", callback_data="plot_tree")],
        [InlineKeyboardButton(text="📊 Learning Curve", callback_data="plot_learning")],
    ])
    return keyboard