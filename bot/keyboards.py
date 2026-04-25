from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

def main_menu():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🍷 Предсказать вино"), KeyboardButton(text="📊 Метрики модели")],
            [KeyboardButton(text="📈 Графики"), KeyboardButton(text="ℹ️ Как работает RF?")]
        ],
        resize_keyboard=True
    )
    return keyboard

def graphs_menu():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🌟 Feature Importance", callback_data="plot_importance")],
        [InlineKeyboardButton(text="🎯 Confusion Matrix", callback_data="plot_confusion")],
        [InlineKeyboardButton(text="📉 ROC Curve", callback_data="plot_roc")],
        [InlineKeyboardButton(text="🌲 Одно дерево", callback_data="plot_tree")],
        [InlineKeyboardButton(text="📊 Learning Curve", callback_data="plot_learning")],
    ])
    return keyboard