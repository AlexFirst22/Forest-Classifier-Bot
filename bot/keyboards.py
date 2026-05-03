from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton


def main_menu():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="💳 Check Credit Risk"), KeyboardButton(text="📊 Model Metrics")],
            [KeyboardButton(text="📈 Graphs"), KeyboardButton(text="ℹ️ How does RF work?")]
        ],
        resize_keyboard=True
    )
    return keyboard


def home_ownership_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🏠 Own", callback_data="own_OWN")],
        [InlineKeyboardButton(text="🏦 Mortgage", callback_data="own_MORTGAGE")],
        [InlineKeyboardButton(text="🏢 Rent", callback_data="own_RENT")],
        [InlineKeyboardButton(text="🎁 Other", callback_data="own_OTHER")],
    ])


def loan_intent_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎓 Education", callback_data="intent_EDUCATION")],
        [InlineKeyboardButton(text="🏥 Medical", callback_data="intent_MEDICAL")],
        [InlineKeyboardButton(text="🏠 Home Improvement", callback_data="intent_HOMEIMPROVEMENT")],
        [InlineKeyboardButton(text="💼 Venture", callback_data="intent_VENTURE")],
        [InlineKeyboardButton(text="🛍 Personal", callback_data="intent_PERSONAL")],
        [InlineKeyboardButton(text="🔄 Debt Consolidation", callback_data="intent_DEBTCONSOLIDATION")],
    ])


def loan_grade_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="A", callback_data="grade_A"),
            InlineKeyboardButton(text="B", callback_data="grade_B"),
            InlineKeyboardButton(text="C", callback_data="grade_C"),
            InlineKeyboardButton(text="D", callback_data="grade_D"),
        ],
        [
            InlineKeyboardButton(text="E", callback_data="grade_E"),
            InlineKeyboardButton(text="F", callback_data="grade_F"),
            InlineKeyboardButton(text="G", callback_data="grade_G"),
        ]
    ])


def default_history_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ No previous defaults", callback_data="default_N")],
        [InlineKeyboardButton(text="⚠️ Yes, had a default", callback_data="default_Y")],
    ])


def graphs_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🌟 Feature Importance", callback_data="plot_importance")],
        [InlineKeyboardButton(text="🎯 Confusion Matrix", callback_data="plot_confusion")],
        [InlineKeyboardButton(text="📉 ROC Curve", callback_data="plot_roc")],
        [InlineKeyboardButton(text="📊 Learning Curve", callback_data="plot_learning")],
        [InlineKeyboardButton(text="📦 Class Distribution", callback_data="plot_distribution")],
        [InlineKeyboardButton(text="🔥 Correlation Matrix", callback_data="plot_correlation")],
    ])