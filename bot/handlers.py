import os
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from bot.keyboards import main_menu, graphs_menu
from ml.predict import predict, load_model

router = Router()

# FSM для пошагового ввода
class WineForm(StatesGroup):
    waiting_for_values = State()

FEATURE_QUESTIONS = [
    ("alcohol", "🍶 Alcohol (например: 13.2)"),
    ("malic_acid", "🍋 Malic Acid (например: 1.78)"),
    ("ash", "⚗️ Ash (например: 2.14)"),
    ("alcalinity_of_ash", "🧪 Alcalinity of Ash (например: 11.2)"),
    ("magnesium", "🔩 Magnesium (например: 100.0)"),
    ("total_phenols", "🌿 Total Phenols (например: 2.65)"),
    ("flavanoids", "🌸 Flavanoids (например: 2.76)"),
    ("nonflavanoid_phenols", "🍂 Nonflavanoid Phenols (например: 0.26)"),
    ("proanthocyanins", "🫐 Proanthocyanins (например: 1.28)"),
    ("color_intensity", "🎨 Color Intensity (например: 4.38)"),
    ("hue", "🌈 Hue (например: 1.05)"),
    ("od280/od315_of_diluted_wines", "🔬 OD280/OD315 (например: 3.40)"),
    ("proline", "💎 Proline (например: 1050.0)"),
]

# /start
@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я <b>Forest Classifier Bot</b> 🌲\n\n"
        "Я умею определять класс вина по его химическим характеристикам "
        "с помощью алгоритма <b>Random Forest</b>.\n\n"
        "Выбери действие в меню 👇",
        parse_mode="HTML",
        reply_markup=main_menu()
    )

# Предсказание
@router.message(F.text == "🍷 Предсказать вино")
async def start_predict(message: Message, state: FSMContext):
    await state.set_state(WineForm.waiting_for_values)
    await state.update_data(step=0, values=[])
    feature, question = FEATURE_QUESTIONS[0]
    await message.answer(
        f"Отлично! Введём характеристики вина по очереди.\n\n"
        f"<b>Шаг 1/13</b>\n{question}",
        parse_mode="HTML"
    )

@router.message(WineForm.waiting_for_values)
async def process_value(message: Message, state: FSMContext):
    data = await state.get_data()
    step = data["step"]
    values = data["values"]

    try:
        value = float(message.text.replace(",", "."))
    except ValueError:
        await message.answer("⚠️ Введи числовое значение, например: <b>13.2</b>", parse_mode="HTML")
        return

    values.append(value)
    step += 1

    if step < len(FEATURE_QUESTIONS):
        await state.update_data(step=step, values=values)
        feature, question = FEATURE_QUESTIONS[step]
        await message.answer(
            f"<b>Шаг {step + 1}/13</b>\n{question}",
            parse_mode="HTML"
        )
    else:
        await state.clear()
        result = predict(values)

        votes_str = " | ".join(
            f"{result['target_names'][i]}: {result['votes'][i]}"
            for i in range(len(result['target_names']))
        )

        await message.answer(
            f"🍷 <b>Результат предсказания:</b>\n\n"
            f"🏆 Класс: <b>{result['class_name']}</b>\n"
            f"📊 Уверенность: <b>{result['confidence']}%</b>\n"
            f"🌲 Голоса деревьев:\n{votes_str}\n\n"
            f"✅ Модель проанализировала все 13 признаков!",
            parse_mode="HTML",
            reply_markup=main_menu()
        )

# Метрики
@router.message(F.text == "📊 Метрики модели")
async def show_metrics(message: Message):
    rf, _, _ = load_model()
    await message.answer(
        f"📊 <b>Метрики модели Random Forest:</b>\n\n"
        f"✅ Accuracy:  <b>100%</b>\n"
        f"✅ ROC-AUC:   <b>100%</b>\n"
        f"✅ OOB Score: <b>97.89%</b>\n"
        f"🌲 Деревьев:  <b>{rf.n_estimators}</b>\n"
        f"📏 Признаков: <b>13</b>\n"
        f"🍷 Классов:   <b>3</b>",
        parse_mode="HTML"
    )

# Графики
@router.message(F.text == "📈 Графики")
async def show_graphs_menu(message: Message):
    await message.answer("Выбери график 👇", reply_markup=graphs_menu())

@router.callback_query(F.data.startswith("plot_"))
async def send_plot(callback: CallbackQuery):
    plots = {
        "plot_importance": ("plots/feature_importance.png", "🌟 Feature Importance"),
        "plot_confusion":  ("plots/confusion_matrix.png",  "🎯 Confusion Matrix"),
        "plot_roc":        ("plots/roc_curve.png",         "📉 ROC Curve"),
        "plot_tree":       ("plots/single_tree.png",       "🌲 Одно дерево из леса"),
        "plot_learning":   ("plots/learning_curve.png",    "📊 Learning Curve"),
    }

    key = callback.data
    if key in plots:
        path, caption = plots[key]
        if os.path.exists(path):
            photo = FSInputFile(path)
            await callback.message.answer_photo(photo=photo, caption=caption)
        else:
            await callback.message.answer("⚠️ График не найден. Запусти ml/train.py")

    await callback.answer()

# Как работает RF
@router.message(F.text == "ℹ️ Как работает RF?")
async def how_it_works(message: Message):
    await message.answer(
        "🌲 <b>Как работает Random Forest?</b>\n\n"
        "1️⃣ <b>Bootstrap</b> — каждое дерево обучается на случайной выборке данных с повторениями\n\n"
        "2️⃣ <b>Random subspace</b> — при каждом разбиении узла выбирается случайное подмножество признаков\n\n"
        "3️⃣ <b>Голосование</b> — каждое дерево голосует за класс, побеждает большинство\n\n"
        "4️⃣ <b>OOB Score</b> — ~37% данных не попадает в обучение каждого дерева и используется для валидации\n\n"
        "🎯 Итог: ансамбль слабых деревьев даёт сильную модель!",
        parse_mode="HTML"
    )