import os
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from bot.keyboards import main_menu, graphs_menu
from ml.predict import predict, load_model

router = Router()

WINE_CLASSES = {
    0: "🍷 Barolo",
    1: "🍇 Grignolino",
    2: "🌹 Barbera"
}

class WineForm(StatesGroup):
    waiting_for_values = State()

FEATURE_QUESTIONS = [
    ("alcohol",                      "🍶 Alcohol (e.g. 13.2)"),
    ("malic_acid",                   "🍋 Malic Acid (e.g. 1.78)"),
    ("ash",                          "⚗️ Ash (e.g. 2.14)"),
    ("alcalinity_of_ash",            "🧪 Alcalinity of Ash (e.g. 11.2)"),
    ("magnesium",                    "🔩 Magnesium (e.g. 100.0)"),
    ("total_phenols",                "🌿 Total Phenols (e.g. 2.65)"),
    ("flavanoids",                   "🌸 Flavanoids (e.g. 2.76)"),
    ("nonflavanoid_phenols",         "🍂 Nonflavanoid Phenols (e.g. 0.26)"),
    ("proanthocyanins",              "🫐 Proanthocyanins (e.g. 1.28)"),
    ("color_intensity",              "🎨 Color Intensity (e.g. 4.38)"),
    ("hue",                          "🌈 Hue (e.g. 1.05)"),
    ("od280/od315_of_diluted_wines", "🔬 OD280/OD315 (e.g. 3.40)"),
    ("proline",                      "💎 Proline (e.g. 1050.0)"),
]

# /start
@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Welcome to <b>Forest Classifier Bot</b> 🌲\n\n"
        "I can identify the type of Italian wine based on its chemical characteristics "
        "using the <b>Random Forest</b> algorithm.\n\n"
        "🍷 Wines I can classify:\n"
        "• <b>Barolo</b> — bold, tannic, full-bodied\n"
        "• <b>Grignolino</b> — light, delicate, aromatic\n"
        "• <b>Barbera</b> — fruity, low tannin, high acidity\n\n"
        "Choose an action below 👇",
        parse_mode="HTML",
        reply_markup=main_menu()
    )

# Predict
@router.message(F.text == "🍷 Predict Wine")
async def start_predict(message: Message, state: FSMContext):
    await state.set_state(WineForm.waiting_for_values)
    await state.update_data(step=0, values=[])
    feature, question = FEATURE_QUESTIONS[0]
    await message.answer(
        "Let's classify your wine! Enter the chemical characteristics one by one.\n\n"
        f"<b>Step 1/13</b>\n{question}",
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
        await message.answer("⚠️ Please enter a numeric value, e.g. <b>13.2</b>", parse_mode="HTML")
        return

    values.append(value)
    step += 1

    if step < len(FEATURE_QUESTIONS):
        await state.update_data(step=step, values=values)
        feature, question = FEATURE_QUESTIONS[step]
        await message.answer(
            f"<b>Step {step + 1}/13</b>\n{question}",
            parse_mode="HTML"
        )
    else:
        await state.clear()
        result = predict(values)

        class_id = result["class_id"]
        wine_name = WINE_CLASSES[class_id]

        votes_str = "\n".join(
            f"  {WINE_CLASSES[i]}: {result['votes'][i]} votes"
            for i in range(len(result['target_names']))
        )

        await message.answer(
            f"🍷 <b>Prediction Result:</b>\n\n"
            f"🏆 Wine type: <b>{wine_name}</b>\n"
            f"📊 Confidence: <b>{result['confidence']}%</b>\n\n"
            f"🌲 Forest voting:\n{votes_str}\n\n"
            f"✅ All 13 features analyzed by 100 trees!",
            parse_mode="HTML",
            reply_markup=main_menu()
        )

# Metrics
@router.message(F.text == "📊 Model Metrics")
async def show_metrics(message: Message):
    rf, _, _ = load_model()
    await message.answer(
        f"📊 <b>Random Forest Model Metrics:</b>\n\n"
        f"✅ Accuracy:   <b>100%</b>\n"
        f"✅ ROC-AUC:    <b>100%</b>\n"
        f"✅ OOB Score:  <b>97.89%</b>\n\n"
        f"⚙️ <b>Model Parameters:</b>\n"
        f"🌲 Trees:      <b>{rf.n_estimators}</b>\n"
        f"📏 Features:   <b>13</b>\n"
        f"🍷 Classes:    <b>3 (Barolo, Grignolino, Barbera)</b>\n\n"
        f"📦 Dataset:    <b>UCI Wine Dataset (178 samples)</b>",
        parse_mode="HTML"
    )

# Graphs
@router.message(F.text == "📈 Graphs")
async def show_graphs_menu(message: Message):
    await message.answer("Choose a graph to display 👇", reply_markup=graphs_menu())

@router.callback_query(F.data.startswith("plot_"))
async def send_plot(callback: CallbackQuery):
    plots = {
        "plot_importance": ("plots/feature_importance.png", "🌟 Feature Importance — top predictive features"),
        "plot_confusion":  ("plots/confusion_matrix.png",  "🎯 Confusion Matrix — prediction errors"),
        "plot_roc":        ("plots/roc_curve.png",         "📉 ROC Curve — model discrimination ability"),
        "plot_tree":       ("plots/single_tree.png",       "🌲 Single tree from the Random Forest (max_depth=3)"),
        "plot_learning":   ("plots/learning_curve.png",    "📊 Learning Curve — accuracy vs number of trees"),
    }

    key = callback.data
    if key in plots:
        path, caption = plots[key]
        if os.path.exists(path):
            photo = FSInputFile(path)
            await callback.message.answer_photo(photo=photo, caption=caption)
        else:
            await callback.message.answer("⚠️ Graph not found. Please run ml/train.py first.")

    await callback.answer()

# How RF works
@router.message(F.text == "ℹ️ How does RF work?")
async def how_it_works(message: Message):
    await message.answer(
        "🌲 <b>How does Random Forest work?</b>\n\n"
        "1️⃣ <b>Bootstrap</b>\n"
        "Each tree trains on a random sample of data with replacement. "
        "About 37% of samples are left out (OOB).\n\n"
        "2️⃣ <b>Random Subspace</b>\n"
        "At each node split, only a random subset of features is considered. "
        "This decorrelates the trees.\n\n"
        "3️⃣ <b>Voting</b>\n"
        "Each tree independently predicts a class. "
        "The final prediction is the majority vote.\n\n"
        "4️⃣ <b>OOB Score</b>\n"
        "Out-of-bag samples serve as a free validation set — "
        "no need for a separate cross-validation!\n\n"
        "🎯 <b>Result:</b> An ensemble of weak trees creates a strong, robust model!",
        parse_mode="HTML"
    )