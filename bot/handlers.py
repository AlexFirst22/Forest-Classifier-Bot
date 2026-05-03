import os
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from bot.keyboards import (main_menu, home_ownership_keyboard,
                            loan_intent_keyboard, loan_grade_keyboard,
                            default_history_keyboard, graphs_menu)
from ml.predict import predict

router = Router()


class CreditForm(StatesGroup):
    age = State()
    income = State()
    home_ownership = State()
    emp_length = State()
    loan_intent = State()
    loan_grade = State()
    loan_amnt = State()
    loan_int_rate = State()
    default_on_file = State()
    cred_hist_length = State()


# /start
@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Welcome to <b>💳 Credit Risk Bot</b>!\n\n"
        "I use a <b>Random Forest</b> model trained on 32,000+ real loan records "
        "to predict whether a loan application is likely to be approved or flagged as high risk.\n\n"
        "🔍 <b>What I can do:</b>\n"
        "• Analyze your loan application in seconds\n"
        "• Show confidence level of the prediction\n"
        "• Explain which factors influenced the decision\n"
        "• Give tips to improve your chances\n\n"
        "Press <b>💳 Check Credit Risk</b> to get started 👇",
        parse_mode="HTML",
        reply_markup=main_menu()
    )


# Start prediction flow
@router.message(F.text == "💳 Check Credit Risk")
async def start_credit_check(message: Message, state: FSMContext):
    await state.set_state(CreditForm.age)
    await message.answer(
        "Let's analyze your loan application! I'll ask you a few questions.\n\n"
        "<b>Step 1/10</b>\n"
        "🎂 What is your <b>age</b>?\n\n"
        "<i>Enter a number (e.g. 35)</i>",
        parse_mode="HTML"
    )


# Age
@router.message(CreditForm.age)
async def process_age(message: Message, state: FSMContext):
    try:
        age = int(message.text)
        if not 18 <= age <= 99:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Please enter a valid age between 18 and 99.")
        return

    await state.update_data(person_age=age)
    await state.set_state(CreditForm.income)
    await message.answer(
        "<b>Step 2/10</b>\n"
        "💰 What is your <b>annual income</b> (in $)?\n\n"
        "<i>Enter a number (e.g. 55000)</i>",
        parse_mode="HTML"
    )


# Income
@router.message(CreditForm.income)
async def process_income(message: Message, state: FSMContext):
    try:
        income = float(message.text.replace(",", ""))
        if income <= 0:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Please enter a valid income amount (e.g. 55000).")
        return

    await state.update_data(person_income=income)
    await state.set_state(CreditForm.home_ownership)
    await message.answer(
        "<b>Step 3/10</b>\n"
        "🏠 What is your <b>home ownership</b> status?",
        parse_mode="HTML",
        reply_markup=home_ownership_keyboard()
    )


# Home ownership
@router.callback_query(F.data.startswith("own_"))
async def process_home_ownership(callback: CallbackQuery, state: FSMContext):
    value = callback.data.split("_")[1]
    await state.update_data(person_home_ownership=value)
    await state.set_state(CreditForm.emp_length)
    await callback.message.answer(
        "<b>Step 4/10</b>\n"
        "💼 How many <b>years have you been employed</b>?\n\n"
        "<i>Enter a number (e.g. 5)</i>",
        parse_mode="HTML"
    )
    await callback.answer()


# Employment length
@router.message(CreditForm.emp_length)
async def process_emp_length(message: Message, state: FSMContext):
    try:
        emp = float(message.text.replace(",", "."))
        if emp < 0:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Please enter a valid number of years (e.g. 5).")
        return

    await state.update_data(person_emp_length=emp)
    await state.set_state(CreditForm.loan_intent)
    await message.answer(
        "<b>Step 5/10</b>\n"
        "🎯 What is the <b>purpose</b> of the loan?",
        parse_mode="HTML",
        reply_markup=loan_intent_keyboard()
    )


# Loan intent
@router.callback_query(F.data.startswith("intent_"))
async def process_loan_intent(callback: CallbackQuery, state: FSMContext):
    value = callback.data.split("_")[1]
    await state.update_data(loan_intent=value)
    await state.set_state(CreditForm.loan_grade)
    await callback.message.answer(
        "<b>Step 6/10</b>\n"
        "🏅 What is your <b>loan grade</b>?\n\n"
        "<i>A = best credit, G = highest risk</i>",
        parse_mode="HTML",
        reply_markup=loan_grade_keyboard()
    )
    await callback.answer()


# Loan grade
@router.callback_query(F.data.startswith("grade_"))
async def process_loan_grade(callback: CallbackQuery, state: FSMContext):
    value = callback.data.split("_")[1]
    await state.update_data(loan_grade=value)
    await state.set_state(CreditForm.loan_amnt)
    await callback.message.answer(
        "<b>Step 7/10</b>\n"
        "💵 What is the <b>loan amount</b> you are requesting (in $)?\n\n"
        "<i>Enter a number (e.g. 10000)</i>",
        parse_mode="HTML"
    )
    await callback.answer()


# Loan amount
@router.message(CreditForm.loan_amnt)
async def process_loan_amnt(message: Message, state: FSMContext):
    try:
        amnt = float(message.text.replace(",", ""))
        if amnt <= 0:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Please enter a valid loan amount (e.g. 10000).")
        return

    await state.update_data(loan_amnt=amnt)
    await state.set_state(CreditForm.loan_int_rate)
    await message.answer(
        "<b>Step 8/10</b>\n"
        "📈 What is the <b>interest rate</b> on the loan (%)?\n\n"
        "<i>Enter a number (e.g. 12.5)</i>",
        parse_mode="HTML"
    )


# Interest rate
@router.message(CreditForm.loan_int_rate)
async def process_int_rate(message: Message, state: FSMContext):
    try:
        rate = float(message.text.replace(",", "."))
        if not 1 <= rate <= 30:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Please enter a valid interest rate between 1 and 30 (e.g. 12.5).")
        return

    await state.update_data(loan_int_rate=rate)
    await state.set_state(CreditForm.default_on_file)
    await message.answer(
        "<b>Step 9/10</b>\n"
        "📋 Do you have any <b>previous defaults</b> on record?",
        parse_mode="HTML",
        reply_markup=default_history_keyboard()
    )


# Default on file
@router.callback_query(F.data.startswith("default_"))
async def process_default(callback: CallbackQuery, state: FSMContext):
    value = callback.data.split("_")[1]
    await state.update_data(cb_person_default_on_file=value)
    await state.set_state(CreditForm.cred_hist_length)
    await callback.message.answer(
        "<b>Step 10/10</b>\n"
        "🗓 How many <b>years of credit history</b> do you have?\n\n"
        "<i>Enter a number (e.g. 4)</i>",
        parse_mode="HTML"
    )
    await callback.answer()


# Credit history length → final prediction
@router.message(CreditForm.cred_hist_length)
async def process_cred_hist(message: Message, state: FSMContext):
    try:
        hist = int(message.text)
        if hist < 0:
            raise ValueError
    except ValueError:
        await message.answer("⚠️ Please enter a valid number of years (e.g. 4).")
        return

    await state.update_data(cb_person_cred_hist_length=hist)
    data = await state.get_data()
    await state.clear()

    await message.answer("🔍 Analyzing your application...")

    try:
        result = predict(data)
    except Exception as e:
        await message.answer(f"⚠️ Prediction error: {str(e)}")
        return

    prediction = result["prediction"]
    prob_approved = result["prob_approved"]
    prob_default = result["prob_default"]
    top_features = result["top_features"]

    # Top factors
    factors_str = "\n".join(
        f"  • {feat}: {round(imp * 100, 1)}% influence"
        for feat, imp in top_features
    )

    if prediction == 0:
        verdict = "✅ <b>APPROVED — Low Risk</b>"
        verdict_detail = (
            f"The model is <b>{prob_approved}%</b> confident this loan is low risk.\n\n"
            f"🌲 Top factors in your favor:\n{factors_str}"
        )
    else:
        verdict = "❌ <b>DENIED — High Risk</b>"
        verdict_detail = (
            f"The model is <b>{prob_default}%</b> confident this loan is high risk.\n\n"
            f"⚠️ Top risk factors:\n{factors_str}\n\n"
            f"💡 <b>Tips to improve your chances:</b>\n"
            f"  • Reduce the loan amount\n"
            f"  • Improve your loan grade (pay off existing debts)\n"
            f"  • Build a longer credit history\n"
            f"  • Lower your debt-to-income ratio"
        )

    await message.answer(
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💳 <b>Credit Risk Assessment</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{verdict}\n\n"
        f"{verdict_detail}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Based on Random Forest model trained on 32,000+ loan records</i>",
        parse_mode="HTML",
        reply_markup=main_menu()
    )


# Metrics
@router.message(F.text == "📊 Model Metrics")
async def show_metrics(message: Message):
    await message.answer(
        "📊 <b>Random Forest Model Metrics:</b>\n\n"
        "✅ Accuracy:   <b>93.50%</b>\n"
        "✅ ROC-AUC:    <b>93.75%</b>\n"
        "✅ OOB Score:  <b>93.01%</b>\n\n"
        "⚙️ <b>Model Parameters:</b>\n"
        "🌲 Trees:      <b>100</b>\n"
        "📏 Features:   <b>11</b>\n"
        "🎯 Task:       <b>Binary Classification</b>\n\n"
        "📦 <b>Dataset:</b>\n"
        "📁 Records:    <b>32,000+</b>\n"
        "✅ Approved:   <b>78.2%</b>\n"
        "❌ Default:    <b>21.8%</b>",
        parse_mode="HTML"
    )


# Graphs
@router.message(F.text == "📈 Graphs")
async def show_graphs_menu(message: Message):
    await message.answer("Choose a graph to display 👇", reply_markup=graphs_menu())


@router.callback_query(F.data.startswith("plot_"))
async def send_plot(callback: CallbackQuery):
    plots = {
        "plot_importance":   ("plots/feature_importance.png",  "🌟 Feature Importance"),
        "plot_confusion":    ("plots/confusion_matrix.png",    "🎯 Confusion Matrix"),
        "plot_roc":          ("plots/roc_curve.png",           "📉 ROC Curve"),
        "plot_learning":     ("plots/learning_curve.png",      "📊 Learning Curve"),
        "plot_distribution": ("plots/class_distribution.png",  "📦 Class Distribution"),
        "plot_correlation":  ("plots/correlation_matrix.png",  "🔥 Correlation Matrix"),
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
        "Each tree trains on a random sample of the 32,000 loan records. "
        "About 37% of records are left out (OOB) for free validation.\n\n"
        "2️⃣ <b>Random Subspace</b>\n"
        "At each node, only a random subset of the 11 features is considered. "
        "This prevents trees from being too similar.\n\n"
        "3️⃣ <b>Voting</b>\n"
        "All 100 trees independently vote: Approved or Default. "
        "The majority wins.\n\n"
        "4️⃣ <b>OOB Score</b>\n"
        "Records not used for training each tree are used to validate it — "
        "giving us a free accuracy estimate of 93.01%.\n\n"
        "🎯 <b>Result:</b> 100 trees working together achieve 93.5% accuracy "
        "on real-world credit risk data!",
        parse_mode="HTML"
    )