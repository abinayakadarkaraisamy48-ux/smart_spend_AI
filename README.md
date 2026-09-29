# SmartSpend AI

Student Expense Analytics & Savings Assistant built with Streamlit.

## Files
- `app.py` — main Streamlit application
- `data_utils.py` — dataset loading, validation, preprocessing, feature engineering
- `ml_model.py` — Random Forest training and evaluation
- `insights.py` — SMS parsing, insights, and saving recommendations
- `requirements.txt` — Python dependencies
- `data/SmartSpendAI_cleaned_dataset.csv` — dataset

## Run
1. Install Python 3.10+.
2. In a terminal, move to this folder.
3. Run `pip install -r requirements.txt`
4. Run `streamlit run app.py`

Manual transaction additions are session-only unless you download the updated CSV. The budget planner compares the entered budget with all loaded records.
