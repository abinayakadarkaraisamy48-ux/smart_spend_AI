import re

def extract_sms_transaction(sms):
    text = sms.lower()
    amount_match = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", sms, re.IGNORECASE)
    amount = float(amount_match.group(1).replace(",", "")) if amount_match else None

    payee = "Unknown"
    for pattern in [r"to\s+([A-Za-z0-9 &.-]+)", r"at\s+([A-Za-z0-9 &.-]+)", r"for\s+([A-Za-z0-9 &.-]+)"]:
        match = re.search(pattern, sms, re.IGNORECASE)
        if match:
            payee = match.group(1).strip()
            break

    category = "Others"
    rules = {
        "Food": ["swiggy", "zomato", "restaurant", "food", "cafe"],
        "Transport": ["uber", "ola", "bus", "train", "metro", "travel"],
        "Shopping": ["amazon", "flipkart", "myntra", "shopping"],
        "Entertainment": ["movie", "netflix", "spotify", "entertainment"],
        "Education": ["college", "book", "education", "course"],
        "Health": ["hospital", "clinic", "pharmacy", "medicine"],
        "Bills": ["bill", "electricity", "water", "internet"]
    }
    for label, words in rules.items():
        if any(word in text for word in words):
            category = label
            break
    return {"amount": amount, "payee": payee, "category": category}

def generate_insights(df):
    if df.empty:
        return []
    total = df["amount"].sum()
    category_spending = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_category = category_spending.index[0]
    top_amount = category_spending.iloc[0]
    insights = [
        f"Total spending recorded is ₹{total:,.2f}.",
        f"Average transaction amount is ₹{df['amount'].mean():,.2f}.",
        f"Highest transaction amount is ₹{df['amount'].max():,.2f}.",
        f"{top_category} is the highest spending category with ₹{top_amount:,.2f}.",
        f"{top_category} contributes approximately {(top_amount / total * 100) if total else 0:.1f}% of total spending."
    ]
    if "weekend" in df.columns:
        weekend = df.loc[df["weekend"] == 1, "amount"].sum()
        weekday = df.loc[df["weekend"] == 0, "amount"].sum()
        insights.append("Weekend spending is higher than weekday spending." if weekend > weekday
                        else "Weekday spending is higher than weekend spending.")
    return insights

def generate_recommendations(df):
    if df.empty:
        return ["Add transaction data to generate recommendations."]
    total = df["amount"].sum()
    if total <= 0:
        return ["No positive spending total is available for recommendations."]
    category_spending = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_category = category_spending.index[0]
    top_amount = category_spending.iloc[0]
    recommendations = [
        f"Monitor {top_category} spending because it accounts for approximately {top_amount / total * 100:.1f}% of total spending."
    ]
    if "Food" in category_spending.index and category_spending["Food"] / total * 100 > 15:
        recommendations.append("Consider reducing frequent food purchases and setting a weekly food budget.")
    if "Shopping" in category_spending.index:
        recommendations.append("Review shopping transactions before making non-essential purchases.")
    if "Entertainment" in category_spending.index:
        recommendations.append("Set a monthly entertainment budget to control discretionary spending.")
    recommendations.append("Set a monthly savings target and track progress using SmartSpend AI.")
    return recommendations
