
from time import perf_counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from mythings_ai_recommender.models.relationship_classifier import (
    classify_one_relationship,
)
from mythings_ai_recommender.schemas.product import Product


# Development evaluation examples.
# These are manually assigned expected relationships.
CASES = [
    ("Chicken Shawarma", "Hummus", "complement"),
    ("Margherita Pizza", "Orange Juice", "complement"),
    ("Pasta", "Tomato Pasta Sauce", "complement"),
    ("Pasta", "Another Brand of Pasta", "substitute"),
    ("Coffee", "Tea", "substitute"),
    ("Shampoo", "Hair Conditioner", "complement"),
    ("Shampoo", "Phone Charger", "unrelated"),
    ("Laptop", "Laptop Sleeve", "complement"),
    ("Laptop", "Another Laptop", "substitute"),
    ("Camera", "Compatible Memory Card", "complement"),
    ("Camera", "Chocolate Bar", "unrelated"),
    ("شاورما دجاج", "مخلل خيار", "complement"),
    ("برجر لحم", "بطاطا مقلية", "complement"),
    ("شامبو للشعر", "بلسم للشعر", "complement"),
    ("هاتف ذكي", "غطاء هاتف متوافق", "complement"),
    ("هاتف ذكي", "هاتف ذكي آخر", "substitute"),
    ("منسف", "صوص شوكولاتة للحلويات", "unrelated"),
    ("Chicken Burger", "ثومية", "complement"),
    ("Pasta", "صوص شوكولاتة", "unrelated"),
    ("غسول وجه", "مرطب وجه", "complement"),
]


LABELS = [
    "complement",
    "substitute",
    "unrelated",
    "uncertain",
]


def make_product(product_id, name):
    return Product(
        id=product_id,
        merchant_id="DEMO",
        name_en=name,
    )


y_true = []
y_pred = []
latencies = []

print("\nMODEL RELATIONSHIP EVALUATION\n")

for index, (cart_name, candidate_name, expected) in enumerate(
    CASES, start=1
):
    start = perf_counter()

    prediction = classify_one_relationship(
        [make_product("cart", cart_name)],
        make_product("candidate", candidate_name),
    )

    elapsed = perf_counter() - start

    y_true.append(expected)
    y_pred.append(prediction)
    latencies.append(elapsed)

    status = "PASS" if prediction == expected else "FAIL"

    print(
        f"{index:02d}. {status} | "
        f"{cart_name} + {candidate_name} | "
        f"expected={expected}, predicted={prediction} | "
        f"{elapsed:.2f}s",
        flush=True,
    )


# ---------------------------------------------------------
# 1. Overall classification metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 60)
print("CLASSIFICATION EVALUATION REPORT")
print("=" * 60)

print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print("Label order:", LABELS)

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )
)


# ---------------------------------------------------------
# 2. Binary complement detection metrics
# ---------------------------------------------------------

true_binary = [
    label == "complement"
    for label in y_true
]

pred_binary = [
    label == "complement"
    for label in y_pred
]

precision, recall, f1, _ = precision_recall_fscore_support(
    true_binary,
    pred_binary,
    average="binary",
    zero_division=0,
)

print("\n" + "=" * 60)
print("COMPLEMENT DETECTION METRICS")
print("=" * 60)

print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")


# ---------------------------------------------------------
# 3. Latency metrics
# ---------------------------------------------------------

sorted_latencies = sorted(latencies)

def latency_percentile(values, percentile):
    if not values:
        return 0.0

    index = (len(values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


print("\n" + "=" * 60)
print("INFERENCE PERFORMANCE")
print("=" * 60)

print(f"Total time: {sum(latencies):.2f}s")
print(f"Average latency: {sum(latencies) / len(latencies):.3f}s")
print(
    "Median latency:",
    f"{latency_percentile(sorted_latencies, 0.50):.3f}s",
)
print(
    "P95 latency:",
    f"{latency_percentile(sorted_latencies, 0.95):.3f}s",
)
