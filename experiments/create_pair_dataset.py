
import json
from pathlib import Path

# Each entry:
# (cart product, positive candidates, negative candidates)
#
# Products are bilingual where available.
# No category or subcategory metadata is required.

EXAMPLES = {
    "food": [
        (
            "Chicken Shawarma / شاورما دجاج",
            ["Garlic Sauce / ثومية", "French Fries / بطاطا مقلية"],
            ["Beef Shawarma / شاورما لحم", "Chocolate Topping / صوص شوكولاتة"],
        ),
        (
            "Beef Burger / برجر لحم",
            ["French Fries / بطاطا مقلية", "Cola / مشروب غازي"],
            ["Chicken Burger / برجر دجاج", "Cake Frosting / كريمة كيك"],
        ),
        (
            "Pasta / معكرونة",
            ["Garlic Bread / خبز بالثوم", "Parmesan Cheese / جبنة بارميزان"],
            ["Another Pasta / معكرونة أخرى", "Chocolate Syrup / شراب شوكولاتة"],
        ),
    ],
    "electronics": [
        (
            "USB-C Smartphone / هاتف USB-C",
            ["USB-C Charging Cable / سلك شحن USB-C", "Phone Case / غطاء هاتف"],
            ["Another Smartphone / هاتف آخر", "Lightning Cable / سلك لايتنينغ"],
        ),
        (
            "Laptop / حاسوب محمول",
            ["Wireless Mouse / فأرة لاسلكية", "Laptop Sleeve / حقيبة لابتوب"],
            ["Another Laptop / لابتوب آخر", "Shampoo / شامبو"],
        ),
        (
            "Digital Camera / كاميرا رقمية",
            ["Camera Strap / حزام كاميرا", "Compatible Memory Card / بطاقة ذاكرة مناسبة"],
            ["Another Camera / كاميرا أخرى", "Face Cream / كريم وجه"],
        ),
    ],
    "personal_care": [
        (
            "Shampoo / شامبو",
            ["Hair Conditioner / بلسم للشعر", "Hair Mask / ماسك للشعر"],
            ["Another Shampoo / شامبو آخر", "Phone Charger / شاحن هاتف"],
        ),
        (
            "Facial Cleanser / غسول وجه",
            ["Face Moisturizer / مرطب وجه", "Facial Sunscreen / واقي شمس للوجه"],
            ["Another Facial Cleanser / غسول آخر", "USB Cable / سلك USB"],
        ),
        (
            "Toothbrush / فرشاة أسنان",
            ["Toothpaste / معجون أسنان", "Dental Floss / خيط أسنان"],
            ["Another Toothbrush / فرشاة أخرى", "Coffee Filters / فلاتر قهوة"],
        ),
    ],
    "household": [
        (
            "Vacuum Cleaner / مكنسة كهربائية",
            ["Compatible Dust Bags / أكياس غبار مناسبة", "Vacuum Filter / فلتر مكنسة"],
            ["Another Vacuum Cleaner / مكنسة أخرى", "Face Serum / سيروم وجه"],
        ),
        (
            "Coffee Maker / ماكينة قهوة",
            ["Compatible Coffee Filters / فلاتر قهوة مناسبة", "Ground Coffee / قهوة مطحونة"],
            ["Another Coffee Maker / ماكينة أخرى", "Phone Case / غطاء هاتف"],
        ),
        (
            "Printer / طابعة",
            ["Compatible Ink Cartridge / خرطوشة حبر مناسبة", "Printer Paper / ورق طباعة"],
            ["Another Printer / طابعة أخرى", "Hair Conditioner / بلسم شعر"],
        ),
    ],
}


def make_records():
    training = []
    evaluation = []

    for domain, groups in EXAMPLES.items():
        for group_index, (cart, positives, negatives) in enumerate(groups):

            # Keep the final cart-product family in each
            # domain for evaluation only.
            destination = (
                evaluation
                if group_index == 2
                else training
            )

            for candidate in positives:
                destination.append({
                    "domain": domain,
                    "cart_product": cart,
                    "candidate": candidate,
                    "label": 1,
                })

            for candidate in negatives:
                destination.append({
                    "domain": domain,
                    "cart_product": cart,
                    "candidate": candidate,
                    "label": 0,
                })

    return training, evaluation


def main():
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    training, evaluation = make_records()

    (output_dir / "train_pairs.json").write_text(
        json.dumps(training, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (output_dir / "eval_pairs.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Training pairs: {len(training)}")
    print(f"Evaluation pairs: {len(evaluation)}")
    print("Saved dataset files in data/")


if __name__ == "__main__":
    main()
