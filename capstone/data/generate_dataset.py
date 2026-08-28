"""
generate_dataset.py
Generates a dataset of product/service reviews, with sentiment labels
(positive / negative / neutral), for training the Sentiment Analysis model.

Note: Since the development environment has no internet access to download a
public dataset (e.g. from the Hugging Face Hub), a synthetic dataset is
generated instead, resembling real reviews by mixing several styles of
positive/negative/neutral sentences — enough to fully demonstrate the
end-to-end pipeline (data -> model -> API -> UI).
"""
import csv
import random

random.seed(42)

positive_en = [
    "I absolutely love this product, it works perfectly!",
    "Amazing quality and fast shipping, highly recommend.",
    "This is the best purchase I've made this year.",
    "The customer service was fantastic and very helpful.",
    "Great value for money, exceeded my expectations.",
    "The app is smooth, fast, and very easy to use.",
    "Excellent build quality, feels premium and durable.",
    "I'm so happy with this, will buy again for sure.",
    "The staff were friendly and the service was quick.",
    "Five stars, everything works exactly as described.",
    "Superb design and the battery life is incredible.",
    "This made my daily routine so much easier, thank you.",
]

negative_en = [
    "This product broke after just two days of use.",
    "Terrible customer service, nobody responded to my emails.",
    "Waste of money, it doesn't work as advertised.",
    "The app keeps crashing every time I open it.",
    "Very disappointed with the quality, feels cheap.",
    "Delivery was extremely late and the box was damaged.",
    "I regret buying this, it stopped working immediately.",
    "The instructions were confusing and support was useless.",
    "Poor performance and the battery drains too fast.",
    "This is the worst experience I've had with a company.",
    "Overpriced for what you actually get, not worth it.",
    "The screen cracked on the first day, very fragile.",
]

neutral_en = [
    "The product arrived on time, nothing special to note.",
    "It works as expected, neither great nor bad.",
    "The packaging was standard, similar to other orders.",
    "I received the item and it matches the description.",
    "The app has basic features, does the job for now.",
    "Average experience overall, might consider other options.",
    "The color is slightly different from the photos online.",
    "Setup took about ten minutes, fairly straightforward.",
    "The price is reasonable compared to similar products.",
    "It is okay for casual use but not for heavy tasks.",
    "The manual explains most of the functions clearly.",
    "Shipping took the usual five to seven business days.",
]

positive_en_2 = [
    "The product quality is great, works exactly as I expected.",
    "Excellent customer service, staff replied quickly and politely.",
    "Really impressed, fast shipping and great packaging.",
    "This app is very easy to use, smooth with no glitches.",
    "Great value for money, quality exceeds the price paid.",
    "I really like it, will definitely buy again.",
    "Beautiful design, materials are sturdy and very durable.",
    "Battery life is excellent, lasts all day without frequent charging.",
    "The team gave great advice and solved the problem quickly.",
    "Five stars, everything matches what was advertised.",
]

negative_en_2 = [
    "The product broke after just two days, really terrible.",
    "Customer service is awful, I emailed and nobody replied.",
    "Total waste of money, it doesn't work as advertised at all.",
    "The app freezes constantly, crashes every time I open it.",
    "Quality is worse than expected, feels really cheap.",
    "Shipping was very slow and the box arrived dented.",
    "Regret buying this, it stopped working after barely any use.",
    "The manual instructions were confusing and support was no help.",
    "Poor performance, the battery drains way too fast.",
    "Way too expensive compared to what you actually get.",
]

neutral_en_2 = [
    "The product arrived on time, nothing particularly exciting.",
    "Works as usual, not especially good or bad.",
    "The packaging was the same as other stores I've ordered from.",
    "Received the product exactly as described.",
    "The app has all the basic features, fine for general use.",
    "Overall an average experience, might try other options too.",
    "The color received is slightly different from the website photos.",
    "Setup took about ten minutes, not complicated.",
    "The price is fairly reasonable compared to other models.",
    "Fine for everyday use, but not suited for heavy tasks.",
]

rows = []
for text in positive_en + positive_en_2:
    rows.append((text, "positive"))
for text in negative_en + negative_en_2:
    rows.append((text, "negative"))
for text in neutral_en + neutral_en_2:
    rows.append((text, "neutral"))

augmented = list(rows)
connectors = ["Honestly, ", "Overall, "]
for text, label in rows:
    for c in connectors:
        augmented.append((c + text[0].lower() + text[1:], label))

random.shuffle(augmented)

with open("data/reviews_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(augmented)

print(f"Generated {len(augmented)} rows -> data/reviews_dataset.csv")
