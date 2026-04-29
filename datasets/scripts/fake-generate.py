import pandas as pd
import random

# -------------------------------
# Fake Data Generator
# -------------------------------
subjects = [
    "Government", "Army", "Supreme Court", "Election Commission",
    "Prime Minister", "Police"
]

verbs = [
    "secretly approves", "bans", "leaks", "confirms",
    "plans", "hides"
]

objects = [
    "new law overnight",
    "hidden agreement with foreign country",
    "rigged election results",
    "mass surveillance program",
    "secret economic deal",
    "nationwide internet shutdown"
]

def generate_fake(n=1000):
    data = []
    for _ in range(n):
        sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
        data.append({"text": sentence, "label": "fake"})
    return data


# -------------------------------
# Generate + Save
# -------------------------------
fake_data = generate_fake(2000)   # change number if needed

df = pd.DataFrame(fake_data)

# remove duplicates (important)
df.drop_duplicates(inplace=True)

# save to csv
df.to_csv("./datasets/generated_fakenews/fake_news_dataset.csv", index=False)

print("Fake dataset saved!")
print(df.head())
print("\nTotal samples:", len(df))