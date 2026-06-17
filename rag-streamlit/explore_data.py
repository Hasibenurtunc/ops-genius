from datasets import load_dataset

# Bitext'i indir
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# İlk 5 satıra bak
df = dataset["train"].to_pandas()
print("Toplam satır sayısı:", len(df))
print("\nSütunlar:", df.columns.tolist())
print("\nİlk 5 satır:")
print(df.head())
print("\nIntent kategorileri:")
print(df["intent"].value_counts())