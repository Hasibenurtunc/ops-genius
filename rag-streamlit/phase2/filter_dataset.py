from datasets import load_dataset
import pandas as pd

# Veri setini yükle
print(" Bitext yükleniyor...")
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)
df = dataset["train"].to_pandas()

# 3 intent seç 
selected_intents = [
    "cancel_order",      # ORDER kategorisi
    "recover_password",  # ACCOUNT kategorisi  
    "delivery_options"   # DELIVERY kategorisi
]

# Filtrele
filtered = df[df["intent"].isin(selected_intents)]

print(f"\n Filtreleme sonucu:")
print(filtered["intent"].value_counts())

# Her intentten ilk 100 satırı al
final_df = filtered.groupby("intent").head(100).reset_index(drop=True)

print(f"\n Final veri seti: {len(final_df)} satır")
print(final_df["intent"].value_counts())

# Kaydet
final_df.to_csv("phase2/filtered_dataset.csv", index=False)
print("\n Kaydedildi: phase2/filtered_dataset.csv")

# İlk 5 satıra bak
print("\n İlk 5 satır:")
print(final_df[["instruction", "intent", "flags"]].head())