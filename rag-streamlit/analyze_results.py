import pandas as pd

# ============================================================
# 1. SONUÇLARI YÜKLE
# ============================================================
df = pd.read_csv("test_results.csv")
print(f"✅ {len(df)} sonuç yüklendi\n")

# ============================================================
# 2. GENEL BAKIŞ
# ============================================================
print("=" * 60)
print("📊 GENEL BAKIŞ")
print("=" * 60)

# Kaç unique intent var?
print(f"Toplam intent sayısı : {df['intent'].nunique()}")
print(f"Toplam soru sayısı   : {len(df)}")

# Kaynak dağılımı
print("\n📄 Hangi dökümandan kaç cevap geldi?")
print(df['source'].value_counts())

# ============================================================
# 3. BAŞARISIZ CEVAPLARI BUL
# ============================================================
print("\n" + "=" * 60)
print("❌ BAŞARISIZ CEVAPLAR")
print("=" * 60)

failed = df[df['rag_answer'].str.contains(
    "could not find", case=False, na=False
)]
print(f"Toplam başarısız: {len(failed)} / {len(df)}")
print(f"Başarısızlık oranı: %{round(len(failed)/len(df)*100, 1)}")

print("\nBaşarısız olan intent'ler:")
print(failed['intent'].value_counts())

# ============================================================
# 4. BAŞARILI CEVAPLARI BUL
# ============================================================
print("\n" + "=" * 60)
print("✅ BAŞARILI CEVAPLAR")
print("=" * 60)

success = df[~df['rag_answer'].str.contains(
    "could not find", case=False, na=False
)]
print(f"Toplam başarılı: {len(success)} / {len(df)}")
print(f"Başarı oranı: %{round(len(success)/len(df)*100, 1)}")

print("\nBaşarılı olan intent'ler:")
print(success['intent'].value_counts())

# ============================================================
# 5. PLACEHOLDER SORUNU
# ============================================================
print("\n" + "=" * 60)
print("⚠️  PLACEHOLDER SORUNU")
print("=" * 60)

placeholder = df[df['question'].str.contains(
    "{{", case=False, na=False
)]
print(f"Placeholder içeren soru sayısı: {len(placeholder)}")
print(f"Bunların kaçı başarısız?")

placeholder_failed = placeholder[placeholder['rag_answer'].str.contains(
    "could not find", case=False, na=False
)]
print(f"Başarısız: {len(placeholder_failed)} / {len(placeholder)}")

# ============================================================
# 6. ÖRNEK BAŞARILI VE BAŞARISIZ
# ============================================================
print("\n" + "=" * 60)
print("🔍 ÖRNEK BAŞARILI CEVAP")
print("=" * 60)
if len(success) > 0:
    row = success.iloc[0]
    print(f"Intent : {row['intent']}")
    print(f"Soru   : {row['question'][:100]}")
    print(f"Cevap  : {row['rag_answer'][:200]}")
    print(f"Kaynak : {row['source']}")

print("\n" + "=" * 60)
print("🔍 ÖRNEK BAŞARISIZ CEVAP")
print("=" * 60)
if len(failed) > 0:
    row = failed.iloc[0]
    print(f"Intent : {row['intent']}")
    print(f"Soru   : {row['question'][:100]}")
    print(f"Cevap  : {row['rag_answer'][:200]}")
    print(f"Kaynak : {row['source']}")

# ============================================================
# 7. ÖZET RAPOR
# ============================================================
print("\n" + "=" * 60)
print("📋 ÖZET RAPOR")
print("=" * 60)
print(f"✅ Başarı oranı    : %{round(len(success)/len(df)*100, 1)}")
print(f"❌ Başarısızlık   : %{round(len(failed)/len(df)*100, 1)}")
print(f"⚠️  Placeholder    : {len(placeholder)} soru etkilendi")
print("\nBaşlıca başarısızlık nedenleri:")
print("1. {{placeholder}} içeren sorular")
print("2. Kaba dil / slang ifadeler")
print("3. Çok spesifik hesap türleri (platinum, premium vb.)")