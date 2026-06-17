# import pandas as pd

# df = pd.read_csv("phase2/results/controlled_results.csv")
# failed = df[(df["intent"] == "recover_password") & (df["failed"] == True)]

# print("BASARISIZ RECOVER PASSWORD SORULARI:")
# print(f"Toplam basarisiz: {len(failed)}")
# print()

# for i, row in failed.head(20).iterrows():
#     print(f"Soru : {row['clean_question']}")
#     print(f"Cevap: {row['rag_answer'][:100]}")
#     print("-" * 50)


import pandas as pd

df = pd.read_csv("phase2/results/manipulated_results.csv")
failed = df[df["failed"] == True]

print(f"Basarisiz: {len(failed)}")
print()
for i, row in failed.iterrows():
    print(f"Tip  : {row['manipulation_type']}")
    print(f"Soru : {row['manipulated_question']}")
    print(f"Cevap: {row['rag_answer'][:120]}")
    print("-" * 50)