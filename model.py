import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("books.csv", on_bad_lines="skip")

features = ["ratings_count", "text_reviews_count","  num_pages"]
df = df.dropna(subset=features + ["average_rating"])

#reducimos carga de memoria para evitar problemas en Render
df_elegido, df_noelegido = train_test_split(df, test_size=0.3, random_state=42)

X = df_elegido[features]
y = df_elegido["average_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)