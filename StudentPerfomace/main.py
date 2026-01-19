import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../StudentPerfomace/StudentPerformance.csv")
df.columns = df.columns.str.replace(" ", "_")

y = df["Previous_Scores"]
X = df.drop("Previous_Scores", axis=1)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror",
    eval_metric="rmse",
)

model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5, scoring="r2")

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(scores)
print("R² médio:", scores.mean())
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
