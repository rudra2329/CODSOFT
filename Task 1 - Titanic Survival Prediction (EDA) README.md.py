# ------------------------------------------------------------------
# 🚢 Titanic Survival Prediction | CodSoft Data Science Internship

# 👩‍💻 Created by: Siva Rudra V
# ------------------------------------------------------------------

# 🎯 Objective:
# Predict survival of passengers using features like age, sex, class, etc.

# -------------------------
# 🔹 Step 1: Import Libraries
# -------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------
# 🔹 Step 2: Load the Dataset
# --------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

print("✅ Dataset loaded successfully!")
print("\n📄 First 5 rows of the dataset:")
print(data.head())

# ---------------------------------------------
# 🔹 Step 3: Handle Missing Values & Clean Data
# ---------------------------------------------
print("\n🔍 Missing values:")
print(data.isnull().sum())

# Fill missing Age with mean
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Fill missing Embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop unused columns
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# -----------------------------
# 🔹 Step 4: Encode Categorical Data
# -----------------------------
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# -----------------------------
# 🔹 Step 5: Exploratory Analysis
# -----------------------------
# Survival count
sns.countplot(x='Survived', data=data)
plt.title("🎯 Survival Count")
plt.show()

# Survival by class
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title("🧳 Survival by Passenger Class")
plt.show()

# Survival by gender
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title("🧍‍♀️ Survival by Gender")
plt.show()

# -----------------------------------
# 🔹 Step 6: Split Data and Train Model
# -----------------------------------
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 🔹 Step 7: Predictions & Results
# -----------------------------
y_pred = model.predict(X_test)

print("\n✅ Model Performance:")
print("🔸 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n🔸 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔸 Classification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 🔚 Final Conclusion (By Me)
# --------------------------
print("\n📌 Conclusion (by Siva Rudra V):")
print("→ The logistic regression model gave good accuracy.")
print("→ Gender and class had a strong effect on survival chances.")
print("→ This project helped me learn data cleaning, visualization, and machine learning basics.")
