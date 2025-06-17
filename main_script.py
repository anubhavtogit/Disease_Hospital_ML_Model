import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
# data collection 

df=pd.read_csv('symptom_classification_dataset.csv')

#print(df.head())

no = 0
mild = 25
severe = 50
df.replace(to_replace="no", value=0, inplace=True)
df.replace(to_replace="mild", value=25, inplace=True)
df.replace(to_replace="severe", value=50, inplace=True)
df.to_csv('outputfile.csv', index=False)

#print(df.head())

X = df.drop(columns=['label'])
y = df['label']

print("X example:", X[:3])  # first 3 rows of X
print("y example:", y[:3])  # first 3 labels

#Training the classifier

clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X, y)

#prediction

preds = clf.predict([[100.9,25,50,50,25,0,0,0,0,1]])
print(preds)

X_encoded = pd.get_dummies(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("y_enc example:", y_encoded[:])  # first 3 labels

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

#model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
#model.fit(X_train_scaled, y_train)

#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

#logistic_model = LogisticRegression(solver='lbfgs', max_iter=500)
#logistic_model.fit(X_train_scaled, y_train)
#logistic_acc = accuracy_score(y_test, logistic_model.predict(X_test_scaled))

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)
#tree_acc = accuracy_score(y_test, tree_model.predict(X_test))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
#rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

#print("Model Accuracy Comparison:")
#print(f"Logistic Regression: {logistic_acc * 100:.2f}%")
#print(f"Decision Tree:        {tree_acc * 100:.2f}%")
#print(f"Random Forest:        {rf_acc * 100:.2f}%")

new_input = pd.DataFrame([{
    'temperature': 100.9,
    'cough_no': 0, 'cough_mild': 1, 'cough_severe': 0,
    'runny_nose_no': 0, 'runny_nose_mild': 0, 'runny_nose_severe': 1,
    'sore_throat_no': 0, 'sore_throat_mild': 0, 'sore_throat_severe': 1,
    'headache_no': 0, 'headache_mild': 1, 'headache_severe': 0,
    'chills': 0,
    'shortness_of_breath_no': 1,'shortness_of_breath_mild': 0, 'shortness_of_breath_severe': 0,
    'night_sweating': 0,
    'loss_of_taste': 0,
    'loss_of_smell': 1
}])


for col in X_encoded.columns:
    if col not in new_input.columns:
        new_input[col] = 0
new_input = new_input[X_encoded.columns]

new_input_scaled = scaler.transform(new_input)
#probs = logistic_model.predict_proba(new_input_scaled)[0]
probs = tree_model.predict_proba(new_input_scaled)[0]
#probs = rf_model.predict_proba(new_input_scaled)[0]
print("\n")
probs
print("\n")
for cls, p in zip(le.classes_, probs):
   print(f"{cls}: {p*100:.2f}%")

print("\n")
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

softmax_probs = softmax(probs)

print(softmax_probs)

for disease, prob in zip(le.classes_, softmax_probs):
   print(f"{disease}: {prob * 100:.2f}%")

print("\n")

X_test_scaled = scaler.fit_transform(X_test)

y_proba = tree_model.predict_proba(X_test_scaled)



#def softmax(x):
 #   e_x = np.exp(x - np.max(x))
 #   return e_x / e_x.sum()

#softmax_probs = softmax(y_proba[0])

true_label_probs = [probs[label] for probs, label in zip(y_proba, y_test)]

# 8. Extract true label probabilities

# 9. Threshold check
threshold = 0.050  # 10%
correct_preds = [1 if p >= threshold else 0 for p in true_label_probs]

#y_proba[0]
#softmax_probs
#correct_preds

total_samples = len(y_test)
error_free_count = sum(correct_preds)
error_count = total_samples - error_free_count
error_percentage = (error_count / total_samples) * 100
print("\n")
# 11. Output
print("=== Threshold-based Error Evaluation ===")
print(f"Threshold: {threshold * 100:.1f}%")
print(f"Total Test Samples: {total_samples}")
print(f"Error-Free Predictions (≥ {threshold * 100:.0f}%): {error_free_count}")
print(f"Erroneous Predictions (< {threshold * 100:.0f}%): {error_count}")
print(f"Error Percentage: {error_percentage:.2f}%")

correct_top3 = 0
for probs, label in zip(y_proba, y_test):
    top3_indices = np.argsort(probs)[-3:]  # Indices of top 3 probabilities
    if label in top3_indices:
        correct_top3 += 1

# 9. Calculate error metrics
total_samples = len(y_test)
error_top3 = total_samples - correct_top3
error_percentage = (error_top3 / total_samples) * 100

# 10. Output results
print("=== Top-3 Evaluation ===")
print(f"Total Test Samples: {total_samples}")
print(f"Correct Predictions (Top-3): {correct_top3}")
print(f"Incorrect Predictions (Not in Top-3): {error_top3}")
print(f"Error Percentage: {error_percentage:.2f}%")

df= pd.read_csv('all_cities_hospitals_extended_diseases.csv')
#print(df)

# take input city of the user

input_city=input("Enter name of your city: ")

#print(val)

patient_vector = np.array([17.40, 13.36, 18.61, 12.33, 12.47, 12.81, 13.01])

city_distances = {
    'MetroCity':    {'MetroCity': 0,  'RiverTown': 50,  'HillVille': 70,  'GreenBay': 80,  'SunPort': 100},
    'RiverTown':    {'MetroCity': 50, 'RiverTown': 0,   'HillVille': 60,  'GreenBay': 70,  'SunPort': 90},
    'HillVille':    {'MetroCity': 70, 'RiverTown': 60,  'HillVille': 0,   'GreenBay': 40,  'SunPort': 70},
    'GreenBay':     {'MetroCity': 80, 'RiverTown': 70,  'HillVille': 40,  'GreenBay': 0,   'SunPort': 50},
    'SunPort':      {'MetroCity': 100,'RiverTown': 90,  'HillVille': 70,  'GreenBay': 50,  'SunPort': 0}
}
alpha = 0.0025

disease_cols = ['COVID-19', 'Influenza', 'Viral Fever', 'Malaria', 'Dengue', 'Cough', 'Tuberculosis']
hospital_vectors = df[disease_cols].values

df[disease_cols].values

similarities = cosine_similarity([patient_vector], hospital_vectors)[0]
df['Similarity'] = similarities


def get_city(row):
    city_scores = {city: row[city] for city in city_distances}
    return max(city_scores, key=city_scores.get)
df['City'] = df.apply(get_city, axis=1)

df['Distance_km'] = df['City'].apply(lambda c: city_distances[input_city][c])
df['Penalty'] = df['Distance_km'] * alpha
df['Final Score'] = df['Similarity'] - df['Penalty']

top_k = 5
recommended = df.sort_values('Final Score', ascending=False)[['Hospital Name', 'City', 'Similarity', 'Distance_km', 'Penalty', 'Final Score']].head(top_k)

print(f"✅ Top {top_k} Hospital Recommendations for patient from **{input_city}**:\n")
print(recommended.to_string(index=False))