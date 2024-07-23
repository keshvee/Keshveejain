import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Sample IP address data with labels
data = [
    {"ip": "192.168.1.1", "label": 1},  # Malicious
    {"ip": "192.168.1.2", "label": 0},  # Benign
    {"ip": "192.168.1.3", "label": 1},  # Malicious
    {"ip": "192.168.1.4", "label": 0},  # Benign
    {"ip": "192.168.1.5", "label": 1},  # Malicious
    {"ip": "192.168.1.6", "label": 0},  # Benign
    {"ip": "192.168.1.7", "label": 1},  # Malicious
    {"ip": "192.168.1.8", "label": 0},  # Benign
    {"ip": "192.168.1.9", "label": 1},  # Malicious
    {"ip": "192.168.1.10", "label": 0}, # Benign
]

df = pd.DataFrame(data)

# Vectorize IP address data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['ip'])

# Vectorize IP address data using Count Vectorizer
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(df['ip'])

# Combine features from both vectorizers
X_combined = np.hstack((X_tfidf.toarray(), X_count.toarray()))

# Labels for supervised learning
y = df['label']

# Train-test split for supervised learning
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Supervised Learning: Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
y_pred = log_reg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Supervised Learning - Logistic Regression Accuracy: {accuracy}')
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Unsupervised Learning: K-means Clustering
n_clusters = 2  # assuming 2 clusters: malicious and benign
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_combined)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(X_combined)
colors = ['r' if label == 1 else 'b' for label in kmeans.labels_]
x_axis = scatter_plot_points[:, 0]
y_axis = scatter_plot_points[:, 1]
plt.scatter(x_axis, y_axis, c=colors)
plt.title('Unsupervised Learning - K-means Clustering')
plt.show()

# Reinforcement Learning: Q-learning
state_space_size = len(df)
action_space_size = 2  # 0 for benign, 1 for malicious
Q = np.zeros((state_space_size, action_space_size))
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

def get_reward(action, true_label):
    return 1 if action == true_label else -1

for episode in range(1000):
    for i in range(state_space_size):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(action_space_size)
        else:
            action = np.argmax(Q[i])
        
        true_label = y.iloc[i]
        reward = get_reward(action, true_label)
        Q[i, action] = Q[i, action] + alpha * (reward + gamma * np.max(Q[i]) - Q[i, action])

print("Reinforcement Learning - Q-values:")
print(Q)

# Combining all three methods for prediction
def combined_predict(ip):
    # Vectorize the IP using both vectorizers
    ip_tfidf_vector = tfidf_vectorizer.transform([ip])
    ip_count_vector = count_vectorizer.transform([ip])
    ip_combined_vector = np.hstack((ip_tfidf_vector.toarray(), ip_count_vector.toarray()))
    
    # Supervised learning prediction
    supervised_pred = log_reg_model.predict(ip_combined_vector)[0]
    
    # Unsupervised learning prediction
    cluster = kmeans.predict(ip_combined_vector)[0]
    unsupervised_pred = 1 if cluster == 0 else 0  # Assuming cluster 0 is malicious
    
    # Reinforcement learning prediction
    state_index = df[df['ip'] == ip].index[0]
    reinforcement_pred = np.argmax(Q[state_index])
    
    # Majority voting
    predictions = [supervised_pred, unsupervised_pred, reinforcement_pred]
    final_prediction = max(set(predictions), key=predictions.count)
    
    # Descriptive result
    if final_prediction == 1:
        result = "malicious"
    else:
        result = "safe"
    
    return result

# Example prediction
new_ip = "192.168.1.1"  # Example IP address
print(f'Combined Prediction for "{new_ip}":', combined_predict(new_ip))
