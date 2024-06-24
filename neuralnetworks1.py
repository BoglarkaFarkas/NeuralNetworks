from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

#Zdroje: Seminar 1 (cvicenie1.py), Seminar 2 (seminar2.py), Seminar 3 (main.py), ChatGPT

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('zadanie1_dataset.csv')
df.rename({
    'danceability': 'danceability',
    'energy': 'energy',
    'loudness': 'loudness',
    'speechiness': 'speechiness',
    'acousticness': 'acousticness',
    'instrumentalness': 'instrumentalness',
}, inplace=True, axis=1)

df['danceability'] = df['danceability'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['loudness'] = df['loudness'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['energy'] = df['energy'].apply(lambda x: x*100 if np.random.randint(0, 1000) < 1 else x)
df['speechiness'] = df['speechiness'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['acousticness'] = df['acousticness'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['instrumentalness'] = df['instrumentalness'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['liveness'] = df['liveness'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['valence'] = df['valence'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)
df['popularity'] = df['popularity'].apply(lambda x: -x if np.random.randint(0, 1000) < 1 else x)

print("*"*20, "Before removing outliers", "*"*20)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

df = df[(df['danceability'] <= 1) & (df['danceability'] >= 0)]
df = df[df['loudness'] <= 0]
df = df[df['energy'] <= 1]
df = df[df['speechiness'] >= 0]
df = df[df['acousticness'] >= 0]
df = df[df['instrumentalness'] >= 0]
df = df[df['liveness'] >= 0]
df = df[df['valence'] >= 0]
df = df[df['popularity'] >= 0]

print("*"*20, "After removing outliers", "*"*20)
print("-"*10, "Min", "-"*10)
print(df.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(df.max(numeric_only=True))

print("*"*20, "Missing values", "*"*20)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

df_with_topgenre = df[df['top_genre'].notnull()]
df.drop(columns=['top_genre'], inplace=True)
df.dropna(inplace=True)

print("*"*20, "Missing values after removing them", "*"*20)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

print("*"*20, "Column types", "*"*20)
print(df.dtypes)

le = LabelEncoder()
df_with_topgenre['labelEncoding'] = le.fit_transform(df_with_topgenre['emotion'])
print("*"*20, "Label encoding", "*"*20)
print(df_with_topgenre[['emotion', 'labelEncoding']].head(10))
emotions_df = df_with_topgenre['emotion']

df_with_topgenre = pd.get_dummies(df_with_topgenre, columns=['emotion'], prefix='', prefix_sep='')
df_with_topgenre['emotion'] = emotions_df

myemotions = list(df_with_topgenre['emotion'].unique())
show_columns = ['emotion'] + myemotions
print("*"*20, "Dummy encoding", "*"*20)
print(df_with_topgenre[show_columns].head(10))

df.drop(columns=['explicit'], inplace=True)
df.drop(columns=['name'], inplace=True)
df.drop(columns=['url'], inplace=True)
df.drop(columns=['genres'], inplace=True)
df.drop(columns=['filtered_genres'], inplace=True)

top_5_countries = df.groupby('emotion').sum().sort_values(by='popularity', ascending=False).head(5).index
df = df[df['emotion'].isin(top_5_countries)]

le = LabelEncoder()
df['emotion'] = le.fit_transform(df['emotion'])
X = df.drop(columns=['emotion'])
y = df['emotion']

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)  #42
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

print("*"*20, "Dataset shapes", "*"*20)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms before scaling/standardizing')
plt.show()

print("*"*20, "Before scaling/standardizing", "*"*20)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()

print("*"*20, "After scaling/standardizing", "*"*20)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

print("*"*20, "MLP", "*"*20)
print(f"Random accuracy: {1/len(y_train.unique())}")

clf = MLPClassifier(
    hidden_layer_sizes=(100, 100, 5, 6, 90),
    random_state=1,
    max_iter=100,
    validation_fraction=0.2,
    early_stopping=True,
    learning_rate='adaptive',
    learning_rate_init=0.001,
).fit(X_train, y_train)

y_pred = clf.predict(X_train)
print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred))
cm_train = confusion_matrix(y_train, y_pred)

y_pred = clf.predict(X_test)
print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred))
cm_test = confusion_matrix(y_test, y_pred)

class_names = list(le.inverse_transform(clf.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['popularity'], bins=20, kde=True,color="#f23a84")
plt.title("Distribution of 'popularity'")
plt.xlabel("Value")
plt.ylabel("Popularity")
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
df = pd.read_csv('zadanie1_dataset.csv')
plt.figure(figsize=(10, 6))
sns.boxplot(x='emotion', y='popularity', data=df, color="#f74d91")
plt.title("Box Plot of 'popularity' by 'emotion'")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='energy', y='loudness', color="#f74d91")
plt.title('Scatter plot medzi energy a loudness')
plt.show()

category_counts = df['emotion'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
plt.title('Percentuálne zastúpenie hodnôt v emotion')
plt.legend(category_counts.index, title='Emotion', loc='lower right')
plt.show()

plt.figure(figsize=(8, 6))
sns.stripplot(x='emotion', y='energy', data=df, jitter=True, hue='emotion', legend=True,palette='Set3')
plt.title("Strip Plot of 'energy' by 'emotion'")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='emotion', y='tempo', data=df, palette='Set3',hue="emotion")
plt.title("Violin Plot of 'tempo' by 'emotion'")
plt.xticks(rotation=45)
plt.show()

df.drop(columns=['explicit'], inplace=True)
df.drop(columns=['name'], inplace=True)
df.drop(columns=['url'], inplace=True)
df.drop(columns=['genres'], inplace=True)
df.drop(columns=['filtered_genres'], inplace=True)
df_with_topgenre = df[df['top_genre'].notnull()]
df.drop(columns=['top_genre'], inplace=True)

df.dropna(inplace=True)
X = df.drop(columns=['emotion'])
y = df['emotion']

y = pd.get_dummies(y)
y = y.astype(int)
print(y)

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

model = Sequential()
model.add(Dense(24, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32,callbacks=[earlystopping])

test_scores = model.evaluate(X_test, y_test, verbose=0)

print("*"*20, "Test accuracy", "*"*20)
print(f"Test accuracy: {test_scores[1]:.4f}")

train_scores = model.evaluate(X_train, y_train, verbose=0)

print("*"*20, "Train accuracy", "*"*20)
print(f"Train accuracy: {train_scores[1]:.4f}")
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5)
print(f"y_test: {y_test.shape}")
print(f"y_train: {y_train.shape}")
class_names = df['emotion'].unique().tolist()

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
class_names = df['emotion'].unique().tolist()
cm = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Plot loss")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Plot accuracy")
plt.legend()
plt.show()