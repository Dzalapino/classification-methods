import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression


def prepare_dataset() -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """
    Prepare the Titanic dataset for training and testing
    :return: Training and testing features and labels
    """
    titanic = sns.load_dataset('titanic')  # Load the Titanic dataset

    # Drop redundant columns (e.g. 'class' is the same as 'pclass' but in string format)
    titanic = titanic.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], axis=1)
    # Drop rows with missing embarked values (only 2 rows)
    titanic = titanic.dropna(subset=['embarked'])

    # Convert categorical columns to numerical
    titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
    titanic['embarked'] = titanic['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Impute missing age values using KNN (around 20% of the data is missing in age column)
    imputer = KNNImputer(n_neighbors=5)
    titanic = pd.DataFrame(imputer.fit_transform(titanic), columns=titanic.columns)

    # Split the dataset into training and testing sets
    titanic_train, titanic_test = train_test_split(titanic, test_size=0.2, random_state=42)
    train_x = titanic_train.drop(columns='survived')
    train_y = titanic_train['survived']
    test_x = titanic_test.drop(columns='survived')
    test_y = titanic_test['survived']

    return train_x, train_y, test_x, test_y


def show_results(accuracy, pred_y, test_x, test_y) -> None:
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Example predictions:')
    for i in range(5):
        print(f'For the following features:'
              f' {test_x.iloc[i].to_dict()}')
        print(f'    Predicted: {"Survived" if pred_y[i] == 1.0 else "Did not survive"},'
              f' Actual: {"Survived" if test_y.iloc[i] == 1.0 else "Did not survive"}')


def classify_kNN(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    """
    Classify the Titanic dataset using k-Nearest Neighbors
    :param train_x: Training features
    :param train_y: Training labels
    :param test_x: Testing features
    :param test_y: Testing labels
    :return: None
    """

    # Train the k-Nearest Neighbors model
    k_nn = KNeighborsClassifier(n_neighbors=3)
    k_nn.fit(train_x, train_y)

    # Test the k-Nearest Neighbors model
    pred_y = k_nn.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)

    print('\n==================== k-Nearest Neighbors ====================')
    show_results(accuracy, pred_y, test_x, test_y)


def classify_SVM(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    """
    Classify the Titanic dataset using Support Vector Machines
    :param train_x: Training features
    :param train_y: Training labels
    :param test_x: Testing features
    :param test_y: Testing labels
    :return: None
    """

    # Train the Support Vector Machines model
    svm = SVC(kernel='linear')
    svm.fit(train_x, train_y)

    # Test the Support Vector Machines model
    pred_y = svm.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)

    print('\n==================== Support Vector Machines ====================')
    show_results(accuracy, pred_y, test_x, test_y)


def classify_linear_regression(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    """
    Classify the Titanic dataset using Linear Regression
    :param train_x: Training features
    :param train_y: Training labels
    :param test_x: Testing features
    :param test_y: Testing labels
    :return: None
    """

    # Train the Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(train_x, train_y)

    # Test the Linear Regression model
    pred_y = lin_reg.predict(test_x)
    pred_y = np.round(pred_y)  # Round to the nearest integer
    accuracy = accuracy_score(test_y, pred_y)

    print('\n==================== Linear Regression ====================')
    show_results(accuracy, pred_y, test_x, test_y)


def classify_logistic_regression(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    """
    Classify the Titanic dataset using Logistic Regression
    :param train_x: Training features
    :param train_y: Training labels
    :param test_x: Testing features
    :param test_y: Testing labels
    :return: None
    """

    # Train the Logistic Regression model, increase the max iterations to avoid convergence warning (acc before 0.80)
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(train_x, train_y)

    # Test the Logistic Regression model
    pred_y = log_reg.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)

    print('\n==================== Logistic Regression ====================')
    show_results(accuracy, pred_y, test_x, test_y)


def main():
    train_x, train_y, test_x, test_y = prepare_dataset()

    classify_kNN(train_x, train_y, test_x, test_y)
    classify_SVM(train_x, train_y, test_x, test_y)
    classify_linear_regression(train_x, train_y, test_x, test_y)
    classify_logistic_regression(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
