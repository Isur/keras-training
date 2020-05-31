import pandas as pd
import pathlib
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix as cm
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


ABS_PATH = pathlib.Path(__file__).parent.absolute()


class Dataset(object):
    def __init__(self):
        self.folder = f"{ABS_PATH}/dataset"
        self.file = f"{self.folder}/winequalityN.csv"
        self.dataset = None
        self.db_prepared = None

    def get_data(self):
        dataset = pd.read_csv(self.file)
        self.dataset = dataset
        return dataset

    def show_dataset(self):
        if self.dataset is None:
            print("no dataset loaded")
        else:
            print(self.dataset)

    def prepare_data(self):
        self.show_nulls(self.dataset)
        self.fill_nulls()
        class_tp = LabelEncoder()
        y_tp = class_tp.fit_transform(self.db_prepared.type.values)
        self.db_prepared['color'] = y_tp
        self.db_prepared['quality_label'] = self.db_prepared.quality.apply(
            lambda q: 'low' if q <= 4 else 'medium' if q <= 7 else 'high')

    def show_nulls(self, dataset):
        Sum = dataset.isnull().sum()
        Percentage = (dataset.isnull().sum() / dataset.isnull().count())
        stat = pd.concat([Sum, Percentage], axis=1, keys=['Sum', 'Percentage'])
        print(stat)

    def fill_nulls(self):
        db = self.dataset
        total_missing_values = db.isnull().sum()
        missing_values_per = db.isnull().sum() / db.isnull().count()
        null_values = pd.concat([total_missing_values, missing_values_per], axis=1,
                                keys=['total_null', 'total_null_perc'])
        null_values = null_values.sort_values('total_null', ascending=False)

        fill_list = null_values[null_values['total_null'] > 0].index
        db_mean = db.copy()
        for col in fill_list:
            db_mean.loc[:, col].fillna(db_mean.loc[:, col].mean(), inplace=True)
        self.show_nulls(db_mean)
        self.db_prepared = db_mean

    def graphs(self):
        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Wine Type & Quality")
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

        ax[0, 0].set_title("Red Wine")
        ax[0, 0].set_xlabel("Quality")
        ax[0, 0].set_ylabel("Freqeuncy")
        x = self.db_prepared.quality[self.db_prepared.type == 'red'].value_counts()
        x = (list(x.index), list(x.values))
        ax[0, 0].set_ylim([0, 2500])
        ax[0, 0].bar(x[0], x[1], color='red', edgecolor='black')

        ax[0, 1].set_title("White Wine")
        ax[0, 1].set_xlabel("Quality")
        ax[0, 1].set_ylabel("Freqeuncy")
        x = self.db_prepared.quality[self.db_prepared.type == 'white'].value_counts()
        x = (list(x.index), list(x.values))
        ax[0, 1].set_ylim([0, 2500])
        ax[0, 1].bar(x[0], x[1], color='white', edgecolor='black')

        ax[1, 0].set_title("Red Wine")
        ax[1, 0].set_xlabel("Quality")
        ax[1, 0].set_ylabel("Freqeuncy")
        x = self.db_prepared.quality_label[self.db_prepared.type == 'red'].value_counts()
        x = (list(x.index), list(x.values))
        ax[1, 0].set_ylim([0, 3500])
        ax[1, 0].bar(x[0], x[1], color='red', edgecolor='black')

        ax[1, 1].set_title("White Wine")
        ax[1, 1].set_xlabel("Quality")
        ax[1, 1].set_ylabel("Freqeuncy")
        x = self.db_prepared.quality_label[self.db_prepared.type == 'white'].value_counts()
        x = (list(x.index), list(x.values))
        ax[1, 1].set_ylim([0, 3500])
        ax[1, 1].bar(x[0], x[1], color='white', edgecolor='black')

        plt.show()

    def data_description(self):
        db = self.db_prepared
        # by type
        red = round(db.loc[db.type == 'red', db.columns].describe(), 2)
        white = round(db.loc[db.type == 'white', db.columns].describe(), 2)
        type_table = pd.concat([red, white], axis=0, keys=['Red Wine', 'White Wine']).T
        # by quality
        low = round(db.loc[db.quality_label == 'low', db.columns].describe(), 2)
        medium = round(db.loc[db.quality_label == 'medium', db.columns].describe(), 2)
        high = round(db.loc[db.quality_label == 'high', db.columns].describe(), 2)
        quality_table = pd.concat([low, medium, high], axis=0, keys=['Low Quality', 'Medium Quality', 'High Quality']).T

        print(type_table)
        print(quality_table)

    def correlation_color(self):
        print("Don't worry! It will take some time!")
        db = self.db_prepared

        corr = db.corr()
        top_corr_cols = corr.color.sort_values(ascending=False).keys()
        top_corr = corr.loc[top_corr_cols, top_corr_cols]
        drop_self = np.zeros_like(top_corr)
        drop_self[np.triu_indices_from(drop_self)] = True
        plt.figure(figsize=(12, 7))
        plt.title("Correlation")
        sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=drop_self)
        sns.set(font_scale=1.5)

        sns.set(font_scale=1.0)
        g = sns.pairplot(data=db, hue='type', palette={'red': '#FF0000', 'white': '#FFFFFF'},
                         plot_kws=dict(edgecolor='black', linewidth=0.5))
        fig = g.fig
        fig.subplots_adjust(top=0.99, wspace=0.3)
        plt.show()

    def correlation_quality(self):
        print("Don't worry! It will take some time!")
        db = self.db_prepared
        corr = db.corr()
        top_corr_cols = corr.quality.sort_values(ascending=False).keys()
        top_corr = corr.loc[top_corr_cols, top_corr_cols]
        drop_self = np.zeros_like(top_corr)
        drop_self[np.triu_indices_from(drop_self)] = True
        plt.figure(figsize=(12, 7))
        plt.title("Correlation")
        sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f",
                    mask=drop_self)
        sns.set(font_scale=1.5)

        sns.set(font_scale=1.0)
        cols = db.columns
        cols = cols.drop('quality')
        g = sns.pairplot(data=db.loc[:, cols], hue='quality_label')
        fig = g.fig
        fig.subplots_adjust(top=0.99, wspace=0.3)
        plt.show()

    def quality_alcohol_relation(self):
        db = self.db_prepared
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Wine type/quality/alcohol")
        sns.boxplot(x='quality', y='alcohol', hue='type', data=db, ax=ax[0],
                    palette={'red': "#FF0000", 'white': "#FFFFFF"})
        ax[0].set_xlabel("Quality")
        ax[0].set_ylabel("Alcohol")
        sns.boxplot(x='quality_label', y='alcohol', hue='type', data=db, ax=ax[1],
                    palette={'red': "#FF0000", 'white': "#FFFFFF"})
        ax[1].set_xlabel("Quality Class")
        ax[1].set_ylabel("Alcohol")
        plt.show()

    def quality_acidity_relation(self):
        db = self.db_prepared
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("Wine type/quality/acidity")
        sns.violinplot(x='quality', y='volatile acidity', hue='type', data=db,
                       split=True, inner='quart', linewidth=1.3, ax=ax[0],
                       palette={'red': "#FF0000", 'white': "#FFFFFF"})

        ax[0].set_xlabel("Quality")
        ax[0].set_ylabel("Fixed Acidity")
        sns.violinplot(x='quality_label', y='volatile acidity', hue='type', data=db,
                       split=True, inner='quart', linewidth=1.3, ax=ax[1],
                       palette={'red': "#FF0000", 'white': "#FFFFFF"})
        ax[1].set_xlabel("Quality Class")
        ax[1].set_ylabel("Fixed Acidity")
        plt.show()

    def type_alcohol_quality_acidity(self):
        db = self.db_prepared
        g = sns.FacetGrid(db, col='type', hue='quality_label',
                          col_order=['red', 'white'], hue_order=['low', 'medium', 'high'])
        g.map(plt.scatter, 'volatile acidity', 'alcohol')
        fig = g.fig
        fig.subplots_adjust(hspace=1, wspace=0.3)
        fig.suptitle("Type - Alcohol - Quality - Acidity")
        g.add_legend(title='Quality Class')

        g = sns.FacetGrid(db, col='type', hue='quality_label',
                          col_order=['red', 'white'], hue_order=['low', 'medium', 'high'])
        g.map(plt.scatter, 'volatile acidity', 'total sulfur dioxide')
        fig = g.fig
        fig.subplots_adjust(hspace=1, wspace=0.3)
        fig.suptitle("Type - Sulfur Dioxide - Quality - Acidity")
        g.add_legend(title='Quality Class')

        plt.show()

    def classifier(self):
        db = self.db_prepared.copy()
        db['quality_range'] = db.quality.apply(
            lambda q: 0 if q <= 4 else 1 if q <= 7 else 2)
        db['type'] = db.type.apply(lambda q: 0 if q == 'white' else 1)
        X = db[['type', 'alcohol', 'density', 'volatile acidity', 'chlorides',
                'citric acid', 'fixed acidity', 'free sulfur dioxide',
                'total sulfur dioxide', 'sulphates', 'residual sugar', 'pH']]
        y = db.quality_range
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        lr = LogisticRegression(random_state=40)
        lr.fit(X_train, y_train)
        train_accuracy = lr.score(X_train, y_train)
        test_accuracy = lr.score(X_test, y_test)
        print('One-vs-rest', '-' * 35,
              'Accuracy in Train Group   : {:.2f}'.format(train_accuracy),
              'Accuracy in Test  Group   : {:.2f}'.format(test_accuracy), sep='\n')
        predictions = lr.predict(X_test)
        score = round(accuracy_score(y_test, predictions), 3)
        cm1 = cm(y_test, predictions)
        sns.heatmap(cm1, annot=True, fmt=".0f")
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Accuracy Score: {0}'.format(score), size=15)
        plt.show()

        pred_test = lr.predict(X_test)
        pred_train = lr.predict(X_train)

        quality_pred = LogisticRegression(random_state=40)
        quality_pred.fit(X_train, y_train)

        confusion_matrix_train = cm(y_train, pred_train)
        confusion_matrix_test = cm(y_test, pred_test)

        TN = confusion_matrix_test[0][0]
        TP = confusion_matrix_test[1][1]
        FP = confusion_matrix_test[0][1]
        FN = confusion_matrix_test[1][0]

        print("(Total) True Negative       :", TN)
        print("(Total) True Positive       :", TP)
        print("(Total) Negative Positive   :", FP)
        print("(Total) Negative Negative   :", FN)

        print("Accuracy Score of Our Model     : ",  quality_pred.score(X_test, y_test))
        Error_Rate = 1 - (accuracy_score(y_test, pred_test))
        print("Error rate: ", Error_Rate)


if __name__ == "__main__":
    db = Dataset()
    db.get_data()
    db.prepare_data()
    db.graphs()
    db.data_description()
    db.correlation_color()
    db.correlation_quality()
    db.quality_alcohol_relation()
    db.quality_acidity_relation()
    db.type_alcohol_quality_acidity()
    db.classifier()
