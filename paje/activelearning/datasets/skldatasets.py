# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from paje.base.data import Data


def iris_dataset(perc_label=0.3):
        dataset = datasets.load_iris()
        unlabeled_features, labeled_features, unlabeled_labels, labeled_labels = train_test_split(dataset.data,
            dataset.target, test_size=perc_label, random_state=42)
        unlabeled_features_0 = pd.DataFrame(unlabeled_features, columns=dataset.feature_names)
        labeled_features_0 = pd.DataFrame(labeled_features, columns=dataset.feature_names)
        unlabeled_labels_0 = pd.DataFrame(unlabeled_labels, columns=["label"])
        labeled_labels_0 = pd.DataFrame(labeled_labels, columns=["label"])

        unlabeled_data = Data(name='IrisUnlabeled', X=unlabeled_features_0.values, 
                      Y=unlabeled_labels_0.values, 
                      columns=list(unlabeled_features_0.columns) + list(unlabeled_labels_0.columns), 
                      history=None)

        labeled_data = Data(name='IrisLabeled', X=labeled_features_0.values, 
                            Y=labeled_labels_0.values, 
                            columns=list(labeled_features_0.columns) + list(labeled_labels_0.columns), 
                            history=None)

        unlabeled_iris = Data(name='IrisUnlabeled', X=unlabeled_features_0.values, 
                            columns=list(unlabeled_features_0.columns), 
                            history=None)

        return unlabeled_data, labeled_data, unlabeled_iris