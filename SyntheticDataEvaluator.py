import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import CatBoostEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, precision_score
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.utils import shuffle


class SyntheticDataEvaluator:
    """Compare differences between 2 datasets
    
    SyntheticDataEvaluator creates useful methods to compare 2 datasets, created
    to allow ease of estimating the fidelity of a synthetic dataset to its
    corresponding real dataset. This may also be used to generically explore
    differences between 2 datasets, eg. for data drift detection.
    
    Methods
    -------
    plot_numeric_to_numeric:
        creates a pairplot between the synthetic and real dataset
        
    plot_categorical_to_numeric: 
        creates a pairgrid violin plot between the synthetic and real dataset
        
    compare_ml_efficacy:
        compares the ML efficacy of a model built with the real dataset 
        vs that of the synthetic dataset

    Args
    ----
    real_data:
        pandas dataframe of the real dataset

    synth_data:
        pandas dataframe of the synthetic dataset

    categorical_cols:
        a list of categorical columns in the dataset, will be determined by
        column types if not provided

    numeric_cols:
        a list of numerical columns in the dataset, will be determined by
        column types if not provided

    """
    
    def __init__(self,
                 real_data,
                 synth_data,
                 categorical_cols=None,
                 numeric_cols=None):
        
        real_data_ = real_data.copy()
        synth_data_ = synth_data.copy()
        
        if categorical_cols is None:
            categorical_cols = [i for i in real_data_.columns if
                                real_data_.dtypes[i]=='object']
        
        print("Categorical column(s): ", categorical_cols)
        
        real_data_[categorical_cols] = real_data_[categorical_cols].astype(str)
        synth_data_[categorical_cols] = synth_data_[categorical_cols].astype(str)
            
        if numeric_cols is None:
            numeric_cols = [i for i in real_data_.columns if
                            real_data_.dtypes[i] in ['float64','float32',
                                                    'int32','int64','uint8']]
        
        print("\nNumerical column(s): ", numeric_cols)  
        
        real_data_[numeric_cols] = real_data_[numeric_cols].astype(float)
        synth_data_[numeric_cols] = synth_data_[numeric_cols].astype(float)
            
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        
        self.real_data = real_data_
        self.synth_data = synth_data_
        
        
    def plot_categorical_to_numeric(self,
                                    plot_categorical_cols=None,
                                    plot_numeric_cols=None,
                                    categorical_on_y_axis=True,
                                    height=4,
                                    output_path=None):
        """Plots charts to compare categorical to numerical columns pairwise.
        
        Plots a pairgrid violin plot of categorical columns to numerical
        columns, split and colored by the source of dataset (real vs 
        synthetic).
        
        Args
        ----
        plot_categorical_cols:
            list of categorical columns to plot, uses all if no specified
            
        plot_numeric_cols: 
            list of numerical columns to plot, uses all if not specified
            
        categorical_on_y_axis:
            determines layout of resulting image - if True, categorical
            columns will be on the y axis
            
        height:
            height (in inches) of each facet
            
        output_path:
            file path to save the resulting image (optional)
        
        Returns
        -------
        Resulting plot
        """
        
        real_data = self.real_data.copy()
        synth_data = self.synth_data.copy()
        
        col_nunique = real_data.nunique()
        
        if plot_categorical_cols is None:
            plot_categorical_cols = [col for col in col_nunique.index if
                            (col_nunique[col] <= 20) &
                                     (col in self.categorical_cols)]
            
        if plot_numeric_cols is None:
            plot_numeric_cols = self.numeric_cols
        
        real_data["source"] = "Real"
        synth_data["source"] = "Synthesized"
        
        plot_df = pd.concat([real_data, synth_data])
        
        print("Plotting the following categorical column(s):\n",
              plot_categorical_cols)
        print("\nAgainst the following numerical column(s):\n",
              plot_numeric_cols)
        print("\nCategorical columns with high cardinality (>20 unique values) are not plotted.\n")
        
        # violinplot does not treat numeric string cols as string - error
        # sln: added a `_` to ensure it is read as a string
        plot_df[plot_categorical_cols] = plot_df[plot_categorical_cols].astype(str) + "_"
        
        if categorical_on_y_axis:
            y_cols = plot_categorical_cols
            x_cols = plot_numeric_cols
        else:
            y_cols = plot_numeric_cols
            x_cols = plot_categorical_cols

        g = sns.PairGrid(data=plot_df,
                         x_vars=x_cols, 
                         y_vars=y_cols,
                         height=height)
        
        g.map(sns.violinplot,
              hue=plot_df["source"], 
              split=True,
              palette="muted",
              bw=0.1)
        
        plt.legend()
        
        if output_path is not None:
            plt.savefig(output_path)
        
        return g

    
    def plot_numeric_to_numeric(self,
                                plot_numeric_cols=None,
                                alpha=1,
                                output_path=None):
        """Plots charts to compare numerical columns pairwise.
        
        Plots a pairplot (from seaborn) of numeric columns, with a distribution
        plot on the diagonal and a scatter plot for all other charts
        
        Args
        ----
        plot_numeric_cols:
            list of numerical columns to plot, uses all if not specified
            
        alpha:
            transparency of the scatter plot
            
        output_path:
            file path to save the resulting image
        
        Returns
        -------
        Resulting plot
        """
        
        if plot_numeric_cols is None:
            plot_numeric_cols = self.numeric_cols
            
        real_data = self.real_data[plot_numeric_cols].copy()
        synth_data = self.synth_data[plot_numeric_cols].copy()
        
        real_data['source'] = "Real"
        synth_data['source'] = "Synthesized"
        
        plot_df = pd.concat([real_data, synth_data])
        
        print("Plotting the following numeric column(s):\n",
              plot_numeric_cols, "\n")
        
        g = sns.pairplot(data=plot_df,
                         hue='source',
                         plot_kws={'alpha': alpha})
        
        plt.legend()
        
        if output_path is not None:
            plt.savefig(output_path)
        
        return g


    # WIP
    # def plot_categorical_to_categorical(self):
        
    
    def rmse(self, targets, predictions):
        return np.sqrt(np.mean((predictions-targets)**2))
    
    
    def compare_ml_efficacy(self, 
                            target_col,
                            test_data=None,
                            ohe_cols=None,
                            high_cardinality_cols=None,
                            random_state=None,
                            train_size=0.7,
                            cv=3,
                            n_iter=5,
                            param_grid={'n_estimators': [100, 200],
                                        'max_samples': [0.6, 0.8, 1],
                                        'max_depth': [3, 4, 5]}):
        """Compares the ML efficacy of the synthetic data to the real data
        
        For a given `target_col`, this builds a ML model separately with the 
        real dataset and the synthetic dataset, and compares the performance
        between the 2 models on a test dataset.
        
        Args
        ----
        target_col:
            target column to be used for ML
            
        test_data:
            pandas dataframe of test data, to do a train test split on the
            real_data if not provided
            
        ohe_cols: 
            list of columns to be one hot encoded, will be determined 
            if not provided
            
        high_cardinality_cols:
            list of columns to be cat boost encoded, will be
            determined if not provided
            
        random_state:
            random state for the RandomizedSearchCV & the model fitting
            
        train_size:
            proportion to split the real_data by, if test_data is not provided
            
        cv: 
            number of cross validation folds to be used in the 
            RandomizedSearchCV
            
        n_iter: 
            number of interations for the RandomizedSearchCV
            
        param_grid: 
            dictionary of hyperparameter values to be iterated by
            the RandomizedSearchCV
                               
        Returns
        -------
        Returns a report of ML metrics between the real model and the
        synthetic model
        """
        
        # TODO: - Allow choice of model?
        #       - Allow choice of encoding for high cardinality cols?
        
        self.target_col = target_col
        self.train_size = train_size
        self.random_state = random_state
        self.cv = cv
        self.n_iter = n_iter
        self.param_grid = param_grid
        
        col_nunique = self.real_data.nunique()
        
        if ohe_cols is None:
            ohe_cols = [col for col in col_nunique.index if
                        (col_nunique[col] <= 20) &
                        (col in self.categorical_cols)]
            
        if high_cardinality_cols is None:
            high_cardinality_cols = [col for col in col_nunique.index if
                                     (col_nunique[col] > 20) &
                                     (col in self.categorical_cols)]
        
        self.ohe_cols = ohe_cols
        self.high_cardinality_cols = high_cardinality_cols
        
        test_data_ = test_data.copy()
        
        if test_data_ is not None:
            test_data_[self.numeric_cols] = test_data_[self.numeric_cols].astype(float)
            test_data_[self.categorical_cols] = test_data_[self.categorical_cols].astype(str)
            
        self.test_data = test_data_
        
        self.ml_data_prep()
        
        if target_col in self.categorical_cols:
            self.build_classifier()
            self.eval_classifier()
            
        elif target_col in self.numeric_cols:
            self.build_regressor()
            self.eval_regressor()
            
        print("\nMetric Report \n=============\n\n",
              self.ml_report)
        
        return self.ml_report
        
    
    def ml_data_prep(self):
        """Prepares datasets for ML
        
        This does one hot encoding, cat boost encoding, and train test
        split (if necessary).
        """
        
        synth_data = self.synth_data.copy()
        
        # create test data if not provided
        if self.test_data is None:
            
            msg = "No test data was provided. Test data will be created with "
            msg += "a {}-{} ".format(self.train_size*100,
                                     (1-self.train_size)*100)
            msg += "shuffle split from the real data set."
            
            print(msg)
            
            real_data = self.real_data.copy()
            
            real_data = shuffle(real_data)
            n_split = int(len(real_data)*self.train_size)

            train = real_data.iloc[:n_split]
            test = real_data.iloc[n_split:]
            
        else:
            test = self.test_data.copy()
            train = self.real_data.copy()
        
        # determine columns for OHE & CatBoost
        ohe_cols = [col for col in self.ohe_cols if col != self.target_col]
        high_cardinality_cols = [col for col in self.high_cardinality_cols
                                 if col != self.target_col]
        
        if len(ohe_cols) > 0:
            print("One hot encoded columns: ", ohe_cols)
        if len(high_cardinality_cols) > 0:
            print("\nCat boost encoded columns: ", high_cardinality_cols)
        
        # concat and then OHE to ensure columns match
        train['source'] = "Real Train"
        test['source'] = "Real Test"
        synth_data['source'] = "Synthesized"
        
        df = pd.concat([train, test, synth_data])
        df = pd.get_dummies(data=df, columns=ohe_cols)
        
        train = df[df.source == 'Real Train'].drop('source', axis=1)
        test = df[df.source == 'Real Test'].drop('source', axis=1)
        train_synth = df[df.source == 'Synthesized'].drop('source', axis=1)
        
        # CatBoostEncoder for high cardinality columns
        test_synth = test.copy()
        
        tf_real = CatBoostEncoder(cols=high_cardinality_cols,
                                  random_state=self.random_state)
        tf_synth = CatBoostEncoder(cols=high_cardinality_cols,
                                        random_state=self.random_state)
        
        train[high_cardinality_cols] = tf_real.fit_transform(train[high_cardinality_cols],
                                                              train[self.target_col])
        test[high_cardinality_cols] = tf_real.transform(test[high_cardinality_cols],
                                                        test[self.target_col])
        train_synth[high_cardinality_cols] = tf_synth.fit_transform(train_synth[high_cardinality_cols],
                                                                    train_synth[self.target_col])
        test_synth[high_cardinality_cols] = tf_synth.transform(test_synth[high_cardinality_cols],
                                                            test_synth[self.target_col])
        
        X_train = train.drop(self.target_col, axis=1).astype(float)
        y_train = train[self.target_col].astype(float)
        X_test = test.drop(self.target_col, axis=1).astype(float)
        y_test = test[self.target_col].astype(float)
        
        X_train_synth = train_synth.drop(self.target_col, axis=1).astype(float)
        y_train_synth = train_synth[self.target_col].astype(float)
        X_test_synth = test_synth.drop(self.target_col, axis=1).astype(float)
        y_test_synth = test_synth[self.target_col].astype(float)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_synth = X_train_synth
        self.y_train_synth = y_train_synth
        self.X_test_synth = X_test_synth
        self.y_test_synth = y_test_synth
        
        
    def build_regressor(self):
        """
        Builds a random forest regressor with a RandomizedSearchCV
        """

        model_real_ = RandomForestRegressor(random_state=self.random_state)
        model_synth_ = RandomForestRegressor(random_state=self.random_state)

        model_real = RandomizedSearchCV(model_real_,
                                        self.param_grid,
                                        n_iter=self.n_iter,
                                        cv=self.cv,
                                        random_state=self.random_state)
        model_synth = RandomizedSearchCV(model_synth_,
                                         self.param_grid,
                                         n_iter=self.n_iter,
                                         cv=self.cv,
                                         random_state=self.random_state)

        model_real.fit(self.X_train, self.y_train)
        model_synth.fit(self.X_train_synth, self.y_train_synth)
        
        print("A RandomForestRegressor with a RandomizedSearchCV was trained.")
        print("\nThe final model (trained with real data) parameters are:\n",
              model_real.best_estimator_)
        print("\nThe final model (trained with synthetic data) parameters are:\n",
              model_synth.best_estimator_)
        
        self.model_real = model_real
        self.model_synth = model_synth
    
    
    def build_classifier(self):
        """
        Build a random forest classifier with a RandomizedSearchCV
        """

        model_real_ = RandomForestClassifier(random_state=self.random_state)
        model_synth_ = RandomForestClassifier(random_state=self.random_state)

        model_real = RandomizedSearchCV(model_real_,
                                        self.param_grid,
                                        n_iter=self.n_iter,
                                        cv=self.cv,
                                        random_state=self.random_state)
        model_synth = RandomizedSearchCV(model_synth_,
                                         self.param_grid,
                                         n_iter=self.n_iter,
                                         cv=self.cv,
                                         random_state=self.random_state)

        model_real.fit(self.X_train, self.y_train)
        model_synth.fit(self.X_train_synth, self.y_train_synth)
        
        print("A RandomForestClassifier with a RandomizedSearchCV was trained.")
        print("\nThe final model (trained with real data) parameters are:\n",
              model_real.best_estimator_)
        print("\nThe final model (trained with synthetic data) parameters are:\n",
              model_synth.best_estimator_)
        
        self.model_real = model_real
        self.model_synth = model_synth
        
    
    def eval_regressor(self):
        """
        Calculates the RMSE, MAE & R2 score of the real and synthetic model.
        
        Returns a pandas dataframe of the results.
        """
        
        y_pred_real = self.model_real.predict(self.X_test)
        y_pred_synth = self.model_synth.predict(self.X_test_synth)
        
        rmse_real = self.rmse(self.y_test, y_pred_real)
        mae_real = mean_absolute_error(self.y_test, y_pred_real)
        r2_real = r2_score(self.y_test, y_pred_real)

        rmse_synth = self.rmse(self.y_test_synth, y_pred_synth)
        mae_synth = mean_absolute_error(self.y_test_synth, y_pred_synth)
        r2_synth = r2_score(self.y_test_synth, y_pred_synth)

        res = pd.DataFrame({'RMSE': [rmse_real, rmse_synth],
                            'MAE': [mae_real, mae_synth],
                            'R2': [r2_real, r2_synth]},
                            index=['Real', 'Synthesized'])
        
        self.ml_report = res


    def eval_classifier(self):
        """
        Calculates the accuracy, precision, recall, F1 score & AUC of the
        real and synthetic model.
        
        Returns a pandas dataframe of the result.
        """
        
        y_pred_real = self.model_real.predict(self.X_test)
        y_pred_synth = self.model_synth.predict(self.X_test_synth)
        
        y_test = pd.DataFrame(self.y_test)
        y_pred_real = pd.DataFrame(y_pred_real, columns=y_test.columns)
        y_pred_synth = pd.DataFrame(y_pred_synth, columns=y_test.columns)
        
        y_pred_real['source'] = "real"
        y_pred_synth['source'] = "synth"
        y_test['source'] = "test"
        
        y_ = pd.concat([y_pred_real, y_pred_synth, y_test])
        cols = [col for col in y_.columns if col != "source"]
        y_ = pd.get_dummies(y_, columns=cols)
        
        y_pred_real = y_[y_.source == 'real'].drop('source', axis=1).values
        y_pred_synth = y_[y_.source == 'synth'].drop('source', axis=1).values
        y_test = y_[y_.source == 'test'].drop('source', axis=1).values
        
        y_.drop('source', axis=1, inplace=True)
        class_labels = y_.columns
        
        res = pd.DataFrame([])
        
        if (len(y_test[0]) == 2): 
            # for binary classification
            # only take position 1, assuming position 1 is the true label
            iters = [1] 
        else:
            # for multiclass classification
            iters = range(len(y_test[0]))
        
        for i in iters: 
        
            precision_real = precision_score(y_test[:,i],
                                             y_pred_real[:,i])
            recall_real = recall_score(y_test[:,i],
                                       y_pred_real[:,i])
            acc_real = accuracy_score(y_test[:,i],
                                      y_pred_real[:,i])
            f1_real = f1_score(y_test[:,i],
                               y_pred_real[:,i])
            try:
                auc_real = roc_auc_score(y_test[:,i],
                                         y_pred_real[:,i])
            except ValueError:
                auc_real = "NA"

            precision_synth = precision_score(y_test_synth[:,i],
                                              y_pred_synth[:,i])
            recall_synth = recall_score(y_test_synth[:,i],
                                        y_pred_synth[:,i])
            acc_synth = accuracy_score(y_test_synth[:,i],
                                       y_pred_synth[:,i])
            f1_synth = f1_score(y_test_synth[:,i],
                                y_pred_synth[:,i])
            try:
                auc_synth = roc_auc_score(y_test_synth[:,i],
                                          y_pred_synth[:,i])
            except ValueError:
                auc_synth = "NA"
            
            multiindex = [(str(class_labels[i]), 'Real'),
                         (str(class_labels[i]), 'Synthesized')]
            
            index = pd.MultiIndex.from_tuples(multiindex,
                                              names=['Class', 'Data Type'])
            
            score = pd.DataFrame({'Accuracy': [acc_real,
                                               acc_synth],
                                  'Precision': [precision_real,
                                                precision_synth],
                                  'Recall': [recall_real,
                                             recall_synth],
                                  'F1': [f1_real,
                                         f1_synth],
                                  'AUC': [auc_real,
                                          auc_synth]},
                                  index=index)
            
            res = pd.concat([res, score])
        
        self.ml_report = res
