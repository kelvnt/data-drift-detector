import copy
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from category_encoders import CountEncoder

from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, ks_2samp, chisquare, wasserstein_distance

from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_absolute_percentage_error, 
    explained_variance_score, precision_score, recall_score, accuracy_score,
    f1_score, roc_auc_score, confusion_matrix
    )


import logging
logger = logging.getLogger()


class DataDriftDetector:
    """Compare differences between 2 datasets
    DataDriftDetector creates useful methods to compare 2 datasets,
    created to allow ease of measuring the fidelity between 2 datasets.
    
    Methods
    ----
    calculate_drift:
        Calculates the distribution distance for each column between the
        datasets
    plot_numeric_to_numeric:
        Creates a pairplot between the 2 datasets
    plot_categorical_to_numeric:
        Creates a pairgrid violin plot between the 2 datasets
    plot_categorical:
        Creates a proportion histogram between the 2 datasets for categorical
        columns
    compare_ml_efficacy:
        Compares the ML efficacy of a model built between the 2 datasets
    Args
    ----
    df_prior: <pandas.DataFrame>
        Pandas dataframe of the prior dataset. In practice, this would be the
        dataset used to train a live model
    df_post: <pandas.DataFrame>
        Pandas dataframe of the post dataset. In practice, this would be the
        current dataset that's flowing into a live model
    categorical_columns: <list of str>
        A list of categorical columns in the dataset, will be determined by
        column types if not provided
    numeric_columns: <list of str>
        A list of numeric columns in the dataset, will be determined by
        column types if not provided
    """
    def __init__(
            self,
            df_prior,
            df_post,
            categorical_columns=None,
            numeric_columns=None
            ):
        assert isinstance(df_prior, pd.DataFrame),\
            "df_prior should be a pandas dataframe"
        assert isinstance(df_post, pd.DataFrame),\
            "df_post should be a pandas dataframe"
        assert sorted(df_prior.columns) == sorted(df_post.columns),\
            "df_prior and df_post should have the same column names"
        assert all(df_prior.dtypes.sort_index() == df_post.dtypes.sort_index()),\
            "df_prior and df_post should have the same column types"
        assert isinstance(categorical_columns, (list, type(None))),\
            "categorical_columns should be of type list"
        assert isinstance(numeric_columns, (list, type(None))),\
            "numeric_columns should be of type list"

        df_prior_ = df_prior.copy()
        df_post_ = df_post.copy()

        if categorical_columns is None:
            categorical_columns = [
                c for c in df_prior_.columns if df_prior_.dtypes[c] == 'object'
            ]
            logger.info(f"Identified categorical column(s): {categorical_columns}")

        df_prior_[categorical_columns] = df_prior_[categorical_columns].astype(str)
        df_post_[categorical_columns] = df_post_[categorical_columns].astype(str)

        if numeric_columns is None:
            num_types = ['float64','float32','int32','int64','uint8']
            numeric_columns = [
                c for c in df_prior_.columns if df_prior_.dtypes[c] in num_types
            ]
            logger.info(f"Identified numeric column(s): {numeric_columns}")

        df_prior_[numeric_columns] = df_prior_[numeric_columns].astype(float)
        df_post_[numeric_columns] = df_post_[numeric_columns].astype(float)

        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

        self.df_prior = df_prior_
        self.df_post = df_post_[df_prior_.columns]


    def calculate_drift(self, steps=100):
        """Calculates metrics and test of similarity between the 2 datasets
        For categorical columns, the probability of each category will be
        computed separately for `df_prior` and `df_post`, and the distance 
        between the 2 probability arrays will be computed. For
        numeric columns, the values will first be fitted into a gaussian KDE
        separately for `df_prior` and `df_post`, and a probability array
        will be sampled from them
        
        Args
        ----
        steps: int
            Number of steps to take to sample for the fitted KDE for numeric
            columns

        Returns
        ----
        Dictionary of results
        """
        cat_res = {}
        num_res = {}

        for col in self.categorical_columns:
            # to ensure similar order, concat before computing probability
            col_prior = self.df_prior[col].to_frame()
            col_post = self.df_post[col].to_frame()
            col_prior['_source'] = 'prior'
            col_post['_source'] = 'post'

            col_ = pd.concat([col_prior, col_post], ignore_index=True)

            # aggregate and convert to probability array
            arr = (
                col_.groupby([col, '_source'])
                    .size()
                    .to_frame()
                    .reset_index()
                    .pivot(index=col, columns='_source')
                    .droplevel(0, axis=1)
                    .pipe(lambda df: df.loc[df.sum(axis=1).sort_values(ascending=False).index, :])
                )

            arr_ = arr.div(arr.sum(axis=0),axis=1)
            arr_.fillna(0, inplace=True)

            # calculate statistical distances
            kl_post_prior = sum(
                rel_entr(arr_['post'].to_numpy(), arr_['prior'].to_numpy())
            )
            kl_prior_post = sum(
                rel_entr(arr_['prior'].to_numpy(), arr_['post'].to_numpy())
            )

            jsd = jensenshannon(arr_['prior'].to_numpy(), arr_['post'].to_numpy())
            wd = wasserstein_distance(arr_['prior'].to_numpy(), arr_['post'].to_numpy())

            arr['post'] = arr['post'] / arr['post'].sum() * arr['prior'].sum()
            arr.fillna(0, inplace=True)
            
            # calculate test of similarity
            cs_test = chisquare(arr['post'].to_numpy(), arr['prior'].to_numpy())

            cat_res.update({
                col: {
                    'chi_square_test_statistic': cs_test[0],
                    'chi_square_test_p_value': cs_test[1],
                    'kl_divergence_post_given_prior': kl_post_prior,
                    'kl_divergence_prior_given_post': kl_prior_post,
                    'jensen_shannon_distance': jsd,
                    'wasserstein_distance': wd
                }
            })

        for col in self.numeric_columns:
            # fit gaussian_kde
            col_prior = self.df_prior[col].dropna()
            col_post = self.df_post[col].dropna()
            kde_prior = gaussian_kde(col_prior)
            kde_post = gaussian_kde(col_post)

            # get range of values
            min_ = min(col_prior.min(), col_post.min())
            max_ = max(col_prior.max(), col_post.max())
            range_ = np.linspace(start=min_, stop=max_, num=steps)

            # sample range from KDE
            arr_prior_ = kde_prior.evaluate(range_)
            arr_post_ = kde_post.evaluate(range_)

            arr_prior = arr_prior_ / np.sum(arr_prior_)
            arr_post = arr_post_ / np.sum(arr_post_)

            # calculate statistical distances
            jsd = jensenshannon(arr_prior, arr_post)
            wd = wasserstein_distance(arr_prior, arr_post)
            
            # calculate test of similarity
            ks_test = ks_2samp(arr_prior, arr_post)

            num_res.update({
                col: {
                    'ks_2sample_test_statistic': ks_test[0],
                    'ks_2sample_test_p_value': ks_test[1],
                    'jensen_shannon_distance': jsd,
                    'wasserstein_distance': wd
                }
            })

        return {
            'categorical': dict(cat_res),
            'numerical': dict(num_res)
            }


    def plot_categorical_to_numeric(
            self,
            plot_categorical_columns=None,
            plot_numeric_columns=None,
            categorical_on_y_axis=True,
            grid_kws={'height':5},
            plot_kws={}
            ):
        """Plots charts to compare categorical to numeric columns pairwise.
        Plots a pairgrid violin plot of categorical columns to numeric
        columns, split and colored by the source of datasets
        
        Args
        ----
        plot_categorical_columns: <list of str>
            List of categorical columns to plot, uses all if no specified
        plot_numeric_columns: <list of str>
            List of numeric columns to plot, uses all if not specified
        categorical_on_y_axis: <boolean>
            Determines layout of resulting image - if True, categorical
            columns will be on the y axis
        grid_kws: <dict>
            arguments to pass into the pair grid plot
        plot_kws: <dict>
            Arguments to pass into the violin plot
        Returns
        ----
        Resulting plot
        """
        assert isinstance(plot_categorical_columns, (list, type(None))),\
            "plot_categorical_columns should be of type list"
        assert isinstance(plot_numeric_columns, (list, type(None))),\
            "plot_numeric_columns should be of type list"
        assert isinstance(categorical_on_y_axis, bool),\
            "categorical_on_y_axis should be a boolean value"

        df_prior = self.df_prior.copy()
        df_post = self.df_post.copy()

        col_nunique = df_prior.nunique()

        if plot_categorical_columns is None:
            plot_categorical_columns = [
                col for col in col_nunique.index if
                (col_nunique[col] <= 20) & (col in self.categorical_columns)
            ]

        if plot_numeric_columns is None:
            plot_numeric_columns = self.numeric_columns

        df_prior["_source"] = "Prior"
        df_post["_source"] = "Post"

        plot_df = pd.concat([df_prior, df_post])
        
        msg = (
            "Plotting the following categorical column(s): " +
            ", ".join(plot_categorical_columns) +
            "\nAgainst the following numeric column(s):" +
            ", ".join(plot_numeric_columns) +
            "\nCategorical columns with high cardinality (>20 unique values)" +
            " are not plotted."
        )

        logger.info(msg)

        # violinplot does not treat numeric string cols as string - error
        # sln: added a whitespace to ensure it is read as a string
        plot_df[plot_categorical_columns] = (
            plot_df[plot_categorical_columns].astype(str) + " "
        )

        if categorical_on_y_axis:
            y_cols = plot_categorical_columns
            x_cols = plot_numeric_columns
        else:
            y_cols = plot_numeric_columns
            x_cols = plot_categorical_columns

        g = sns.PairGrid(
            data=plot_df,
            x_vars=x_cols,
            y_vars=y_cols,
            hue='_source',
            hue_kws={'split':True},
            **grid_kws
        )

        g.map(
            sns.violinplot,
            hue=plot_df['_source'],
            split=True,
            **plot_kws
        )

        g.add_legend()

        return g


    def plot_numeric_to_numeric(
            self,
            kind='scatter',
            diag_kind='kde',
            plot_kws=None,
            grid_kws=None,
            diag_kws={'common_norm':False},
            plot_numeric_columns=None,
            **kwargs
            ):
        """Plots charts to compare numeric columns pairwise.
        Plots a pairplot (from seaborn) of numeric columns, with a distribution
        plot on the diagonal and a scatter plot for all other charts
        Args
        ----
        plot_numeric_columns: <list of str>
            List of numeric columns to plot, uses all if not specified
        kind: <str>
            Plot kind for the pair plot
        diag_kind: <str>
            Plot kind for the diagonal plots
        plot_kws: <dict>
            Arguments for both the diagonal and grid plots
        grid_kws: <dict>
            Arguments for the grid plots
        diag_kws: <dict>
            Arguments for the diagonal plots
        Returns
        ----
        Resulting plot
        """
        assert isinstance(plot_numeric_columns, (list, type(None))),\
            "plot_numeric_columns should be of type list"

        if plot_numeric_columns is None:
            plot_numeric_columns = self.numeric_columns

        df_prior = self.df_prior[plot_numeric_columns].copy()
        df_post = self.df_post[plot_numeric_columns].copy()

        df_prior['_source'] = "Prior"
        df_post['_source'] = "Post"

        plot_df = pd.concat([df_prior, df_post])
        plot_df.reset_index(drop=True, inplace=True)

        logger.info(f"Plotting the following numeric column(s): {plot_numeric_columns}")

        g = sns.pairplot(
            data=plot_df,
            kind=kind,
            diag_kind=diag_kind,
            hue='_source',
            plot_kws=plot_kws,
            diag_kws=diag_kws,
            grid_kws=grid_kws,
            **kwargs
        )

        return g


    def plot_categorical(self, plot_categorical_columns=None, **kwargs):
        """Plot histograms to compare categorical columns
        Args
        ----
        plot_categorical_columns: <list of str>
            List of categorical columns to plot, uses all if no specified
        Returns
        ----
        Resulting plot
        """
        assert isinstance(plot_categorical_columns, (list, type(None))),\
            "plot_categorical_columns should be of type list"

        col_nunique = self.df_prior.nunique()
        if plot_categorical_columns is None:
            plot_categorical_columns = [
                col for col in col_nunique.index if
                (col_nunique[col] <= 20) & (col in self.categorical_columns)
            ]

        logger.info(
            f"Plotting the following categorical column(s): {plot_categorical_columns}"
        )

        fig, ax = plt.subplots(len(plot_categorical_columns), 1,
                               figsize=(10, 5*len(plot_categorical_columns)))
        
        for i, col in enumerate(plot_categorical_columns):
            if len(plot_categorical_columns) == 1:
                _ax = ax
            elif len(plot_categorical_columns) > 1:
                _ax = ax[i]

            _p1 = (self.df_prior[col].value_counts(normalize=True)
                                     .rename("Proportion")
                                     .sort_index()
                                     .reset_index())
            _p2 = (self.df_post[col].value_counts(normalize=True)
                                    .rename("Proportion")
                                    .sort_index()
                                    .reset_index())
            _p1['_source'] = 'Prior'
            _p2['_source'] = 'Post'
            _p = pd.concat([_p1, _p2])

            sns.barplot(
                x=_p.index,
                y="Proportion",
                hue="_source",
                data=_p,
                ax=_ax,
                **kwargs
            )
            _ax.legend(loc='upper right', title='_source')
            _ax.set_xlabel(col)
            _ax.tick_params(axis='x', labelrotation=90)

        plt.tight_layout()
        plt.close(fig)

        return fig


    def _rmse(self, targets, predictions):
        return np.sqrt(np.mean((predictions-targets)**2))


    def compare_ml_efficacy(
            self,
            target_column,
            test_data=None,
            OHE_columns=None,
            high_cardinality_columns=None,
            OHE_columns_cutoff=5,
            train_size=0.7,
            model_prior=None,
            model_post=None
            ):
        """Compares the ML efficacy of the prior data to the post data
        For a given `target_column`, this builds a ML model separately with
        `df_prior` and `df_post`, and compares the performance
        between the 2 models on a test dataset. Test data will be drawn
        from `df_post` if it is not provided.
        Args
        ----
        target_column: <str>
            Target column to be used for ML
        test_data: <pandas.DataFrame>
            Pandas dataframe of test data, to do a train test split on the
            df_post if not provided
        OHE_columns: <list of str>
            List of columns to be one hot encoded, will be determined
            if not provided
        high_cardinality_columns: <list of str>
            List of columns to be count-encoded, will be
            determined if not provided
        OHE_columns_cutoff: <int>
            Number of unique labels in a column to determine OHE_columns &
            high_cardinality_columns if not provided.
        train_size: <float>
            Proportion to split the df_post by, if test_data is not provided
        model_prior: <class>
            A model with minimally a `fit` and `predict` methods used to train the prior
            dataset. If none is provided, a default model will be used
        model_post: <class>
            A model with minimally a `fit` and `predict` methods used to train the post
            dataset. If none is provided, a default model will be used
        Returns
        ----
        Returns a report of ML metrics between the prior model and the
        post model
        """
        assert isinstance(target_column, str),\
            "target_column should be of type string"
        assert target_column in self.df_prior.columns,\
            "target_column does not exist in df_prior"
        assert isinstance(test_data, (pd.DataFrame, type(None))),\
            "test_data should be a pandas dataframe"
        assert isinstance(OHE_columns, (list, type(None))),\
            "OHE_columns should be of type list"
        assert isinstance(high_cardinality_columns, (list, type(None))),\
            "high_cardinality_columns should be of type list"

        self.target_column = target_column
        self.train_size = train_size
        self.model_prior = model_prior
        self.model_post = model_post

        col_nunique = self.df_prior.nunique()

        if OHE_columns is None:
            OHE_columns = [col for col in col_nunique.index if
                        (col_nunique[col] <= OHE_columns_cutoff) &
                        (col in self.categorical_columns)]

        if high_cardinality_columns is None:
            high_cardinality_columns = [col for col in col_nunique.index if
                                     (col_nunique[col] > OHE_columns_cutoff) &
                                     (col in self.categorical_columns)]

        self.OHE_columns = OHE_columns
        self.high_cardinality_columns = high_cardinality_columns

        test_data_ = copy.deepcopy(test_data)

        if test_data_ is not None:
            test_data_[self.numeric_columns] = (
                test_data_[self.numeric_columns].astype(float)
            )
            test_data_[self.categorical_columns] = (
                test_data_[self.categorical_columns].astype(str)
            )

        self.test_data = test_data_

        self._ml_data_prep()

        if target_column in self.categorical_columns:
            # set default model if not provided
            if self.model_prior is None:
                self.model_prior = self._default_classifier()
            if self.model_post is None:
                self.model_post = self._default_classifier()
            # actually fit and evaluate model
            self._fit_model()
            self._eval_classifier()

        elif target_column in self.numeric_columns:
            # set default model if not provided
            if self.model_prior is None:
                self.model_prior = self._default_regressor()
            if self.model_post is None:
                self.model_post = self._default_regressor()
            # actually fit and evaluate model
            self._fit_model()
            self._eval_regressor()

        return self.ml_report
    

    def _default_regressor(self):
        """Default regression model to be used if not specified by user
        """
        return RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_distributions={
                "n_estimators": [100, 200],
                "max_samples": [0.6, 0.8, 1],
                "max_depth": [3, 4, 5]
            },
            n_iter=10,
            cv=10,
            random_state=42
        )
    

    def _default_classifier(self):
        """Default classification model to be used if not specified by user
        """
        return RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions={
                "n_estimators": [100, 200],
                "max_samples": [0.6, 0.8, 1],
                "max_depth": [3, 4, 5]
            },
            n_iter=10,
            cv=10,
            random_state=42
        )


    def _ml_data_prep(self):
        """Prepares datasets for ML. This does one hot encoding, count encoding
        and train test split (if necessary)
        """

        df_post = self.df_post.copy()
        train_prior = self.df_prior.copy()
        
        # drop NAs
        cols = self.categorical_columns+self.numeric_columns
        df_post = df_post[cols].dropna(how='any')
        train_prior = train_prior[cols].dropna(how='any')

        # create test data if not provided
        if self.test_data is None:

            msg = (
                "No test data was provided. Test data will be created with " +
                "a {}-{} shuffle split from the post data set.".format(
                    str(round(self.train_size*100, 0)),
                    str(round((1-self.train_size)*100, 0)))
            )
            logger.info(msg)

            df_post = shuffle(df_post)
            n_split = int(len(df_post)*self.train_size)

            train_post = df_post.iloc[:n_split].copy()
            test = df_post.iloc[n_split:].copy()

        else:
            test = self.test_data.copy()
            test = test[cols].dropna(how='any')
            train_post = df_post

        # determine columns for OHE & High Cardinality
        OHE_columns = [
            col for col in self.OHE_columns if col != self.target_column
        ]
        high_cardinality_columns = [
            col for col in self.high_cardinality_columns if col != self.target_column
        ]

        # concat and then OHE to ensure columns match
        train_prior['source'] = "Train Prior"
        test['source'] = "Test"
        train_post['source'] = "Train Post"

        df = pd.concat([train_prior, test, train_post])
        if len(OHE_columns) > 0:
            logger.info(f"One hot encoded columns: {OHE_columns}")
            df = pd.get_dummies(data=df, columns=OHE_columns)

        train_prior = df[df.source == 'Train Prior'].drop('source', axis=1)
        test = df[df.source == 'Test'].drop('source', axis=1)
        train_post = df[df.source == 'Train Post'].drop('source', axis=1)

        test_prior = test.copy()
        test_post = test.copy()

        if len(high_cardinality_columns) > 0:
            logger.info(f"High cardinality columns: {high_cardinality_columns}")
            # CountEncoder for high cardinality columns
            tf_prior = CountEncoder(cols=high_cardinality_columns)
            tf_post = CountEncoder(cols=high_cardinality_columns)

            train_prior[high_cardinality_columns] = tf_prior.fit_transform(
                train_prior[high_cardinality_columns],
                train_prior[self.target_column]
            )
            test_prior[high_cardinality_columns] = tf_prior.transform(
                test_prior[high_cardinality_columns],
                test_prior[self.target_column]
            )
            train_post[high_cardinality_columns] = tf_post.fit_transform(
                train_post[high_cardinality_columns],
                train_post[self.target_column]
            )
            test_post[high_cardinality_columns] = tf_post.transform(
                test_post[high_cardinality_columns],
                test_post[self.target_column]
            )

        X_train_prior = train_prior.drop(self.target_column, axis=1).astype(float)
        y_train_prior = train_prior[self.target_column]
        X_test_prior = test_prior.drop(self.target_column, axis=1).astype(float)
        y_test = test[self.target_column]

        X_train_post = train_post.drop(self.target_column, axis=1).astype(float)
        y_train_post = train_post[self.target_column]
        X_test_post = test_post.drop(self.target_column, axis=1).astype(float)

        self.X_train_prior = X_train_prior
        self.y_train_prior = y_train_prior
        self.X_test_prior = X_test_prior
        self.y_test = y_test
        self.X_train_post = X_train_post
        self.y_train_post = y_train_post
        self.X_test_post = X_test_post


    def _fit_model(self):
        """Fits the ML models
        """
        self.model_prior.fit(self.X_train_prior, self.y_train_prior)
        self.model_post.fit(self.X_train_post, self.y_train_post)


    def _eval_regressor(self):
        """Calculates the regression metrics of the prior and post model.
        Returns a pandas dataframe of the results.
        """

        y_pred_prior = self.model_prior.predict(self.X_test_prior)
        y_pred_post = self.model_post.predict(self.X_test_post)

        rmse_prior = self._rmse(self.y_test, y_pred_prior)
        mae_prior = mean_absolute_error(self.y_test, y_pred_prior)
        r2_prior = r2_score(self.y_test, y_pred_prior)
        mape_prior = mean_absolute_percentage_error(self.y_test, y_pred_prior)
        evs_prior = explained_variance_score(self.y_test, y_pred_prior)

        rmse_post = self._rmse(self.y_test, y_pred_post)
        mae_post = mean_absolute_error(self.y_test, y_pred_post)
        r2_post = r2_score(self.y_test, y_pred_post)
        mape_post = mean_absolute_percentage_error(self.y_test, y_pred_post)
        evs_post = explained_variance_score(self.y_test, y_pred_post)

        res = pd.DataFrame({
            'root_mean_squared_error': [rmse_prior, rmse_post],
            'mean_absolute_error': [mae_prior, mae_post],
            'mean_absolute_percentage_error': [mape_prior, mape_post],
            'explained_variance_score': [evs_prior, evs_post],
            'r2_score': [r2_prior, r2_post]
            },
            index=['Prior', 'Post']
        )

        self.ml_report = res


    def _eval_classifier(self):
        """Calculates the classification metrics of the prior and post model.
        Returns a pandas dataframe of the result.
        """

        y_pred_prior_arr = self.model_prior.predict(self.X_test_prior)
        y_pred_post_arr = self.model_post.predict(self.X_test_post)

        y_test = pd.DataFrame(self.y_test)
        y_pred_prior = pd.DataFrame(y_pred_prior_arr, columns=y_test.columns)
        y_pred_post = pd.DataFrame(y_pred_post_arr, columns=y_test.columns)

        y_pred_prior['source'] = "prior"
        y_pred_post['source'] = "post"
        y_test['source'] = "test"

        y_ = pd.concat([y_pred_prior, y_pred_post, y_test])
        cols = [col for col in y_.columns if col != "source"]
        y_ = pd.get_dummies(y_, columns=cols)

        y_pred_prior = y_[y_.source == 'prior'].drop('source', axis=1).values
        y_pred_post = y_[y_.source == 'post'].drop('source', axis=1).values
        y_test = y_[y_.source == 'test'].drop('source', axis=1).values

        y_.drop('source', axis=1, inplace=True)
        class_labels = y_.columns

        res = pd.DataFrame([])

        if len(y_test[0]) == 2:
            # for binary classification
            # only take position 1, assuming position 1 is the true label
            iters = [1]
        elif len(y_test[0]) > 2:
            # for multiclass classification
            iters = range(len(y_test[0]))

        for i in iters:
            N = np.count_nonzero(y_test[:,i])

            precision_prior = precision_score(y_test[:,i], y_pred_prior[:,i])
            recall_prior = recall_score(y_test[:,i], y_pred_prior[:,i])
            acc_prior = accuracy_score(y_test[:,i], y_pred_prior[:,i])
            f1_prior = f1_score(y_test[:,i], y_pred_prior[:,i])
            conf_mat_prior = confusion_matrix(y_test[:,i], y_pred_prior[:,i])
            tn_pr, fp_pr, fn_pr, tp_pr = conf_mat_prior.ravel()
            try:
                auc_prior = roc_auc_score(y_test[:,i], y_pred_prior[:,i])
            except ValueError:
                auc_prior = "NA"

            precision_post = precision_score(y_test[:,i], y_pred_post[:,i])
            recall_post = recall_score(y_test[:,i], y_pred_post[:,i])
            acc_post = accuracy_score(y_test[:,i], y_pred_post[:,i])
            f1_post = f1_score(y_test[:,i], y_pred_post[:,i])
            conf_mat_post = confusion_matrix(y_test[:,i], y_pred_post[:,i])
            tn_pt, fp_pt, fn_pt, tp_pt = conf_mat_post.ravel()
            try:
                auc_post = roc_auc_score(y_test[:,i], y_pred_post[:,i])
            except ValueError:
                auc_post = "NA"

            multiindex = [
                (str(class_labels[i]), 'Prior'), (str(class_labels[i]), 'Post')
            ]

            index = pd.MultiIndex.from_tuples(
                multiindex,
                names=['Class', 'Data Type']
            )

            score = pd.DataFrame({
                'accuracy': [acc_prior, acc_post],
                'precision': [precision_prior, precision_post],
                'recall': [recall_prior, recall_post],
                'f1_score': [f1_prior, f1_post],
                'roc_auc_score': [auc_prior, auc_post],
                'TN': [tn_pr, tn_pt],
                'FP': [fp_pr, fp_pt],
                'FN': [fn_pr, fn_pt],
                'TP': [tp_pr, tp_pt],
                'N': [N, N]
                },
                index=index
            )

            res = pd.concat([res, score])

        # if multiclass, calculate macro, micro & weight averaging of scores
        if len(y_test[0]) > 2:
            N = len(y_test)
            
            # macro average
            ind = "_MACRO_AVERAGE"
            index = pd.MultiIndex.from_tuples(
                [(ind, 'Prior'), (ind, 'Post')],
                names=['Class', 'Data Type']
            )
            prior_avg = res[res.index.get_level_values(1) == "Prior"].mean()
            post_avg = res[res.index.get_level_values(1) == "Post"].mean()
            prior_sum = res[res.index.get_level_values(1) == "Prior"].sum()
            post_sum = res[res.index.get_level_values(1) == "Post"].sum()
            macro_roc_auc_ovr_prior = roc_auc_score(
                y_test,
                y_pred_prior,
                multi_class="ovr",
                average="macro",
            )
            macro_roc_auc_ovr_post = roc_auc_score(
                y_test,
                y_pred_post,
                multi_class="ovr",
                average="macro",
            )

            macro_avg_res = pd.DataFrame({
                'accuracy': [prior_avg["accuracy"], post_avg["accuracy"]],
                'precision': [prior_avg["precision"], post_avg["precision"]],
                'recall': [prior_avg["recall"], post_avg["recall"]],
                'f1_score': [prior_avg["f1_score"], post_avg["f1_score"]],
                'roc_auc_score': [macro_roc_auc_ovr_prior, macro_roc_auc_ovr_post],
                'TN': [prior_sum["TN"], post_sum["TN"]],
                'FP': [prior_sum["FP"], post_sum["FP"]],
                'FN': [prior_sum["FN"], post_sum["FN"]],
                'TP': [prior_sum["TP"], post_sum["TP"]],
                'N': [N, N]
                },
                index=index
            )

            # micro average
            ind = "_MICRO_AVERAGE"
            index = pd.MultiIndex.from_tuples(
                [(ind, 'Prior'), (ind, 'Post')],
                names=['Class', 'Data Type']
            )
            recall_prior = prior_sum["TP"] / (prior_sum["TP"] + prior_sum["FN"])
            recall_post = post_sum["TP"] / (post_sum["TP"] + post_sum["FN"])
            precision_prior = prior_sum["TP"] / (prior_sum["TP"] + prior_sum["FP"])
            precision_post = post_sum["TP"] / (post_sum["TP"] + post_sum["FP"])
            f1_prior = (
                (2 * precision_prior * recall_prior) / (precision_prior + recall_prior)
            )
            f1_post = (
                (2 * precision_post * recall_post) / (precision_post + recall_post)
            )
            micro_roc_auc_ovr_prior = roc_auc_score(
                y_test,
                y_pred_prior,
                multi_class="ovr",
                average="micro",
            )
            micro_roc_auc_ovr_post = roc_auc_score(
                y_test,
                y_pred_post,
                multi_class="ovr",
                average="micro",
            )

            micro_avg_res = pd.DataFrame({
                'accuracy': [recall_prior, recall_post],
                'precision': [precision_prior, precision_post],
                'recall': [recall_prior, recall_post],
                'f1_score': [f1_prior, f1_post],
                'roc_auc_score': [micro_roc_auc_ovr_prior, micro_roc_auc_ovr_post],
                'TN': [prior_sum["TN"], post_sum["TN"]],
                'FP': [prior_sum["FP"], post_sum["FP"]],
                'FN': [prior_sum["FN"], post_sum["FN"]],
                'TP': [prior_sum["TP"], post_sum["TP"]],
                'N': [N, N]
                },
                index=index
            )

            # weighted average
            ind = "_WEIGHTED_AVERAGE"
            index = pd.MultiIndex.from_tuples(
                [(ind, 'Prior'), (ind, 'Post')],
                names=['Class', 'Data Type']
            )
            cols = [c for c in res.columns if c != "N"]
            _res = res[cols].multiply(res["N"], axis=0)

            prior_avg = _res[_res.index.get_level_values(1) == "Prior"].sum() / N
            post_avg = _res[_res.index.get_level_values(1) == "Post"].sum() / N

            weighted_avg_res = pd.DataFrame({
                'accuracy': [prior_avg["accuracy"], post_avg["accuracy"]],
                'precision': [prior_avg["precision"], post_avg["precision"]],
                'recall': [prior_avg["recall"], post_avg["recall"]],
                'f1_score': [prior_avg["f1_score"], post_avg["f1_score"]],
                'roc_auc_score': [prior_avg["roc_auc_score"], post_avg["roc_auc_score"]],
                'TN': [prior_sum["TN"], post_sum["TN"]],
                'FP': [prior_sum["FP"], post_sum["FP"]],
                'FN': [prior_sum["FN"], post_sum["FN"]],
                'TP': [prior_sum["TP"], post_sum["TP"]],
                'N': [N, N]
                },
                index=index
            )
            res = pd.concat([res, micro_avg_res, macro_avg_res, weighted_avg_res])

        self.ml_report = res