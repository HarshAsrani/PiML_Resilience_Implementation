import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score, log_loss, brier_score_loss, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import wasserstein_distance, gaussian_kde
from typing import List, Tuple, Optional, Union, Dict
import pandas as pd


class MetricCalculator(ABC):
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class MSEMetric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

class AccuracyMetric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

class AUCMetric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

class F1Metric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

class LogLossMetric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return log_loss(y_true, y_pred)

class BrierMetric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return brier_score_loss(y_true, y_pred)

class MAEMetric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

class R2Metric(MetricCalculator):
    def calculate(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

class SampleSelector(ABC):
    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series, model: BaseEstimator, 
               alpha: float, n_clusters: int, metric: MetricCalculator, immu_feature: str) -> np.ndarray:
        """Abstract Sample selector method that takes the X and y objects, returns the worst samples based on the specific selector

        Args:
            X (pd.DataFrame): Independent variables dataframe, containing all columns. Numerical columns should be scaled and categorical features should be encoded ordinally
            y (pd.Series): Target variable series
            model (BaseEstimator): The model to calculate the residuals
            alpha (float): The ratio of worst samples
            n_clusters (int): Number of Clusters for running k_means
            metric (MetricCalculator): Metric to check for finding out the worst clusters
            immu_feature (str): Immutable Feature

        Returns:
            np.ndarray: The indices of the worst samples based on the method and metric
        """
        pass

    @property
    @abstractmethod
    def requires_alpha(self) -> bool:
        pass

class WorstSampleSelector(SampleSelector):
    def select(self, X, y, task_type, model, alpha, n_clusters, metric, immu_feature):
        """
        Selects worst samples.

        Args:
            X (pd.DataFrame): Independent variables dataframe.
            y (pd.Series): Target variable series.
            task_type (str): Type of task (regression or classification).
            model (BaseEstimator): Model to calculate residuals.
            alpha (float): Ratio of worst samples.
            n_clusters (int): Number of clusters (not used).
            metric (MetricCalculator): Metric calculator (not used).
            immu_feature (str): Immutable feature.

        Returns:
            np.ndarray: Indices of worst samples.
        """
        if task_type == 'regression':
            residuals = np.abs(y.values - model.predict(X.values))
        elif task_type == 'classification':
            proba = model.predict_proba(X.values)
            y = y.values.astype(int)
            residuals = proba[np.arange(len(y)), y]
        if immu_feature is not None:
            bins = np.quantile(X[immu_feature], [0, 0.2, 0.4, 0.6, 0.8, 1])
            bins[0] = bins[0] - 1e-7
            bins[-1] = bins[-1] + 1e-7
            
            selected_indices = []
            for bin_min, bin_max in zip(bins[:-1], bins[1:]):
                bin_mask = (X[immu_feature] >= bin_min) & (X[immu_feature] < bin_max)
                bin_residuals = residuals[bin_mask]
                if task_type == 'regression':
                    bin_indices = np.argsort(bin_residuals)[::-1][:int(len(bin_residuals) * alpha)]
                else:
                    bin_indices = np.argsort(bin_residuals)[:int(len(bin_residuals) * alpha)]
                bin_selected_indices = np.where(bin_mask)[0][bin_indices]
                selected_indices.extend(bin_selected_indices)
            return np.array(selected_indices)
        
        else:
            n_select = int(len(X) * alpha)
            if task_type == 'regression':
                return np.argsort(residuals)[::-1][:n_select]
            else:
                return np.argsort(residuals)[:n_select]
                


    @property
    def requires_alpha(self) -> bool:
        return True

class OuterSampleSelector(SampleSelector):
    def select(self, X, y, task_type, model, alpha, n_clusters, metric, immu_feature):
        """
        Selects outer samples.

        Args:
            X (pd.DataFrame): Independent variables dataframe.
            y (pd.Series): Target variable series (not used).
            task_type (str): Type of task (regression or classification) (not used).
            model (BaseEstimator): Model to calculate residuals (not used).
            alpha (float): Ratio of outer samples.
            n_clusters (int): Number of clusters (not used).
            metric (MetricCalculator): Metric calculator (not used).
            immu_feature (str): Immutable feature.

        Returns:
            np.ndarray: Indices of outer samples.
        """
        midpoint = np.mean(X.values, axis=0)
        distances = np.linalg.norm(X.values - midpoint, axis=1)
        if immu_feature is not None:
            bins = np.quantile(X[immu_feature], [0, 0.2, 0.4, 0.6, 0.8, 1])
            bins[0] = bins[0] - 1e-7
            bins[-1] = bins[-1] + 1e-7
            
            selected_indices = []
            for bin_min, bin_max in zip(bins[:-1], bins[1:]):
                bin_mask = (X[immu_feature] >= bin_min) & (X[immu_feature] < bin_max)
                bin_distances = distances[bin_mask]
                bin_indices = np.argsort(bin_distances)[::-1][:int(len(bin_distances) * alpha)]
                bin_selected_indices = np.where(bin_mask)[0][bin_indices]
                selected_indices.extend(bin_selected_indices)
            return np.array(selected_indices)
        
        else:
            return np.argsort(distances)[::-1][:int(len(X) * alpha)]

    @property
    def requires_alpha(self) -> bool:
        return True

class HardSampleSelector(SampleSelector):
    def select(self, X: pd.DataFrame, y: pd.Series, task_type: str, model: BaseEstimator, 
               alpha: float, n_clusters: int, metric: MetricCalculator, immu_feature: str) -> np.ndarray:
        """
        Selects hard samples.

        Args:
            X (pd.DataFrame): Independent variables dataframe.
            y (pd.Series): Target variable series.
            task_type (str): Type of task (regression or classification).
            model (BaseEstimator): Model to calculate residuals.
            alpha (float): Ratio of hard samples.
            n_clusters (int): Number of clusters (not used).
            metric (MetricCalculator): Metric calculator (not used).
            immu_feature (str): Immutable feature.

        Returns:
            np.ndarray: Indices of hard samples.
        """

        if task_type == 'regression':
            xgb_a = XGBRegressor(n_estimators=100)
            xgb_a.fit(X.values, y.values)
            y_pred_a = xgb_a.predict(X.values)
            residuals = np.abs(y.values - y_pred_a)
            worst_30_percent = np.argsort(residuals)[::-1][:int(0.3 * len(X))]
        elif task_type == 'classification':
            xgb_a = XGBClassifier(n_estimators=100)
            xgb_a.fit(X.values, y.values)
            y_pred_a = xgb_a.predict_proba(X.values)
            y = y.values.astype(int)
            residuals = y_pred_a[np.arange(len(y)), y]
            worst_30_percent = np.argsort(residuals)[:int(0.3 * len(X))]

        y_b = np.zeros(len(X))
        y_b[worst_30_percent] = 1

        xgb_b = XGBClassifier()
        xgb_b.fit(X.values, y_b)

        hardness_scores = xgb_b.predict_proba(X.values)[:, 1]

        if immu_feature is not None:
            bins = np.quantile(X[immu_feature], [0, 0.2, 0.4, 0.6, 0.8, 1])
            bins[0] = bins[0] - 1e-7
            bins[-1] = bins[-1] + 1e-7
            
            selected_indices = []
            for bin_min, bin_max in zip(bins[:-1], bins[1:]):
                bin_mask = (X[immu_feature] >= bin_min) & (X[immu_feature] < bin_max)
                bin_hardness_scores = hardness_scores[bin_mask]
                bin_indices = np.argsort(bin_hardness_scores)[::-1][:int(len(bin_hardness_scores) * alpha)]
                bin_selected_indices = np.where(bin_mask)[0][bin_indices]
                selected_indices.extend(bin_selected_indices)
            
            return np.array(selected_indices)
        
        else:
            return np.argsort(hardness_scores)[::-1][:int(len(X) * alpha)]


    @property
    def requires_alpha(self) -> bool:
        """Checks if alpha is required."""
        return True

def _update_worst_metric(metric: MetricCalculator, worst_metric: float, metric_value: float, indices: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """Update the worst metric for kMeans clustering based on the metric type

    Args:
        metric (MetricCalculator): The instance of the specific metric being used
        worst_metric (float): Current worst value of the metric
        metric_value (float): The new received value of the metric to compare with worst_metric
        indices (np.ndarray): 1D array containing the indices corresponding to the metric_value

    Returns:
        Tuple[float, Optional[np.ndarray]]: The updated worst metric, is returned. Optionally, if the metric value changes, the updated indices are returned
    """
    if isinstance(metric, (MSEMetric, MAEMetric, BrierMetric, LogLossMetric)):
        if metric_value > worst_metric:
            return metric_value, indices
        return worst_metric, None
    if isinstance(metric, (R2Metric, AccuracyMetric, AUCMetric, F1Metric)):
        if metric_value < worst_metric:
            return metric_value, indices
        return worst_metric, None

class WorstClusterSelector(SampleSelector):
    def select(self, X, y, task_type, model, alpha, n_clusters, metric, immu_feature):
        """
        Selects samples based on the worst cluster.

        Args:
            X (pd.DataFrame): Independent variables dataframe.
            y (pd.Series): Target variable series.
            task_type (str): Type of task (regression or classification) (not used).
            model (BaseEstimator): Model to calculate residuals.
            alpha (float): Ratio of worst samples (not used).
            n_clusters (int): Number of clusters.
            metric (MetricCalculator): Metric calculator.
            immu_feature (str): Immutable feature.

        Returns:
            np.ndarray: Indices of worst samples.
        """
        worst_samples = None
        if isinstance(metric, (MSEMetric, MAEMetric, BrierMetric, LogLossMetric)):
            worst_metric = float('-inf')
        elif isinstance(metric, (R2Metric, AccuracyMetric, AUCMetric, F1Metric)):
            worst_metric = float('inf')
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(X.values)
        
        if immu_feature is not None:
            bins = np.quantile(X[immu_feature], [0, 0.2, 0.4, 0.6, 0.8, 1])
            bins[0] = bins[0] - 1e-7
            bins[-1] = bins[-1] + 1e-7
            
            selected_indices = []
            for bin_min, bin_max in zip(bins[:-1], bins[1:]):
                bin_mask = (X[immu_feature] >= bin_min) & (X[immu_feature] < bin_max)
                bin_X = X.values[bin_mask]
                bin_y = y.values[bin_mask]
                bin_cluster_labels = cluster_labels[bin_mask]
                
                bin_worst_samples = None
                if isinstance(metric, (MSEMetric, MAEMetric, BrierMetric, LogLossMetric)):
                    bin_worst_metric = float('-inf')
                elif isinstance(metric, (R2Metric, AccuracyMetric, AUCMetric, F1Metric)):
                    bin_worst_metric = float('inf')
                
                for cluster in np.unique(bin_cluster_labels):
                    cluster_indices = np.where(bin_cluster_labels == cluster)[0]
                    cluster_y_pred = model.predict(bin_X[cluster_indices])
                    cluster_metric_value = metric.calculate(bin_y[cluster_indices], cluster_y_pred)
                    bin_worst_metric, temp_ind = _update_worst_metric(metric, bin_worst_metric, cluster_metric_value, cluster_indices)
                    if temp_ind is not None:
                        bin_worst_samples = temp_ind
                
                bin_worst_samples = np.where(bin_mask)[0][bin_worst_samples]
                selected_indices.extend(bin_worst_samples)
            
            return np.array(selected_indices)
        
        else:
            for cluster in np.unique(cluster_labels):
                indices = np.where(cluster_labels == cluster)[0]
                y_pred = model.predict(X.values[indices])
                metric_value = metric.calculate(y.values[indices], y_pred)
                worst_metric, temp_ind = _update_worst_metric(metric, worst_metric, metric_value, indices)
                if temp_ind is not None:
                    worst_samples = temp_ind
            
            return worst_samples

    @property
    def requires_alpha(self) -> bool:
        return False

class DistanceMetric(ABC):
    @abstractmethod
    def calculate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        pass

class PSIDistance(DistanceMetric):
    def calculate(self, x1, x2, bins=10):
        """Return the PSI Distance between two arrays

        Args:
            x1 (numpy.ndarray): The first nd array
            x2 (numpy.ndarray): The second nd array
            bins (int, optional): _description_. Defaults to 10.

        Returns:
            float: Distance value
        """
        
        min_val = min(np.min(x1), np.min(x2))
        max_val = max(np.max(x1), np.max(x2))
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        hist1, _ = np.histogram(x1, bins=bin_edges)
        hist2, _ = np.histogram(x2, bins=bin_edges)

        p1 = hist1 / len(x1)
        p2 = hist2 / len(x2)

        epsilon = 1e-10
        p1 = np.maximum(p1, epsilon)
        p2 = np.maximum(p2, epsilon)

        return float(np.sum((p2 - p1) * np.log(p2 / p1)))

class WassersteinDistance(DistanceMetric):
    def calculate(self, x1, x2):
        return wasserstein_distance(x1, x2)

class Experiment:
    def __init__(self, data: pd.DataFrame, task_type: str, target: str, exclude: List[str] = None):
        if task_type == 'classification':
            y = data[target]
        if task_type == 'regression':
            y = data[[target]]
        if exclude:
            exclude_columns = exclude + [target]
            X = data.drop(columns=exclude_columns)
        else:
            X = data.drop(columns=[target])
        categorical_features = []
        for col in X.columns:
            if X[col].dtype in ['object','string', 'category']:
                categorical_features.append(col)
            elif X[col].dtype in ['int64', 'float64'] and X[col].nunique() < 5:
                categorical_features.append(col)
            else:
                X[col] = X[col].astype(float)

        numerical_features = [col for col in X.columns if col not in categorical_features]

        encoder = OrdinalEncoder()
        X[categorical_features] = encoder.fit_transform(X[categorical_features])

        scaler = MinMaxScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        
        if task_type == 'regression':
            y = scaler.fit_transform(y) # Ideally this should not be scaled
            y = pd.Series(y.flatten())
            
        self.X = X
        self.y = y
        self.task_type = task_type
        self.model = None
        self._initialize_components()

    def _initialize_components(self):
        self.metrics = {
            'mse': MSEMetric(),
            'acc': AccuracyMetric(),
            'auc': AUCMetric(),
            'f1': F1Metric(),
            'logloss': LogLossMetric(),
            'brier': BrierMetric(),
            'mae': MAEMetric(),
            'r2': R2Metric()
        }
        self.sample_selectors = {
            'worst-sample': WorstSampleSelector(),
            'outer-sample': OuterSampleSelector(),
            'hard-sample': HardSampleSelector(),
            'worst-cluster': WorstClusterSelector()
        }
        self.distance_metrics = {
            'psi': PSIDistance(),
            'wd1': WassersteinDistance()
        }


    def model_train(self, model: BaseEstimator, name: Optional[str] = None) -> None:
        self.model = model
        self.model.fit(self.X, self.y)
        print(f"Trained: {name}")

    def model_diagnose(self, show: str = "resilience_perf", 
                      resilience_method: str = "worst-sample",
                      metric_name: str = None,
                      distance_metric: str = "psi", 
                      alpha: float = 0.3,
                      n_clusters: int = 10,
                      figsize: Tuple[int, int] = (5, 4),
                      immu_feature: Optional[str] = None,
                      show_feature: Optional[str] = None,
                      original_scale: bool = True) -> None:
        
        if self.model is None:
            raise ValueError("Model must be trained before diagnosis")

        plot_methods = {
            "resilience_perf": self._plot_resilience_performance,
            "resilience_distance": self._plot_resilience_distance,
            "resilience_shift_histogram": self._plot_resilience_shift_histogram,
            "resilience_shift_density": self._plot_resilience_shift_density
        }

        if show in plot_methods:
            plot_methods[show](resilience_method, metric_name, distance_metric, 
                              alpha, n_clusters, immu_feature, show_feature, 
                              original_scale, figsize)

    def _plot_resilience_performance(self, *args):
        metric_name = args[1]
        if not metric_name:
            metric_name = self._get_default_metrics()
        metric = self.metrics.get(metric_name)
        selector = self.sample_selectors[args[0]]
        metric_values = []
        if selector.requires_alpha:
            alphas = np.linspace(0.1, 1, 10)
            for alpha in alphas:
                worst_samples = selector.select(self.X, self.y, self.task_type, self.model, alpha, args[4], metric, args[5])
                y_pred = self.model.predict(self.X.iloc[worst_samples])
                metric_value = metric.calculate(self.y.iloc[worst_samples], y_pred)
                metric_values.append(metric_value)
            
            plt.figure(figsize=args[8])
            plt.plot(alphas, metric_values)
            plt.axhline(y=metric_values[-1], color='r', linestyle='--')
            plt.xlabel('Worst Sample Ratio')
        else:
            clusters = np.arange(2, 11)
            for n_clusters in clusters:
                worst_samples = selector.select(self.X, self.y, self.task_type, self.model, args[3], n_clusters, metric, args[5])
                y_pred = self.model.predict(self.X.iloc[worst_samples])
                metric_value = metric.calculate(self.y.iloc[worst_samples], y_pred)
                metric_values.append(metric_value)
            plt.figure(figsize=args[8])
            plt.plot(range(2, 11), metric_values)
            plt.axhline(y=metric_values[-1], color='r', linestyle='--')
            plt.xlabel('# Clusters')

        plt.ylabel(metric_name.upper())
        plt.title('Resilience Test')
        plt.show()

    def _get_default_metrics(self):
        if self.task_type == 'classification':
            metric_name = 'acc'
        elif self.task_type == 'regression':
            metric_name = 'mse'
        return metric_name
    
    def _plot_resilience_distance(self, method, metric_name, distance_metric_name, alpha, n_clusters,
                                 immu_feature, *args):
        selector = self.sample_selectors[method]
        if not metric_name:
            metric_name = self._get_default_metrics()
        worst_samples = selector.select(
            self.X, self.y, self.task_type, self.model, 
            alpha, n_clusters,
            self.metrics[metric_name], immu_feature
        )
        remaining_samples = np.setdiff1d(range(len(self.X)), worst_samples)
        
        distance_metric = self.distance_metrics[distance_metric_name]
        distances = []
        labels = []

        for feature in self.X.columns:
            dist = distance_metric.calculate(
                self.X[feature].values[remaining_samples], 
                self.X[feature].values[worst_samples]
            )
            distances.append(dist)
            labels.append(feature)


        sorted_indices = np.argsort(distances)[::-1]
        distances = [distances[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        num_features = min(10, len(labels))
        distances = distances[:num_features]
        labels = labels[:num_features]


        plt.figure(figsize=args[2])
        plt.bar(labels, distances)
        plt.ylabel(distance_metric_name)
        plt.title(f"Distribution Shift: {'Worst Cluster' if method == 'worst-cluster' else f'{int(alpha*100)}%-Worst'} vs Remaining")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def _plot_resilience_shift_histogram(self, method, metric_name, *args):
        selector = self.sample_selectors[method]
        if not metric_name:
            metric_name = self._get_default_metrics()
        worst_samples = selector.select(self.X, self.y, self.task_type, self.model, args[1], args[2], self.metrics[metric_name], args[3])
        remaining_samples = np.setdiff1d(range(len(self.X)), worst_samples)

        show_feature = args[4]
        feature_data = self.X[show_feature].values if show_feature in self.X.columns else self.y.values

        plt.figure(figsize=args[6])

        all_data = np.concatenate([feature_data[remaining_samples], feature_data[worst_samples]])
        bins = np.linspace(min(all_data), max(all_data), 11) 
        
        remaining_hist, _ = np.histogram(feature_data[remaining_samples], bins=bins)
        worst_hist, _ = np.histogram(feature_data[worst_samples], bins=bins)
        
        remaining_percent = remaining_hist / len(remaining_samples) * 100
        worst_percent = worst_hist / len(worst_samples) * 100
        
        bar_width = (bins[1] - bins[0]) * 0.35
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        plt.bar(bin_centers - bar_width/2, remaining_percent, width=bar_width, 
                label='90%-Remaining Sample', color='lightblue')
        plt.bar(bin_centers + bar_width/2, worst_percent, width=bar_width, 
                label='10%-Worst Sample', color='orange')
        
        plt.xlabel(f"{show_feature}" if show_feature is not None else 'Target')
        plt.ylabel('Percent')
        plt.title('Distribution Shift: 10%-Worst vs Remaining')
        plt.legend()
        plt.show()


    def _plot_resilience_shift_density(self, method, *args):
        selector = self.sample_selectors[method]
        if not args[0]:
            metric_name = self._get_default_metrics()
        else:
            metric_name = args[0]
        worst_samples = selector.select(self.X, self.y, self.task_type, self.model, args[2], args[3], metric_name, args[4])
        remaining_samples = np.setdiff1d(range(len(self.X)), worst_samples)

        show_feature = args[5]
        feature_data = self.X[show_feature].values if show_feature in self.X.columns else self.y.values

        kde_remaining = gaussian_kde(feature_data[remaining_samples])
        kde_worst = gaussian_kde(feature_data[worst_samples])

        x = np.linspace(feature_data.min(), feature_data.max(), 100)

        plt.figure(figsize=args[7])
        plt.plot(x, kde_remaining(x), label='90%-Remaining Sample')
        plt.plot(x, kde_worst(x), label='10%-Worst Sample')
        plt.xlabel(f"{show_feature}" if show_feature is not None else 'Target')
        plt.ylabel('Density')
        plt.title('Distribution Shift: 10%-Worst vs Remaining')
        plt.legend()
        plt.show()