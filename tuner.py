import pandas as pd 
import numpy as np 

from datetime import date,time,timedelta,datetime

from sklearn.pipeline import Pipeline 

from sklearn.linear_model import LassoLarsCV
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.preprocessing import QuantileTransformer,StandardScaler,MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA,KernelPCA,FastICA

from sklearn.metrics import mean_absolute_error as mae 

import optuna
 
class Tuning():
    """
    The Tuning class is responsible for tuning regression models using various methods of data transformation, 
    decomposition, and hyperparameter tuning with Optuna.
    
    Attributes
    ----------
    X_train : pandas.DataFrame
        Training data for the predictors.
    X_test : pandas.DataFrame
        Testing data for the predictors.
    y_train : pandas.Series or numpy.ndarray
        Training data for the target values.
    y_test : pandas.Series or numpy.ndarray
        Testing data for the target values.
    """
    def __init__(self,X_train, X_test,y_train,y_test):
        """
        Initializes the Tuning class with training and testing data.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training data for the predictors.
        X_test : pandas.DataFrame
            Testing data for the predictors.
        y_train : pandas.Series or numpy.ndarray
            Training data for the target values.
        y_test : pandas.Series or numpy.ndarray
            Testing data for the target values.
        """
        
        self.X_train = X_train
        self.X_test =  X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def create_pipe(self,regressor,transformer,decomposition,targetTransformer,**_params):
        """
    Creates a pipeline for data processing and regression model with optional transformation and decomposition steps.

    Parameters
    ----------
    regressor : Estimator
        The regressor to be used in the pipeline.
    transformer : Class or None
        The data transformation class, such as StandardScaler, MinMaxScaler. Can be None if no transformation is needed.
    decomposition : Class or None
        The class for dimensionality reduction, such as PCA, FastICA. Can be None if dimensionality reduction is not required.
    targetTransformer : Class or None
        The transformation for target values. Can be None if no transformation is required.
    **_params : dict
        Parameters for each step in the pipeline, such as transformer, decomposition, and regressor.

    Returns
    -------
    Pipeline
        Returns a pipeline object containing the data processing steps and regression model.
    """
        transformer_params = _params.get("transformer_params",{})
        decomposition_params = _params.get("decomposition_params",{})
        regressor_params = _params.get("regresor_params",{})
        targetTransformer_params = _params.get("targetTransformer_params",{})
        steps = []
        if transformer != None:
            steps.append(('transform',transformer(**transformer_params)))
        if decomposition != None:
            steps.append(('decomposition',decomposition(**decomposition_params)))
        if targetTransformer != None:
            steps.append(('targetTransformer',TransformedTargetRegressor(regressor=regressor(**regressor_params),                                                                 
                                                                    transformer=targetTransformer(**targetTransformer_params))))
        else:
            steps.append(('regressor',regressor(**regressor_params)))
        return Pipeline(steps)
                    
    def tuning(self,evaluation=mae,storage = f"sqlite:///optuna_study.db",study_name = f"Study_{date.today()}",direction='minimize',n_trials = 100):
        """
        Performs regression model tuning using Optuna.

        Parameters
        ----------
        evaluation : function, default mean_absolute_error
            Function used to evaluate the model performance.
        storage : str, default "sqlite:///optuna_study.db"
            Path to the Optuna database.
        study_name : str, default "Study_{current_date}"
            Name of the Optuna study.
        direction : str, default 'minimize'
            Optimization direction ('minimize' or 'maximize').
        n_trials : int, default 100
            Number of tuning trials.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Raises an error if issues occur during tuning.
        """
        def objective(trial):
            
            # Wybór regressora
            try:
                regressor = trial.suggest_categorical('regressor', ["Lasso", "MLP", "SVM", "KNeighbors", "Forest", "Gradient"])

                if regressor == "Lasso":
                    regressor = LassoLarsCV
                    regressor_params = dict(cv=5, n_jobs=10)

                elif regressor == "MLP":
                    regressor = MLPRegressor
                    hidden_layer_sizes = trial.suggest_categorical(
                        'hidLay_sizes', [str((i, j)) for i in range(2, 22, 2) for j in range(2, 22, 2)]
                    )
                    hidden_layer_sizes = eval(hidden_layer_sizes)
                    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
                    solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd'])
                    learning_rate = 'adaptive'

                    regressor_params = dict(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        learning_rate=learning_rate
                    )

                elif regressor == "SVM":
                    regressor = SVR
                    kernel = trial.suggest_categorical('kernel', ['rbf', 'sigmoid', 'poly'])
                    if kernel == "poly":
                        degree = trial.suggest_int('degree', 2, 7)
                    else:
                        degree = 3
                    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                    if kernel == 'rbf':
                        coef0 = 0.0
                    else:
                        coef0 = trial.suggest_float('coef0', -100, 100)
                    C = trial.suggest_float('C', 1e-5, 1e5, log=True)
                    max_iter = 3000
                    regressor_params = dict(
                        kernel=kernel,
                        degree=degree,
                        gamma=gamma,
                        coef0=coef0,
                        C=C,
                        max_iter=max_iter
                    )

                elif regressor == "KNeighbors":
                    regressor = KNeighborsRegressor
                    n_neighbors = trial.suggest_int('n_neighbors', 3, 10)
                    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])

                    regressor_params = dict(
                        n_neighbors=n_neighbors,
                        weights=weights
                    )

                elif regressor == "Forest":
                    regressor = RandomForestRegressor
                    n_estimators = trial.suggest_int('n_estimators', 10, 50)
                    max_depth = trial.suggest_int('max_depth', 1, 20)
                    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 1e-5, 1e-2, log=True)

                    regressor_params = dict(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_impurity_decrease=min_impurity_decrease
                    )

                elif regressor == "Gradient":
                    regressor = GradientBoostingRegressor
                    loss = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile'])
                    alpha = 0.5
                    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
                    n_estimators = trial.suggest_int('n_estimators', 10, 50)
                    criterion = trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error'])
                    max_depth = trial.suggest_int('max_depth', 2, 20)
                    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 1e-4, 1e-1, log=True)
                    regressor_params = dict(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_impurity_decrease=min_impurity_decrease,
                        criterion=criterion,
                        learning_rate=learning_rate,
                        alpha=alpha,
                        loss=loss
                    )
                transformer = trial.suggest_categorical('transformer', ["StandardScaler", "MinMaxScaler", "QuantileTransformer",None])
                if transformer == "StandardScaler":
                    transformer = StandardScaler
                    transformer_params = {}
                elif transformer == "MinMaxScaler":
                    transformer = MinMaxScaler
                    transformer_params = {}
                elif transformer == "QuantileTransformer":
                    transformer = QuantileTransformer
                    n_quantiles = trial.suggest_int('n_quantiles',20,min(len(self.X_train)-1,300))
                    transformer_params = dict(n_quantiles = n_quantiles)
                else:
                    transformer_params={}
                
                decomposition = trial.suggest_categorical('decomposition', ["PCA", "FastICA",None])
                if decomposition == "PCA":
                    decomposition = PCA
                    n_components = trial.suggest_int('n_components',2,len(self.X_train.columns)-1)
                    decomposition_params = dict(n_components=n_components)

                elif decomposition == "FastICA":
                    decomposition = FastICA
                    n_components = trial.suggest_int('n_components',2,len(self.X_train.columns)-2)
                    decomposition_params = dict(n_components=n_components)
                else:
                    decomposition_params={}

                targetTransformer = trial.suggest_categorical('targetTransformer', ["StandardScaler", "MinMaxScaler", "QuantileTransformer", None])
                if targetTransformer == "StandardScaler":
                    targetTransformer = StandardScaler
                    targetTransformer_params = {}
                elif targetTransformer == "MinMaxScaler":
                    targetTransformer = MinMaxScaler
                    targetTransformer_params = {}
                elif targetTransformer == "QuantileTransformer":
                    targetTransformer = QuantileTransformer
                    n_quantiles = trial.suggest_int('n_quantiles',20,min(len(self.X_train)-1,300))
                    targetTransformer_params = dict(n_quantiles = n_quantiles)
                else:
                    targetTransformer_params={}
                params = dict(transformer_params = transformer_params,
                            regressor_params = regressor_params,
                            decomposition_params = decomposition_params,
                            targetTransformer_params = targetTransformer_params)
                
                

                model = self.create_pipe(regressor,transformer,decomposition,targetTransformer,**params)
                model.fit(self.X_train,self.y_train)
                preds = model.predict(self.X_test)
                score = evaluation(self.y_test,preds)
                return score
            except Exception as e:
                # Obsługa dowolnych nieoczekiwanych błędów
                print(f"Unexpected error occurred: {e}")
                return np.inf
        study = optuna.create_study(study_name=study_name,direction = direction,storage = storage,load_if_exists = True)
        study.optimize(objective,n_trials=n_trials)
        self.best_params = study.best_params

    def build_best_model(self):
        """
        Builds the best model based on the parameters obtained from the Optuna tuning process.

        This method uses the best parameters stored in the `self.best_params` attribute to create a pipeline 
        with a regressor, transformer, decomposition technique, and target transformer. The pipeline is then 
        configured with the parameters that were found to be optimal during the tuning process.

        Returns
        -------
        Pipeline
            A scikit-learn pipeline object that includes the best regressor, data transformations, 
            dimensionality reduction, and target value transformation steps.

        Raises
        ------
        AttributeError
            If the tuning process has not been run and `self.best_params` is not set, an AttributeError is raised.

        Notes
        -----
        - The method first identifies the types of regressor, transformer, decomposition, and target transformer 
        based on the best parameters obtained from the Optuna study.
        - It constructs a dictionary of parameters for each component of the pipeline, such as the regressor's 
        hyperparameters, transformer settings, decomposition configuration, and target transformer parameters.
        - The method calls `self.create_pipe` to assemble the pipeline using the specified components and their 
        corresponding parameters.
        """
        try:
            bp = self.best_params
            regressor_name = bp['regressor']            
            if regressor_name == "Lasso":
                regressor = LassoLarsCV
            elif regressor_name == "MLP":
                regressor = MLPRegressor
            elif regressor_name == "SVM":
                regressor = SVR
            elif regressor_name == "KNeighbors":
                regressor = KNeighborsRegressor
            elif regressor_name == "Forest":
                regressor = RandomForestRegressor
            elif regressor_name == "Gradient":
                regressor = GradientBoostingRegressor
                
            transformer_name = bp['transformer']
            if transformer_name == "StandardScaler":
                transformer = StandardScaler
            elif transformer_name == "MinMaxScaler":
                transformer = MinMaxScaler
            elif transformer_name == "QuantileTransformer":
                transformer = QuantileTransformer
            else:
                transformer = None
                
            decomposition_name = bp['decomposition']
            if decomposition_name == "PCA":
                decomposition = PCA
            elif decomposition_name == "FastICA":
                decomposition = FastICA
            else:
                decomposition = None
                
            targetTransformer_name = bp['targetTransformer']
            if transformer_name == "StandardScaler":
                targetTransformer = StandardScaler
            elif transformer_name == "MinMaxScaler":
                targetTransformer = MinMaxScaler
            elif transformer_name == "QuantileTransformer":
                targetTransformer = QuantileTransformer
            else:
                targetTransformer = None
            regressor_params = {}
            transformer_params = {}
            decomposition_params = {}
            targetTransformer_params = {}
            for param_name, param_value in bp.items():
                if param_name in ['hidLay_sizes','activation','solver','learning_rate','kernel','degree','gamma','coef0','C','max_iter','n_neighbors','weights','n_estimators','max_depth','min_impurity_decrease','criterion','learning_rate','alpha','loss']:
                    regressor_params[param_name]=param_value
                if transformer_name == QuantileTransformer and param_name in ['n_quantiles']:                    
                    transformer_params[param_name]=param_value
                if param_name in ['n_components']:
                    decomposition_params[param_name]=param_value
                if targetTransformer_name == QuantileTransformer and param_name in ['n_quantiles']:
                    targetTransformer_params[param_name]=param_value
                    
            params = dict(  regressor_params = regressor_params,
                            transformer_params = transformer_params,
                            decomposition_params = decomposition_params,
                            targetTransformer_params = targetTransformer_params)
            
            model = self.create_pipe(regressor,transformer,decomposition,targetTransformer,**params)
            model.fit(self.X_train,self.y_train)
            return model
        except AttributeError as a:
            print("Tuning haven't be runed. There are no the best parameters")
        
        
