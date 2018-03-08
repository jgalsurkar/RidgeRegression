import numpy as np

class Ridge(object):
    """Linear least squares with l2 regularization
    
    Parameters
    ----------
    X : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data
    
    y : array-like, shape = [n_samples]
        Training targets
    
    """
    def __init__(self, X, y):
        #Singlular-value decomposition is done for efficiency
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices = False)
        self.y = y
        self.info = {}
        
    def train(self, lambda_par):
        """Train the Ridge Regression model

        Parameters
        ----------
        lambda_par : float
            Regularization strength; must be a positive float.
        """
        if lambda_par not in self.info:
            # Compute S inverse
            s_inv = np.diag(list(map((lambda x: x / (lambda_par + x**2)), self.S)))
            # Compute the weights
            w = np.dot(np.dot(np.dot(self.V.T, s_inv), self.U.T), self.y) 
            # Compute degrees of freedom
            df = sum(list(map((lambda x: x**2 / (lambda_par + x**2)), self.S))) 
            # Save the weights and degrees of freedom for the lambda
            self.info[lambda_par] = (w, df) 
    
    def predict(self, X_test, lambda_par):
        """Use the trained model to make predictions on the test set

        Parameters
        ----------
        X_test : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            Test data
        
        lambda_par : float
            Regularization strength; must be a positive float.
        
        Returns
        ----------
        prediction : float
            Model prediction on test data
        """
        if lambda_par in self.info:
            prediction = np.dot(X_test, self.info[lambda_par][0])
            return prediction
    
    def multi_train(self, lambdas):
        """Train multiple Ridge Regression models with different regularization

        Parameters
        ----------
        lambdas : array-like
            Multiple regularization strength values
        """
        for lambda_par in lambdas:
            self.train(lambda_par)
    
    def multi_predict(self, X_test, lambdas):
        """Use the weights for the given lambdas to make predictions on the test set

        Parameters
        ----------
        X_test : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            Test data
        
        lambdas : array-like
            Multiple regularization strength values
            
        Returns
        ----------
        predictions : dict
            Predictions associated with each lambda parameter
        """
        predictions = {lambda_par : self.predict(X_test, lambda_par) for lambda_par in lambdas}
        return predictions