#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:50:42 2019

@author: afinneg2
"""
from __future__ import  division
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.optimize import root as find_root 
from numpy.linalg import multi_dot
import numpy.linalg as LA
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


#########################################################################################################################################
## CLASSES

class LogReg_Continuous(object):
    def __init__(self):
        """
        To do: -[x] lmbda should scale with number of observations
               -[x] stardarziation of inputs (see LRC_preprocessor class)
               -[x] method to get prediction accuracy
               -[x] cross validation (separate)
               -[] add auROC method
               -[] lambda should have dependence on number of bins (add this)
               -[] implement observation weighting
        """
        self.beta = None
        return None
     
    def fit(self, X,  y, lmbda, method = "scipy", standardize = False):
        """
        Inputs:
        -------
            X - ndarray rows are observations columns are features
            y - 1d array  (XX 1d array of class lambes or 2d array of one-hot class encoding)
        """
        self.X = X
        self.y = y
        self.lmbda = lmbda
#         if len(y.shape) == 1:
#             self.y = self.to_onehot(y)
#         else:
#             self.y = y
        self._X = np.hstack( [np.ones(X.shape[0], dtype = float)[:,None], self.X] )
        self._DtD = self._make_DtD(n_bins = self.X.shape[1] )
        self.nObs_train = self._X.shape[0]
        f_newton = self.get_f_newton(X = self._X, y=self.y, lmbda = self.lmbda, DtD = self._DtD, nObs = self.nObs_train )
        fprime_newton = self.get_fprime_newton(X = self._X, lmbda  = self.lmbda, DtD = self._DtD, nObs = self.nObs_train )
        beta0 = np.zeros(shape = self._X.shape[1], dtype = float)
        if method.lower()  == "scipy":
            soln = find_root(fun = f_newton, jac = fprime_newton, x0 = beta0, method = "hybr", options = {"maxfev": 1000*len(beta0)} ) 
            if not soln["success"]:
                raise Exception("find_root failed with message\n {}\n fun is\n {}".format(soln["message"] , soln["fun"]))
            self.beta = soln["x"]
        elif method.lower() == "custom":
            soln = self.findRoot_newton(f= f_newton, fprime = fprime_newton, x0= beta0,  max_iter= 10000, tol = 0.0000001 )
            if not soln[3]:
                raise Exception("my_newton failed, fun is\n {}".format(soln["message"] , soln[1]))
            self.beta = soln[0]
        else:
            raise NotImplementedError()
        return soln
    
    def predict(self, X, y = None ):
        """
        Inputs
            X - ndarray rows are observations columns are features
            y - 1d array of 0, 1 class labels
        """
        if self.beta is None:
            raise Exception("run .fit method first")
        X_pred =  np.hstack( [np.ones(X.shape[0], dtype = float)[:,None], X] )
        preds = self.p_pos(X_pred, self.beta )
        if y is not None:
            y_pred = (preds > 0.5).astype(float)
            acc = np.isclose(y_pred, y).sum() / len(y)
            return preds, acc
        else:
            return preds
        
    def ROC(self , X , y, ax = None):
        preds = self.predict(X)
        fpr, tpr,  thresholds = roc_curve(y,  preds ) 
        auc_val = auc(fpr, tpr)
        if ax is None:
            return auc_val, [fpr, tpr,  thresholds]
        else:
            ax.plot(fpr, tpr, color='darkorange',
                         lw=2, label='ROC curve (area = %0.2f)' % auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.xlim([0.0, 1.0])
            ax.ylim([0.0, 1.05])
            ax.xlabel('False Positive Rate')
            ax.ylabel('True Positive Rate')
            ax.title('Receiver operating characteristic example')
            ax.legend(loc="lower right")
            return auc_val, [fpr, tpr,  thresholds], ax
            
    @staticmethod
    def _make_DtD(n_bins):
        """
        DtD is matrix that gives discrete approximation of \int( (beta'')**2 )
        """
        D = np.zeros( shape= (n_bins-1, n_bins+1) ,dtype = float )
        for i in range(1, D.shape[0]):
            D[i, [i, i+1 , i+2] ] = 1,-2, 1
        DtD = np.dot( np.transpose(D), D)
        return DtD
    
    @staticmethod
    def get_f_newton(X, y, lmbda, DtD, nObs):
        X_t = np.transpose(X)
        def f_newton(beta):
            """
            beta - ndarray with shape (1 + n_features, )
            """
            r_val = np.dot(X_t,
                           y -  LogReg_Continuous.p_pos(X, beta) 
                          ) - lmbda*nObs*np.dot(DtD, beta)
            return r_val
        return f_newton
    
    @staticmethod
    def get_fprime_newton(X,  lmbda, DtD, nObs):
        X_t = np.transpose(X)
        def fprime_newton(beta):
            p = LogReg_Continuous.p_pos(X, beta) 
            W = np.diag(p*(1.0 - p ))
            r_val = -1.0*multi_dot([X_t , W , X] ) - lmbda*nObs*DtD
            return r_val
        return fprime_newton
    
    @staticmethod
    def p_pos(X, beta ):
        r_val = np.divide(1.0, 1 + np.exp(-1*np.dot(X, beta)) )
        return r_val
             
    @staticmethod    
    def to_onehot(y):
        nClasses = np.max(y)
        onehot = np.stack( [np.eye(nClasses)[i, :] for i in y ] )
        return onehot
    
    @staticmethod
    def findRoot_newton( f, fprime, x0, max_iter= 10000, tol = 0.0000001  ):
        """
        simple root finder for testing purpose. Sometimes throws non-invertable
        error when DtD is much larger that -1.0*multi_dot([X_t , W , X] ).
        """
        x= x0
        for i in range(max_iter):
            f_val = f(x)
            if np.all(np.abs(f_val) < tol):
                return x , f_val, i, True, jac
            else:
                jac = fprime( x )
                x = x - np.dot(LA.inv(jac), f_val) 
        print("Failed to converge after {} iteratoins".format( max_iter))
        return x, f_val, i, False, jac
    
class LRC_preprocessor(object):
    """
    Preprocess data for training/prediction using LogReg_Continuous object
    
    To preprocess a single dataset use 
        lrc_pp = LRC_preprocessor( n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1)
        lrc_pp.set_stdz_params(data)
        lrc_pp.add_data(dataTrain_byClass, dataTest_byClass)
        dataset = lrc_pp.dataset  ## the preprocessed data
    To preprocess for cross-validation use:
        lrc_pp = LRC_preprocessor( n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1)
        stdz_params_CV = lrc_pp.add_CV_datasets(data_byClass, n_folds , stdz_name) ## the parameters used for preprocssing of each fold
        dataset = lrc_pp.dataset  ## the preprocessed data
    """
    def __init__(self,  n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1 ):
        self.n_features = n_features 
        self.start = start
        self.stop = stop
        self.stdz_width = stdz_width
        self.stdz_params = None
        self.stdz_name = None
        self.bins = np.hstack([np.linspace(start ,stop, n_features+1)[0:-1,None],
                                 np.linspace(start ,stop, n_features+1)[1:,None]])
        self.bin_centers = np.round( self.bins.mean(axis = 1) , 3)
        self.dataset =  pd.DataFrame(index = [], 
                                    columns =  pd.MultiIndex.from_tuples([("feature", x ) for x in  self.bin_centers ] + \
                                             [("metaData", "y"), ("metaData", "cls_name"), ("metaData", "geneID"), 
                                              ("metaData",  "stdz_name"), ("metaData", "fold"), ("metaData" , "train/test")])
                                    )
        self.n_folds = None
                                        
    def set_stdz_params(self, data, name = None):
        """
        Inputs
        ------
            data - list of dictionaries of observations, e.g.
                        [ OrderedDict([ ("geneA1", data_df ),
                                        ("geneA2", data_df), ... ]) ,
                          OrderedDict([ ("geneB1", data_df ),
                                        ("geneB2", data_df), ... ]) 
                        ]
                        where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        """
        
        self.stdz_params = self.calc_stdz_params(data, n_features = self.n_features, 
                                                 start = self.start, stop = self.stop, stdz_width = self.stdz_width )
        if name is not None:
            self.stdz_name = name
        else:
            if self.stdz_name is None:
                self.stdz_name = "0"
            else:
                self.stdz_name = str(int(self.stdz_name) +1 )
        return None
    
    def add_data(self, dataTrain_byClass, dataTest_byClass, nan_to_zero = True,  metadata = {"fold": 0, "stdz_type": "0"}):
        """
        Append observations to self.X
        Inputs
        --------
        dataTrain_byClass - OrderedDict with class names as keys and dictionaries 
                        of observations as values, e.g.
                        OrderedDict([  ("className0", OrderedDict([ ("geneA1", data_df ),
                                                                    ("geneA2", data_df), ... ])
                                        ),
                                        ("className1", OrderedDict([ ("geneB1", data_df ),
                                                                    ("geneB2", data_df), ... ])
                                        ),
                                    ])
                         where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        dataTest_byClass - Like datatrain_byClass
        """
        if self.stdz_params is None:
            raise Exception("run set_stdz_params method first")
        data_train = self.preprocess_data( dataTrain_byClass, self.stdz_params, self.bin_centers, nan_to_zero = True)
        data_train.loc[: , ("metaData", "train/test")]  = "train"  
        for k, v in  metadata.items():
            data_train.loc[:, ("metaData", k)] = v                         
        data_test = self.preprocess_data( dataTest_byClass, self.stdz_params, self.bin_centers, nan_to_zero = True)                           
        data_test.loc[: , ("metaData", "train/test")]  = "test"
        for k, v in  metadata.items():
            data_test.loc[:, ("metaData", k)] = v                            
        self.dataset = pd.concat([ self.dataset , data_train , data_test], axis = 0, ignore_index = True)                   
        return None                                                                   

    def add_CV_datasets(self, data_byClass, n_folds , stdz_name):
        """
        data_byClass - OrderedDict with class names as keys and dictionaries 
                        of observations as values, e.g.
                        OrderedDict([  ("className0", OrderedDict([ ("geneA1", data_df ),
                                                                    ("geneA2", data_df), ... ])
                                        ),
                                        ("className1", OrderedDict([ ("geneB1", data_df ),
                                                                    ("geneB2", data_df), ... ])
                                        ),
                                    ])
                         where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        n_folds - int
        """
        if self.n_folds is None:
            self.n_folds = n_folds
        else:
            if self.n_folds != n_folds:
                raise Exception( "self.n_folds is already set" )
            
        geneIDs = [geneID for x in data_byClass.values() for geneID in x.keys() ]
        np.random.shuffle(geneIDs)
        fold_size = -( (-len(geneIDs))//self.n_folds ) 
        fold_to_geneID = OrderedDict([(i, geneIDs[i*fold_size : (i+1)*fold_size]) for i in range(self.n_folds)  ] )
        print(OrderedDict([ (k, len(v) ) for k,v in fold_to_geneID.items()] ))
        stdz_params_CV = pd.DataFrame(index = [], columns = [ "bin_lbound" , "bin_ubound", "mean", "std" , "stdz_name" , "fold" ])
        for fold , genes in  fold_to_geneID.items():
                                       
            dataTrain_byClass = OrderedDict([ (className, OrderedDict([(geneID, x[geneID]) for geneID in  x.keys() if geneID not in genes ]))
                                              for className, x in  data_byClass.items() ] )
            print([ len(x) for x in dataTrain_byClass.values() ])
            dataTest_byClass = OrderedDict([ (className, OrderedDict([(geneID, x[geneID]) for geneID in  x.keys() if geneID in genes ]))
                                              for className, x in  data_byClass.items() ] )
            self.stdz_params = self.calc_stdz_params([ x for x in dataTrain_byClass.values() ], 
                                           n_features = self.n_features,  start = self.start, stop = self.stop, stdz_width = self.stdz_width )
                                    
            self.add_data(dataTrain_byClass, dataTest_byClass, nan_to_zero = True,  metadata = {"fold": fold, "stdz_name": stdz_name } )
                  
            self.stdz_params.loc[: , "stdz_name"] = stdz_name   
            self.stdz_params.loc[: , "fold"] = fold
            stdz_params_CV = pd.concat( [ stdz_params_CV ,  self.stdz_params] , axis = 0 )
                
        return  stdz_params_CV    
                          
    @staticmethod
    def preprocess_data( data_byClass, stdz_params, bin_centers, nan_to_zero = True):
        """
        Inputs
        --------
        data_byClass - OrderedDict with class names as keys and dictionaries 
                        of observations as values, e.g.
                        OrderedDict([  ("className0", OrderedDict([ ("geneA1", data_df ),
                                                                    ("geneA2", data_df), ... ])
                                        ),
                                        ("className1", OrderedDict([ ("geneB1", data_df ),
                                                                    ("geneB2", data_df), ... ])
                                        ),
                                    ])
                         where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
        """
        n_obs = sum([len(v) for v in data_byClass.values()])
        bins = stdz_params.loc[:, ["bin_lbound" , "bin_ubound" ] ].values
        bin_centers = np.round( bins.mean(axis = 1) , 3)
        ds = pd.DataFrame(data = np.zeros( shape=(n_obs, len(bin_centers)+3), dtype = float ) ,
                          columns =  pd.MultiIndex.from_tuples([("feature", x ) for x in bin_centers ] + \
                                                 [("metaData", "y"), ("metaData", "cls_name"), ("metaData", "geneID") ] )
                         )
        ds_row_idx = 0
        for clsID, cls_obsDict in enumerate(data_byClass.items()):
            cls_name , obs_dict =  cls_obsDict
            for geneID, df in obs_dict.items():
                for j, bin_ends in enumerate(bins):
                    bin_lb , bin_ub =bin_ends
                    est_p = df.loc[ (df["position_rel_scaled"] >= bin_lb) & (df["position_rel_scaled"] < bin_ub), "est_p" ].values
                    if len(est_p) > 0:
                        est_p = np.mean((est_p -  stdz_params.loc[j, "mean"] ) / stdz_params.loc[j, "std"]  )
                    else:
                        est_p = np.nan
                    ds.iloc[ds_row_idx, j ] = est_p
                ds.loc[ds_row_idx ,("metaData", "y") ] = clsID
                ds.loc[ds_row_idx ,("metaData", "cls_name") ] = cls_name
                ds.loc[ds_row_idx ,("metaData", "geneID") ] = geneID
                ds_row_idx +=1
        if nan_to_zero:
            ds = ds.fillna(0.0)
        return ds

    @staticmethod
    def calc_stdz_params(data, n_features = 30, start = -0.1, stop = 1.1, stdz_width = 0.1 ):
        """
        Inputs
        ------
            data - list of dictionaries of observations, e.g.
                        [ OrderedDict([ ("geneA1", data_df ),
                                        ("geneA2", data_df), ... ]) ,
                          OrderedDict([ ("geneB1", data_df ),
                                        ("geneB2", data_df), ... ]) 
                        ]
                        where data_df is pd.DataFrame with columns: [position_rel_scaled, est_p]
            n_features - int
            stat - left endpoint of 1st feture bin
            stop- right endpoint of last feature bin
            stdz_width - width of position_rel_scaled for computing mean and standardization
        Returns:
        --------
            stdz_params - pd.DataFrame with columns [ "bin_lbound" , "bin_ubound", "mean","std" ]
        """
        bins = np.hstack([np.linspace(start ,stop, n_features+1)[0:-1,None],
                         np.linspace(start ,stop, n_features+1)[1:,None]])
        if not isinstance(data, list):
            data = [data]
        positions_scaled = []
        est_p_vals = []
        for obs_dict in data:
            for v in obs_dict.values():
                positions_scaled.append(v.loc[:, "position_rel_scaled"].values)
                est_p_vals.append(v.loc[:, "est_p"].values)
        positions_scaled = np.concatenate( positions_scaled )
        est_p_vals = np.concatenate( est_p_vals )
        bins_stdz = np.array([ [center -  stdz_width/2.0 , center +  stdz_width/2.0 ] for center in bins.mean(axis = 1)])
        df = pd.DataFrame( data = np.hstack( [positions_scaled[:,None], est_p_vals[:,None] ] ),
                                  columns = ['position_rel_scaled', 'est_p'] ).sort_values(by = 'position_rel_scaled')  
        stdz_params = pd.DataFrame( data = [ [  bins[i,0],  bins[i,1] ,
                                              df.loc[ (df["position_rel_scaled"]>= bins_stdz[i,0] ) & \
                                                    (df["position_rel_scaled"] < bins_stdz[i,1] ), 'est_p' ].mean() ,
                                              df.loc[ (df["position_rel_scaled"]>= bins_stdz[i,0] ) & \
                                                    (df["position_rel_scaled"] < bins_stdz[i,1] ), 'est_p' ].std(),
                                                 ]
                                            for i in range(len(bins_stdz)) ] ,
                                     columns = [ "bin_lbound" , "bin_ubound", "mean","std"])
        return  stdz_params
    
##############################################################################################################################
## UTILITY FUNCTIONS - Cross validation

def run_CV(CV_data, lmbda_vals =[ 0.001, 0.1,1.0 , 2.0 , 4.0 , 8.0, 16.0 , 32.0, 64.0, 128.0 ], performanceStat = ["acc", "auROC" ] ):
    
    if CV_data.isna().any(axis = None):
        raise ValueError("CV_data should not contain Nans" )
    fold_IDs = sorted( CV_data.loc[:, ("metaData" , "fold")].unique(), key = lambda x: int(x) )
    if isinstance(performanceStat , str ):
        performanceStat = [performanceStat]
    CV_results = pd.DataFrame( index = fold_IDs, 
                                columns =  pd.MultiIndex.from_product([tuple(performanceStat), tuple(lmbda_vals)]) ,
                                  dtype = float )
    for lmbda in lmbda_vals:
        results_lmbda = { stat: [] for stat in performanceStat  }
        for fold_id in  CV_results.index:
            mask_train = (CV_data.loc[:, ("metaData", "fold")] == fold_id) & (CV_data.loc[:, ("metaData", "train/test") ] == "train")
            X_train = CV_data.loc[ mask_train , ("feature" , slice(None)) ].values.copy()
            y_train = CV_data.loc[ mask_train , ("metaData" , "y") ].values.copy()
            
            mask_test = (CV_data.loc[:, ("metaData", "fold")] == fold_id) & (CV_data.loc[:, ("metaData", "train/test") ] == "test")
            X_test = CV_data.loc[ mask_test , ("feature" , slice(None)) ].values.copy()
            y_test = CV_data.loc[ mask_test , ("metaData" , "y") ].values.copy()
            
            logreg = LogReg_Continuous()
            _ = logreg.fit(X= X_train, y = y_train, lmbda = lmbda )         
            
            for stat in performanceStat:
                if stat == "acc":
                    pred, acc = logreg.predict(X= X_test, y = y_test)
                    results_lmbda["acc"].append(acc)
                elif stat == "auROC":
                    pred, acc = logreg.predict(X= X_test, y = y_test)
                    fpr, tpr, _ = roc_curve(y_test,  pred ) 
                    results_lmbda["auROC"].append(auc(fpr, tpr))
                else:
                     raise NotImplementedError()
        for stat in performanceStat:
            CV_results.loc[:, (stat , lmbda)] = results_lmbda[stat]
            
    if len(performanceStat) ==1:
        ## drop unnecesary level9 of CV_results
        CV_results  =  CV_results.drop(axis = 1, level = 0)
    return CV_results

def plot_CV_results(CV_results, ax_height = 3 , figwidth= 8.5, ylabel = "" ):
    
    if CV_results.columns.nlevels == 1: 
        ## assume columns describe lambda values only
        CV_results.columns = pd.MultiIndex.from_tuples([ (ylabel , colname) for colname in CV_results.columns])
    level0_unique = list(set(CV_results.columns.get_level_values(0)))
    nrows = len(level0_unique )
    fig , axes = plt.subplots(nrows = nrows, ncols = 2, figsize=( figwidth, ax_height*nrows))
    for ax_row , level0_val in zip(axes, level0_unique):
        ## Box plot
        ax  = ax_row[0]
        plotData = CV_results.loc[: , (level0_val, slice(None))].melt(var_name = "lambda" , value_name = level0_val, col_level = 1)
        plotData.loc[: , "lambda"] = plotData.loc[:, "lambda"].apply(lambda x: "{:.1e}".format(x))
        ax = sns.boxplot(data = plotData ,x = "lambda" , y = level0_val, notch = True,
                            order = sorted( np.unique(plotData.loc[: , "lambda"]) , key = lambda x: float(x) ), ax = ax) 
        _ = ax.set_xticklabels([float(x.get_text()) for x in ax.get_xticklabels()] ,rotation=90)
        ## Plot Mean acc
        ax  =  ax_row[1]
        y_vals = CV_results.loc[: , (level0_val, slice(None))].mean(axis = 0).values
        ax.plot( np.arange(len(y_vals)),  y_vals, marker = "o")
        ax.set_xticks( np.arange(len(y_vals)))
        ax.set_xticklabels( CV_results.loc[: , (level0_val, slice(None))].columns.get_level_values(1), rotation =90)
        ax.set_xlabel("lambda")
    fig.tight_layout()  
    return fig
    

