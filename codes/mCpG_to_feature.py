#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 21:14:41 2019

@author: afinneg2
"""
from __future__ import  division
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import gzip
import pickle
from scipy.stats import beta
from numpy import percentile

import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
## Classes 
class Gene_Methyldata(object, ):
    def __init__(self, gff_file, feature_type, CpG_fns ,
                 chroms_allowed = ["chr1" , "chr2" , "chr3" , "chr4"], est_p = "p_mle" , est_p_kws = {}):
        """
        Inputs
        ------
            gff_file - path
            feature_type - string ("gene" is a good choice)
            CpG_fns - OrderedDict of from  [(sample1_name, fileName1), (sample2_name, fileName2), ... ]
                        where fileNamei is a CpGreport 

        TODO: -Gene_Methyldata.map_mCpG_to_gene separates dataframe column into dictionary of arrays then
                Gene_Methyldata.windowAgg_mCpG_genes_scaled recombines into dataframe. Rewrite with no 
                separation.    
        """
        self.est_p = est_p.lower()
        self.chroms_allowed = chroms_allowed 
        self.mCpG_genome = None
        self.mCpG_genes = None
        self.extend_abs = None
        self.extend_rel = None
        self.mCpG_genesgroups = None
        self.mCpG_genesgroups_WA_scaled = None
        self.genes_df = self.load_gff( gff_file, feature_type, 
                                              chroms_allowed = self.chroms_allowed  )
        if CpG_fns is not None:
            self.mCpG_genome = self.load_CpGRpt(CpG_fns , chroms_allowed = self.chroms_allowed,
                                                est_p= self.est_p, est_p_kws = est_p_kws)
        
    @staticmethod
    def load_gff(gff_file, feature_type, chroms_allowed = ["chr1" , "chr2" , "chr3" , "chr4"]  ):
        """
        Parse gff file into pandas data frame with columns
        "seqName" , "source" , "feature" , "start" , "end", strand", "gene_name" ,"attribute"
        index is ID
        Inputs
        -----
            gff_file - path
             feature_type - string ("gene" is a good choice)
        Returns
        ------
            df - pd.DataFrame of Gff file
        """
        print("Loading gff file {}".format(gff_file))
        df = pd.read_csv( gff_file, sep = "\t", comment = "#", 
                         names = ["seqName" , "source" , "feature" , "start" , "end" , 
                                  "score", "strand", "frame" , "attribute"] )
        df = df.drop( columns = ["score" , "frame"] )
        df = df.loc[ df.loc[:, "feature"] =="gene", :  ].copy()
        df = df.loc[ df.loc[:, "seqName"].map(lambda x: x in chroms_allowed), :  ].copy()
        df["start"] = df["start"]  - 1  ## change to 0 based convention
        df["end"] = df["end"] ##-1 +1 change to 0 based and right end open convention
        df["ID"] = df["attribute"].map(lambda x: re.search( r'ID=([^;]+)', x).group(1))
        df["gene_name"] =  df["attribute"].map(lambda x: re.search( r'Name=([^;]+)', x).group(1))
        df = df.set_index("ID", verify_integrity=True) 
        return df 
    
    @staticmethod
    def load_CpGRpt(fn_dict, chroms_allowed, est_p = "p_mle", est_p_kws = {} ):
        """
        Inputs
        ------
            fn_dict - OrderedDict of from [(sample1_name, fileName1), (sample2_name, fileName2), ... ]
            chroms_allowed - list
        Returns
        -------
            CpG_df
        """
        CpG_dfs = []
        for fn in fn_dict.values():
            print("Loading CpG report file {}".format(fn))
            df = pd.read_csv(fn , header = None , sep = "\t", usecols=[0,1,2,3,4],
                        names= ["seqName" , "position", "strand" , "count_ME" , "count_uME"])
            df = df.loc[ df.loc[:, "seqName"].map(lambda x: x in chroms_allowed), :  ].copy()
            df = df.set_index(keys= ["seqName", "position"] )
            if est_p  == "p_mle":  ## make MLE estimtes for each sample
                df["p_mle"] = df.apply(
                            lambda x: Gene_Methyldata.est_p_mle(x["count_ME"], x["count_uME"]), 
                                        axis = 1 ) 
            elif est_p  == "p_bayes":  
                pass  ## will estimate beta distribution parameters by summing observations across samples to get posterior
            else:
                raise NotImplementedError()
            CpG_dfs.append(df)
        strand_df = CpG_dfs[0].loc[: , "strand"]
        CpG_dfs = [ df.drop(columns = "strand") for df in   CpG_dfs ]
        CpG_df = pd.concat( [strand_df] + CpG_dfs ,  axis = 1, 
                           keys = ["mdata"] + list(fn_dict.keys()), verify_integrity = True)
        if est_p == "p_bayes":
            counts_summed = CpG_df.loc[:, pd.MultiIndex.from_product([tuple(fn_dict.keys()), ("count_ME", "count_uME" ) ])
                                          ].groupby(axis = 1, level = 1).sum(axis =1 )
            counts_summed.columns = pd.MultiIndex.from_product([ ("s_combined",), counts_summed.columns ])
            CpG_df = pd.concat([CpG_df.loc[: , ("mdata", "strand")] , counts_summed  ], axis =1 )
            CpG_df[("s_combined", "p_postmean")], \
            CpG_df[("s_combined", "alpha1") ], \
            CpG_df[("s_combined", "beta1") ] = \
                 zip(*CpG_df.apply( lambda x: Gene_Methyldata.est_p_bayes(x.loc[("s_combined","count_ME")],
                                                                          x.loc[("s_combined","count_uME")], **est_p_kws ), 
                                   axis = 1 )  )
        return  CpG_df 
        
    def map_mCpG_to_gene(self, extend_abs = None, extend_rel = 0.1):
        """
        Make self.mCpG_genes = OrderedDict([ (geneID, {data: 2d_array , 
                                                       extend_abs: int } ) ])
            where 2d_array = [  [Stranded_pos_rel_TSS , mCpG_rate_estimate_s1 , mCpG_rate_estimate_s2 ] , 
                            [Stranded_pos_rel_TSS , mCpG_rate_estimate_s1, mCpG_rate_estimate_s2 ], ... ]
        """
        self.extend_abs = extend_abs
        self.extend_rel = extend_rel 
        self.mCpG_genes = OrderedDict([  ])
        for geneID, series in self.genes_df.iterrows():
            seqName = series["seqName"]
            length =  series["end"] - series["start"]
            if self.extend_rel is not None:
                extend_abs = int(np.ceil(length*self.extend_rel))
            mCpG_rel= self.filter_rel_to_feature(self.mCpG_genome, seqName, 
                                            series["start"], series["end"],
                                            series["strand"], extend_abs)
            if self.est_p == "p_mle":
                self.mCpG_genes[ geneID ] = { "position_rel" : mCpG_rel.index.values[:,None].copy() , 
                                             "position_rel_scaled" :  mCpG_rel.index.values[:,None] / length , 
                                             "est_p": mCpG_rel.loc[: , (slice(None), "p_mle")].values.copy() , 
                                             "extend_abs": extend_abs }
            elif self.est_p == "p_bayes":
                self.mCpG_genes[ geneID ] = { "position_rel" : mCpG_rel.index.values[:,None].copy() , 
                                             "position_rel_scaled" :  mCpG_rel.index.values[:,None] / length , 
                                             "est_p": mCpG_rel.loc[: , ("s_combined" , "p_postmean")].values.copy() , 
                                             "alpha1" : mCpG_rel.loc[: ,  ("s_combined" , "alpha1")].values.copy() ,
                                             "beta1" : mCpG_rel.loc[: , ("s_combined" , "beta1")].values.copy() ,
                                             "extend_abs": extend_abs }   
            else:
                raise NotImplementedError()
        return None
    
    def windowAgg_mCpG_genes_scaled(self, genegroups, w_width , w_step, w_start = None, w_stop = None, 
                                    kws_agg = {"intervalType": "bootstrap", "alpha" : 0.95, "n_boot" : 100 }):
        """
        genegroups - OrderedDict([ (group_name1, geneList1), ...   ])
        """
        if self.mCpG_genes is None:
            raise Exception("run method map_mCpG_to_gene() first")
        if (w_start is None) or (w_stop is None):
            if  self.extend_rel is None:
                raise Exception("run method map_mCpG_to_gene() with float value for extent_rel")
            w_start = -1*self.extend_rel 
            w_stop = 1 + self.extend_rel
        self.mCpG_genesgroups  = genegroups
        bins = np.array([[x -w_width/2, x + w_width/2] for x in np.arange(w_start, w_stop+w_step, w_step)])
        df_WA_list = []
        for group_name, genes in self.mCpG_genesgroups.items():
            if self.est_p == "p_mle":
                data = np.concatenate( [self.mCpG_genes[g]['est_p'] for g in genes ] , axis = 0 )
                pos_scaled = np.concatenate( [self.mCpG_genes[g]['position_rel_scaled'] for g in genes ] , axis = 0 )
                df_WA = Gene_Methyldata.calc_binMeans_scaled(pos_scaled, est_p= data, bins = bins)
            elif self.est_p == "p_bayes": 
                pos_scaled = np.concatenate( [self.mCpG_genes[g]['position_rel_scaled'] for g in genes ] , axis = 0 )
                alpha1 = np.concatenate( [self.mCpG_genes[g]['alpha1'] for g in genes ] , axis = 0 )
                beta1 = np.concatenate( [self.mCpG_genes[g]['beta1'] for g in genes ] , axis = 0 )
                df_WA = Gene_Methyldata.aggBayes_bins_scaled(pos_scaled, alpha1, beta1,bins = bins, **kws_agg )
            else:
                raise NotImplementedError()
            df_WA_list.append(df_WA)
        self.mCpG_genesgroups_WA_scaled =  pd.concat(df_WA_list, axis =1 , keys = self.mCpG_genesgroups.keys(),
                                                    names = ["group" , "sample"])
        return None            
                             
    ## Utilities
    @staticmethod
    def calc_binMeans_scaled(pos_scaled, est_p, bins ):
        """
        Inputs:
            pos_scaled - 1d array
            est_p - (pos_scaled.shape[0], n_samples ) array
            bins - (nBins ,2 ) array. Each bin is [bins[i,0], bins[i,1])
        Returns
            df_WA - dataFrame of window average values. Index is window centers columns are samples
        """
        samples = ["s{}".format(i) for i in range(est_p.shape[-1])]
        df = pd.DataFrame( data = np.hstack( [pos_scaled, est_p] ),
                              columns = ['position_rel_scaled'] +  samples )
        df = df.sort_values(by = 'position_rel_scaled')
        df_WA = pd.DataFrame(index = bins.mean(axis = 1) ,
                            data = [ df.loc[ (df["position_rel_scaled"]>= b_l ) & \
                                        (df["position_rel_scaled"] < b_u ), samples ].mean(axis = 0).values \
                                for b_l, b_u in bins ] ,
                             columns = samples)
        return df_WA
    @staticmethod
    def aggBayes_bins_scaled(pos_scaled, alpha1, beta1, bins, intervalType = "bootstrap", alpha = 0.95, n_boot = 100  ):
        df = pd.DataFrame( data = np.hstack( [pos_scaled, alpha1[:,None], beta1[:, None]] ),
                              columns = ['position_rel_scaled', 'alpha1' , 'beta1' ] )
        df = df.sort_values(by = 'position_rel_scaled')
        df_WA = pd.DataFrame(index = bins.mean(axis = 1) ,
                             data = np.array([ Gene_Methyldata.combine_betaPDF( df.loc[ (df["position_rel_scaled"]>= b_l ) & \
                                                                             (df["position_rel_scaled"] < b_u ), ["alpha1", "beta1"] ],
                                                                     intervalType = intervalType, alpha = alpha, n_boot = n_boot ) \
                                                for b_l, b_u in bins 
                                             ] ) ,
                             columns = ["est_p", "lower", "upper"])
        return df_WA
    
    @staticmethod
    def combine_betaPDF( betaParams, intervalType = "bootstrap", alpha = 0.95, n_boot = 100 ):
        """
        betaParams_df — pd.DataFrame: columns are alpha1 and beta1, 
                                     rows are different distributions
        """
        if intervalType == "bootstrap":
            est_p = betaParams.apply(lambda x: x["alpha1"] / (x["alpha1"]+ x["beta1"]), axis = 1 )
            means_resample = [est_p.sample(frac = 1.0, replace = True,).mean() for i in range(n_boot) ]
            lower, upper = percentile(means_resample, 100.0*np.array([(1.0-alpha)/2, alpha+((1-alpha)/2)] ) )
            est_p_mean = est_p.mean()
        else:
            raise NotImplementedError()
        return est_p_mean, lower, upper 
                              
#    @staticmethod
#    def combine_betaPDF_OLD( betaParams, intervalType = "mean_symmetric", CI =0.95 ):
#        """
#        This gives Too big CI there is something conceptually wrong with my approach
#        betaParams_df — pd.DataFrame: columns are alpha1 and beta1, 
#                                     rows are different distributions
#        """
#        if intervalType == "mean_symmetric":
#            def f(x, alpha1,beta1, center, CI=0.95):
#                y = (beta.cdf( np.ones_like(alpha1)*(center + x), alpha1, beta1 ) - \
#                        beta.cdf( np.ones_like(alpha1)*(center - x), alpha1, beta1 )).mean()
#                return y - CI
#            center = betaParams.apply(lambda x: x["alpha1"] / (x["alpha1"]+ x["beta1"]), axis = 1 ).mean()
#            width, _, success, mesg = fsolve( func =f, x0= 10.0**(-5) , 
#                                             args = (betaParams.loc[:, "alpha1"].values, 
#                                                     betaParams.loc[:, "beta1"].values,
#                                                    center,
#                                                    CI),
#                                            full_output = True,
#                                            )
#            if success != 1:
#                raise Exception("fsolve for CI construction failed with message\n{}".format(mesg))
#        else:
#            raise NotImplementedError()
#        return center, width
                                                                                        
    @staticmethod
    def est_p_mle(count_Me, count_uMe ):
        try:
            if (count_Me == 0) and (count_uMe == 0):
                rval = np.nan
            else:
                rval = count_Me/(count_Me + count_uMe )
        except:
            print(count_Me, count_uMe)
            raise Exception("problem")
        return rval
    @staticmethod
    def est_p_bayes(count_Me, count_uMe, alpha0 , beta0 ):
        alpha1  =  alpha0 + count_Me
        beta1 = beta0 + count_uMe
        p_postmean = alpha1 / (alpha1 + beta1)
        return p_postmean, alpha1, beta1 
        
    @staticmethod
    def filter_rel_to_feature(df, seqName, start, end, strand, extend_abs):
        """
        Filter df indexed by (seqName, position) to include only entries between: 
            start - extend_avs and
            end + extend abs 
            (accounting for stand). 
        Returned df is index by position relative to start.
        """
        if strand == "+" :
            df_rel = df.loc[ (seqName, slice(start - extend_abs, end + extend_abs) ), : ].copy()
            df_rel["position_rel"] =  df_rel.index.get_level_values("position").values - start
            df_rel = df_rel.set_index("position_rel")
        elif strand == "-" :
            df_rel = df.loc[ (seqName, slice(start - extend_abs, end + extend_abs) ), : ].copy()
            df_rel["position_rel"] = end -1 - df_rel.index.get_level_values("position").values ## -1 b/c right end open
            df_rel = df_rel.set_index("position_rel")
        else:
            raise Exception("strand {} not recognized for seq {}, start {}".format(strand,seqName, start))
        return df_rel 
    
    def write_mCpG_gene(self,fn ):
        f = gzip.open(fn , 'wb')
        pickle.dump(self.mCpG_genes, f)
        f.close()
        
    def load_mCpG_genes(self, fn ):
        f = gzip.open(fn,'rb')
        self.mCpG_genes = pickle.load(f)
        f.close()   
            
    @staticmethod    
    def plot_mCpG_genesgroups( data, groups, colors= None, ax = None , figsize = (8.5,3),
                             xlabel = "Position (metagene)"):
        """
        Inputs
            data — pd.DataFrame: index are genomic positions (abs or scaled)
                                 columns if 1 level then each columns is a group
                                        if 2 levels 0th level is group 1st level is replicate
            groups — list: the columns of data to plot
            colors — list: the colors for plotting each group
        Returns
            ax
        """
        colors_default = None ## TODO good default
        data = data.reset_index()
        data = data.rename(columns= {"index":  xlabel})
        if data.columns.nlevels ==1:
            data_plot =  data.loc[: , [xlabel] + groups].melt( id_vars = [xlabel], value_name = "mCpG rate" , var_name = "group" ) 
        elif data.columns.nlevels == 2:
            data_plot = data.loc[: , [xlabel] + groups].melt( id_vars = [xlabel], value_name = "mCpG rate"  , var_name = ["lvl0" , "lvl1"]  )
            data_plot["group"] =  data_plot.apply(lambda x: "_".join( [x["lvl0"] ,x["lvl1"] ]), axis = 1 )
            groups = [ "_".join(x) for x in groups ]
        else:
            raise NotImplementedError()
        ax = sns.lineplot(x = xlabel, y = "mCpG rate", hue = "group" , data = data_plot, ax = ax ) ## add color
        return ax

##############################################
## PLOTTING FUNCTIONS
def FacetGrid_day_DEd_DEdp1(mCpG_byDE, colors ,data_only = False, sep_levels = "__"):
    """
    Inputs
    --------
        mCpG_byDE — OrderedDict([ ('d1' , mCpG_d1_df) , ('d2' , mCpG_d1_df), ... ] )
                    where mCpG_df_df is pd.DataFrame with
                        index — bin centers
                        columns — if nlevel ==1 "DE.<dirction>__d.plus1_DE.<direction>"
                                  if nlevel ==2 ( "DE.<dirction>__d.plus1_DE.<direction>" , "s1") , 
                                                ( "DE.<dirction>__d.plus1_DE.<direction>" , "s2") , ...
        data_only — just return the datafrae, don't plot
        sep_levels — separator between DE.<dirction> and d.plus1_DE.<direction> in mCpG_d1_df.columns
    Returns
    -------
        g
        df
    """
    #DE_d_keys =  ["DE.up" , "DE.down" , "DE.neither"]
    #DE_dp1_keys = ["d.plus1_DE.up", "d.plus1_DE.down", "d.plus1_DE.neither"]
    plotData = []
    for cond, mCpG_WA_df in mCpG_byDE.items():
        d =  cond.split("-")[-1]
        if mCpG_WA_df.columns.nlevels == 2:
            mCpG_WA_df = mCpG_WA_df.rename(columns = lambda x: d + sep_levels + x, level = 0)
            plotData.append( mCpG_WA_df.groupby(axis = 1, level = 0).apply( lambda x: x.mean(axis = 1)) )
        elif mCpG_WA_df.columns.nlevels == 1:
            mCpG_WA_df = mCpG_WA_df.rename(columns = lambda x: d + sep_levels + x )
            plotData.append(mCpG_WA_df)
        else:
            raise NotImplementedError()
    plotData =  pd.concat(plotData, axis = 1)
    #print(plotData.head())
    plotData.columns = pd.MultiIndex.from_tuples( [tuple(c.split(sep_levels)) for  c in plotData.columns],
                                                    names = ["day" , "DE current" , "DE d plus 1"])
    plotData = plotData.reset_index(col_fill = "index")
    plotData = plotData.melt(id_vars = [("index" , "index" , "index")], value_name = "mCpG rate")
    plotData = plotData.rename(columns = {("index" , "index" , "index"): "Position (metagene)"  } )
    
    g = sns.FacetGrid(data = plotData, row = "day" , col = "DE current" , hue = "DE d plus 1",  margin_titles=True)
    g.map(plt.plot , "Position (metagene)" , "mCpG rate" )
    g.add_legend()
    
    return g , plotData

def FacetGrid_day_DEd_DEdp1_shaded(mCpG_byDE, colors= None ,data_only = False, sep_levels = "__",
                                  ylim = (0.005,0.03)):
    """
    Inputs
    --------
        mCpG_byDE — OrderedDict([ ('d1' , mCpG_d1_df) , ('d2' , mCpG_d1_df), ... ] )
                    where mCpG_df_df is pd.DataFrame with
                        index — bin centers
                        columns — if nlevel ==1 "DE.<dirction>__d.plus1_DE.<direction>"
                                  if nlevel ==2 ( "DE.<dirction>__d.plus1_DE.<direction>" , "s1") , 
                                                ( "DE.<dirction>__d.plus1_DE.<direction>" , "s2") , ...
        data_only — just return the datafrae, don't plot
        sep_levels — separator between DE.<dirction> and d.plus1_DE.<direction> in mCpG_d1_df.columns
        colors = [color_for_d.plus1_DE.down, color_for_d.plus1_DE.neither ,  color_for_d.plus1_DE.up]
    Returns
    -------
        g
        df
    """
    plotData = []
    for cond, mCpG_WA_df in mCpG_byDE.items():
        d =  cond.split("-")[-1]
        mCpG_WA_df = mCpG_WA_df.rename(columns = lambda x: d + sep_levels + x, level = 0)
        plotData.append(mCpG_WA_df)
    plotData =  pd.concat(plotData, axis = 1)
    plotData.columns = pd.MultiIndex.from_tuples( [tuple(c[0].split(sep_levels)) + (c[1],) for  c in plotData.columns.values],
                                                        names = ["day" , "DE current" , "DE d plus 1", "y"])
    if data_only:
        return plotData
    plotData = plotData.stack(level = [0,1,2]).reset_index().rename(columns = {"level_0" : "Position (metagene)" , "est_p": "mCpG rate"}).sort_values(
                                                by = ["day" , "DE current", "DE d plus 1", "Position (metagene)"])
    def plot_shaded(x,y,lower,upper, color, label):
        plt.plot( x, y, label = label, color = color )
        plt.fill_between(x, lower, upper, color = color , alpha = 0.2)
        return None
    if colors is None:
        colors = ["b" , "orange" , "r"]
    g = sns.FacetGrid(data = plotData, row = "day" , col = "DE current" , hue = "DE d plus 1",  
                      margin_titles=True, aspect = 1.5, 
                      palette = { "d.plus1_DE.down": colors[0] , 
                                 "d.plus1_DE.neither": colors[1] ,
                                 "d.plus1_DE.up": colors[2] }, ylim = ylim )
    g.map(plot_shaded , "Position (metagene)" , "mCpG rate" , "lower" , "upper" )
    g.add_legend()
    
    return plotData, g