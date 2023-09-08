#! /usr/local/bin/python3.7

# FIXME: write a native `eval_func` R function.

import os, zipfile, pathlib
import numpy as np
import pandas as pd
import functools
import warnings

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import rpy2

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType


##--------------------------------------------------------------------------------------
# R version
##--------------------------------------------------------------------------------------
# eval_func_rcode = zipfile.ZipFile(pathlib.Path(__file__).parents[1]).open("darima/R/eval_func.R").read().decode("utf-8")
eval_func_rcode = open("/home/ole/IdeaProjects/darima/darima/R/eval_func.R").read()
robjects.r.source(exprs=rpy2.rinterface.parse(eval_func_rcode), verbose=False)
eval_func=robjects.r['eval_func']


##--------------------------------------------------------------------------------------
# Python version
##--------------------------------------------------------------------------------------
def model_eval(x, xx, period,
               pred, lower, upper, level = 95):
    '''
    Evaluation
    '''

    # Get series data as numpy array (pdf -> numpy array)
    #--------------------------------------
    h = len(xx)
    x = x
    xx = xx
    pred = pred
    lower = lower
    upper = upper

    # Forecasting
    #--------------------------------------
    eval_result = eval_func(x = robjects.FloatVector(x),
                      xx = robjects.FloatVector(xx),
                      period = period,
                      pred = robjects.FloatVector(pred),
                      lower = robjects.FloatVector(lower),
                      upper = robjects.FloatVector(upper),
                      level = level)

    # Extract returns
    #--------------------------------------
    mase = robjects.FloatVector(eval_result.rx2("mase"))
    smape = robjects.FloatVector(eval_result.rx2("smape"))
    msis = robjects.FloatVector(eval_result.rx2("msis"))

    # R object to python object
    #--------------------------------------
    mase = np.array(mase).reshape(h, 1) # h-by-1
    smape = np.array(smape).reshape(h, 1) # h-by-1
    msis = np.array(msis).reshape(h, 1) # h-by-1

    # Out
    #--------------------------------------
    out_np = np.concatenate((mase, smape, msis),1) # h-by-3
    out_pdf = pd.DataFrame(out_np,
                           columns=pd.Index(["mase", "smape", "msis"]))
    out = out_pdf

    if out.isna().values.any():
        warnings.warn("NAs appear in the final output")

    return out
