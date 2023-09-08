from pyspark.sql.pandas.functions import PandasUDFType, pandas_udf

from darima.model import darima_model
from mycode.splitted.settings import *
from mycode.splitted.types import schema_beta


@pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
def darima_model_udf(sample_df):
    return darima_model(sample_df=sample_df, Y_name='demand', period=period, tol=tol,
                        order=order, seasonal=seasonal,
                        max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                        max_order=max_order, max_d=max_d, max_D=max_D,
                        allowmean=allowmean, allowdrift=allowdrift, method=method,
                        approximation=approximation, stepwise=stepwise,
                        parallel=parallel, num_cores=num_cores)
