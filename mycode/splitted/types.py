import numpy as np
from pyspark.sql.types import StructField, DoubleType, StringType, StructType, IntegerType

from mycode.splitted.settings import tol

# Model
schema_sdf = StructType([
    StructField('demand', DoubleType(), True),
    StructField('time', StringType(), True)
])

# AR Coefficients
usecoef_ar = ['c0', 'c1'] + ["pi" + str(i+1) for i in np.arange(tol)]
schema_fields = []
for i in usecoef_ar:
    schema_fields.append(StructField(i, DoubleType(), True))

schema_beta = StructType(
    [StructField('par_id', IntegerType(), True),
     StructField('Sig_inv_value', DoubleType(), True)]
    + schema_fields)
