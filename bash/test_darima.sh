#! /bin/bash

# https://help.aliyun.com/document_detail/28124.html

# Fat executors: one executor per node
# EC=16
# EM=30g

MODEL_DESCRIPTION=$1

# Tiny executors: one executor per core
EC=1
EM=6g

# MODEL_FILE
MODEL_FILE=test_darima1
OUTPATH=/home/student/xiaoqian-darima/darima/result/

# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# for i in 1 {4..100..4} # 1, 5, 10, 15, ... , 100

# for i in {256..4..-4}
tic0=`date +"%Y-%m-%d-%T"`
executors=2
tic=`date +%s`
PYSPARK_PYTHON=/usr/local/bin/python3.7 ARROW_PRE_0_15_IPC_FORMAT=1 spark-submit \
              $DIR/../${MODEL_FILE}.py  \
toc=`date +%s`
runtime=$((toc-tic))
echo ${MODEL_FILE}.NE${executors}.EC${EC} finished, "Time used (s):" $runtime

exit 0;