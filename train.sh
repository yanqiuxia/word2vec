#!/bin/bash
PYTHON_HOME=" /home/eleanor/anaconda3/envs/test_cuda/bin"
APP_HOME="/home/eleanor/yan.qiuxia/PycharmProjects/word2vec"
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_HOME/python -u $APP_HOME/train2.py > $APP_HOME/logs/log_big_data.txt 2>&1 &
