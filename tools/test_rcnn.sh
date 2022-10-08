export CUDA_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
python test_rcnn.py --train_mode rcnn --batch_size 10 --epochs 100 > log.txt