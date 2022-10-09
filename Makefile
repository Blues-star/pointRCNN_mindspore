clean:
	find .  -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -r
	find . -name "*.egg" | xargs rm -f
	find . -name "build" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name "*.npy" | xargs rm -f
	find . -name "*.so" | xargs rm -f

build:
	bash build_and_install.sh

testeval:
	export CUDA_LAUNCH_BLOCKING=1
	export LD_LIBRARY_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib:$LD_LIBRARY_PATH
	cd tools/ && python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../PointRCNN.ckpt --batch_size 1 --eval_mode rcnn && cd ..

test_network:
	bash tests/test.sh