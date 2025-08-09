CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=1 \
main_retrieval.py \
--do_train 1 \
--workers 1 \
--n_display 100 \
--epochs 10 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 24 \
--batch_size_val 24 \
--anno_path ActivityNet \
--video_path ActivityNet/videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir experiments/ActivityNet

