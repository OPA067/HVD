CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=1 \
main_retrieval.py \
--do_eval 1 \
--workers 8 \
--n_display 100 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 32 \
--batch_size_val 32 \
--anno_path DiDeMo \
--video_path DiDeMo/videos \
--datatype didemo \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--split_batch 17 \
--output_dir experiments/DiDeMo \
--init_model experiments/DiDeMo/2025-05-30_03:47:48/pytorch_model.bin.4

