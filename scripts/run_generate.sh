python generate.py \
--task t2v-1.3B \
--size 832*480 \
--original_ckpt_dir ./weights/Wan2.1-T2V-1.3B \
--finetuned_ckpt_dir ./outputs/run1_3dgs_dissolve/checkpoint-1000 \ 
--sample_shift 8 \
--sample_guide_scale 6 \
--prompt "A small, cute unicorn in a 3D appearance against a black background. The unicorn gradually begins to glow and expand, filling the screen with vibrant colors. As it grows larger, it starts to emit a bright, fiery light. The unicorn then evaporates into a burst of red sparks, creating a dynamic and visually striking effect."


