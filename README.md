# Submission QWRX21

All code is in the /Solution/ directory.
To recreate the final output video:
1. `cd Solution; ./get_cgan_repo.sh`
2. `python Q2.3_apply.py -i <input video> -o <working and output directory>`
3. The results will be in `<working and output directory>/output`

Figures used in the report can be generated using scripts found in `/Solution/Analysis`.
NOTE. these scripts have dataset requirements and model checkpoint requirements.
Datasets can be recreated using by running:
1. `mkdir Dataset\Train` and place training videos in here
2. `Scripts/video_to_images.py`
3. `Q2.1_make_background_dataset.py`
4. `Q1.1_apply_human_patch_extract.py`
5. `Q1.2_apply_pose_estimation.py`

However, the movie to game model checkpoints have not been included in the submission, so some scripts will fail.
