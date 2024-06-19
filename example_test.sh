pint="/home/vikrant/Downloads/MMFUSION-github_main/myenv/bin/python"
exp="./experiments/ec_example_phase2.yaml"
ckpt="./ckpt/early_fusion_localization.pth"



INPUT_DIR=/home/vikrant/Downloads/MMFUSION-github_main/data/photos
pwd
last_word=$(basename "$INPUT_DIR")
$pint inference.py --path ${INPUT_DIR} --score_file ${last_word}
$pint accuracy_MM.py --score_file ${last_word}
