@echo off
REM Set the Python interpreter path
set pint=C:\Users\souvik\Downloads\MMFUSION_FINAL-main\myenv\Scripts\python.exe

REM Set paths to other variables
set exp=.\experiments\ec_example_phase2.yaml
set ckpt=.\ckpt\early_fusion_localization.pth
set INPUT_DIR=C:\Users\souvik\Downloads\MMFUSION_FINAL-main\data\photos

REM Get last word of INPUT_DIR path
for %%I in ("%INPUT_DIR%") do set "last_word=%%~nxI"

REM Run scripts using Python interpreter
"%pint%" inference.py --path "%INPUT_DIR%" --score_file "%last_word%"
"%pint%" accuracy_MM.py --score_file "%last_word%"


