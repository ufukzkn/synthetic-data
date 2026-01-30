@echo off
REM Quick training launcher for RTX 5060
REM Run this script to start training

echo ====================================
echo  Curve Segmentation Training
echo  RTX 5060 Optimized
echo ====================================
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Check CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo Starting training...
echo.

REM Training parameters optimized for RTX 5060 (16GB VRAM assumed)
python train_curve_segmentation.py ^
    --epochs 100 ^
    --batch_size 12 ^
    --img_size 512 ^
    --lr 0.0001 ^
    --samples_per_epoch 500 ^
    --save_dir checkpoints

echo.
echo ====================================
echo Training complete!
echo Model saved to: checkpoints/best_model.pt
echo ====================================
pause
