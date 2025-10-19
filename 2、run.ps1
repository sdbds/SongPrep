# Parameters
$input_wav = "Happy Maria!.mp3"  # Path to input file (required)
$qwen_ckpt = "SongPrep-7B/"  # Path to Qwen checkpoint
$ssl_ckpt = "SongPrep-7B/muencoder.pt"  # Path to SSL checkpoint
$codec_ckpt = "SongPrep-7B/mucodec.safetensors"  # Path to codec checkpoint

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================

# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    ./.venv/bin/activate.ps1
}

# Set environment variables
$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG = "1"

# Build command arguments
$py_args = @("-i", $input_wav, "-q", $qwen_ckpt, "-s", $ssl_ckpt, "-c", $codec_ckpt)

# Run the Python script
Write-Output "Running run.py with arguments: $($py_args -join ' ')"
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 "run.py" @py_args