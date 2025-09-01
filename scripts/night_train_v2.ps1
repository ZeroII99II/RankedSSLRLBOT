$ErrorActionPreference = 'Stop'
while ($true) {
    try {
        python -m src.training.train_v2 --envs 1 --steps 1000000 --ckpt_dir models/checkpoints
        python -m src.inference.export --sb3 --ckpt models/checkpoints/best_sb3.zip --out ssl_policy.ts
        Copy-Item ssl_policy.ts models/exported/ssl_policy.ts -Force
        Start-Sleep -Seconds 5
    }
    catch {
        Write-Host "[night_train] restart due to error: $_"
        Start-Sleep -Seconds 5
    }
}
