# Invoke-Expression "conda activate eval-env"
& C:\Users\kztod\Anaconda3\envs\eval-env\python.exe post-ocr-test.py
if($lastexitcode -ne '0')
{
    while($lastexitcode -ne '0')
    {
        Start-Sleep -s 2
        Write-Host "retrying..."
        & C:\Users\kztod\Anaconda3\envs\eval-env\python.exe post-ocr-test.py
    }
}