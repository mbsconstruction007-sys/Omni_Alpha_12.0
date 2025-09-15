Write-Host "🤖 CLAUDE-GITHUB COLLABORATION TOOL" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

function Send-CodeToGitHub {
    Write-Host "`n📝 Paste the code from Claude (press Enter twice when done):" -ForegroundColor Yellow
    $code = @()
    while ($true) {
        $line = Read-Host
        if ([string]::IsNullOrEmpty($line)) {
            $lastLine = Read-Host
            if ([string]::IsNullOrEmpty($lastLine)) { break }
            $code += ""
            $code += $lastLine
        } else {
            $code += $line
        }
    }
    
    $filename = Read-Host "`n📁 Enter filename (e.g., backend/app/new_feature.py)"
    $message = Read-Host "💬 Enter commit message"
    
    # Create directory if needed
    $dir = Split-Path $filename -Parent
    if ($dir -and !(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    
    # Save file
    $code -join "`n" | Out-File -FilePath $filename -Encoding UTF8
    
    # Git operations
    git add $filename
    git commit -m "$message"
    git push origin main
    
    Write-Host "`n✅ SUCCESS! Code pushed to GitHub!" -ForegroundColor Green
    Write-Host "📎 View at: https://github.com/mbsconstruction007-sys/Omni_Alpha_12.0/blob/main/$filename" -ForegroundColor Cyan
}

function Get-CodeForClaude {
    $filename = Read-Host "`n📁 Enter filename to read"
    
    if (Test-Path $filename) {
        $content = Get-Content $filename -Raw
        Write-Host "`n📄 Content of $filename:" -ForegroundColor Cyan
        Write-Host "====== COPY BELOW TO CLAUDE ======" -ForegroundColor Yellow
        Write-Host $content
        Write-Host "====== END OF CONTENT ======" -ForegroundColor Yellow
        
        $content | Set-Clipboard
        Write-Host "`n✅ Content copied to clipboard!" -ForegroundColor Green
    } else {
        Write-Host "❌ File not found: $filename" -ForegroundColor Red
    }
}

function Show-Status {
    Write-Host "`n📊 Repository Status:" -ForegroundColor Cyan
    git status --short
    Write-Host "`n📜 Recent Commits:" -ForegroundColor Cyan
    git log --oneline -5
}

# Main Menu
while ($true) {
    Write-Host "`n🎯 Choose Action:" -ForegroundColor Yellow
    Write-Host "1. Send Claude's code to GitHub"
    Write-Host "2. Get code from GitHub for Claude"
    Write-Host "3. Show repository status"
    Write-Host "4. Pull latest changes"
    Write-Host "5. Exit"
    
    $choice = Read-Host "`nSelect (1-5)"
    
    switch ($choice) {
        "1" { Send-CodeToGitHub }
        "2" { Get-CodeForClaude }
        "3" { Show-Status }
        "4" { 
            git pull origin main
            Write-Host "✅ Repository updated!" -ForegroundColor Green
        }
        "5" { 
            Write-Host "👋 Goodbye!" -ForegroundColor Cyan
            break 
        }
        default { Write-Host "❌ Invalid choice" -ForegroundColor Red }
    }
    
    if ($choice -eq "5") { break }
}
