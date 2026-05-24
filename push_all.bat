@echo off
git push origin main
git push hf main
git push ms main:master
echo.
echo === Pushed to GitHub, HF, and ModelScope ===
echo === 提醒: MS 创空间需去网页点 "上线" 才会重新部署; HF 自动重建 ===
pause