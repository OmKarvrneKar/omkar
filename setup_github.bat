@echo off
REM GitHub Setup Script for Titanic Data Cleaning Project
REM Run this script after installing Git

echo ========================================
echo   TITANIC DATA CLEANING - GITHUB SETUP
echo ========================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/
    echo After installation, restart your terminal and run this script again
    pause
    exit /b 1
)

echo ✓ Git is installed

REM Initialize repository
echo.
echo Initializing Git repository...
git init
if errorlevel 1 (
    echo ERROR: Failed to initialize Git repository
    pause
    exit /b 1
)

echo ✓ Git repository initialized

REM Add all files
echo.
echo Adding files to staging area...
git add .
if errorlevel 1 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)

echo ✓ Files added to staging area

REM Create initial commit
echo.
echo Creating initial commit...
git commit -m "Initial commit: Titanic data cleaning and preprocessing project"
if errorlevel 1 (
    echo ERROR: Failed to create commit
    pause
    exit /b 1
)

echo ✓ Initial commit created

echo.
echo ========================================
echo   SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Copy the repository URL
echo 3. Run the following commands:
echo.
echo    git remote add origin YOUR_GITHUB_URL
echo    git branch -M main
echo    git push -u origin main
echo.
echo Replace YOUR_GITHUB_URL with your actual GitHub repository URL
echo.
echo Example:
echo    git remote add origin https://github.com/OmKarvrneKar/omkar.git
echo    git branch -M main
echo    git push -u origin main
echo.
pause