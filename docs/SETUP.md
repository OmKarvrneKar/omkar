# Setup Instructions for GitHub

This file contains instructions to prepare your project for GitHub after installing Git.

## Prerequisites

1. **Install Git**: Download and install Git from [git-scm.com](https://git-scm.com/)
2. **Create GitHub Account**: Sign up at [github.com](https://github.com) if you don't have one

## Steps to Push to GitHub

### 1. Install Git (if not already installed)
- Download Git from: https://git-scm.com/
- Run the installer with default settings
- Restart your terminal/PowerShell

### 2. Configure Git (first time only)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Initialize Repository and Make First Commit
```bash
# Navigate to your project directory
cd "c:\Users\OmSai\OneDrive\Desktop\New folder"

# Initialize git repository
git init

# Add all files to staging
git add .

# Make your first commit
git commit -m "Initial commit: Titanic data cleaning project"
```

### 4. Create Repository on GitHub
1. Go to [github.com](https://github.com)
2. Click "New repository" or the "+" icon
3. Repository name: `omkar`
4. Description: `Data cleaning and preprocessing for Titanic dataset`
5. Keep it **Public** (or Private if you prefer)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### 5. Connect Local Repository to GitHub
```bash
# Add the remote repository
git remote add origin https://github.com/OmKarvrneKar/omkar.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

### 6. Verify Your Repository
- Go to your GitHub repository URL
- You should see all your files uploaded
- The README.md will be displayed automatically

## Alternative: Using GitHub Desktop
If you prefer a GUI:
1. Download GitHub Desktop from [desktop.github.com](https://desktop.github.com/)
2. Sign in with your GitHub account
3. Click "Add an Existing Repository from your Hard Drive"
4. Select your project folder
5. Publish repository to GitHub

## Project Structure After Setup
```
omkar/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── SETUP.md
│   └── USAGE.md
├── data/
│   ├── titanic.csv
│   └── titanic_cleaned.csv
└── scripts/
    ├── titanic_data_cleaning.py
    └── titanic_model_training.py
```

## Next Steps After Pushing to GitHub
1. Update the GitHub repository description
2. Add topics/tags to your repository
3. Consider adding a license file
4. Star your own repository 😄
5. Share with the community!

## Troubleshooting
- **Git not recognized**: Make sure Git is installed and restart your terminal
- **Permission denied**: Check your GitHub credentials or use personal access token
- **Repository already exists**: Use a different name or delete the existing repository

## Additional Resources
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [GitHub Docs](https://docs.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)