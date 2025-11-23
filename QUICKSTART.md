# 🚀 Quick Start Guide

Welcome! Let's get your blog up and running in 5 minutes.

## Step 1: Install Dependencies

```bash
npm install
```

This installs `marked` (Markdown parser) and `gray-matter` (frontmatter parser).

## Step 2: Test the Build

```bash
npm run build
```

You should see:
```
🚀 Building blog...
📝 Found 2 posts
  📎 Copied asset: example.png
  📎 Copied asset: example-video.mp4
✅ Build complete!
```

Open `dist/index.html` in your browser to see your blog locally!

## Step 3: Write Your First Post

Create a new folder in `posts/`:

```bash
mkdir posts/2024-01-15-my-first-post
```

Create `posts/2024-01-15-my-first-post/index.md`:

```markdown
---
title: "My First Post"
date: "2024-01-15"
description: "This is my very first blog post!"
---

# Hello World!

This is my first blog post. I'm excited to start writing!

## What I'll Write About

- My daily thoughts
- Things I'm learning
- Cool projects I'm working on

Let's go! 🚀
```

Build again to see it:

```bash
npm run build
```

## Step 4: Add Images or Videos (Optional)

Just drop files in your post folder:

```bash
# Copy an image to your post folder
cp ~/Pictures/photo.jpg posts/2024-01-15-my-first-post/
```

Then reference it in your markdown:

```markdown
![My Photo](photo.jpg)
```

For videos:

```html
<video controls width="100%">
  <source src="video.mp4" type="video/mp4">
</video>
```

## Step 5: Deploy to GitHub

### First Time Setup

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial blog setup"

# Add your GitHub repository
git remote add origin https://github.com/YOUR-USERNAME/andrew-bu1.github.io.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select **GitHub Actions**

Wait ~1 minute and your blog will be live at:
```
https://YOUR-USERNAME.github.io
```

## Daily Workflow

Every day, just:

1. **Create a new folder** with today's date:
   ```bash
   mkdir posts/$(date +%Y-%m-%d)-daily-thoughts
   ```

2. **Write your post**:
   ```bash
   nano posts/$(date +%Y-%m-%d)-daily-thoughts/index.md
   ```

3. **Add images/videos** if needed (just copy them to the folder)

4. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Daily post"
   git push
   ```

That's it! Your blog updates automatically! 🎉

## Tips

- **Folder naming**: Must start with `YYYY-MM-DD-` (e.g., `2024-01-15-post-title`)
- **File naming**: Must have `index.md` inside each post folder
- **Frontmatter**: At minimum, include a `title` in the frontmatter
- **Images**: Place them in the same folder as `index.md` and use relative paths
- **Build time**: GitHub Actions takes about 30-60 seconds to deploy

## Troubleshooting

**Q: My post doesn't show up**
- Check folder name starts with `YYYY-MM-DD-`
- Make sure there's an `index.md` file inside
- Verify frontmatter has at least a `title`

**Q: Images don't load**
- Use relative paths: `![Alt](image.png)` not absolute paths
- Check file is in the same folder as `index.md`
- File extensions are case-sensitive

**Q: Build fails**
- Run `npm run build` locally to see the error
- Check your markdown syntax
- Ensure frontmatter is valid YAML

Need help? Check `README.md` for more details!

Happy blogging! ✍️