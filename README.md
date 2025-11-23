# My Blog 📝

A simple, elegant blog built with Node.js and deployed automatically with GitHub Actions. Write in Markdown, push to GitHub, and your blog updates automatically!

## 🚀 Quick Start

1. **Clone this repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/andrew-bu1.github.io.git
   cd andrew-bu1.github.io
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Build the site**
   ```bash
   npm run build
   ```

4. **View locally**
   - The generated site will be in the `dist` folder
   - Open `dist/index.html` in your browser

## ✍️ Writing a New Post

### Step 1: Create a New Folder

Create a new folder in the `posts` directory with the naming convention:
```
YYYY-MM-DD-your-post-title
```

For example:
```
posts/2024-01-15-my-daily-thoughts/
```

### Step 2: Create index.md

Inside your new folder, create an `index.md` file with frontmatter:

```markdown
---
title: "Your Post Title"
date: "2024-01-15"
description: "A brief description of your post"
---

# Your Post Title

Your content here...
```

### Step 3: Add Media Files (Optional)

You can include images and videos by placing them in the same folder:

```
posts/2024-01-15-my-daily-thoughts/
├── index.md
├── photo.png
├── diagram.jpg
└── video.mp4
```

Then reference them in your markdown:

```markdown
![My Photo](photo.png)

<video controls width="100%">
  <source src="video.mp4" type="video/mp4">
</video>
```

### Step 4: Commit and Push

```bash
git add .
git commit -m "Add new post: my daily thoughts"
git push
```

GitHub Actions will automatically build and deploy your blog! 🎉

## 📁 Project Structure

```
andrew-bu1.github.io/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow
├── posts/                       # Your blog posts
│   ├── 2024-01-01-welcome/
│   │   └── index.md
│   └── 2024-01-02-example-with-images/
│       ├── index.md
│       ├── example.png
│       └── video.mp4
├── templates/                   # HTML templates
│   ├── index.html              # Home page template
│   └── post.html               # Post page template
├── build.js                     # Build script
├── package.json                 # Dependencies
└── README.md                    # This file
```

## 🎨 Customization

### Change Blog Title and Description

Edit `templates/index.html` and `templates/post.html` to change:
- Blog title
- Tagline/description
- Header styling

### Modify Styles

The CSS is generated in `build.js` in the `generateStyles()` function. Edit that function to customize:
- Colors
- Fonts
- Layout
- Spacing

### Add Features

The build script (`build.js`) is well-commented and easy to extend. You can add:
- Tags/categories
- Search functionality
- RSS feed
- Table of contents
- Comments system

## 🔧 GitHub Setup

### 1. Create Repository

Create a new repository on GitHub named `andrew-bu1.github.io`

### 2. Push Your Code

```bash
git remote add origin https://github.com/YOUR-USERNAME/andrew-bu1.github.io.git
git branch -M main
git push -u origin main
```

### 3. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select **GitHub Actions**

That's it! Your blog will be available at `https://YOUR-USERNAME.github.io`

## 📝 Post Format Reference

### Frontmatter (Required)

```yaml
---
title: "Your Post Title"        # Required
date: "2024-01-15"              # Optional (uses folder date if not provided)
description: "Brief summary"     # Optional (shows on home page)
---
```

### Markdown Features Supported

- **Headings**: `#`, `##`, `###`
- **Bold**: `**text**`
- **Italic**: `*text*`
- **Links**: `[text](url)`
- **Images**: `![alt](image.png)`
- **Lists**: `- item` or `1. item`
- **Code**: `` `code` `` or ` ```language ... ``` `
- **Blockquotes**: `> quote`
- **Tables**: Standard markdown tables
- **HTML**: You can use inline HTML

### Image Best Practices

- Use descriptive filenames
- Optimize images before uploading (keep under 1MB)
- Use JPG for photos, PNG for graphics
- Add alt text for accessibility: `![Description](image.png)`

### Video Best Practices

- Keep videos short or host on YouTube/Vimeo
- Use MP4 format for best compatibility
- Try to keep under 10MB
- Use the HTML5 video tag with controls

## 🚦 Daily Workflow

1. **Morning**: Create a new post folder with today's date
   ```bash
   mkdir posts/2024-01-15-todays-thoughts
   ```

2. **Write**: Create `index.md` and write your content
   ```bash
   nano posts/2024-01-15-todays-thoughts/index.md
   ```

3. **Add media**: Copy any images/videos to the post folder

4. **Test locally** (optional):
   ```bash
   npm run build
   # Open dist/index.html in browser
   ```

5. **Publish**:
   ```bash
   git add .
   git commit -m "Daily post for Jan 15"
   git push
   ```

6. **Done!** Check your blog in ~1 minute

## 🐛 Troubleshooting

### Build fails on GitHub Actions

- Check the Actions tab in your repository for error details
- Make sure all markdown files have proper frontmatter
- Verify `package.json` dependencies are correct

### Images not showing

- Ensure images are in the same folder as `index.md`
- Use relative paths: `![Alt](image.png)` not `![Alt](/posts/.../image.png)`
- Check file extensions match (case-sensitive on Linux)

### Post not appearing

- Folder name must start with `YYYY-MM-DD-`
- Must have an `index.md` file inside
- Frontmatter must have at least a `title` field

### Local build works but GitHub Pages doesn't

- Check GitHub Pages is set to "GitHub Actions" in Settings
- Verify the workflow file is in `.github/workflows/deploy.yml`
- Check workflow permissions in repository settings

## 📚 Resources

- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## 🤝 Contributing

Feel free to fork this project and customize it for your own needs!

## 📄 License

MIT License - feel free to use this for your own blog!

---

**Happy Blogging!** 🎉 Write something awesome every day!
