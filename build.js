import fs from "fs";
import path from "path";
import { marked } from "marked";
import matter from "gray-matter";

// Configuration
const POSTS_DIR = "./posts";
const CONTENT_DIR = "./content";
const DIST_DIR = "./dist";
const TEMPLATES_DIR = "./templates";

// Ensure directories exist
if (!fs.existsSync(DIST_DIR)) {
  fs.mkdirSync(DIST_DIR, { recursive: true });
}

if (!fs.existsSync(CONTENT_DIR)) {
  fs.mkdirSync(CONTENT_DIR, { recursive: true });
}

// Read template files
const homeTemplate = fs.readFileSync(
  path.join(TEMPLATES_DIR, "home.html"),
  "utf-8",
);
const blogListTemplate = fs.readFileSync(
  path.join(TEMPLATES_DIR, "blog-list.html"),
  "utf-8",
);
const postTemplate = fs.readFileSync(
  path.join(TEMPLATES_DIR, "post.html"),
  "utf-8",
);
const pageTemplate = fs.readFileSync(
  path.join(TEMPLATES_DIR, "page.html"),
  "utf-8",
);

// Navigation HTML (same across all pages)
const navigation = `
<nav class="main-nav">
  <div class="nav-container">
    <a href="/" class="nav-logo">✨ Andrew Bui</a>
    <div class="nav-links">
      <a href="/">🏠 Home</a>
      <a href="/blog">📝 Blog</a>
      <a href="/projects">🚀 Projects</a>
      <a href="/cv">📄 CV</a>
      <a href="/profile">👤 Profile</a>
    </div>
  </div>
</nav>
`;

// Get all blog posts
function getAllPosts() {
  if (!fs.existsSync(POSTS_DIR)) {
    return [];
  }

  const entries = fs.readdirSync(POSTS_DIR, { withFileTypes: true });
  const postFolders = entries
    .filter((entry) => entry.isDirectory())
    .sort()
    .reverse();

  const posts = postFolders
    .map((folder) => {
      const folderPath = path.join(POSTS_DIR, folder.name);
      const markdownFile = path.join(folderPath, "index.md");

      if (!fs.existsSync(markdownFile)) {
        console.warn(`⚠️  No index.md found in ${folder.name}`);
        return null;
      }

      const content = fs.readFileSync(markdownFile, "utf-8");
      const { data, content: markdown } = matter(content);
      const html = marked(markdown);

      const match = folder.name.match(/^(\d{4}-\d{2}-\d{2})-(.+)$/);
      const date = match ? match[1] : new Date().toISOString().split("T")[0];
      const slug = match ? match[2] : folder.name;

      return {
        title: data.title || slug.replace(/-/g, " "),
        date: data.date || date,
        description: data.description || "",
        slug,
        html,
        folderName: folder.name,
        folderPath,
      };
    })
    .filter((post) => post !== null);

  return posts;
}

// Generate individual blog post pages
function generatePostPages(posts) {
  const blogDir = path.join(DIST_DIR, "blog");
  if (!fs.existsSync(blogDir)) {
    fs.mkdirSync(blogDir, { recursive: true });
  }

  posts.forEach((post) => {
    const postHtml = postTemplate
      .replace(/{{navigation}}/g, navigation)
      .replace(/{{title}}/g, post.title)
      .replace(/{{date}}/g, post.date)
      .replace(/{{content}}/g, post.html);

    const postDistDir = path.join(blogDir, post.slug);
    if (!fs.existsSync(postDistDir)) {
      fs.mkdirSync(postDistDir, { recursive: true });
    }

    fs.writeFileSync(path.join(postDistDir, "index.html"), postHtml);

    // Copy all assets
    const entries = fs.readdirSync(post.folderPath, { withFileTypes: true });
    entries.forEach((entry) => {
      if (entry.isFile() && entry.name !== "index.md") {
        const srcFile = path.join(post.folderPath, entry.name);
        const destFile = path.join(postDistDir, entry.name);
        fs.copyFileSync(srcFile, destFile);
        console.log(`  📎 Copied asset: ${entry.name}`);
      }
    });
  });
}

// Generate blog listing page
function generateBlogListPage(posts) {
  const blogDir = path.join(DIST_DIR, "blog");
  if (!fs.existsSync(blogDir)) {
    fs.mkdirSync(blogDir, { recursive: true });
  }

  const postsList = posts
    .map(
      (post) => `
    <article class="post-preview">
      <h2><a href="/blog/${post.slug}/">${post.title}</a></h2>
      <time datetime="${post.date}">${new Date(post.date).toLocaleDateString(
        "en-US",
        {
          year: "numeric",
          month: "long",
          day: "numeric",
        },
      )}</time>
      ${post.description ? `<p>${post.description}</p>` : ""}
    </article>
  `,
    )
    .join("\n");

  const blogHtml = blogListTemplate
    .replace(/{{navigation}}/g, navigation)
    .replace(/{{posts}}/g, postsList);

  fs.writeFileSync(path.join(blogDir, "index.html"), blogHtml);
}

// Generate home page
function generateHomePage() {
  const homeHtml = homeTemplate.replace(/{{navigation}}/g, navigation);
  fs.writeFileSync(path.join(DIST_DIR, "index.html"), homeHtml);
}

// Generate content pages (CV, Projects, Profile)
function generateContentPages() {
  const pages = ["cv", "projects", "profile"];

  pages.forEach((pageName) => {
    const pageDir = path.join(DIST_DIR, pageName);
    if (!fs.existsSync(pageDir)) {
      fs.mkdirSync(pageDir, { recursive: true });
    }

    const contentFile = path.join(CONTENT_DIR, `${pageName}.md`);
    let content = "";
    let title = pageName.charAt(0).toUpperCase() + pageName.slice(1);

    if (fs.existsSync(contentFile)) {
      const fileContent = fs.readFileSync(contentFile, "utf-8");
      const { data, content: markdown } = matter(fileContent);
      content = marked(markdown);
      title = data.title || title;
    } else {
      // Generate default content
      content = generateDefaultContent(pageName);
    }

    const pageHtml = pageTemplate
      .replace(/{{navigation}}/g, navigation)
      .replace(/{{title}}/g, title)
      .replace(/{{content}}/g, content);

    fs.writeFileSync(path.join(pageDir, "index.html"), pageHtml);
  });
}

// Generate default content for pages
function generateDefaultContent(pageName) {
  const defaults = {
    cv: `
      <div class="cv-section">
        <h2>📄 Curriculum Vitae</h2>
        <p>Download my CV or view online.</p>

        <div class="cv-download">
          <a href="#" class="button">📥 Download PDF</a>
        </div>


    `,
    projects: `
      <div class="projects-section">
        <h2>🚀 My Projects</h2>
        <p>Here are some of the projects I've worked on.</p>

        // <div class="project-card">
        //   <h3>Project Name 1</h3>
        //   <p class="project-meta">🔗 <a href="#">GitHub</a> • 🌐 <a href="#">Live Demo</a></p>
        //   <p>Description of your project. What it does, technologies used, and your role.</p>
        //   <div class="project-tags">
        //     <span class="tag">React</span>
        //     <span class="tag">Node.js</span>
        //     <span class="tag">MongoDB</span>
        //   </div>
        // </div>

        // <div class="project-card">
        //   <h3>Project Name 2</h3>
        //   <p class="project-meta">🔗 <a href="#">GitHub</a> • 🌐 <a href="#">Live Demo</a></p>
        //   <p>Another cool project you've built.</p>
        //   <div class="project-tags">
        //     <span class="tag">Python</span>
        //     <span class="tag">Django</span>
        //     <span class="tag">PostgreSQL</span>
        //   </div>
        // </div>
      </div>
    `,
    profile: `
      <div class="profile-section">
        <h2>👤 About Me</h2>

        <div class="profile-intro">
          <p>Hi! I'm a developer who loves building things and writing about them.</p>
        </div>

        <h3>🌐 Connect With Me</h3>
        <div class="social-links">
          <a href="https://github.com/andrew-bu1" class="social-link github" target="_blank">
            <svg viewBox="0 0 24 24" width="24" height="24">
              <path fill="currentColor" d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub
          </a>

          <a href="https://www.linkedin.com/in/your-profile" class="social-link linkedin" target="_blank">
            <svg viewBox="0 0 24 24" width="24" height="24">
              <path fill="currentColor" d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            LinkedIn
          </a>

          <a href="https://www.facebook.com/your-profile" class="social-link facebook" target="_blank">
            <svg viewBox="0 0 24 24" width="24" height="24">
              <path fill="currentColor" d="M9 8h-3v4h3v12h5v-12h3.642l.358-4h-4v-1.667c0-.955.192-1.333 1.115-1.333h2.885v-5h-3.808c-3.596 0-5.192 1.583-5.192 4.615v3.385z"/>
            </svg>
            Facebook
          </a>
        </div>

        <h3>📧 Contact</h3>
        <p>Feel free to reach out via email: <a href="mailto:your.email@example.com">your.email@example.com</a></p>
      </div>
    `,
  };

  return defaults[pageName] || "<p>Content coming soon...</p>";
}

// Generate CSS
function generateStyles() {
  const css = `
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --primary: #667eea;
  --secondary: #764ba2;
  --accent: #f093fb;
  --success: #4facfe;
  --warning: #f6d365;
  --danger: #fa709a;
  --dark: #2d3748;
  --gray: #718096;
  --light-gray: #e2e8f0;
  --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --card-shadow: 0 10px 30px rgba(102, 126, 234, 0.1);
  --card-shadow-hover: 0 20px 40px rgba(102, 126, 234, 0.2);
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.7;
  color: var(--dark);
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  min-height: 100vh;
}

/* Navigation */
.main-nav {
  background: white;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  position: sticky;
  top: 0;
  z-index: 1000;
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav-logo {
  font-size: 1.5rem;
  font-weight: 800;
  background: var(--bg-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-decoration: none;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-links a {
  color: var(--dark);
  text-decoration: none;
  font-weight: 500;
  transition: all 0.3s;
  padding: 0.5rem 1rem;
  border-radius: 8px;
}

.nav-links a:hover {
  background: var(--bg-gradient);
  color: white;
  transform: translateY(-2px);
}

/* Container */
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

/* Hero Section (Home Page) */
.hero {
  text-align: center;
  padding: 4rem 2rem;
  background: var(--bg-gradient);
  color: white;
  border-radius: 30px;
  margin: 2rem auto;
  max-width: 1000px;
  box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
  animation: pulse 15s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(10px, 10px) scale(1.1); }
}

.hero-content {
  position: relative;
  z-index: 1;
}

.hero h1 {
  font-size: 3.5rem;
  margin-bottom: 1rem;
  text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}

.hero p {
  font-size: 1.3rem;
  margin-bottom: 2rem;
  opacity: 0.95;
}

.hero-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.button {
  display: inline-block;
  padding: 1rem 2rem;
  background: white;
  color: var(--primary);
  text-decoration: none;
  border-radius: 30px;
  font-weight: 600;
  transition: all 0.3s;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.button:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.button.secondary {
  background: transparent;
  color: white;
  border: 2px solid white;
}

.button.secondary:hover {
  background: white;
  color: var(--primary);
}

/* Features Grid (Home Page) */
.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}

.feature-card {
  background: white;
  padding: 2rem;
  border-radius: 20px;
  box-shadow: var(--card-shadow);
  text-align: center;
  transition: all 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: var(--card-shadow-hover);
}

.feature-card h3 {
  color: var(--primary);
  margin: 1rem 0;
  font-size: 1.5rem;
}

.feature-icon {
  font-size: 3rem;
}

/* Page Header */
header.page-header {
  background: var(--bg-gradient);
  color: white;
  padding: 3rem 2rem;
  margin: -2rem -1rem 3rem -1rem;
  border-radius: 0 0 30px 30px;
  box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
  text-align: center;
  position: relative;
  overflow: hidden;
}

header.page-header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
  animation: pulse 15s ease-in-out infinite;
}

header.page-header h1 {
  font-size: 3rem;
  margin-bottom: 0.5rem;
  font-weight: 800;
  text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
  position: relative;
  z-index: 1;
}

header.page-header p {
  font-size: 1.2rem;
  opacity: 0.95;
  position: relative;
  z-index: 1;
}

/* Content Area */
.content {
  background: white;
  padding: 3rem;
  border-radius: 20px;
  box-shadow: var(--card-shadow);
  margin-bottom: 2rem;
}

h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  background: var(--bg-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: 700;
}

h2 {
  font-size: 1.8rem;
  margin-top: 2rem;
  margin-bottom: 1rem;
  color: var(--primary);
  font-weight: 600;
}

h3 {
  font-size: 1.4rem;
  margin-top: 1.5rem;
  margin-bottom: 0.8rem;
  color: var(--secondary);
  font-weight: 600;
}

time {
  display: inline-flex;
  align-items: center;
  color: white;
  background: linear-gradient(135deg, var(--success) 0%, var(--primary) 100%);
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
  box-shadow: 0 4px 10px rgba(79, 172, 254, 0.3);
}

time::before {
  content: '📅';
  margin-right: 0.4rem;
}

/* Blog Post Preview */
.post-preview {
  background: white;
  padding: 2rem;
  margin-bottom: 2rem;
  border-radius: 20px;
  box-shadow: var(--card-shadow);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 2px solid transparent;
  position: relative;
  overflow: hidden;
}

.post-preview::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--bg-gradient);
}

.post-preview:hover {
  transform: translateY(-8px);
  box-shadow: var(--card-shadow-hover);
  border-color: var(--primary);
}

.post-preview h2 {
  margin-top: 0.5rem;
  margin-bottom: 0.8rem;
}

.post-preview a {
  color: var(--dark);
  text-decoration: none;
  font-weight: 600;
  transition: color 0.3s;
}

.post-preview a:hover {
  color: var(--primary);
}

.post-preview p {
  margin-top: 0.8rem;
  color: var(--gray);
  line-height: 1.6;
}

/* Post Content */
article.post-content {
  background: white;
  padding: 3rem;
  border-radius: 20px;
  box-shadow: var(--card-shadow);
  position: relative;
  overflow: hidden;
}

article.post-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 6px;
  background: var(--bg-gradient);
}

article.post-content h1 {
  margin-bottom: 1rem;
}

article.post-content img {
  max-width: 100%;
  height: auto;
  border-radius: 12px;
  margin: 2rem 0;
  display: block;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  transition: transform 0.3s;
}

article.post-content img:hover {
  transform: scale(1.02);
}

article.post-content video {
  max-width: 100%;
  height: auto;
  border-radius: 12px;
  margin: 2rem 0;
  display: block;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

article.post-content p {
  margin: 1.2rem 0;
  font-size: 1.05rem;
}

article.post-content ul, article.post-content ol {
  margin: 1.2rem 0;
  padding-left: 2rem;
}

article.post-content li {
  margin: 0.8rem 0;
  padding-left: 0.5rem;
}

article.post-content li::marker {
  color: var(--primary);
  font-weight: bold;
}

article.post-content blockquote {
  border-left: 5px solid var(--accent);
  padding: 1rem 1.5rem;
  margin: 2rem 0;
  background: linear-gradient(90deg, rgba(240, 147, 251, 0.1) 0%, transparent 100%);
  border-radius: 0 10px 10px 0;
  font-style: italic;
  color: var(--gray);
}

code {
  background: linear-gradient(135deg, #f093fb20 0%, #667eea20 100%);
  color: var(--secondary);
  padding: 0.3rem 0.6rem;
  border-radius: 6px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
  font-weight: 500;
  border: 1px solid var(--light-gray);
}

pre {
  background: var(--dark);
  color: #f8f8f2;
  padding: 1.5rem;
  border-radius: 12px;
  overflow-x: auto;
  margin: 2rem 0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  border: 1px solid rgba(102, 126, 234, 0.3);
}

pre code {
  background: none;
  padding: 0;
  color: inherit;
  border: none;
}

a {
  color: var(--primary);
  text-decoration: none;
  transition: all 0.3s;
  font-weight: 500;
}

a:hover {
  color: var(--secondary);
  text-decoration: underline;
}

.back-link {
  display: inline-flex;
  align-items: center;
  margin-bottom: 1.5rem;
  padding: 0.6rem 1.2rem;
  background: white;
  border-radius: 30px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  transition: all 0.3s;
  border: 2px solid var(--light-gray);
}

.back-link::before {
  content: '←';
  margin-right: 0.5rem;
  font-size: 1.2rem;
  color: var(--primary);
}

.back-link:hover {
  transform: translateX(-5px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
  border-color: var(--primary);
  text-decoration: none;
}

/* CV Specific */
.cv-section {
  padding: 2rem 0;
}

.cv-download {
  margin: 2rem 0;
}

.experience-item {
  margin: 1.5rem 0;
  padding: 1.5rem;
  background: linear-gradient(135deg, #f5f7fa 0%, #e2e8f0 100%);
  border-radius: 15px;
  border-left: 4px solid var(--primary);
}

.experience-item h4 {
  color: var(--primary);
  font-size: 1.3rem;
  margin-bottom: 0.5rem;
}

.company {
  color: var(--gray);
  font-weight: 600;
  margin-bottom: 0.8rem;
}

.skills-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-top: 1rem;
}

.skill-tag {
  background: var(--bg-gradient);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 500;
  font-size: 0.9rem;
  box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
}

/* Projects Specific */
.projects-section {
  padding: 2rem 0;
}

.project-card {
  background: linear-gradient(135deg, #f5f7fa 0%, #e2e8f0 100%);
  padding: 2rem;
  border-radius: 20px;
  margin-bottom: 2rem;
  border-left: 5px solid var(--primary);
  transition: all 0.3s;
}

.project-card:hover {
  transform: translateX(10px);
  box-shadow: -5px 5px 20px rgba(102, 126, 234, 0.2);
}

.project-card h3 {
  margin-top: 0;
  color: var(--primary);
}

.project-meta {
  color: var(--gray);
  margin: 0.5rem 0 1rem 0;
}

.project-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}

.tag {
  background: white;
  color: var(--primary);
  padding: 0.3rem 0.8rem;
  border-radius: 15px;
  font-size: 0.85rem;
  font-weight: 500;
  border: 1px solid var(--primary);
}

/* Profile Specific */
.profile-section {
  padding: 2rem 0;
}

.profile-intro {
  background: linear-gradient(135deg, #f5f7fa 0%, #e2e8f0 100%);
  padding: 2rem;
  border-radius: 20px;
  margin-bottom: 2rem;
  font-size: 1.1rem;
  border-left: 5px solid var(--accent);
}

.social-links {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.social-link {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 1rem 1.5rem;
  background: white;
  border-radius: 15px;
  border: 2px solid var(--light-gray);
  transition: all 0.3s;
  font-weight: 600;
}

.social-link:hover {
  transform: translateY(-3px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
  text-decoration: none;
}

.social-link.github {
  border-color: #333;
  color: #333;
}

.social-link.github:hover {
  background: #333;
  color: white;
}

.social-link.linkedin {
  border-color: #0077b5;
  color: #0077b5;
}

.social-link.linkedin:hover {
  background: #0077b5;
  color: white;
}

.social-link.facebook {
  border-color: #1877f2;
  color: #1877f2;
}

.social-link.facebook:hover {
  background: #1877f2;
  color: white;
}

/* Footer */
footer {
  text-align: center;
  margin-top: 4rem;
  padding: 2rem 1rem;
  color: var(--gray);
  border-top: 2px solid var(--light-gray);
}

footer p {
  font-size: 0.95rem;
  margin: 0.5rem 0;
}

/* Responsive */
@media (max-width: 768px) {
  .nav-container {
    flex-direction: column;
    gap: 1rem;
  }

  .nav-links {
    gap: 1rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .nav-links a {
    padding: 0.4rem 0.8rem;
    font-size: 0.9rem;
  }

  .container {
    padding: 1rem;
  }

  .hero h1 {
    font-size: 2rem;
  }

  .hero p {
    font-size: 1rem;
  }

  header.page-header h1 {
    font-size: 2rem;
  }

  .content {
    padding: 1.5rem;
  }

  .post-preview, article.post-content {
    padding: 1.5rem;
  }

  h1 {
    font-size: 2rem;
  }

  h2 {
    font-size: 1.5rem;
  }

  .features {
    grid-template-columns: 1fr;
  }
}

/* Smooth scroll */
html {
  scroll-behavior: smooth;
}

/* Selection color */
::selection {
  background: var(--accent);
  color: white;
}

::-moz-selection {
  background: var(--accent);
  color: white;
}
`;

  fs.writeFileSync(path.join(DIST_DIR, "style.css"), css);
}

// Copy static assets (PDFs, images, etc.)
function copyStaticAssets() {
  const staticFiles = ["resume.pdf", "cv.pdf"];

  staticFiles.forEach((file) => {
    const srcPath = path.join(".", file);
    const destPath = path.join(DIST_DIR, file);

    if (fs.existsSync(srcPath)) {
      fs.copyFileSync(srcPath, destPath);
      console.log(`  📄 Copied: ${file}`);
    }
  });
}

// Build process
console.log("🚀 Building personal website...");

const posts = getAllPosts();
console.log(`📝 Found ${posts.length} blog posts`);

generateHomePage();
console.log("✅ Generated home page");

generateBlogListPage(posts);
console.log("✅ Generated blog listing page");

generatePostPages(posts);
console.log("✅ Generated blog post pages");

generateContentPages();
console.log("✅ Generated content pages (CV, Projects, Profile)");

generateStyles();
console.log("✅ Generated styles");

copyStaticAssets();
console.log("✅ Copied static assets");

console.log("🎉 Build complete!");
console.log("\n📂 Your site is ready in the 'dist' folder");
console.log("🌐 Deploy it to see your personal website!");
