---
title: "How to Add Images and Videos to Your Posts"
date: "2024-01-02"
description: "Learn how to include media files in your blog posts"
---

# Adding Images and Videos 📸

This post demonstrates how to include images and videos in your blog posts!

## Adding Images

To add an image to your post, simply place the image file in the same folder as your `index.md` file, then reference it using standard Markdown syntax:

```markdown
![Alt text](image-name.png)
```

### Example

You can add images inline with your text. For example, here's a placeholder for an image:

![Example Image](example.png)

You can also add a caption by putting text right after the image:

*Caption: This is where your image would appear*

## Adding Videos

Videos work the same way! Place your video file (mp4, webm, etc.) in the post folder and embed it using HTML:

```html
<video controls width="100%">
  <source src="your-video.mp4" type="video/mp4">
  Your browser doesn't support videos.
</video>
```

### Video Example

<video controls width="100%" style="max-width: 100%; border-radius: 8px;">
  <source src="example-video.mp4" type="video/mp4">
  Your browser doesn't support the video tag.
</video>

*Caption: This is where your video would appear*

## Folder Structure

Your post folder should look like this:

```
posts/2024-01-02-example-with-images/
├── index.md
├── image1.png
├── image2.jpg
└── video.mp4
```

## Tips

1. **Use descriptive names** for your files (e.g., `sunset-photo.jpg` instead of `IMG_001.jpg`)
2. **Optimize your images** before uploading to keep your site fast
3. **Keep videos short** or consider hosting large videos on YouTube/Vimeo and embedding them
4. **Use alt text** for images to improve accessibility

## Best Practices

- **Images**: JPG for photos, PNG for graphics with transparency
- **Video formats**: MP4 is most compatible across browsers
- **File sizes**: Try to keep images under 1MB and videos under 10MB
- **Dimensions**: Resize images to appropriate dimensions before uploading

That's it! Just drop your media files in the post folder and reference them in your markdown. The build script will automatically copy them to the right place when deploying.

Happy blogging! 🚀