#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get today's date
DATE=$(date +%Y-%m-%d)

# Ask for post title
echo -e "${BLUE}📝 Create a new blog post${NC}"
echo ""
read -p "Enter post title (e.g., 'My Daily Thoughts'): " TITLE

# Convert title to slug (lowercase, replace spaces with hyphens)
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g' | sed 's/[^a-z0-9-]//g')

# Create folder name
FOLDER_NAME="${DATE}-${SLUG}"
FOLDER_PATH="posts/${FOLDER_NAME}"

# Check if folder already exists
if [ -d "$FOLDER_PATH" ]; then
    echo -e "${YELLOW}⚠️  Folder already exists: ${FOLDER_PATH}${NC}"
    read -p "Do you want to overwrite? (y/n): " CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Create the folder
mkdir -p "$FOLDER_PATH"

# Ask for description
read -p "Enter a brief description (optional): " DESCRIPTION

# Create the index.md file
cat > "$FOLDER_PATH/index.md" << EOF
---
title: "$TITLE"
date: "$DATE"
description: "$DESCRIPTION"
---

# $TITLE

Write your content here...

## Section 1

Your thoughts go here.

## Section 2

More content...

---

*Written on $(date +"%B %d, %Y")*
EOF

echo ""
echo -e "${GREEN}✅ Post created successfully!${NC}"
echo ""
echo -e "${BLUE}📁 Location:${NC} $FOLDER_PATH/index.md"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit the post: nano $FOLDER_PATH/index.md"
echo "2. Add images/videos: cp your-image.png $FOLDER_PATH/"
echo "3. Build locally: npm run build"
echo "4. Publish: git add . && git commit -m 'New post: $TITLE' && git push"
echo ""
echo -e "${GREEN}Happy writing! ✍️${NC}"
