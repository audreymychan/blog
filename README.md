# blog
Source code for my personal blog website, https://audreymychan.com.
This website is powered by Pelican, a static site generator written in Python.

## Usage
1. In the **jupyter** directory, create new blog post as a Jupyter notebook (i.e. blog_post_3.ipynb)

2. In **blog** directory, execute following commands in terminal:
- `python blog.py convert <blog_post_name>` (i.e. blog_post_name would be blog_post_3)
- `python blog.py local` allows spinning up blog website for viewing and checking on localhost:8000
- Once happy with the changes and content, `python blog.py publish` publishes blog website to https://audreymychan.com (or audreymychan.github.io/blog)
