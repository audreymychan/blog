# blog
Source code for my personal blog website, https://audreymychan.com. This website is powered by Pelican, a static site generator written in Python.

## Usage
1. In the **jupyter** directory, create new blog post as a Jupyter notebook (i.e. blogpost3.ipynb)

2. In **blog** directory, execute following commands in terminal:
- `python blog.py convert <blogpostname>` (i.e. blogpostname would be blogpost3)
- `python blog.py local` allows spinning up blog website for viewing and checking on localhost:8000
- `python blog.py publish` once ready, this allows publishing of blog website to https://audreymychan.com (or audreymychan.github.io/blog)
