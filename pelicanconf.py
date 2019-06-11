#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Audrey Chan'
SITENAME = 'Audrey Chan'
SITEURL = 'https://audreymychan.github.io/blog'
TAGLINE = '''Data scientist • Engineer • National athlete
 • Always finding joy in constant improvement and helping
  anyone who needs it'''
PATH = 'content'
STATIC_PATHS = ['images']
TIMEZONE = 'America/Toronto'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10
# set to False for Production, True for Development
if os.environ.get('PELICAN_ENV') == 'DEV':
    RELATIVE_URLS = True
else:
    RELATIVE_URLS = False
