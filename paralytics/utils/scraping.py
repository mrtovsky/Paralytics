"""Utilities for web scraping"""


from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from ..decorators import force_context_manager


@force_context_manager
class BaseSeleniumBrowser(object):
    """Based on the Selenium package, it automates the use of the
    browser of user's choice."""
    def __init__(self, browser_name, executable_path=None):
        try:
            self.browser = getattr(webdriver, browser_name)()
        except FileNotFoundError:
            print(
                'Executable not found in the PATH. Trying execution with '
                'the path defined in the `executable_path` instead.'
            )
            self.browser = getattr(webdriver, browser_name)(executable_path)

    def __enter__(self):
        return self

    def open_page(self, url, title):
        """Opens page with use of the passed url."""
        try:
            self.browser.get(url)
        except WebDriverException as e:
            print('URL provided is not existing!')
            raise
        if title.lower() not in self.browser.title.lower():
            raise ValueError(
                f'Expected title: "{title}" is not included in the '
                'page title.'
            )

    def __exit__(self, exc_type, exc_value, traceback):
        self.browser.close()
