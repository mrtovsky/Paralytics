"""Utilities for web scraping"""


from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from ..decorators import force_context_manager


@force_context_manager
class BaseSeleniumBrowser(object):
    """Based on the Selenium package, it automates the use of the browser of
    user's choice.

    Base class for Selenium scrapers that implements universal methods for
    web scraping with the Selenium package.

    Parameters
    ----------
    browser_name: str {Firefox, Chrome}
        Name of the browser that will be used to the web scraping with the
        Selenium package.

    executable_path: str
        Path to the executable file adequate for the browser of your choice.
        If not specified then the only attempt to find an executable is made in
        the PATH.
    """
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
        except WebDriverException:
            print('URL provided is not existing!')
            raise
        page_title = self.browser.title
        if title.lower() not in page_title.lower():
            raise ValueError(
                'Expected title: "{}" is not included in the '
                'page title: "{}".'.format(title, page_title)
            )

    def __exit__(self, exc_type, exc_value, traceback):
        self.browser.close()
