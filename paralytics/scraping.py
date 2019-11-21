"""Utilities for web scraping."""


try:
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
except ImportError as e:
    _has_selenium = e
else:
    _has_selenium = True


__all__ = [
    "BaseSeleniumBrowser"
]


class BaseSeleniumBrowser(object):
    """Base class for all scrapers built on top of the Selenium package.

    Automates the use of the browser of user's choice by implementing universal
    methods for web scraping with the Selenium package.

    Parameters
    ----------
    browser_name: str {Firefox, Chrome}
        Name of the browser that will be used to the web scraping with the
        Selenium package.

    executable_path: str, optional (default=None)
        Path to the executable file adequate for the browser of your choice.
        If not specified then the only attempt to find an executable is made in
        the PATH.

    Notes
    -----
    It is recommended to decorate all classes inheriting from
    BaseSeleniumBrowser with `paralytics.force_context_manager` function.

    See also
    --------
    paralytics.force_context_manager

    """
    def __init__(self, browser_name, executable_path=None):
        if isinstance(_has_selenium, ImportError):
            raise ImportError(
                "`BaseSeleniumBrowser` requires extra requirements installed. "
                "Reinstall paralytics package with 'browser' extra "
                "specified or install the dependencies directly "
                "from the source."
            ).with_traceback(_has_selenium.__traceback__)
        try:
            self.browser = getattr(webdriver, browser_name)()
        except FileNotFoundError:
            print(
                "Executable not found in the PATH. Trying execution with "
                "the path defined in the `executable_path` instead."
            )
            self.browser = getattr(webdriver, browser_name)(executable_path)

    def __enter__(self):
        return self

    def open_page(self, url, title):
        """Opens page with use of the passed url."""
        try:
            self.browser.get(url)
        except WebDriverException:
            print("URL provided is not existing!")
            raise
        page_title = self.browser.title
        if title.lower() not in page_title.lower():
            raise ValueError(
                "Expected title: '{}' is not included in the "
                "page title: '{}'."
                .format(title, page_title)
            )

    def __exit__(self, exc_type, exc_value, traceback):
        self.browser.close()
