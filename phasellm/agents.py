"""
Agents to help with workflows.
"""

import re
import os
import sys
import time
import docker
import smtplib
import requests
import subprocess
import feedparser

from queue import Queue

from io import StringIO

from pathlib import Path

from warnings import warn

from threading import Thread

from functools import partial

from bs4 import BeautifulSoup

from dataclasses import dataclass

from abc import ABC, abstractmethod

from fake_useragent import UserAgent

from contextlib import contextmanager

from datetime import datetime, timedelta

from playwright.sync_api import sync_playwright

from docker import DockerClient
from docker.models.containers import Container, ExecResult

from typing import Generator, Union, Dict, List, Optional, NamedTuple, Callable

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .exceptions import LLMCodeException


class Agent(ABC):

    @abstractmethod
    def __init__(self, name: str = ""):
        """
        Abstract class for an agent.

        Args:
            name: The name of the agent.

        """
        self.name = name

    def __repr__(self):
        return f"Agent(name='{self.name}')"


@contextmanager
def stdout_io(stdout=None):
    """
    Used to hijack printing to screen so we can save the Python code output for the LLM (or any other arbitrary code).

    Args:
        stdout: The stdout to use.

    """
    pass


class CodeExecutionAgent(Agent):

    def __init__(self, name: str = ""):
        """
        Creates a new CodeExecutionAgent.

        This agent is NOT sandboxed and should only be used for trusted code.

        Args:
            name: The name of the agent.

        """
        super().__init__(name=name)

    def __repr__(self):
        return f"CodeExecutionAgent(name={self.name})"

    @staticmethod
    def execute_code(code: str, globals=None, locals=None) -> str:
        """
        Executes arbitrary Python code and saves the output (or error!) to a variable.

        Returns the variable and a boolean (is_error) depending on whether an error took place.

        Args:
            code: Python code to execute.
            globals: python globals
            locals: python locals

        Returns:
            The logs from the code execution.

        """
        pass


class ExecCommands(NamedTuple):
    requirements: str
    python: str


class SandboxedCodeExecutionAgent(Agent):
    CODE_FILENAME = "sandbox_code.py"

    def __init__(
        self,
        name: str = "",
        docker_image: str = "python:3",
        scratch_dir: Union[Path, str] = None,
        module_package_mappings: Dict[str, str] = None,
    ):
        """
        Creates a new SandboxedCodeExecutionAgent.

        This agent is for executing arbitrary code in a sandboxed environment. We choose to use docker for this, so if
        you're running this code, you'll need to have docker installed and running.

        Examples:
            >>> from typing import Generator
            >>> from phasellm.agents import SandboxedCodeExecutionAgent

            Managing the docker client yourself:
                >>> agent = SandboxedCodeExecutionAgent()
                >>> logs = agent.execute_code('print("Hello World!")')
                >>> for log in logs:
                ...     print(log)
                Hello World!
                >>> agent.close()

            Using the context manager:
                >>> with SandboxedCodeExecutionAgent() as agent:
                ...     logs: Generator = agent.execute_code('print("Hello World!")')
                ...     for log in logs:
                ...         print(log)
                Hello World!

            Code with custom packages is possible! Note that the package must exist in the module_package_mappings dictionary:
                >>> module_package_mappings = {
                ...     "numpy": "numpy"
                ...}
                >>> with SandboxedCodeExecutionAgent(module_package_mappings=module_package_mappings) as agent:
                ...     logs = agent.execute_code('import numpy as np; print(np.__version__)')
                ...     for log in logs:
                ...         print(log)
                1.24.3

            Disable log streaming (waits for code to finish executing before returning logs):
                >>> with SandboxedCodeExecutionAgent() as agent:
                ...     logs = agent.execute_code('print("Hello World!")', stream=False)
                ...     print(logs)
                Hello World!

            Custom docker image:
                >>> with SandboxedCodeExecutionAgent(docker_image='python:3.7') as agent:
                Hello World!

            Custom scratch directory:
                >>> with SandboxedCodeExecutionAgent(scratch_dir='my_dir') as agent:

            Stop container after each call to agent.execute_code()
                >>> with SandboxedCodeExecutionAgent() as agent:
                ...     logs = agent.execute_code('print("Hello 1")', auto_stop_container=True)
                ...     assert agent._container is None
                ...     logs = agent.execute_code('print("Hello 2")', auto_stop_container=True)
                ...     assert agent._container is None
        Args:
            name: Name of the agent.
            docker_image: Docker image to use for the sandboxed environment.
            scratch_dir: Scratch directory to use for copying files (bind mounting) to the sandboxed environment.
            module_package_mappings: Dictionary of module to package mappings. This is used to determine
            which packages are allowed to be installed in the sandboxed environment.

        """
        super().__init__(name=name)

        if module_package_mappings is None:
            module_package_mappings = {
                "numpy": "numpy",
                "pandas": "pandas",
                "scipy": "scipy",
                "sklearn": "scikit-learn",
                "matplotlib": "matplotlib",
                "seaborn": "seaborn",
                "statsmodels": "statsmodels",
                "tensorflow": "tensorflow",
                "torch": "torch",
            }
        self.module_package_mappings = module_package_mappings

        self.docker_image = docker_image

        if scratch_dir is None:
            scratch_dir = f".tmp/sandboxed_code_execution"
        self.scratch_dir = scratch_dir

        # Pre-compile regexes for performance (helps if executing code in a loop).
        self._module_regex = re.compile(
            r"(?:(?<=^import\s)|(?<=^from\s))\w+", flags=re.MULTILINE
        )

        # Create the docker client.
        self._client: DockerClient = docker.from_env()
        self._ping_client()

        # Get the docker image.
        self._client.images.pull(self.docker_image)

        # Placeholder for the container.
        self._container: Optional[Container] = None

    def __repr__(self):
        return (
            f"SandboxedCodeExecutionAgent("
            f"name={self.name}, "
            f"docker_image={self.docker_image}, "
            f"scratch_dir={self.scratch_dir})"
        )

    def __enter__(self):
        """
        Runs When entering the context manager.

        Returns:
            SandboxedCodeExecutionAgent()

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Runs when exiting the context manager.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.

        """
        self.close()

    def _ping_client(self) -> None:
        """
        Pings the docker client to make sure it's running.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        """
        pass

    def _create_scratch_dir(self) -> None:
        """
        Creates the scratch directory if it doesn't exist.
        """
        pass

    def _write_code_file(self, code: str) -> None:
        """
        Writes the code to a file in the scratch directory.

        Args:
            code: The code string to write to the file.

        Returns:

        """
        pass

    def _write_requirements_file(self, packages: List[str]) -> None:
        """
        Writes a requirements.txt file to the scratch directory.

        Args:
            packages: List of packages to write to the requirements.txt file.

        Returns:

        """
        pass

    def _modules_to_packages(self, code: str) -> List[str]:
        """
        Scans the code for modules and maps them to a package. If no package is specified in the mapping whitelist,
        then the package is ignored.

        Args:
            code: The code to scan for modules.

        Returns:
            A list of packages to install in the sandboxed environment.

        """
        pass

    def _prep_commands(self, packages: List[str]) -> ExecCommands:
        """
        Prepares the commands to be run in the docker container.

        Args:
            packages: List of packages to install in the docker container.

        Returns:
            A tuple containing the requirements command and the python command in the form
            (requirements_command, python_command).

        """
        pass

    @staticmethod
    def _handle_exec_errors(output: str, exit_code: int, code: str) -> None:
        """
        Handles errors that occur during code execution.

        Args:
            output: The output of the code execution.
            exit_code: The exit code of the code execution.
            code: The code that was executed.

        Returns:

        """
        pass

    def _execute(self, code: str, auto_stop_container: bool) -> Generator:
        """
        Starts the container, installs packages defined in the code (if they are provided in the
        module_package_mappings), and executes the provided code inside the container.

        Args:
            code: The code string to execute.
            auto_stop_container: Whether to automatically stop the container after execution.

        Returns:
            A Generator that yields the stdout and stderr of the code execution.

        """
        pass

    def close(self) -> None:
        """
        Stops all containers and closes client sessions. This should be called when you're done using the agent.

        This method automatically runs when exiting the context manager. If you do not use a context manager, you
        should call this method manually.

        Returns:

        """
        pass

    def start_container(self) -> None:
        """
        Starts the docker container.

        Returns:

        """
        pass

    def stop_container(self) -> None:
        """
        Stops the docker container and removes it, if it exists.

        Returns:

        """
        pass

    def execute_code(
        self, code: str, stream: bool = True, auto_stop_container: bool = False
    ) -> Union[str, Generator]:
        """
        Executes the provided code inside a sandboxed container.

        Args:
            code: The code string to execute.
            stream: Whether to stream the output of the code execution.
            auto_stop_container: Whether to automatically stop the container after the code execution.

        Returns:
            A string output of the whole code execution stdout and stderr if stream is False, otherwise a Generator
            that yields the stdout and stderr of the code execution.

        """
        pass


class EmailSenderAgent(Agent):

    def __init__(
        self,
        sender_name: str,
        smtp: str,
        sender_address: str,
        password: str,
        port: int,
        name: str = "",
    ):
        """
        Create an EmailSenderAgent.

        Sends emails via an SMPT server.

        Args:
            sender_name: Name of the sender (i.e., "Wojciech")
            smtp: The smtp server (e.g., smtp.gmail.com)
            sender_address: The sender's email address
            password: The password for the email account
            port: The port used by the SMTP server
            name: The name of the agent (optional)

        """

        super().__init__(name=name)
        self.sender_name = sender_name
        self.smtp = smtp
        self.sender_address = sender_address
        self.password = password
        self.port = port

    def __repr__(self):
        return f"EmailSenderAgent(name={self.name})"

    def sendPlainEmail(self, recipient_email: str, subject: str, content: str) -> None:
        """
        DEPRECATED: see send_plain_email

        Args:
            recipient_email: The person receiving the email
            subject: Email subject
            content: The plain text context for the email

        """
        pass

    def send_plain_email(
        self, recipient_email: str, subject: str, content: str
    ) -> None:
        """
        Sends an email encoded as plain text.

        Args:
            recipient_email: The person receiving the email
            subject: Email subject
            content: The plain text context for the email

        """
        pass


class NewsSummaryAgent(Agent):

    def __init__(self, apikey: str = None, name: str = ""):
        """
        Create a NewsSummaryAgent.

        Takes a query, calls the API, and gets news articles.

        Args:
            apikey: The API key for newsapi.org
            name: The name of the agent (optional)

        """
        super().__init__(name=name)
        self.apikey = apikey

    def __repr__(self):
        return f"NewsSummaryAgent(name={self.name})"

    def getQuery(
        self,
        query: str,
        days_back: int = 1,
        include_descriptions: bool = True,
        max_articles: int = 25,
    ) -> str:
        """
        DEPRECATED: see get_query

        Args:
            query: What keyword to look for in news articles
            days_back: How far back we go with the query
            include_descriptions: Will include article descriptions as well as titles; otherwise only titles
            max_articles: How many articles to include in the summary

        Returns:
            A news summary string

        """
        pass

    def get_query(
        self,
        query: str,
        days_back: int = 1,
        include_descriptions: bool = True,
        max_articles: int = 25,
    ) -> str:
        """
        Gets all articles for a query for the # of days back. Returns a String with all the information so that an LLM
        can summarize it. Note that obtaining too many articles will likely cause an issue with prompt length.

        Args:
            query: What keyword to look for in news articles
            days_back: How far back we go with the query
            include_descriptions: Will include article descriptions as well as titles; otherwise only titles
            max_articles: How many articles to include in the summary

        Returns:
            A news summary string

        """
        pass


class WebpageAgent(Agent):

    def __init__(self, name: str = ""):
        """
        Create a WebpageAgent.

        This agent helps you scrape webpages.

        Examples:
            >>> from phasellm.agents import WebpageAgent

            Use default parameters:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection')

            Keep html tags:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', text_only=False, body_only=False)

            Keep html tags, but only return body content:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', text_only=False, body_only=True)

            Use a headless browser to enable scraping of dynamic content:
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', text_only=False, body_only=True,
                ...                     use_browser=True)

            Pass custom headers:
                >>> agent = WebpageAgent()
                >>> headers = {'Example': 'header'}
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', headers=headers)

            Wait for a selector to load (useful for dynamic content, only works when use_browser=True):
                >>> agent = WebpageAgent()
                >>> text = agent.scrape('https://10millionsteps.com/ai-inflection', use_browser=True,
                ...                     wait_for_selector='#dynamic')

        Args:
            name: The name of the agent (optional)

        """
        super().__init__(name=name)

        self.session = requests.Session()

    def __repr__(self):
        return f"WebpageAgent(name={self.name})"

    @staticmethod
    def _validate_url(url: str) -> None:
        """
        This method validates that a url can be used by the agent.
        """
        pass

        # TODO consider adding more validations.

    @staticmethod
    def _handle_errors(res: requests.Response) -> None:
        """
        This method handles errors that occur during a request.

        Args:
            res: The response from the request.

        """
        pass

    @staticmethod
    def _parse_html(html: str, text_only: bool = True, body_only: bool = False) -> str:
        """
        This method parses the given html string.

        Args:
            html: The html to parse.
            text_only: If True, only the text of the webpage is returned. If False, the entire HTML is returned.
            body_only: If True, only the body of the webpage is returned. If False, the entire HTML is returned.

        Returns:
            The string containing the webpage text or html.

        """
        pass

    @staticmethod
    def _prep_headers(headers: Dict = None) -> Dict:
        """
        This method prepares the headers for a request. It fills in missing headers with default values. It also
        adds a fake user agent to reduce the likelihood of being blocked.

        Args:
            headers: The headers to use for the request.

        Returns:
            The headers to use for the request.
        """
        pass

    def _scrape_html(self, url: str, headers: Dict = None) -> str:
        """
        This method scrapes a webpage and returns a string containing the html of the webpage.

        Args:
            url: The URL of the webpage to scrape.
            headers: A dictionary of headers to use for the request.

        Returns:
            A string containing the html of the webpage.

        """
        pass

    @staticmethod
    def _scrape_html_and_js(
        url: str, headers: Dict, wait_for_selector: str = None
    ) -> str:
        """
        This method scrapes a webpage and returns a string containing the html of the webpage. It uses a headless
        browser to render the webpage and execute javascript.

        Args:
            url: The URL of the webpage to scrape.
            headers: A dictionary of headers to use for the request.
            wait_for_selector: The selector to wait for before returning the HTML. Useful for when you know something
            should be on the page, but it is not there yet since it needs to be rendered by javascript.

        Returns:
            A string containing the html of the webpage.

        """
        pass

    def scrape(
        self,
        url: str,
        headers: Dict = None,
        use_browser: bool = False,
        wait_for_selector: str = None,
        text_only: bool = True,
        body_only: bool = True,
    ) -> str:
        """
        This method scrapes a webpage and returns a string containing the html or text of the webpage.

        Args:
            url: The URL of the webpage to scrape.
            headers: A dictionary of headers to use for the request.
            use_browser: If True, the webpage is rendered using a headless browser, allowing javascript to run and
                hydrate the page. If False, the webpage is scraped as-is.
            wait_for_selector: The selector to wait for before returning the HTML. Useful for when you know something
                should be on the page, but it is not there yet since it needs to be rendered by javascript. Only used when
                use_browser is True.
            text_only: If True, only the text of the webpage is returned. If False, the entire HTML is returned.
            body_only: If True, only the body of the webpage is returned. If False, the entire HTML is returned.

        Returns:
            A string containing the text of the webpage.

        """
        pass


@dataclass
class WebSearchResult:
    """
    This dataclass represents a single search result.
    """

    title: str
    url: str
    description: str
    content: str


class WebSearchAgent(Agent):

    def __init__(
        self,
        name: str = "",
        api_key: str = None,
        rate_limit: float = 1,
        text_only: bool = True,
        body_only: bool = True,
        use_browser: bool = False,
        wait_for_selector: str = None,
    ):
        """
        Create a WebSearchAgent.

        This agent helps you search the web using a web search API. Currently, the agent supports Google and Brave.

        Examples:
            >>> from phasellm.agents import WebSearchAgent

            Search with Google:
                >>> agent = WebSearchAgent(
                ...     name='Google Search Agent',
                ...     api_key='YOUR_API_KEY'
                ... )
                >>> results = agent.search_google(
                ...     query='test'
                ...     custom_search_engine_id='YOUR_CUSTOM_SEARCH_ENGINE_ID'
                ... )

            Search with Brave:
                >>> agent = WebSearchAgent(
                ...     name='Brave Search Agent',
                ...     api_key='YOUR_API_KEY'
                ... )
                >>> results = agent.search_brave(query='test')

            Iterate over the results:
                >>> for result in results:
                ...     print(result.title)
                ...     print(result.url)
                ...     print(result.description)
                ...     print(result.content)

        Args:
            name: The name of the agent (optional).
            api_key: The API key to use for the search engine.
            rate_limit: The number of seconds to wait between requests for webpage content.
            text_only: If True, only the text of the webpage is returned. If False, the entire HTML is returned.
            body_only: If True, only the body of the webpage is returned. If False, the entire HTML is returned.
            use_browser: If True, the webpage is rendered using a headless browser, allowing javascript to run and
                hydrate the page. If False, the webpage is scraped as-is.
            wait_for_selector: The selector to wait for before returning the HTML. Useful for when you know something
                should be on the page, but it is not there yet since it needs to be rendered by javascript. Only used if
                use_browser is True.

        """
        super().__init__(name=name)

        self.api_key = api_key
        self.rate_limit = rate_limit

        self.webpage_agent = WebpageAgent()
        self.session = requests.Session()

        # Parameters for the WebpageAgent
        self.text_only = text_only
        self.body_only = body_only
        self.use_browser = use_browser
        self.wait_for_selector = wait_for_selector

    def __repr__(self):
        return f"WebSearchAgent(name={self.name})"

    @staticmethod
    def _prepare_url(base_url: str, params: Dict) -> str:
        """
        This method prepares a URL for a request.

        Args:
            base_url: The base url.
            params: A dictionary of parameters to use for the request.

        Returns:
            The prepared URL.

        """
        pass

    @staticmethod
    def _handle_errors(res: requests.Response) -> None:
        """
        This method handles errors that occur during a request.

        Args:
            res: The response from the request.

        """
        pass

    def _send_request(
        self, base_url: str, headers: Dict = None, params: Dict = None
    ) -> Dict:
        """
        This method sends a request to a URL.

        Args:
            base_url: The base URL to send the request to.
            headers: A dictionary of headers to use for the request.
            params: A dictionary of parameters to use for the request.

        Returns:
            The response from the request.

        """
        pass

    def search_brave(self, query: str, **kwargs) -> List[WebSearchResult]:
        """
        This method performs a web search using Brave.

        Get an API key here (credit card required):
        https://api.search.brave.com/register

        Args:
            query: The query to search for.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A list of WebSearchResult objects.

        """
        pass

    def search_google(
        self, query: str, custom_search_engine_id: str = None, **kwargs
    ) -> List[WebSearchResult]:
        """
        This method performs a web search using Google.

        Get an API key here:
        https://developers.google.com/custom-search/v1/overview

        You must create a custom search engine and pass its ID. To create or view custom search engines, visit:
        https://programmablesearchengine.google.com/u/1/controlpanel/all

        Args:
            query: The search query.
            custom_search_engine_id: The ID of the custom search engine to use.
            **kwargs: Any additional keyword arguments to pass to the API.

        Returns:
            A list of WebSearchResult objects.

        """
        pass


class RSSAgent(Agent):

    def __init__(self, name: str = "", url: str = None, **kwargs):
        """
        Create a RSSAgent

        This agent helps you read data from RSS feeds.

        Args:
            name: The name of the agent.
            url: The URL of the RSS feed.
            **kwargs: Any additional keyword arguments to pass to feedparser.parse(). You may need to pass a user agent
                header or other headers for some RSS feeds. See https://feedparser.readthedocs.io/en/latest/http.html.

        Examples:

            Read an RSS feed once, passing a user agent header:
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs', agent="it's me!")
                >>> data = agent.read()

            Poll the arXiv CS RSS feed every 60 seconds:
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs')
                >>> with agent.poll(interval=60) as poller:
                >>>     for data in poller():
                >>>         print(data)

            Poll the arXiv CS RSS feed every 60 seconds and stop after 5 minutes:
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs')
                >>> def poll_helper(p: Callable[[], Generator[List[Dict], None, None]]):
                >>>     for data in poller():
                >>>         print(data)
                >>> with agent.poll(interval=60) as poller:
                >>>     t = Thread(target=poll_helper, kwargs={'p': poller})
                >>>     t.start()
                >>>     time.sleep(300)
                >>> t.join()

            Poll and print the data and polling time after each update is received.
                >>> from phasellm.agents import RSSAgent
                >>> agent = RSSAgent(url='https://arxiv.org/rss/cs')
                >>> with agent.poll(interval=60) as poller:
                >>>     for data in poller():
                >>>         print(f'data: {data}')
                >>>         print(f'polling time: {agent.poll_time}')

        """
        if not url:
            raise Exception("Must provide a URL for the RSSAgent.")

        super().__init__(name=name)

        self.url = url
        self.kwargs = kwargs

        # Private attribute for tracking polling state of the agent.
        self._polling = False
        self._poll_start_time = None
        self._poll_end_time = None

    def __repr__(self):
        return f"RSSAgent(name={self.name})"

    @staticmethod
    def _yield_data(queue: Queue) -> Generator[List[Dict], None, None]:
        """
        This method is responsible for yielding data from the queue. It stops generating when it receives None.

        Args:
            queue: The queue to yield data from.

        Returns:
            A generator that yields data from the queue.

        """
        pass

    def _poll_thread(self, queue: Queue, interval: int = 60) -> None:
        """
        This method is responsible for polling the RSS feed and putting new data in the queue.

        Args:
            queue: The queue to put data in.
            interval: The number of seconds to wait between polls.

        """
        pass

    def read(self) -> List[Dict]:
        """
        This method reads data from an RSS feed.

        Returns:
            A list of dictionaries containing the data from the RSS feed.

        """
        pass

    @contextmanager
    def poll(
        self, interval: int = 60
    ) -> Generator[Callable[[], Generator[List[str], None, None]], None, None]:
        """
        This method polls an RSS feed for new data.

        Args:
            interval: The number of seconds to wait between polls.

        Returns:
            A generator that yields a list of dictionaries containing the data from the RSS feed.

        """
        pass

    @property
    def poll_time(self) -> timedelta:
        """
        This method calculates the amount of time the agent has been polling.

        Returns:
            A timedelta object.

        """
        pass
