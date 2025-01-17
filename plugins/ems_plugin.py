import json
import logging
import time
from typing import Any, Optional, Union

import requests
import urllib3

from plugins import plugin
from result import RequestResult

urllib3.disable_warnings()
"""
Example plugin config.yaml:

plugin: "ems_plugin"
plugin_options:
  # provide kubernetes service FQDN 
  host: "model-serving-example-intents-bentoml-c0fcdc58.model-serving-ml-apac-beagle.svc.cluster.local"
  endpoint: "/predictions/model"
"""

required_args = ["host", "endpoint"]

logger = logging.getLogger("user")

def deepget(obj: Union[dict, list], *path: Any, default: Any = None) -> Any:
    """
    Acts like .get() but for nested objects.

    Each item in path is recusively indexed on obj. For path of length N,
      obj[path[0]][path[1]]...[path[N-1]][path[N]]

    :param obj: root object to index
    :param path: ordered list of keys to index recursively
    :param default: the default value to return if an indexing fails
    :returns: result of final index or default if Key/Index Error occurs
    """
    current = obj
    for pos in path:
        try:
            current = current[pos]
        except (KeyError, IndexError):
            return default
    return current

class EMSPlugin(plugin.Plugin):
    def __init__(self, args):
        self._parse_args(args)

    def _parse_args(self, args):
        for arg in required_args:
            if arg not in args:
                logger.error("Missing plugin arg: %s", arg)

        self.request_func = self.request_http # no streaming requests supported for this plugin

        self.host = "http://" + args.get("host") + args.get("endpoint")

        logger.debug("Host: %s", self.host)
        

    def request_http(self, query: dict, user_id: int, test_end_time: float = 0):

        result = RequestResult(user_id, query.get("text"))

        result.start_time = time.time()

        headers = {"Content-Type": "application/json"}


        logger.debug("Query: %s", query)
        data = {"inputs":{"text":query.get("text")}} # this is the format that the EMS model serving expects
        response = None
        try:
            response = requests.post(self.host, headers=headers, json=data, verify=False)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as err:
            result.end_time = time.time()
            result.error_text = repr(err)
            if response is not None:
                result.error_code = response.status_code
            logger.exception("Connection error")
            return result
        except requests.exceptions.HTTPError as err:
            result.end_time = time.time()
            result.error_text = repr(err)
            if response is not None:
                result.error_code = response.status_code
            logger.exception("HTTP error")
            return result

        result.end_time = time.time()

        ###########################################
        # DO NOT CALL time.time BEYOND THIS POINT #
        ###########################################

        logger.debug("Response: %s", json.dumps(response.text))

        try:
            message = json.loads(response.text)
            error = message.get("error")
            if error is None:
                result.output_text = "" # not storing the output text to save memory
            else:
                result.error_code = response.status_code
                result.error_text = error
                logger.error("Error received in response message: %s", error)
        except json.JSONDecodeError:
            logger.exception("Response could not be json decoded: %s", response.text)
            result.error_text = f"Response could not be json decoded {response.text}"

        # For non-streaming requests we are keeping output_tokens_before_timeout and output_tokens same.
        result.output_tokens_before_timeout = result.output_tokens
        result.calculate_results()

        return result


    def streaming_request_http(self, query: dict, user_id: int, test_end_time: float):
       pass
