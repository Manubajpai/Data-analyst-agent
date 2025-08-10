import io
import sys
import base64
import duckdb
import pandas as pd
import requests
from langchain.tools import tool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bs4 import BeautifulSoup 
import lxml
import statsmodels

@tool
def python_code_interpreter(code: str) -> dict:
    """
    Executes a python script in a sandboxed environment and returns the result.
    The script MUST assign its final answer (a dictionary or a list) to a variable called `final_result`.
    This tool has access to common data science libraries (pandas, numpy, matplotlib, etc.).
    Use this single tool to perform all steps of a task, including data acquisition, cleaning,
    analysis, and visualization.
    """
    local_vars = {
        "pd": pd, "plt": plt, "np": np, "io": io, "base64": base64,
        "requests": requests, "BeautifulSoup": BeautifulSoup, "duckdb": duckdb
    }
    
    try:
        exec(code, local_vars)
        if 'final_result' in local_vars:
            return {"result": local_vars['final_result']}
        else:
            raise ValueError("The script did not assign a value to the 'final_result' variable.")

    except Exception as e:
        plt.close('all')

        return {"error": f"Error executing code: {type(e).__name__} - {str(e)}"}