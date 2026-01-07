
instr = """
**Output format:**
Your output response should be provided in the following format:

```json
{
    "answer": Union[str, int, float, list, pd.Series, dict, set] <Mention the answer for the query>
    "dtype": str <Mention the data type of the answer you provided. It must belong to one of the options listed above.>
}
```

## Instructions
1) You must provide your answer in one of the following data types: "int", "float", "str", "list", "dict", "set" or "pd.Series".
# Rules on choosing the required dtype:
- If your answer needs to preserve order with only the relevant items, mention them as a list.
- If your answer does not have to preserve order with only the relevant items, mention them as a set.
- If your answer contains items along with their values where the order needs to be preserved, write the answer as a dictionary and mention the dtype as "pd.Series".
- If your answer contains items along with their values where the order need not be preserved, mention them as "dict".
2) In case the answer is in percentage, mention only the figure along with its datatype without the `%` sign.
3) You should not use the code variables in your final response. Only provide the final answer in the specified format.
4) Enclose the dtype value in double quotes.
"""

python_instr = """
**Python code format:**
The program you provide should be presented in the following manner:
```python
<Complete Python program that answers the query>
```
There should be only one code block in your response containing the complete Python program.
"""

def get_response_format() -> str:
    return instr

def get_python_response_format() -> str:
    return python_instr


