
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
"""

def get_response_format() -> str:
    return instr


