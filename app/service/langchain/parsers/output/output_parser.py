from langchain.agents.conversational_chat.output_parser import ConvoOutputParser

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": string, \\\\ The action to take. Must be one of {tool_names}
    "action_input": string \\\\ The input to the action
}}}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": "Final Answer",
    "action_input": "Final Answer: string" \\\\ You should put what you want to return to use here.
}}}}
```"""

class CustomConvoOutputParser(ConvoOutputParser):

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS