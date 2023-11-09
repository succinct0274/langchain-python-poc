from typing import Any, List, Optional
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
import sys
from queue import Queue

class AgentQueueCallbackHandler(FinalStreamingStdOutCallbackHandler):

    def __init__(self, queue: Queue, *, answer_prefix_tokens: List[str] | None = None, strip_tokens: bool = True, stream_prefix: bool = False) -> None:
        self.queue = queue
        super().__init__(answer_prefix_tokens=answer_prefix_tokens, strip_tokens=strip_tokens, stream_prefix=stream_prefix)
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                for t in self.last_tokens:
                    self.queue.put({
                        'event': 'message',
                        'data': t,
                        'retry': 10,
                    })
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            self.queue.put({
                'event': 'message',
                'data': token,
                'retry': 10,
            })
