from .langchain_helper import get_chain


class TextToSql:
    def __init__(self) -> None:
    
        self._chain = get_chain()

    def get_answer(self,question):
        return self._chain.invoke(question)

