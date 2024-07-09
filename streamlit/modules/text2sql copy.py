from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine,inspect
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX,SQLITE_PROMPT
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX,SQLITE_PROMPT

class TextToSql:
    def __init__(self) -> None:
        self._api_key = 'AIzaSyCmpiCWepXd61qTGWPjUiK2Xb532UHsbkM'
        try:
            self._llm = GooglePalm(google_api_key=self._api_key, temperature=0.0)
        except NotImplementedError:
            self._llm = GooglePalm(google_api_key=self._api_key, temperature=0.0)
        self._engine = create_engine("sqlite:///movie.db")
        self._db = SQLDatabase(self._engine)
        self._chain = SQLDatabaseChain.from_llm(self._llm, self._db)

    def get_answer(self,question):
        return self._chain(question)

