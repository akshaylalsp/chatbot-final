
from modules.text2sql import TextToSql
from modules.SetupDb import SetupDb

class TheatreChatBot():
    def __init__(self,location) -> None:
        self.text2sql = TextToSql()
        self.db = SetupDb(location=location)


    def setup_database(self,location):
        self.db.insert_into_db(location=location)

    def get_recommendation():
        pass

    def execute_result(self,nl_question):
        result = self.text2sql.get_answer(nl_question)
        return result
    