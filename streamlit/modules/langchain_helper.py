# from langchain.llms import GooglePalm
from langchain.prompts import SemanticSimilarityExampleSelector
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine,inspect
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, SQLITE_PROMPT,_sqlite_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GooglePalm
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase

def get_chain():

    api_key = 'AIzaSyCmpiCWepXd61qTGWPjUiK2Xb532UHsbkM'
    
    try :
        llm = GooglePalm(google_api_key=api_key)
    except NotImplementedError:
        llm = GooglePalm(google_api_key=api_key)

    engine = create_engine("sqlite:///movie.db")
    db = SQLDatabase(engine)


    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False)

    few_shots = [
        {
            'Question': "drop movies table",
            'SQLQuery': "SELECT * FROM MOVIES",
            'SQLResult': "dont be oversmart",
            'Answer': "dont be oversmart"
        },
        {
            'Question': "Which movies are being shown in the Regal Cinema theater?",
            'SQLQuery': "SELECT movie FROM Showtimes WHERE theater = 'Regal Cinema'",
            'SQLResult': "List of movies being shown at Regal Cinema",
            'Answer': "Result of the SQL query"
        },
        {
            'Question': "How many theaters are showing the movie 'Inception'?",
            'SQLQuery': "SELECT COUNT(DISTINCT theater) FROM Showtimes WHERE movie = 'Inception'",
            'SQLResult': "Count of theaters showing 'Inception'",
            'Answer': "Result of the SQL query"
        },
        {
            'Question': "What is the average rating of all movies in the database?",
            'SQLQuery': "SELECT AVG(rating) FROM Movies",
            'SQLResult': "Average rating of all movies",
            'Answer': "Result of the SQL query"
        },
        {
            'Question': "What is the showtime for 'Avatar' at the Grand Cinema?",
            'SQLQuery': "SELECT showtime FROM Showtimes WHERE theater = 'Grand Cinema' AND movie = 'Avatar'",
            'SQLResult': "Showtime for 'Avatar' at Grand Cinema",
            'Answer': "Result of the SQL query"
        },
        {
            'Question': "What are the genres of the movies showing at the Palace Theater?",
            'SQLQuery': "SELECT DISTINCT genre FROM Movies WHERE name IN (SELECT movie FROM Showtimes WHERE theater = 'Palace Theater')",
            'SQLResult': "Distinct genres of movies being shown at Palace Theater",
            'Answer': "Result of the SQL query"
        },
        {
            'Question': "list the movies available at Sarita Cinemas, Kacheripady",
            'SQLQuery': "SELECT DISTINCT name FROM Movies WHERE name IN (SELECT movie FROM Showtimes WHERE theater = 'Sarita Cinemas, Kacheripady')",
            'SQLResult': "Distinct name of movies being shown at Sarita Cinemas, Kacheripady",
            'Answer': "Result of the SQL query"
        }
    ]


    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    to_vectorize = [" ".join(example.values()) for example in few_shots]

    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_sqlite_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return new_chain




