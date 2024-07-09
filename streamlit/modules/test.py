from langchain_helper import get_chain

chain = get_chain()
print(f'this is answ {chain.invoke("list the movie names")}')