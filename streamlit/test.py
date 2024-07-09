import TheatreChatBot

instance = TheatreChatBot.TheatreChatBot('kochi')

result = instance.execute_result("top movies")

print(result)