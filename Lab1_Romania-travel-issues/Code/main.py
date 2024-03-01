import module


graph = module.Graph('Romania.txt')
file = open("res.txt", 'w').close() # 清空文本
graph.dijkstra('a', 'b')
graph.dijkstra('f', 'd')
graph.dijkstra('m', 's')
city_a = input("请输入起始城市    ")
city_b = input("请输入终点城市    ")
graph.dijkstra(city_a, city_b)
