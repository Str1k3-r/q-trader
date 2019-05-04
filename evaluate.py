import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
actions = []
total_profit_graph = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		actions.append(("BUY", data[t]))
		agent.inventory.append(data[t])
		print("Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		actions.append(("SELL", data[t]))
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
		total_profit_graph.append(total_profit)

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		x = [i for i in range(len(total_profit_graph))]
		plt.plot(x, total_profit_graph)
		plt.title("Total Profits " + stock_name + "_" + model_name)
		plt.legend()
		plt.show()
		plt.savefig("tp_" + stock_name + "_" + model_name + ".png")

		print("--------------------------------")
		print(stock_name + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
