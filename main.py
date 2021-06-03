from time import time

from network import Network

# edit this:
params = {
    "mini batch size": 60,
    "learning rate": 0.1,
    "iterations": 100000,
    "submission output file": "submission.csv"
}

network = Network(batch_size=params["mini batch size"], learning_rate=params["learning rate"])

start = time()
network.gradient_descent(params["iterations"])
end = time()

network.make_submission(params["submission output file"])

print(f"Submission file in \'{params['submission output file']}\'")
print(f"Time elapsed: {round(end - start, 2)} seconds")