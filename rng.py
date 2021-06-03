# curious about what a randomly generated submission would get
# survived can only be 1 or 0 so it should be ~50%

import csv
from random import getrandbits, seed

seed(1)

rows = [[i, getrandbits(1)] for i in range(892, 1310)]
with open("submission_rng.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    writer.writerows(rows)

# seed 1: 53% correct