import random
import string

print(''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=9)))
