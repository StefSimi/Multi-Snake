class Food:
    expirationCounter = 0

    def __init__(self, NM, NExp, fruit_position, poisoned):
        self.NM = NM
        self.NExp = NExp * 60
        self.fruit_position = fruit_position
        self.poisoned = poisoned