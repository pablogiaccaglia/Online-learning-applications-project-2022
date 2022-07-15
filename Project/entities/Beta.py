class Beta:
    # Class containing the beta parameters for the learning algorithm (at the beginning uniform distribution)
    def __init__(self, copy: 'Beta' = None):
        if copy:
            self.a = copy.a
            self.b = copy.b
            self.played = copy.played
        else:
            self.a = 1
            self.b = 1
            self.played = 0

    def __str__(self):
        # return f"Beta with alpha:{self.a}, beta:{self.b}, repetitions:{self.played}"
        return f"a:{self.a}, b:{self.b}, r:{self.played}"
