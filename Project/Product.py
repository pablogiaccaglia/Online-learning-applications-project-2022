class Product:
    def __init__(self, id, price, secondary_list=[]):
        self.id = id
        self.price = price  # correspond to margin
        self.secondary_list = secondary_list  # list of secondary products

    def __str__(self):
        return f"Product {self.id}"

    def add_secondary(self, product):
        """ Set the ordered list of items to visit as secondary after this is shown as first"""
        self.secondary_list.append(product)