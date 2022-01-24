def add(a,b):
    return a+b
def sub(a,b):
    return a-b
class Student:
    def __init__(self,name,roll):
        self.name,self.roll=name,roll
    def get_details(self):
        return self.name,self.roll
